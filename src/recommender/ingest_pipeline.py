"""Ingest pipeline — main entry point: scan dir, extract features, store to SQLite.

Stages (see ARCHITECTURE.md):
  1. Ingest — scan directory for audio files
  2. Extract — run feature extraction scripts via subprocess
  3. Store — persist results to SQLite
"""

import json
import shutil
import sqlite3
from pathlib import Path

from tinytag import TinyTag

from .track import Track
from .feature_extractor import extract_essentia, extract_clap, run_batch, run_clap_batch, ensure_mood_models
from .feature_converter import convert


# Supported audio extensions (lowercase)
AUDIO_EXTENSIONS = {".mp3", ".flac", ".ogg", ".wav", ".m4a"}


def scan_directory(music_dir: Path) -> list[Path]:
    """Recursively scan a directory for audio files.

    Args:
        music_dir: root directory to scan

    Returns:
        list of absolute paths to audio files
    """
    results: list[Path] = []
    for entry in music_dir.rglob("*"):
        if entry.is_file() and entry.suffix.lower() in AUDIO_EXTENSIONS:
            results.append(entry.resolve())
    return results


def is_audio_file(path: Path) -> bool:
    """Check if a file has a supported audio extension."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def read_metadata(audio_file: Path) -> tuple[str | None, str | None]:
    """Read title and artist from ID3/Vorbis tags.

    Args:
        audio_file: path to the audio file

    Returns:
        (title, artist) tuple — each may be None if tag is missing
    """
    try:
        tag = TinyTag.get(str(audio_file))
        title = tag.title if tag.title else None
        artist = tag.artist if tag.artist else None
        return title, artist
    except Exception:
        return None, None


def init_database(db_path: Path) -> sqlite3.Connection:
    """Create the tracks table if it does not exist.

    Args:
        db_path: path to the SQLite database file

    Returns:
        open connection (caller should close when done)
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            title TEXT,
            artist TEXT,
            duration_sec REAL,
            feature_json TEXT,
            clap_embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def _stored_version(feature_json: str | None) -> str | None:
    """Return the extractor version recorded in a stored sidecar, if any."""
    try:
        return json.loads(feature_json or "{}").get("version")
    except (json.JSONDecodeError, AttributeError):
        return None


def _current_extractor_version() -> str | None:
    """Return the current extractor version, or None if undeterminable.

    Single source of truth is EXTRACTOR_VERSION in scripts/extract_essentia.py.
    Returns None (version check disabled) when the scripts package is not
    importable, so ingestion degrades to the legacy skip-if-present behaviour.
    """
    try:
        from scripts.extract_essentia import EXTRACTOR_VERSION

        return EXTRACTOR_VERSION
    except Exception:
        return None


def _run_extraction(
    conn: sqlite3.Connection,
    track_id: int,
    audio_file: Path,
    track: Track,
    extract_clap_flag: bool = False,
    timeout: int | None = None,
    dsp_timeout: int | None = None,
    mood_timeout: int | None = None,
    no_mood: bool = False,
) -> None:
    """Run Essentia (+optional CLAP) extraction and UPDATE the track row.

    Two-phase extraction:
      Phase 1: DSP features (always runs)
      Phase 2: Mood scores (skipped if no_mood, retried on next run if failed)

    Updates feature_json, duration_sec, title and artist in place.
    """
    from .feature_extractor import timeout_for

    # Refresh metadata (cheap; picks up retagged files on re-extract)
    title, artist = read_metadata(audio_file)
    track.set_title(title)
    track.set_artist(artist)
    conn.execute(
        "UPDATE tracks SET title = ?, artist = ? WHERE id = ?",
        (title, artist, track_id),
    )

    # Phase 1: DSP extraction
    essentia_output = Path(f"essentia_{track_id}.json")
    dsp_eff = timeout if timeout is not None else timeout_for("dsp", dsp_timeout)
    try:
        extract_essentia(audio_file, essentia_output, timeout=dsp_eff, no_mood=True)
        feature_json = json.loads(essentia_output.read_text())
        duration = float(feature_json.get("duration_sec", 0.0) or 0.0)
        conn.execute(
            "UPDATE tracks SET feature_json = ?, duration_sec = ? WHERE id = ?",
            (json.dumps(feature_json), duration, track_id),
        )
        track.set_feature_json(json.dumps(feature_json))
        track.set_features(convert(track.get_feature_json()))
        track.set_duration_sec(duration)
    except Exception:
        # DSP failed: drop any partial sidecar so Phase 2 / retries never
        # read stale data, then propagate. On success the sidecar is kept:
        # Phase 2 (--mood-only) needs the file to exist, and its own
        # finally block cleans it up.
        if essentia_output.exists():
            essentia_output.unlink()
        raise

    # Phase 2: Mood extraction (unless no_mood)
    if not no_mood:
        mood_eff = timeout if timeout is not None else timeout_for("mood", mood_timeout)
        try:
            extract_essentia(audio_file, essentia_output, timeout=mood_eff, mood_only=True)
            mood_json = json.loads(essentia_output.read_text())
            parsed = json.loads(track.get_feature_json() or "{}")
            if "mood" in mood_json:
                parsed["mood"] = mood_json["mood"]
            conn.execute(
                "UPDATE tracks SET feature_json = ? WHERE id = ?",
                (json.dumps(parsed), track_id),
            )
            track.set_feature_json(json.dumps(parsed))
        except Exception as e:
            print(f"  mood failed for {audio_file.name} (DSP kept): {e}")
            # leave mood NULL; next run retries
        finally:
            if essentia_output.exists():
                essentia_output.unlink()

    # Optionally run CLAP extraction
    if extract_clap_flag:
        clap_output = Path(f"clap_{track_id}.json")
        try:
            extract_clap(audio_file, clap_output)
            clap_data = json.loads(clap_output.read_text())
            conn.execute(
                "UPDATE tracks SET clap_embedding = ? WHERE id = ?",
                (json.dumps(clap_data.get("embedding", [])), track_id),
            )
            track.set_clap_embedding(clap_data.get("embedding"))
        finally:
            if clap_output.exists():
                clap_output.unlink()

    conn.commit()


def process_file(
    conn: sqlite3.Connection,
    audio_file: Path,
    extract_clap_flag: bool = False,
    force: bool = False,
    timeout: int | None = None,
    dsp_timeout: int | None = None,
    mood_timeout: int | None = None,
    no_mood: bool = False,
) -> tuple[Track, str]:
    """Process a single audio file: insert metadata, run extraction, store features.

    Args:
        conn: open SQLite connection
        audio_file: path to the audio file
        extract_clap_flag: whether to also run CLAP embedding extraction
        force: re-extract even when a current-version row already exists
        timeout: per-file subprocess timeout in seconds (None uses default 180)
        dsp_timeout: DSP-only timeout override (None uses 60)
        mood_timeout: mood-only timeout override (None uses 180)
        no_mood: if True, skip mood extraction entirely

    Rows whose stored extractor version differs from the current
    EXTRACTOR_VERSION are re-extracted automatically (stale features).

    Returns:
        Tuple of (Track, status) where status is "skipped", "re-extracted", or "processed"
    """
    path_str = str(audio_file)

    # Check if already ingested
    row = conn.execute(
        "SELECT * FROM tracks WHERE file_path = ?", (path_str,)
    ).fetchone()
    if row is not None:
        stored = _stored_version(row[5])
        current = _current_extractor_version()
        stale = row[5] is None or (
            current is not None and stored != current
        )
        if not force and not stale:
            # Return existing track (no extraction needed)
            track = Track(
                id=row[0],
                file_path=Path(row[1]),
                title=row[2],
                artist=row[3],
                duration_sec=row[4] if row[4] is not None else 0.0,
                features=convert(row[5]) if row[5] else None,
                feature_json=row[5] if row[5] is not None else None,
                clap_embedding=None if row[6] is None else json.loads(row[6]),
            )
            return track, "skipped"
        track = Track(
            id=row[0],
            file_path=audio_file,
            title=row[2],
            artist=row[3],
            duration_sec=row[4] if row[4] is not None else 0.0,
            features=convert(row[5]) if row[5] else None,
            feature_json=row[5] if row[5] is not None else None,
            clap_embedding=None if row[6] is None else json.loads(row[6]),
        )
        _run_extraction(conn, row[0], audio_file, track, extract_clap_flag=extract_clap_flag, timeout=timeout, dsp_timeout=dsp_timeout, mood_timeout=mood_timeout, no_mood=no_mood)
        return track, "re-extracted"

    # Insert metadata row
    title, artist = read_metadata(audio_file)
    conn.execute(
        "INSERT INTO tracks (file_path, title, artist, duration_sec) VALUES (?, ?, ?, ?)",
        (path_str, title, artist, 0.0),
    )
    conn.commit()
    track_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    track = Track(id=track_id, file_path=audio_file)

    _run_extraction(conn, track_id, audio_file, track, extract_clap_flag=extract_clap_flag, timeout=timeout, dsp_timeout=dsp_timeout, mood_timeout=mood_timeout, no_mood=no_mood)
    return track, "processed"


def run_pipeline(music_dir: Path, db_path: Path, extract_clap: bool = False, force: bool = False, timeout: int | None = None, dsp_timeout: int | None = None, mood_timeout: int | None = None, no_mood: bool = False, batch: bool = False, models_dir: Path | None = None) -> list[tuple[Track, str]]:
    """Run the full ingestion pipeline.

    Args:
        music_dir: root directory containing audio files
        db_path: path to the SQLite database file
        extract_clap: whether to run CLAP embedding extraction
        force: re-extract tracks even when a current-version row exists
        timeout: per-file subprocess timeout in seconds (None uses default 180)
        dsp_timeout: DSP-only timeout override (None uses 60)
        mood_timeout: mood-only timeout override (None uses 180)
        no_mood: if True, skip mood extraction entirely
        batch: if True, use batch worker (single TF graph load)
        models_dir: directory for mood classification models (passed to batch worker)

    Returns:
        list of Track objects that were processed
    """
    music_dir = music_dir.resolve()
    if not music_dir.is_dir():
        raise ValueError(f"Not a directory: {music_dir}")

    # Best-effort prefetch of mood models (warn but don't fail)
    if not no_mood:
        try:
            ensure_mood_models(models_dir=models_dir)
        except Exception as e:
            print(f"Warning: mood model prefetch failed (will retry per-track): {e}")

    audio_files = scan_directory(music_dir)
    print(f"Found {len(audio_files)} audio files")

    conn = init_database(db_path)
    try:
        if batch:
            return _run_pipeline_batch(conn, audio_files, db_path, extract_clap=extract_clap, force=force, timeout=timeout, dsp_timeout=dsp_timeout, mood_timeout=mood_timeout, no_mood=no_mood, models_dir=models_dir)

        tracks: list[Track] = []
        for audio_file in audio_files:
            try:
                track, status = process_file(conn, audio_file, extract_clap_flag=extract_clap, force=force, timeout=timeout, dsp_timeout=dsp_timeout, mood_timeout=mood_timeout, no_mood=no_mood)
                tracks.append(track)
                print(f"  {status}: {audio_file.name} (id={track.get_id()})")
            except Exception as e:
                print(f"  failed: {audio_file.name} — {e}")
                # Continue to next track — single failure does not halt pipeline
        return tracks
    finally:
        conn.close()


def _run_pipeline_batch(
    conn: sqlite3.Connection,
    audio_files: list[Path],
    db_path: Path,
    extract_clap: bool = False,
    force: bool = False,
    timeout: int | None = None,
    dsp_timeout: int | None = None,
    mood_timeout: int | None = None,
    no_mood: bool = False,
    models_dir: Path | None = None,
) -> list[tuple[Track, str]]:
    """Batch path: write manifest, run batch worker once, store results.

    Falls back to single-track extraction per-file on batch failure.
    """
    import tempfile
    from .feature_extractor import extract_essentia, timeout_for

    # Filter to files needing extraction
    pending: list[Path] = []
    existing_tracks: list[tuple[Track, str]] = []
    for audio_file in audio_files:
        path_str = str(audio_file)
        row = conn.execute(
            "SELECT * FROM tracks WHERE file_path = ?", (path_str,)
        ).fetchone()
        if row is not None and not force:
            stored = _stored_version(row[5])
            current = _current_extractor_version()
            stale = row[5] is None or (current is not None and stored != current)
            if not stale:
                existing_tracks.append((Track(
                    id=row[0],
                    file_path=audio_file,
                    title=row[2],
                    artist=row[3],
                    duration_sec=row[4] if row[4] is not None else 0.0,
                    features=convert(row[5]) if row[5] else None,
                    feature_json=row[5] if row[5] is not None else None,
                    clap_embedding=None if row[6] is None else json.loads(row[6]),
                ), "skipped"))
                continue
        pending.append(audio_file)

    if not pending:
        if not extract_clap:
            return existing_tracks
        # If extract_clap is True, return early only if all existing tracks
        # already have a non-NULL clap_embedding and force is False.
        # Otherwise fall through so the CLAP pass can populate missing embeddings.
        all_have_clap = True
        for track_obj, _status in existing_tracks:
            path_str = str(track_obj.get_file_path())
            row = conn.execute(
                "SELECT id, clap_embedding FROM tracks WHERE file_path = ?",
                (path_str,),
            ).fetchone()
            if row is not None and row[1] is None:
                all_have_clap = False
                break
        if all_have_clap and not force:
            return existing_tracks
        # Fall through: pending is empty, CLAP pass will handle CLAP work

    # Write manifest
    if pending:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            manifest_entries = []
            for i, af in enumerate(pending):
                manifest_entries.append({
                    "audio_path": str(af),
                    "output_path": str(Path(f.name).parent / f"batch_{i:04d}_{af.stem}.json"),
                })
            json.dump(manifest_entries, f)
            manifest_path = Path(f.name)

        summary_path = Path(str(manifest_path) + ".summary.json")

        tracks: list[tuple[Track, str]] = list(existing_tracks)
        try:
            summary = run_batch(manifest_path, summary_path, timeout=timeout, no_mood=no_mood, models_dir=models_dir)
        except Exception as e:
            print(f"  batch worker failed: {e} — falling back to single-track")
            summary = {"ok": [], "failed": []}
            # On batch failure, fall back to single-track for each pending file
            for af in pending:
                manifest_entry = next(
                    (m for m in manifest_entries if m["audio_path"] == str(af)), None
                )
                if manifest_entry:
                    summary["failed"].append({
                        "output": manifest_entry["output_path"],
                        "error": str(e),
                    })
        finally:
            # Clean up manifest
            manifest_path.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)
    else:
        # No pending entries — empty manifest, empty summary
        manifest_entries = []
        tracks: list[tuple[Track, str]] = list(existing_tracks)
        summary = {"ok": [], "failed": []}

    # Process ok entries — read sidecars and store
    ok_set = set(summary.get("ok", []))

    for entry in manifest_entries:
        audio_file = Path(entry["audio_path"])
        output_file = Path(entry["output_path"])

        if entry["output_path"] in ok_set and output_file.exists():
            # Read sidecar and store to DB
            try:
                sidecar = json.loads(output_file.read_text())
                output_file.unlink(missing_ok=True)
            except Exception as e:
                print(f"  failed reading sidecar for {audio_file.name}: {e}")
                output_file.unlink(missing_ok=True)
                continue

            path_str = str(audio_file)
            row = conn.execute(
                "SELECT * FROM tracks WHERE file_path = ?", (path_str,)
            ).fetchone()

            title, artist = read_metadata(audio_file)
            duration = float(sidecar.get("duration_sec", 0.0) or 0.0)

            if row is not None:
                conn.execute(
                    "UPDATE tracks SET feature_json = ?, duration_sec = ?, title = ?, artist = ? WHERE id = ?",
                    (json.dumps(sidecar), duration, title, artist, row[0]),
                )
                track_id = row[0]
            else:
                conn.execute(
                    "INSERT INTO tracks (file_path, title, artist, duration_sec, feature_json) VALUES (?, ?, ?, ?, ?)",
                    (path_str, title, artist, duration, json.dumps(sidecar)),
                )
                track_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            track = Track(id=track_id, file_path=audio_file, title=title, artist=artist, duration_sec=duration, features=convert(json.dumps(sidecar)), feature_json=json.dumps(sidecar))
            tracks.append((track, "re-extracted" if row is not None else "processed"))
            print(f"  batch processed: {audio_file.name} (id={track_id})")
        else:
            # Failed entry — fall back to single-track
            output_file.unlink(missing_ok=True)
            try:
                track, status = process_file(conn, audio_file, extract_clap_flag=extract_clap, force=force, timeout=timeout, dsp_timeout=dsp_timeout, mood_timeout=mood_timeout, no_mood=no_mood)
                tracks.append((track, status))
                print(f"  {status}: {audio_file.name} (id={track.get_id()}) [fallback single-track]")
            except Exception as e2:
                print(f"  failed: {audio_file.name} — {e2}")

    # --- CLAP batch pass (after Essentia, before commit) ---
    if extract_clap:
        # Collect tracks that need CLAP embedding
        clap_pending: list[str] = []  # file_path strings
        for audio_file in pending:
            path_str = str(audio_file)
            row = conn.execute(
                "SELECT id, clap_embedding FROM tracks WHERE file_path = ?",
                (path_str,),
            ).fetchone()
            if row is not None and (force or row[1] is None):
                clap_pending.append(path_str)

        # Also check tracks that were skipped (existing, no re-extraction)
        for track_obj, _status in existing_tracks:
            path_str = str(track_obj.get_file_path())
            row = conn.execute(
                "SELECT id, clap_embedding FROM tracks WHERE file_path = ?",
                (path_str,),
            ).fetchone()
            if row is not None and (force or row[1] is None):
                clap_pending.append(path_str)

        if clap_pending:
            clap_tmp = Path(tempfile.mkdtemp())
            try:
                clap_manifest_entries = []
                for i, fp in enumerate(clap_pending):
                    audio_p = Path(fp)
                    clap_manifest_entries.append({
                        "audio_path": fp,
                        "output_path": str(clap_tmp / f"clap_{i:04d}_{audio_p.stem}.json"),
                    })
                clap_manifest_path = clap_tmp / "clap_manifest.json"
                clap_manifest_path.write_text(json.dumps(clap_manifest_entries))
                clap_summary_path = Path(str(clap_manifest_path) + ".summary.json")

                try:
                    clap_summary = run_clap_batch(clap_manifest_path, clap_summary_path, timeout=timeout)
                except Exception as e:
                    print(f"  batch CLAP worker failed: {e} — leaving clap_embedding NULL")
                    clap_summary = {"ok": [], "failed": []}

                # Store successful CLAP embeddings
                for entry in clap_manifest_entries:
                    output_file = Path(entry["output_path"])
                    if entry["output_path"] in set(clap_summary.get("ok", [])) and output_file.exists():
                        try:
                            clap_sidecar = json.loads(output_file.read_text())
                            embedding = clap_sidecar.get("embedding", [])
                            if len(embedding) != 512:
                                print(f"  CLAP embedding for {Path(entry['audio_path']).name} has {len(embedding)} dims, expected 512 — skipping")
                                output_file.unlink(missing_ok=True)
                                continue
                            conn.execute(
                                "UPDATE tracks SET clap_embedding = ? WHERE file_path = ?",
                                (json.dumps(embedding), entry["audio_path"]),
                            )
                        except Exception as e:
                            print(f"  failed reading CLAP sidecar for {Path(entry['audio_path']).name}: {e}")
                    if output_file.exists():
                        output_file.unlink(missing_ok=True)

                # Clean up CLAP manifest + summary
                clap_manifest_path.unlink(missing_ok=True)
                clap_summary_path.unlink(missing_ok=True)
            finally:
                # Clean up CLAP temp directory
                import shutil
                shutil.rmtree(clap_tmp, ignore_errors=True)

    conn.commit()
    return tracks
