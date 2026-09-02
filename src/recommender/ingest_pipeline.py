"""Ingest pipeline — main entry point: scan dir, extract features, store to SQLite.

Stages (see ARCHITECTURE.md):
  1. Ingest — scan directory for audio files
  2. Extract — run feature extraction scripts via subprocess
  3. Store — persist results to SQLite
"""

import json
import sqlite3
from pathlib import Path

from tinytag import TinyTag

from .track import Track
from .feature_extractor import extract_essentia, extract_clap


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


def process_file(conn: sqlite3.Connection, audio_file: Path, extract_clap_flag: bool = False) -> Track:
    """Process a single audio file: insert metadata, run extraction, store features.

    Args:
        conn: open SQLite connection
        audio_file: path to the audio file
        extract_clap_flag: whether to also run CLAP embedding extraction

    Returns:
        the Track object that was inserted
    """
    path_str = str(audio_file)

    # Check if already ingested
    cur = conn.execute("SELECT 1 FROM tracks WHERE file_path = ?", (path_str,))
    if cur.fetchone():
        # Return existing track
        row = conn.execute("SELECT * FROM tracks WHERE file_path = ?", (path_str,)).fetchone()
        return Track(
            id=row[0],
            file_path=Path(row[1]),
            title=row[2],
            artist=row[3],
            duration_sec=row[4] if row[4] is not None else 0.0,
            features=None,
            feature_json=row[5] if row[5] is not None else None,
            clap_embedding=None if row[6] is None else json.loads(row[6]),
        )

    # Insert metadata row
    title, artist = read_metadata(audio_file)
    conn.execute(
        "INSERT INTO tracks (file_path, title, artist, duration_sec) VALUES (?, ?, ?, ?)",
        (path_str, title, artist, 0.0),
    )
    conn.commit()
    track_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    track = Track(id=track_id, file_path=audio_file)

    # Run Essentia extraction
    essentia_output = Path(f"essentia_{track_id}.json")
    try:
        extract_essentia(audio_file, essentia_output)
        feature_json = json.loads(essentia_output.read_text())
        conn.execute(
            "UPDATE tracks SET feature_json = ? WHERE id = ?",
            (json.dumps(feature_json), track_id),
        )
        track.feature_json = json.dumps(feature_json)
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
            track.clap_embedding = clap_data.get("embedding")
        finally:
            if clap_output.exists():
                clap_output.unlink()

    conn.commit()
    return track


def run_pipeline(music_dir: Path, db_path: Path, extract_clap: bool = False) -> list[Track]:
    """Run the full ingestion pipeline.

    Args:
        music_dir: root directory containing audio files
        db_path: path to the SQLite database file
        extract_clap: whether to run CLAP embedding extraction

    Returns:
        list of Track objects that were processed
    """
    music_dir = music_dir.resolve()
    if not music_dir.is_dir():
        raise ValueError(f"Not a directory: {music_dir}")

    audio_files = scan_directory(music_dir)
    print(f"Found {len(audio_files)} audio files")

    conn = init_database(db_path)
    try:
        tracks: list[Track] = []
        for audio_file in audio_files:
            try:
                track = process_file(conn, audio_file, extract_clap_flag=extract_clap)
                tracks.append(track)
                print(f"  processed: {audio_file.name} (id={track.get_id()})")
            except Exception as e:
                print(f"  failed: {audio_file.name} — {e}")
                # Continue to next track — single failure does not halt pipeline
        return tracks
    finally:
        conn.close()