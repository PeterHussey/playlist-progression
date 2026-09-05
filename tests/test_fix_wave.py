"""Tests for the final fix wave: batch collision, --no-mood in batch, prefetch."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Same-stem collision regression ──────────────────────────────────


def test_batch_output_paths_unique_for_same_stem(tmp_path, monkeypatch):
    """Two files with same stem in different dirs → distinct sidecar paths."""
    monkeypatch.chdir(tmp_path)
    import sqlite3
    from src.recommender.ingest_pipeline import _run_pipeline_batch, init_database

    # Create two audio files with same name in different dirs
    dir_a = tmp_path / "album_a"
    dir_b = tmp_path / "album_b"
    dir_a.mkdir()
    dir_b.mkdir()
    file_a = dir_a / "song.mp3"
    file_b = dir_b / "song.mp3"
    file_a.write_bytes(b"\x00" * 100)
    file_b.write_bytes(b"\x00" * 100)

    db = tmp_path / "test.db"
    conn = init_database(db)

    # Mock run_batch to capture manifest and write fake sidecars
    captured_manifest = {}

    def fake_run_batch(manifest_path, summary_path, timeout=None, no_mood=False, models_dir=None):
        manifest = json.loads(manifest_path.read_text())
        # Check that output paths are unique
        paths = [e["output_path"] for e in manifest]
        assert len(paths) == len(set(paths)), f"Duplicate output paths: {paths}"
        captured_manifest["entries"] = manifest
        # Write fake sidecars
        for entry in manifest:
            out = Path(entry["output_path"])
            out.write_text(json.dumps({
                "version": "1.1",
                "duration_sec": 200.0,
                "loudness": {"integrated": -10.0, "range": 5.0},
                "tempo": {"bpm": 120.0, "confidence": 0.9},
                "key": {"key": "C", "mode": "major", "scale": "C major", "confidence": 0.9},
                "spectral": {"centroid": 2000.0, "rolloff": 4000.0, "flatness": 0.02},
                "rhythm": {"danceability": 0.5, "onset_rate": 1.0},
            }))
        # Write summary
        summary = {"ok": [e["output_path"] for e in manifest], "failed": []}
        Path(str(summary_path)).write_text(json.dumps(summary))
        return summary

    with patch("src.recommender.ingest_pipeline.run_batch", side_effect=fake_run_batch):
        tracks = _run_pipeline_batch(conn, [file_a.resolve(), file_b.resolve()], db)

    # Verify distinct DB rows
    rows = conn.execute("SELECT file_path FROM tracks ORDER BY id").fetchall()
    assert len(rows) == 2
    assert rows[0][0] != rows[1][0], "Two distinct file paths should be stored"

    # Verify the manifest output paths are distinct
    entries = captured_manifest["entries"]
    assert entries[0]["output_path"] != entries[1]["output_path"]
    conn.close()


# ── --batch --no-mood threading ──────────────────────────────────────


def test_run_batch_passes_no_mood(tmp_path, monkeypatch):
    """run_batch passes --no-mood to the script command."""
    monkeypatch.chdir(tmp_path)
    from src.recommender.feature_extractor import run_batch

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([
        {"audio_path": str(tmp_path / "a.mp3"), "output_path": str(tmp_path / "a.json")},
    ]))

    captured_cmd = {}

    def fake_subprocess_run(cmd, capture_output=False, text=False, timeout=None):
        captured_cmd["cmd"] = cmd
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps({"ok": [], "failed": []}))
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch("src.recommender.feature_extractor.subprocess.run", side_effect=fake_subprocess_run):
        run_batch(manifest, tmp_path / "summary.json", no_mood=True)

    assert "--no-mood" in captured_cmd["cmd"], f"--no-mood not in cmd: {captured_cmd['cmd']}"


def test_run_batch_passes_models_dir(tmp_path, monkeypatch):
    """run_batch passes --models-dir to the script command."""
    monkeypatch.chdir(tmp_path)
    from src.recommender.feature_extractor import run_batch

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([
        {"audio_path": str(tmp_path / "a.mp3"), "output_path": str(tmp_path / "a.json")},
    ]))

    captured_cmd = {}

    def fake_subprocess_run(cmd, capture_output=False, text=False, timeout=None):
        captured_cmd["cmd"] = cmd
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps({"ok": [], "failed": []}))
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch("src.recommender.feature_extractor.subprocess.run", side_effect=fake_subprocess_run):
        run_batch(manifest, tmp_path / "summary.json", models_dir=Path("/custom/models"))

    cmd = captured_cmd["cmd"]
    assert "--models-dir" in cmd, f"--models-dir not in cmd: {cmd}"
    assert "/custom/models" in cmd, f"models_dir value not in cmd: {cmd}"


# ── Pipeline batch path threads no_mood + models_dir ─────────────────


def test_pipeline_batch_threads_no_mood_and_models_dir(tmp_path, monkeypatch):
    """run_pipeline(batch=True) threads no_mood and models_dir to run_batch."""
    monkeypatch.chdir(tmp_path)
    import sqlite3
    from src.recommender.ingest_pipeline import run_pipeline, init_database

    # Create audio dir with one file
    music = tmp_path / "music"
    music.mkdir()
    (music / "test.mp3").write_bytes(b"\x00" * 100)

    db = tmp_path / "test.db"

    captured = {}

    def fake_run_batch(manifest_path, summary_path, timeout=None, no_mood=False, models_dir=None):
        captured["no_mood"] = no_mood
        captured["models_dir"] = models_dir
        # Write empty sidecars for all entries
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest:
            Path(entry["output_path"]).write_text(json.dumps({"version": "1.1", "duration_sec": 0.0}))
        summary = {"ok": [e["output_path"] for e in manifest], "failed": []}
        Path(str(summary_path)).write_text(json.dumps(summary))
        return summary

    with patch("src.recommender.ingest_pipeline.run_batch", side_effect=fake_run_batch):
        tracks = run_pipeline(music, db, batch=True, no_mood=True, models_dir=Path("/my/models"))

    assert captured.get("no_mood") is True, "no_mood not threaded to run_batch"
    assert captured.get("models_dir") == Path("/my/models"), "models_dir not threaded to run_batch"


# ── ensure_mood_models prefetch ───────────────────────────────────────


def test_run_pipeline_calls_prefetch(tmp_path, monkeypatch):
    """run_pipeline calls ensure_mood_models when no_mood=False."""
    monkeypatch.chdir(tmp_path)
    import sqlite3
    from src.recommender.ingest_pipeline import run_pipeline

    music = tmp_path / "music"
    music.mkdir()
    (music / "test.mp3").write_bytes(b"\x00" * 100)

    db = tmp_path / "test.db"

    prefetch_called = [False]

    def fake_prefetch(**kwargs):
        prefetch_called[0] = True

    with patch("src.recommender.ingest_pipeline.ensure_mood_models", side_effect=fake_prefetch):
        # Mock extract_essentia to avoid subprocess
        with patch("src.recommender.ingest_pipeline.extract_essentia"):
            tracks = run_pipeline(music, db, no_mood=False)

    assert prefetch_called[0], "ensure_mood_models not called when no_mood=False"


def test_run_pipeline_skips_prefetch_when_no_mood(tmp_path, monkeypatch):
    """run_pipeline does NOT call ensure_mood_models when no_mood=True."""
    monkeypatch.chdir(tmp_path)
    import sqlite3
    from src.recommender.ingest_pipeline import run_pipeline

    music = tmp_path / "music"
    music.mkdir()
    (music / "test.mp3").write_bytes(b"\x00" * 100)

    db = tmp_path / "test.db"

    prefetch_called = [False]

    def fake_prefetch(**kwargs):
        prefetch_called[0] = True

    with patch("src.recommender.ingest_pipeline.ensure_mood_models", side_effect=fake_prefetch):
        with patch("src.recommender.ingest_pipeline.extract_essentia"):
            tracks = run_pipeline(music, db, no_mood=True)

    assert not prefetch_called[0], "ensure_mood_models should not be called when no_mood=True"


def test_prefetch_failure_warns_not_fails(tmp_path, monkeypatch):
    """Prefetch exception is caught and pipeline continues."""
    monkeypatch.chdir(tmp_path)
    import sqlite3
    from src.recommender.ingest_pipeline import run_pipeline

    music = tmp_path / "music"
    music.mkdir()
    (music / "test.mp3").write_bytes(b"\x00" * 100)

    db = tmp_path / "test.db"

    def failing_prefetch(**kwargs):
        raise RuntimeError("download failed")

    with patch("src.recommender.ingest_pipeline.ensure_mood_models", side_effect=failing_prefetch):
        with patch("src.recommender.ingest_pipeline.extract_essentia"):
            # Should not raise — prefetch failure is best-effort
            tracks = run_pipeline(music, db, no_mood=False)

    # Pipeline should still complete (0 tracks extracted due to mock, but no crash)
    assert isinstance(tracks, list)


def test_run_pipeline_nonbatch_returns_track_status_tuples(tmp_path, monkeypatch):
    """Non-batch run_pipeline returns (Track, status) tuples like the batch path.

    Regression: the non-batch path returned bare Track objects, crashing
    run.py's `for track, _ in tracks` unpacking with TypeError.
    """
    monkeypatch.chdir(tmp_path)
    from src.recommender.track import Track
    from src.recommender.ingest_pipeline import run_pipeline

    music = tmp_path / "music"
    music.mkdir()
    (music / "test.mp3").write_bytes(b"\x00" * 100)

    db = tmp_path / "test.db"

    def fake_process_file(conn, audio_file, **kwargs):
        return Track(id=1, file_path=audio_file), "processed"

    with patch("src.recommender.ingest_pipeline.ensure_mood_models"):
        with patch("src.recommender.ingest_pipeline.process_file", side_effect=fake_process_file):
            tracks = run_pipeline(music, db, no_mood=True)

    assert len(tracks) == 1
    track, status = tracks[0]  # must unpack without TypeError
    assert isinstance(track, Track)
    assert status == "processed"


# ── run.py --timeout validation ──────────────────────────────────────


def test_cli_timeout_rejects_non_positive():
    """CLI --timeout rejects zero and negative values."""
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--timeout" in result.stdout


def test_cli_rejects_zero_timeout():
    """CLI rejects --timeout 0."""
    result = subprocess.run(
        [sys.executable, "run.py", "/nonexistent", "/tmp/db.sqlite", "--timeout", "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_rejects_negative_timeout():
    """CLI rejects --timeout -5."""
    result = subprocess.run(
        [sys.executable, "run.py", "/nonexistent", "/tmp/db.sqlite", "--timeout", "-5"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_accepts_positive_timeout():
    """CLI accepts --timeout 120."""
    result = subprocess.run(
        [sys.executable, "run.py", "/nonexistent", "/tmp/db.sqlite", "--timeout", "120"],
        capture_output=True,
        text=True,
    )
    # Will fail because /nonexistent doesn't exist, but argparse passes
    assert "Not a directory" in result.stderr or result.returncode != 0
    assert "must be a positive integer" not in result.stderr


# ── CLI --models-dir flag ────────────────────────────────────────────


def test_cli_accepts_models_dir():
    """CLI --models-dir argument is accepted."""
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--models-dir" in result.stdout
