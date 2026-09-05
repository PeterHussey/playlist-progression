import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_db(db_path: Path):
    from src.recommender.ingest_pipeline import init_database
    conn = init_database(db_path)
    conn.close()


_ESSENTIA_SIDECAR = {
    "version": "1.1", "duration_sec": 200.0,
    "loudness": {"integrated": -15.0, "range": 5.0},
    "tempo": {"bpm": 120.0, "confidence": 1.0},
    "key": {"key": "C", "mode": "major", "scale": "C major", "confidence": 0.9},
    "spectral": {"centroid": 1500.0, "rolloff": 700.0, "flatness": 0.05},
    "rhythm": {"danceability": 1.0, "onset_rate": 3.0},
}

_CLAP_SIDECAR = {"version": "1.0", "model": "clap-v1", "embedding": [0.1] * 512}


def _fake_run_batch(manifest_path, summary_path, **kwargs):
    """Side-effect for run_batch: read manifest, write sidecars, return summary."""
    manifest = json.loads(manifest_path.read_text())
    ok = []
    for entry in manifest:
        out = Path(entry["output_path"])
        out.write_text(json.dumps(_ESSENTIA_SIDECAR))
        ok.append(entry["output_path"])
    summary = {"ok": ok, "failed": []}
    summary_path.write_text(json.dumps(summary))
    return summary


def _fake_run_clap_batch(manifest_path, summary_path, **kwargs):
    """Side-effect for run_clap_batch: read manifest, write sidecars, return summary."""
    manifest = json.loads(manifest_path.read_text())
    ok = []
    for entry in manifest:
        out = Path(entry["output_path"])
        out.write_text(json.dumps(_CLAP_SIDECAR))
        ok.append(entry["output_path"])
    summary = {"ok": ok, "failed": []}
    summary_path.write_text(json.dumps(summary))
    return summary


def test_batch_clap_pass_stores_embedding_without_version_churn(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.recommender import ingest_pipeline as ip
    db = tmp_path / "t.db"
    _make_db(db)
    audio = tmp_path / "song.mp3"
    audio.touch()
    conn = sqlite3.connect(str(db))
    with patch.object(ip, "run_batch", side_effect=_fake_run_batch), \
         patch.object(ip, "run_clap_batch", side_effect=_fake_run_clap_batch) as mclap, \
         patch.object(ip, "read_metadata", return_value=("Song", "Artist")):
        tracks = ip._run_pipeline_batch(conn, [audio], db, extract_clap=True, force=True, no_mood=True)
    assert mclap.called, "CLAP batch pass must run when extract_clap=True"
    row = conn.execute("SELECT feature_json, clap_embedding FROM tracks").fetchone()
    assert json.loads(row[0])["version"] == "1.1"
    assert len(json.loads(row[1])) == 512
    statuses = [s for _, s in tracks]
    assert statuses == ["re-extracted"] or statuses == ["processed"]
    conn.close()


def test_batch_clap_failure_leaves_null_and_skips_when_no_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.recommender import ingest_pipeline as ip
    db = tmp_path / "t.db"
    _make_db(db)
    audio = tmp_path / "song.mp3"
    audio.touch()
    conn = sqlite3.connect(str(db))

    # First run: Essentia succeeds, CLAP fails — verify NULL remains
    with patch.object(ip, "run_batch", side_effect=_fake_run_batch), \
         patch.object(ip, "run_clap_batch", return_value={"ok": [], "failed": [{"output": "x", "error": "boom"}]}), \
         patch.object(ip, "read_metadata", return_value=("Song", "Artist")):
        ip._run_pipeline_batch(conn, [audio], db, extract_clap=True, force=True, no_mood=True)
    row = conn.execute("SELECT clap_embedding FROM tracks").fetchone()
    assert row[0] is None, "CLAP failure must leave clap_embedding NULL"

    # Second run: extract_clap=False — CLAP batch must not be called
    with patch.object(ip, "run_batch", side_effect=_fake_run_batch), \
         patch.object(ip, "run_clap_batch") as mc:
        ip._run_pipeline_batch(conn, [audio], db, extract_clap=False, force=False, no_mood=True)
    mc.assert_not_called()
    conn.close()


def test_batch_clap_populates_existing_track_no_churn(tmp_path, monkeypatch):
    """CLAP pass populates missing clap_embedding for version-current tracks without version churn.

    When every row is version-current (force=False) and extract_clap=True but all clap_embedding
    are NULL, the early return must NOT skip the CLAP pass — it should fall through so the CLAP
    pass can populate the missing embeddings.
    """
    monkeypatch.chdir(tmp_path)
    from src.recommender import ingest_pipeline as ip
    db = tmp_path / "t.db"
    _make_db(db)
    audio = tmp_path / "song.mp3"
    audio.touch()
    conn = sqlite3.connect(str(db))
    # Insert a version-current track with feature_json version "1.1" and NULL clap_embedding
    conn.execute(
        "INSERT INTO tracks (file_path, title, artist, duration_sec, feature_json, clap_embedding) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (str(audio), "Song", "Artist", 200.0, json.dumps({"version": "1.1"}), None),
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(str(db))
    with patch.object(ip, "run_batch") as mb, \
         patch.object(ip, "run_clap_batch", side_effect=_fake_run_clap_batch) as mclap, \
         patch.object(ip, "read_metadata", return_value=("Song", "Artist")):
        tracks = ip._run_pipeline_batch(conn, [audio], db, extract_clap=True, force=False, no_mood=True)
    assert not mb.called, "run_batch must not be called when pending is empty"
    assert mclap.called, "CLAP batch pass must run when extract_clap=True"
    row = conn.execute("SELECT feature_json, clap_embedding FROM tracks").fetchone()
    assert json.loads(row[0])["version"] == "1.1"
    assert len(json.loads(row[1])) == 512, f"Expected 512 CLAP dims, got {len(json.loads(row[1]))}"
    statuses = [s for _, s in tracks]
    assert statuses == ["skipped"], f"Expected ['skipped'], got {statuses}"
    conn.close()


def test_batch_clap_failures_are_logged(tmp_path, monkeypatch, capsys):
    """Failed CLAP entries must be reported on stdout, never silently skipped."""
    monkeypatch.chdir(tmp_path)
    from src.recommender import ingest_pipeline as ip
    db = tmp_path / "t.db"
    _make_db(db)
    audio = tmp_path / "song.mp3"
    audio.touch()
    conn = sqlite3.connect(str(db))

    def _fail_all_clap(manifest_path, summary_path, **kwargs):
        manifest = json.loads(manifest_path.read_text())
        summary = {"ok": [], "failed": [
            {"output": e["output_path"], "error": "boom"} for e in manifest
        ]}
        summary_path.write_text(json.dumps(summary))
        return summary

    with patch.object(ip, "run_batch", side_effect=_fake_run_batch), \
         patch.object(ip, "run_clap_batch", side_effect=_fail_all_clap), \
         patch.object(ip, "read_metadata", return_value=("Song", "Artist")):
        ip._run_pipeline_batch(conn, [audio], db, extract_clap=True, force=True, no_mood=True)
    out = capsys.readouterr().out
    assert "boom" in out, f"CLAP failure reason must be logged, got: {out!r}"
    conn.close()
