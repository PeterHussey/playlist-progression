#!/usr/bin/env python3
"""Tests for backlog item 8 hardening: re-extract/version check + playlist CLI."""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _sidecar(version, duration=200.0, bpm=120.0):
    return {
        "version": version,
        "duration_sec": duration,
        "loudness": {"integrated": -10.0, "range": 5.0},
        "tempo": {"bpm": bpm, "confidence": 0.9},
        "key": {"key": "C", "mode": "major", "scale": "C major",
                "confidence": 0.9},
        "spectral": {"centroid": 2000.0, "rolloff": 4000.0,
                     "flatness": 0.02},
        "rhythm": {"danceability": 0.5, "onset_rate": 1.0},
        "mood": {},
    }


def _seed_db(db_path, rows):
    """Create a tracks table with the given rows.

    rows: list of (file_path, title, duration_sec, feature_dict|None)
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL, title TEXT, artist TEXT,
            duration_sec REAL, feature_json TEXT, clap_embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    for fp, title, dur, feat in rows:
        conn.execute(
            "INSERT INTO tracks (file_path, title, duration_sec, feature_json)"
            " VALUES (?, ?, ?, ?)",
            (fp, title, dur,
             json.dumps(feat) if feat is not None else None),
        )
    conn.commit()
    return conn


def test_extractor_version_single_source():
    from scripts.extract_essentia import EXTRACTOR_VERSION
    from src.recommender.ingest_pipeline import _current_extractor_version

    assert _current_extractor_version() == EXTRACTOR_VERSION
    assert EXTRACTOR_VERSION == "1.1"


def test_process_file_skips_current_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.recommender.ingest_pipeline import process_file
    from scripts.extract_essentia import EXTRACTOR_VERSION

    audio = tmp_path / "song.mp3"
    audio.touch()
    conn = _seed_db(
        tmp_path / "t.db",
        [(str(audio), "Old Title", 111.0,
          _sidecar(EXTRACTOR_VERSION, duration=111.0))],
    )
    from unittest.mock import patch

    with patch(
        "src.recommender.ingest_pipeline.extract_essentia",
        side_effect=RuntimeError("must not be called"),
    ):
        track = process_file(conn, audio)
    assert track.get_title() == "Old Title"
    assert track.get_duration_sec() == 111.0
    conn.close()


def test_process_file_reextracts_stale_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.recommender.ingest_pipeline import process_file

    audio = tmp_path / "song.mp3"
    audio.touch()
    conn = _seed_db(
        tmp_path / "t.db",
        [(str(audio), "Old Title", 0.0, _sidecar("0.0"))],
    )
    from unittest.mock import patch

    def fake_extract(audio_path, output_path, **kwargs):
        Path(output_path).write_text(
            json.dumps(_sidecar("1.1", duration=222.0)))

    with patch("src.recommender.ingest_pipeline.extract_essentia",
               side_effect=fake_extract) as m:
        track = process_file(conn, audio)
    assert m.call_count == 1
    assert track.get_duration_sec() == 222.0
    row = conn.execute(
        "SELECT feature_json, duration_sec FROM tracks").fetchone()
    assert json.loads(row[0])["version"] == "1.1"
    assert row[1] == 222.0
    conn.close()


def test_process_file_force_reextracts_current(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.recommender.ingest_pipeline import process_file
    from scripts.extract_essentia import EXTRACTOR_VERSION

    audio = tmp_path / "song.mp3"
    audio.touch()
    conn = _seed_db(
        tmp_path / "t.db",
        [(str(audio), "T", 100.0, _sidecar(EXTRACTOR_VERSION))],
    )
    from unittest.mock import patch

    def fake_extract(audio_path, output_path, **kwargs):
        Path(output_path).write_text(
            json.dumps(_sidecar(EXTRACTOR_VERSION, duration=333.0)))

    with patch("src.recommender.ingest_pipeline.extract_essentia",
               side_effect=fake_extract) as m:
        track = process_file(conn, audio, force=True)
    assert m.call_count == 1
    assert track.get_duration_sec() == 333.0
    conn.close()


def _playlist_db(db_path):
    return _seed_db(
        db_path,
        [
            ("/m/a.mp3", "Alpha", 200.0, _sidecar("1.1", bpm=100.0)),
            ("/m/b.mp3", "Beta", 210.0, _sidecar("1.1", bpm=120.0)),
            ("/m/c.mp3", "Gamma", 220.0, _sidecar("1.1", bpm=140.0)),
        ],
    ).close()


def test_playlist_cli_seed_and_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "pl.db"
    _playlist_db(db)
    import generate_playlist

    out = tmp_path / "out.json"
    summ = tmp_path / "summ.txt"
    generate_playlist.main(["--db", str(db), "--seed-id", "2",
                            "--limit", "2", "--output", str(out),
                            "--summary", str(summ)])
    data = json.loads(out.read_text())
    assert data["seed"]["id"] == 2
    assert data["seed"]["title"] == "Beta"
    assert len(data["playlist"]) == 2
    text = summ.read_text()
    assert "Beta" in text and "Hold axis: tempo.bpm" in text
    capsys.readouterr()


def test_playlist_cli_unknown_seed_exits_1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "pl.db"
    _playlist_db(db)
    import generate_playlist
    import pytest

    with pytest.raises(SystemExit) as e:
        generate_playlist.main(["--db", str(db), "--seed-id", "99"])
    assert e.value.code == 1


def test_playlist_cli_unknown_axis_exits_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import generate_playlist
    import pytest

    with pytest.raises(SystemExit) as e:
        generate_playlist.main(["--hold-axis", "nope.not real"])
    assert e.value.code == 2
