"""Integration tests for PlaylistSampler (synthetic Tracks, no DB)."""
import json
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.recommender.track import Track
from src.recommender.playlist_sampler import PlaylistSampler


def _mk(i, emb, bpm=120.0, mood=None, key="C", mode="major"):
    sidecar = {"version": "1.1", "duration_sec": 200.0,
               "tempo": {"bpm": bpm, "confidence": 1.0},
               "key": {"key": key, "mode": mode, "scale": f"{key} {mode}", "confidence": 2.0},
               "mood": mood or {"sad": 0.8}}
    t = Track(id=i, file_path=Path(f"/m/{i}.mp3"), title=f"T{i}", artist="A", duration_sec=200.0)
    t.set_feature_json(json.dumps(sidecar))
    t.set_clap_embedding(list(emb))
    return t


def _config(**kw):
    base = {"near_quantile": 0.10, "mid_quantile": 0.40, "mood_box": {"mood.sad": {"min": 0.5}},
            "key_max_steps": 2, "tempo_max_step_pct": 0.10, "tempo_drift_per_step": 0.0,
            "band_schedule": ["Near", "Mid", "Far"], "allow_missing_clap": False, "allow_missing_mood": False}
    base.update(kw)
    return base


def test_generate_respects_mood_box_and_returns_entries():
    seed = _mk(1, [1.0, 0.0])
    good = _mk(2, [0.99, 0.01], mood={"sad": 0.9})
    bad = _mk(3, [0.999, 0.001], mood={"sad": 0.0})
    s = PlaylistSampler(_config(band_schedule=["Near"]))
    s.load_library([seed, good, bad])
    entries = s.generate(seed, limit=1)
    assert len(entries) == 1
    assert entries[0]["track_id"] == 2
    assert entries[0]["band"] == "Near"


def test_missing_clap_excluded_by_default():
    seed = _mk(1, [1.0, 0.0])
    no_clap = _mk(2, [1.0, 0.0])
    no_clap.set_clap_embedding(None)
    s = PlaylistSampler(_config(band_schedule=["Near"]))
    s.load_library([seed, no_clap])
    assert s.generate(seed, limit=1) == []


def _sidecar(bpm=120.0):
    return {
        "version": "1.1",
        "duration_sec": 200.0,
        "loudness": {"integrated": -10.0, "range": 5.0},
        "tempo": {"bpm": bpm, "confidence": 0.9},
        "key": {"key": "C", "mode": "major", "scale": "C major",
                "confidence": 0.9},
        "spectral": {"centroid": 2000.0, "rolloff": 4000.0,
                     "flatness": 0.02},
        "rhythm": {"danceability": 0.5, "onset_rate": 1.0},
        "mood": {"sad": 0.9},
    }


def test_clap_embedding_loaded_from_db(tmp_path, monkeypatch):
    """clap_embedding must be deserialized from DB so clap sampler works."""
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "pl.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL, title TEXT, artist TEXT,
            duration_sec REAL, feature_json TEXT, clap_embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    for fp, title, bpm, clap in [
        ("/m/seed.mp3", "Seed", 120.0, [1.0, 0.0]),
        ("/m/near.mp3", "Near", 120.5, [0.99, 0.01]),
        ("/m/far.mp3", "Far", 120.0, [0.0, 1.0]),
    ]:
        conn.execute(
            "INSERT INTO tracks (file_path, title, duration_sec, feature_json, clap_embedding)"
            " VALUES (?, ?, ?, ?, ?)",
            (fp, title, 200.0, json.dumps(_sidecar(bpm)), json.dumps(clap)),
        )
    conn.commit()
    conn.close()

    import generate_playlist
    out = tmp_path / "out.json"
    summ = tmp_path / "summ.txt"
    # Do NOT pass --sampler; default is clap (the path under test).
    generate_playlist.main(["--db", str(db), "--seed-id", "1",
                            "--limit", "1", "--output", str(out),
                            "--summary", str(summ)])
    data = json.loads(out.read_text())
    assert len(data["playlist"]) == 1
    assert data["playlist"][0]["id"] == 2  # the near candidate
