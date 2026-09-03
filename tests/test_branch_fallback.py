#!/usr/bin/env python3
"""Regression tests: fallback picks must be labelled with actual band,
and band selections must be nearest-first (BRANCHING.md)."""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _mk_track(i, vec):
    from src.recommender.track import Track
    t = Track(id=i, file_path=Path(f"/m/{i}.mp3"), title=f"T{i}",
              artist="A", duration_sec=0.0)
    t.set_features(vec)
    return t


def _sampler():
    from src.recommender.branch_sampler import BranchSampler
    return BranchSampler(axis_weights=[1.0], axis_index={"x": 0},
                         axis_means=[0.0], axis_stddevs=[1.0])


def test_select_near_returns_nearest_first():
    sampler = _sampler()
    seed = _mk_track(0, [0.0])
    farther = _mk_track(1, [0.25])
    nearer = _mk_track(2, [0.10])
    res = sampler.select_near(seed, [farther, nearer])
    assert [t.id for t in res] == [2, 1]


def test_select_mid_returns_nearest_first():
    sampler = _sampler()
    seed = _mk_track(0, [0.0])
    farther = _mk_track(1, [0.60])
    nearer = _mk_track(2, [0.40])
    res = sampler.select_mid(seed, [farther, nearer])
    assert [t.id for t in res] == [2, 1]


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
        "mood": {},
    }


def test_generate_playlist_fallback_labels_actual_band(tmp_path, monkeypatch):
    """Scheduled Near with only far candidates must not label pick 'Near'."""
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
    # Seed (id 1) + two far-away tracks: bpm far apart so that with only
    # 3 rows the z-scored distance exceeds the Near 0.3 threshold.
    for fp, title, bpm in [("/m/a.mp3", "Seed", 60.0),
                           ("/m/b.mp3", "Far1", 180.0),
                           ("/m/c.mp3", "Far2", 190.0)]:
        conn.execute(
            "INSERT INTO tracks (file_path, title, duration_sec, feature_json)"
            " VALUES (?, ?, ?, ?)",
            (fp, title, 200.0, json.dumps(_sidecar(bpm))),
        )
    conn.commit()
    conn.close()

    import generate_playlist
    out = tmp_path / "out.json"
    summ = tmp_path / "summ.txt"
    generate_playlist.main(["--db", str(db), "--seed-id", "1",
                            "--limit", "1", "--output", str(out),
                            "--summary", str(summ)])
    data = json.loads(out.read_text())
    assert len(data["playlist"]) == 1
    entry = data["playlist"][0]
    # The pick is a global-nearest fallback: its band label must be
    # consistent with its distance, and the reason must disclose fallback.
    d = entry["distance"]
    if d <= 0.3:
        assert entry["band"] == "Near"
    elif d <= 0.7:
        assert entry["band"] == "Mid", f"mislabeled Mid-range pick: {entry}"
        assert "allback" in entry["reason"], f"reason hides fallback: {entry}"
    else:
        assert entry["band"] == "Far", f"mislabeled Far-range pick: {entry}"
        assert "allback" in entry["reason"], f"reason hides fallback: {entry}"
    # Guard against the reported bug shape: scheduled "Near" with d=0.64.
    assert not (entry["band"] == "Near" and d > 0.3), f"impossible label: {entry}"
    assert not (entry["band"] == "Mid" and (d <= 0.3 or d > 0.7)), f"impossible label: {entry}"
