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


def test_small_library_greedy_skips_bands():
    """Spec §6 guard: <20 candidates → greedy nearest, scheduled label kept."""
    seed = _mk(1, [1.0, 0.0])
    a = _mk(2, [0.0, 1.0])
    b = _mk(3, [0.9, 0.1])
    s = PlaylistSampler(_config(band_schedule=["Far"], mood_box={}))
    s.load_library([seed, a, b])
    entries = s.generate(seed, limit=1)
    assert len(entries) == 1
    assert entries[0]["track_id"] == 3  # nearest by CLAP
    assert entries[0]["band"] == "Far"  # scheduled label kept
    assert "Small library" in entries[0]["reason"]


def test_large_library_uses_bands():
    """≥20 candidates → normal quantile-band path, no greedy reason."""
    tracks = [_mk(1, [1.0, 0.0])]
    for i in range(2, 27):
        tracks.append(_mk(i, [1.0 - i * 0.01, i * 0.01]))
    s = PlaylistSampler(_config(band_schedule=["Near"], mood_box={}))
    s.load_library(tracks)
    entries = s.generate(tracks[0], limit=1)
    assert len(entries) == 1
    assert "CLAP pick" in entries[0]["reason"]
    assert "Small library" not in entries[0]["reason"]


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


def test_tempo_drift_per_step():
    """Tempo drift centres the window on drifted seed BPM, not raw seed BPM."""
    # Seed bpm=100, drift=0.5 → step 0 centre=100, step 1 centre=150.
    # A bpm=105 passes step 0 window (105/100 ≤ 1.10), fails step 1 (|105-150|/150 > 0.10).
    # B bpm=150 fails step 0 (150/100 > 1.10), passes step 1 (|150-150|/150 = 0 ≤ 0.10).
    # B has CLAP slightly closer to seed than A, so without drift B would always win.
    seed = _mk(1, [1.0, 0.0], bpm=100.0)
    a = _mk(2, [0.99, 0.01], bpm=105.0)   # CLAP a bit less close
    b = _mk(3, [0.999, 0.001], bpm=150.0)  # CLAP closest
    cfg = _config(band_schedule=["Near", "Near"], tempo_drift_per_step=0.5)
    s = PlaylistSampler(cfg)
    s.load_library([seed, a, b])
    entries = s.generate(seed, limit=2)
    assert len(entries) == 2
    # Step 0: centre=100 → A passes, B fails → A wins
    assert entries[0]["track_id"] == 2
    # Step 1: centre=150 → B passes (within window) → B should be picked
    assert entries[1]["track_id"] == 3


def test_final_fallback_global_nearest_by_clap():
    """When all candidates are blocked by key/tempo (unrelaxable), fallback picks global nearest."""
    # Seed: C major bpm=120. Candidate: F# major (12 steps = block) bpm=200 (tempo also fails).
    # Key gate blocks F# major regardless of mood relax → fallback should still return 1 entry.
    seed = _mk(1, [1.0, 0.0], bpm=120.0, key="C", mode="major")
    blocked = _mk(2, [0.99, 0.01], bpm=200.0, key="F#", mode="major")
    cfg = _config(band_schedule=["Near"], mood_box={})
    s = PlaylistSampler(cfg)
    s.load_library([seed, blocked])
    entries = s.generate(seed, limit=1)
    assert len(entries) == 1
    assert entries[0]["track_id"] == 2
    assert "Fallback" in entries[0]["reason"]


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
