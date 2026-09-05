"""Integration tests for PlaylistSampler (synthetic Tracks, no DB)."""
import json
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
