"""Tests for Essentia constraint gates."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.recommender import constraint_filter as cf


def _sidecar(bpm=120.0, key="C", mode="major", conf=2.0, mood=None):
    return {
        "version": "1.1", "duration_sec": 200.0,
        "tempo": {"bpm": bpm, "confidence": 0.9},
        "key": {"key": key, "mode": mode, "scale": f"{key} {mode}", "confidence": conf},
        "mood": mood or {},
    }


def test_mood_box_pass_and_fail():
    assert cf.mood_passes({"sad": 0.7}, {"mood.sad": {"min": 0.5}}) is True
    assert cf.mood_passes({"sad": 0.2}, {"mood.sad": {"min": 0.5}}) is False


def test_mood_missing_key_fails_closed():
    assert cf.mood_passes({}, {"mood.sad": {"min": 0.5}}) is False


def test_key_steps_relative_major_minor_is_one():
    a = cf.key_slot("C", "major")
    b = cf.key_slot("A", "minor")
    assert cf.key_steps(a, b) == 1


def test_key_steps_tritone_blocked():
    a = cf.key_slot("C", "major")
    b = cf.key_slot("F#", "major")
    assert cf.key_steps(a, b) == 12
    assert cf.key_verdict(12) == "block"
    assert cf.key_verdict(None) == "allow"


def test_tempo_window():
    assert cf.tempo_ok(120.0, 126.0, max_step_pct=0.10) is True
    assert cf.tempo_ok(120.0, 150.0, max_step_pct=0.10) is False


def test_penalise_demotion_sorts_after_allow():
    """penalise-verdict candidates sort after allow-verdict candidates."""
    from src.recommender.track import Track
    # Seed: C major (slot 0), bpm 120, mood passing
    seed_raw = _sidecar(bpm=120.0, key="C", mode="major", mood={"sad": 0.8})
    seed = Track(id=1, file_path=Path("/m/1.mp3"), title="S", artist="A", duration_sec=200.0)
    seed.set_feature_json(json.dumps(seed_raw))

    # G major (slot 2, steps=2 → allow) bpm=120 mood passing
    g_raw = _sidecar(bpm=120.0, key="G", mode="major", mood={"sad": 0.9})
    g = Track(id=2, file_path=Path("/m/2.mp3"), title="G", artist="A", duration_sec=200.0)
    g.set_feature_json(json.dumps(g_raw))

    # D major (slot 4, steps=4 → penalise) bpm=120 mood passing
    d_raw = _sidecar(bpm=120.0, key="D", mode="major", mood={"sad": 0.9})
    d = Track(id=3, file_path=Path("/m/3.mp3"), title="D", artist="A", duration_sec=200.0)
    d.set_feature_json(json.dumps(d_raw))

    # Verify D major steps == 4
    assert cf.key_steps(cf.key_slot("C", "major"), cf.key_slot("D", "major")) == 4
    assert cf.key_verdict(4) == "penalise"

    flt = cf.ConstraintFilter({"mood_box": {}, "key_max_steps": 2, "tempo_max_step_pct": 0.50})
    result = flt.filter(seed, [d, g])
    # G major (allow) must come before D major (penalise)
    assert result[0].id == 2
    assert result[1].id == 3