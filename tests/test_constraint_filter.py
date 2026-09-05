"""Tests for Essentia constraint gates."""
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