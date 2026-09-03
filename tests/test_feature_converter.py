"""TDD test for feature converter (backlog item 3)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
from src.recommender.feature_converter import AXIS_NAMES, convert


def test_axis_names_length():
    assert len(AXIS_NAMES) == 18


def test_convert_empty_json():
    vec = convert('{}')
    assert len(vec) == 18
    assert all(isinstance(v, float) for v in vec)


def test_convert_full_json():
    raw = {
        "duration_sec": 120.5,
        "loudness": {"integrated": -12.3, "range": 7.8},
        "tempo": {"bpm": 128, "confidence": 0.92},
        "key": {"confidence": 0.85},
        "spectral": {"centroid": 2100, "rolloff": 4500, "flatness": 0.03},
        "rhythm": {"danceability": 0.65, "onset_rate": 1.8},
        "mood": {
            "happy": 0.7, "sad": 0.2, "aggressive": 0.1,
            "relaxed": 0.6, "electronic": 0.5, "party": 0.4, "acoustic": 0.3,
        },
    }
    vec = convert(json.dumps(raw))
    assert len(vec) == 18
    assert vec[0] == 120.5
    assert vec[3] == 128.0
    assert vec[9] == 0.65  # rhythm.danceability
    assert vec[10] == 1.8  # rhythm.onset_rate
