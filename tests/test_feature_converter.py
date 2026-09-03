"""TDD test for feature converter (backlog item 3)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
from src.recommender.feature_converter import AXIS_NAMES, convert


def test_axis_names_length():
    assert len(AXIS_NAMES) == 20
    assert "key.fifths_x" in AXIS_NAMES
    assert "key.fifths_y" in AXIS_NAMES
    # key.confidence retained as reliability signal
    assert "key.confidence" in AXIS_NAMES


def test_convert_empty_json():
    vec = convert('{}')
    assert len(vec) == 20
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
    assert len(vec) == 20
    assert vec[0] == 120.5
    assert vec[3] == 128.0
    assert vec[9] == 0.65  # rhythm.danceability
    assert vec[10] == 1.8  # rhythm.onset_rate


def _key_xy(feature_dict):
    vec = convert(json.dumps(feature_dict))
    ix = AXIS_NAMES.index("key.fifths_x")
    iy = AXIS_NAMES.index("key.fifths_y")
    return vec[ix], vec[iy]


def _base_raw(key, mode=None, scale=None):
    k = {"confidence": 0.9, "key": key}
    if mode is not None:
        k["mode"] = mode
    if scale is not None:
        k["scale"] = scale
    return {
        "duration_sec": 100.0,
        "loudness": {"integrated": -10.0, "range": 5.0},
        "tempo": {"bpm": 120, "confidence": 0.9},
        "key": k,
        "spectral": {"centroid": 2000, "rolloff": 4000, "flatness": 0.02},
        "rhythm": {"danceability": 0.5, "onset_rate": 1.0},
        "mood": {},
    }


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def test_key_fifths_neighbours_close_tritone_far():
    c = _key_xy(_base_raw("C", mode="major"))
    g = _key_xy(_base_raw("G", mode="major"))
    fsharp = _key_xy(_base_raw("F#", mode="major"))
    assert _dist(c, g) < _dist(c, fsharp)
    # C vs F# are opposite sides of the circle (12 of 24 slots)
    assert _dist(c, fsharp) > 1.9


def test_key_relative_minor_adjacent():
    c = _key_xy(_base_raw("C", mode="major"))
    a_min = _key_xy(_base_raw("A", mode="minor"))
    g = _key_xy(_base_raw("G", mode="major"))
    # Am is 1 slot from C; G is 2 slots from C
    assert _dist(c, a_min) < _dist(c, g)


def test_key_enharmonic_equivalent():
    csharp = _key_xy(_base_raw("C#", mode="major"))
    db = _key_xy(_base_raw("Db", mode="major"))
    assert csharp == db


def test_key_scale_string_fallback():
    via_mode = _key_xy(_base_raw("A", mode="minor"))
    via_scale = _key_xy(_base_raw("A", scale="A minor"))
    assert via_mode == via_scale


def test_key_unknown_maps_to_origin():
    x, y = _key_xy(_base_raw("X", mode="major"))
    assert (x, y) == (0.0, 0.0)
    x2, y2 = _key_xy(_base_raw("C", mode="dorian"))
    assert (x2, y2) == (0.0, 0.0)
