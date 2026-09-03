"""Shared feature converter — removes duplicated axis knowledge."""
from __future__ import annotations
import json
import math

AXIS_NAMES = [
    "duration_sec", "loudness.integrated", "loudness.range",
    "tempo.bpm", "tempo.confidence", "key.confidence",
    "spectral.centroid", "spectral.rolloff", "spectral.flatness",
    "rhythm.danceability", "rhythm.onset_rate",
    "mood.happy", "mood.sad", "mood.aggressive", "mood.relaxed",
    "mood.electronic", "mood.party", "mood.acoustic",
    "key.fifths_x", "key.fifths_y",
]

# Circle-of-fifths order (12 pitch classes, enharmonic-normalised).
FIFTHS_ORDER = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
_FIFTHS_INDEX = {name: i for i, name in enumerate(FIFTHS_ORDER)}

# Enharmonic normalisation to the canonical names in FIFTHS_ORDER.
_ENHARMONIC = {
    "C#": "Db", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb",
    "B#": "C", "E#": "F", "Cb": "B", "Fb": "E",
}

_PC = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
       "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
_PC_TO_NAME = {v: k for k, v in _PC.items()}


def _normalise_key_name(name: str | None) -> str | None:
    if not name:
        return None
    s = str(name).strip()
    if not s:
        return None
    canon = s[0].upper() + s[1:]
    canon = _ENHARMONIC.get(canon, canon)
    return canon if canon in _FIFTHS_INDEX else None


def _parse_mode(key_dict: dict) -> str | None:
    mode = key_dict.get("mode")
    if mode:
        m = str(mode).strip().lower()
        if "major" in m:
            return "major"
        if "minor" in m:
            return "minor"
        return None
    scale = key_dict.get("scale", "")
    m = str(scale).strip().lower()
    if "major" in m:
        return "major"
    if "minor" in m:
        return "minor"
    return None


def key_to_circle(key_name: str | None, mode: str | None) -> tuple[float, float]:
    """Map key+mode to 2D circle-of-fifths coordinates.

    24 slots: majors on even slots, relative minors on the following odd
    slot (Am adjacent to C). Returns (cos, sin) of slot angle, or
    (0.0, 0.0) for unknown key/mode.
    """
    canon = _normalise_key_name(key_name)
    m = str(mode).strip().lower() if mode else None
    if m is not None:
        if "major" in m:
            m = "major"
        elif "minor" in m:
            m = "minor"
        else:
            return (0.0, 0.0)
    if canon is None or m not in ("major", "minor"):
        return (0.0, 0.0)
    if m == "major":
        slot = (_FIFTHS_INDEX[canon] * 2) % 24
    else:
        rel_pc = (_PC[canon] + 3) % 12
        rel_name = _PC_TO_NAME[rel_pc]
        slot = (_FIFTHS_INDEX[rel_name] * 2 + 1) % 24
    angle = slot * math.pi / 12
    return (math.cos(angle), math.sin(angle))


def convert(feature_json_str: str | None) -> list[float]:
    raw = json.loads(feature_json_str or "{}")
    vec = []
    vec.append(float(raw.get("duration_sec", 0)))
    loud = raw.get("loudness", {})
    vec.append(float(loud.get("integrated", 0)))
    vec.append(float(loud.get("range", 0)))
    tempo = raw.get("tempo", {})
    vec.append(float(tempo.get("bpm", 0)))
    vec.append(float(tempo.get("confidence", 0)))
    key = raw.get("key", {})
    vec.append(float(key.get("confidence", 0)))
    spec = raw.get("spectral", {})
    vec.append(float(spec.get("centroid", 0)))
    vec.append(float(spec.get("rolloff", 0)))
    vec.append(float(spec.get("flatness", 0)))
    rhythm = raw.get("rhythm", {})
    vec.append(float(rhythm.get("danceability", 0)))
    vec.append(float(rhythm.get("onset_rate", 0)))
    mood = raw.get("mood", {})
    vec.append(float(mood.get("happy", 0)))
    vec.append(float(mood.get("sad", 0)))
    vec.append(float(mood.get("aggressive", 0)))
    vec.append(float(mood.get("relaxed", 0)))
    vec.append(float(mood.get("electronic", 0)))
    vec.append(float(mood.get("party", 0)))
    vec.append(float(mood.get("acoustic", 0)))
    mode = _parse_mode(key)
    x, y = key_to_circle(key.get("key"), mode)
    vec.append(float(x))
    vec.append(float(y))
    return vec
