"""Shared feature converter — removes duplicated axis knowledge."""
from __future__ import annotations
import json

AXIS_NAMES = [
    "duration_sec", "loudness.integrated", "loudness.range",
    "tempo.bpm", "tempo.confidence", "key.confidence",
    "spectral.centroid", "spectral.rolloff", "spectral.flatness",
    "rhythm.danceability", "rhythm.onset_rate",
    "mood.happy", "mood.sad", "mood.aggressive", "mood.relaxed",
    "mood.electronic", "mood.party", "mood.acoustic",
]


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
    return vec
