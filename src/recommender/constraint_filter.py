"""Essentia constraint gates: mood box, key congruence, tempo window."""
from __future__ import annotations

import json
from .feature_converter import FIFTHS_ORDER, _PC, _PC_TO_NAME, _normalise_key_name
from .track import Track

_FIFTHS_INDEX = {name: i for i, name in enumerate(FIFTHS_ORDER)}


def _mood_value(mood: dict[str, float], key: str) -> float | None:
    short = key.split(".", 1)[1] if "." in key else key
    if short in mood:
        return float(mood[short])
    return None


def mood_passes(mood: dict[str, float], mood_box: dict[str, dict[str, float]]) -> bool:
    for axis, bounds in mood_box.items():
        v = _mood_value(mood or {}, axis)
        if v is None:
            return False
        if "min" in bounds and v < float(bounds["min"]):
            return False
        if "max" in bounds and v > float(bounds["max"]):
            return False
    return True


def relax_mood_box(mood_box: dict[str, dict[str, float]], step: float = 0.1) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for axis, bounds in mood_box.items():
        nb: dict[str, float] = {}
        if "min" in bounds:
            nb["min"] = max(0.0, float(bounds["min"]) - step)
        if "max" in bounds:
            nb["max"] = min(1.0, float(bounds["max"]) + step)
        out[axis] = nb
    return out


def key_slot(key: str | None, mode: str | None) -> int | None:
    canon = _normalise_key_name(key)
    m = str(mode).strip().lower() if mode else None
    if m is not None:
        if "major" in m:
            m = "major"
        elif "minor" in m:
            m = "minor"
        else:
            return None
    if canon is None or m not in ("major", "minor"):
        return None
    if m == "major":
        return (_FIFTHS_INDEX[canon] * 2) % 24
    rel_pc = (_PC[canon] + 3) % 12
    rel_name = _PC_TO_NAME[rel_pc]
    return (_FIFTHS_INDEX[rel_name] * 2 + 1) % 24


def key_steps(slot_a: int | None, slot_b: int | None) -> int | None:
    if slot_a is None or slot_b is None:
        return None
    d = abs(slot_a - slot_b) % 24
    return min(d, 24 - d)


def key_verdict(steps: int | None, key_max_steps: int = 2) -> str:
    if steps is None:
        return "allow"
    if steps <= key_max_steps:
        return "allow"
    if steps <= key_max_steps + 2:
        return "penalise"
    return "block"


def tempo_ok(seed_bpm: float, cand_bpm: float, max_step_pct: float = 0.10, centre_bpm: float | None = None) -> bool:
    centre = centre_bpm if centre_bpm is not None else seed_bpm
    if not centre or centre <= 0:
        return False
    return abs(cand_bpm - centre) / centre <= max_step_pct


def _parsed_sidecar(feature_json: str | None) -> dict:
    try:
        return json.loads(feature_json or "{}")
    except (json.JSONDecodeError, AttributeError):
        return {}


class ConstraintFilter:
    def __init__(self, config: dict) -> None:
        self.mood_box: dict = dict(config.get("mood_box", {}))
        self.key_max_steps: int = int(config.get("key_max_steps", 2))
        self.tempo_max_step_pct: float = float(config.get("tempo_max_step_pct", 0.10))
        self.allow_missing_mood: bool = bool(config.get("allow_missing_mood", False))

    def filter(self, seed: Track, candidates: list[Track], centre_bpm: float | None = None) -> list[Track]:
        """Filter candidates through mood/key/tempo gates.

        penalise-verdict tracks sort after allow (soft demotion).
        """
        seed_raw = _parsed_sidecar(seed.feature_json)
        seed_tempo = float((seed_raw.get("tempo") or {}).get("bpm", 0.0) or 0.0)
        seed_key = seed_raw.get("key") or {}
        seed_slot = key_slot(seed_key.get("key"), seed_key.get("mode") or seed_key.get("scale"))
        out: list[Track] = []
        verdicts: list[str] = []
        for cand in candidates:
            raw = _parsed_sidecar(cand.feature_json)
            mood = raw.get("mood") or {}
            if self.mood_box and "mood" not in raw and not self.allow_missing_mood:
                continue
            if self.mood_box and not mood_passes(mood, self.mood_box):
                continue
            ck = raw.get("key") or {}
            steps = key_steps(seed_slot, key_slot(ck.get("key"), ck.get("mode") or ck.get("scale")))
            v = key_verdict(steps, self.key_max_steps)
            if v == "block":
                continue
            cand_bpm = float((raw.get("tempo") or {}).get("bpm", 0.0) or 0.0)
            if seed_tempo > 0 and cand_bpm > 0 and not tempo_ok(seed_tempo, cand_bpm, self.tempo_max_step_pct, centre_bpm=centre_bpm):
                continue
            out.append(cand)
            verdicts.append(v)
        # Stable-sort: allow before penalise (soft demotion)
        paired = list(zip(out, verdicts, range(len(out))))
        paired.sort(key=lambda t: (0 if t[1] == "allow" else 1, t[2]))
        return [t for t, _, _ in paired]
