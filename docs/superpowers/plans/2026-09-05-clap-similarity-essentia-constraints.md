# CLAP Similarity + Essentia Constraints Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Essentia-Euclidean ranking with CLAP-cosine ranking gated by Essentia mood/key/tempo constraints, with old-vs-new eval.

**Architecture:** Three new additive modules (`clap_similarity.py`, `constraint_filter.py`, `playlist_sampler.py`) plus an eval script and ear-test sheet. Legacy `BranchSampler` frozen untouched except a docstring note. CLI gains `--sampler {clap,essentia}` (default `clap`).

**Tech Stack:** Python 3.10+, stdlib only (`math`, `json`, `sqlite3`, `argparse`), pytest, existing `Track` dataclass.

**Spec:** `docs/superpowers/specs/2026-09-05-clap-similarity-essentia-constraints-design.md`

## Global Constraints

- Python 3.10+ (`list[float]`, `str | None` syntax in use).
- No schema changes: `tracks.feature_json` + `tracks.clap_embedding` columns stay as-is.
- Legacy path frozen: no logic changes to `BranchSampler.compute_distance` / band thresholds.
- No network in tests; `tests/sample_audio/` placeholders are not real audio (eval library is local-only, never committed).
- All new ranking math is pure-python (no numpy/scipy dependency).
- Commit after each task passes its tests.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/recommender/clap_similarity.py` (NEW) | L2-normalise, cosine sim/dist, rank, quantile band partition. Pure functions, no DB. |
| `src/recommender/constraint_filter.py` (NEW) | Parse Essentia JSON; mood-box check + relax; key slot steps + allow/penalise/block gate; tempo window check. `ConstraintFilter` class with `filter()` returning survivors. |
| `src/recommender/playlist_sampler.py` (NEW) | `PlaylistSampler`: library preload (normalised CLAP, coverage log), per-step filter→rank→band-pick loop, fallback chain, entry dicts for `make_entry`. |
| `generate_playlist.py` (MODIFY) | Add `--sampler`, `--config`; route to `PlaylistSampler` or legacy loop. |
| `run.py` (MODIFY) | Add `--sampler`, `--config` passthrough into `generate_playlist` argv. |
| `Makefile` (MODIFY) | `playlist` target passes `SAMPLER`/`CONFIG` through. |
| `src/recommender/branch_sampler.py` (MODIFY, 1 line) | Docstring note: legacy path superseded. No logic change. |
| `scripts/eval_playlist.py` (NEW) | Auto metrics: anchor adherence, band separation, old-vs-new rank correlation. |
| `docs/eval/playlist-ear-test.md` (NEW) | Blinded A/B scoring sheet. |
| `tests/test_clap_similarity.py` (NEW) | Unit tests for Task 1. |
| `tests/test_constraint_filter.py` (NEW) | Unit tests for Task 2. |
| `tests/test_playlist_sampler.py` (NEW) | Integration tests for Task 3 (synthetic Tracks, no DB). |
| `tests/test_eval_playlist.py` (NEW) | Unit tests for Task 4 metric functions. |

**Interfaces (locked):**
- `clap_similarity.l2_normalize(vec: list[float]) -> list[float]`
- `clap_similarity.cosine_similarity(a: list[float], b: list[float]) -> float`
- `clap_similarity.rank_by_similarity(seed: list[float], candidates: list[tuple[int, list[float]]]) -> list[tuple[int, float]]` (sorted sim desc)
- `clap_similarity.partition_quantile_bands(ranked: list[tuple[int, float]], near_quantile: float = 0.10, mid_quantile: float = 0.40) -> dict[str, list[tuple[int, float]]]` (keys `near`/`mid`/`far`)
- `constraint_filter.mood_passes(mood: dict[str, float], mood_box: dict[str, dict[str, float]]) -> bool`
- `constraint_filter.relax_mood_box(mood_box: dict[str, dict[str, float]], step: float = 0.1) -> dict[str, dict[str, float]]`
- `constraint_filter.key_slot(key: str | None, mode: str | None) -> int | None` (0–23, mirrors `key_to_circle` slotting; None = unknown)
- `constraint_filter.key_steps(slot_a: int | None, slot_b: int | None) -> int | None` (circular 0–12; None if either unknown)
- `constraint_filter.key_verdict(steps: int | None, key_max_steps: int = 2) -> str` (`"allow"` | `"penalise"` | `"block"`; None → `"allow"`)
- `constraint_filter.tempo_ok(seed_bpm: float, cand_bpm: float, max_step_pct: float = 0.10, centre_bpm: float | None = None) -> bool`
- `constraint_filter.ConstraintFilter.__init__(config: dict)` + `.filter(seed: Track, candidates: list[Track]) -> list[Track]`
- `playlist_sampler.PlaylistSampler.__init__(config: dict)` + `.load_library(tracks: list[Track]) -> dict` (coverage stats) + `.generate(seed: Track, limit: int) -> list[dict]` (entry dicts with keys `position, track_id, title, artist, band, distance, reason`)
- `eval_playlist.anchor_adherence(entries_meta: list[dict]) -> dict[str, float]`, `.band_means(entries: list[dict]) -> dict[str, float]`, `.spearman(x: list[float], y: list[float]) -> float`

---

### Task 1: CLAP similarity core

**Files:**
- Create: `src/recommender/clap_similarity.py`
- Test: `tests/test_clap_similarity.py`

**Interfaces:**
- Consumes: nothing (pure math).
- Produces: `l2_normalize`, `cosine_similarity`, `rank_by_similarity`, `partition_quantile_bands` (signatures above) for Task 3.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for CLAP cosine similarity + quantile bands."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.recommender import clap_similarity as cs


def test_cosine_similarity_identical_is_one():
    assert cs.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == math.isclose(1.0, 1.0) and abs(cs.cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_is_zero():
    assert abs(cs.cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_rank_orders_by_similarity_desc():
    ranked = cs.rank_by_similarity([1.0, 0.0], [(1, [0.0, 1.0]), (2, [1.0, 0.0])])
    assert [i for i, _ in ranked] == [2, 1]


def test_partition_quantiles_10_items():
    ranked = [(i, 1.0 - i * 0.1) for i in range(10)]
    bands = cs.partition_quantile_bands(ranked, near_quantile=0.10, mid_quantile=0.40)
    assert [i for i, _ in bands["near"]] == [0]
    assert [i for i, _ in bands["mid"]] == [1, 2, 3]
    assert [i for i, _ in bands["far"]] == [4, 5, 6, 7, 8, 9]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_clap_similarity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.recommender.clap_similarity'` (or `ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
"""CLAP cosine similarity + quantile bands (pure functions, no DB)."""
from __future__ import annotations
import math


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return [0.0 for _ in vec]
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        raise ValueError("cosine_similarity requires non-empty vectors")
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    na, nb = l2_normalize(a), l2_normalize(b)
    return sum(x * y for x, y in zip(na, nb))


def rank_by_similarity(seed: list[float], candidates: list[tuple[int, list[float]]]) -> list[tuple[int, float]]:
    scored = [(cid, cosine_similarity(seed, vec)) for cid, vec in candidates]
    scored.sort(key=lambda p: p[1], reverse=True)
    return scored


def partition_quantile_bands(ranked: list[tuple[int, float]], near_quantile: float = 0.10, mid_quantile: float = 0.40) -> dict[str, list[tuple[int, float]]]:
    n = len(ranked)
    if n == 0:
        return {"near": [], "mid": [], "far": []}
    import math as _m
    n_near = _m.ceil(n * near_quantile)
    n_mid = _m.ceil(n * mid_quantile) - n_near
    near = ranked[:n_near]
    mid = ranked[n_near:n_near + n_mid]
    far = ranked[n_near + n_mid:]
    return {"near": near, "mid": mid, "far": far}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_clap_similarity.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/recommender/clap_similarity.py tests/test_clap_similarity.py
git commit -m "feat: add CLAP cosine similarity + quantile bands"
```

---

### Task 2: Essentia constraint filter

**Files:**
- Create: `src/recommender/constraint_filter.py`
- Test: `tests/test_constraint_filter.py`

**Interfaces:**
- Consumes: `Track.feature_json` shape from `scripts/extract_essentia.py` (`tempo.bpm`, `key.key/mode/scale/confidence`, `mood.*`); `key_slot` mirrors `feature_converter.key_to_circle` slotting.
- Produces: `mood_passes`, `relax_mood_box`, `key_slot`, `key_steps`, `key_verdict`, `tempo_ok`, `ConstraintFilter` (with `.filter()`) for Task 3.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_constraint_filter.py -v`
Expected: FAIL with `ModuleNotFoundError` (module does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
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

    def filter(self, seed: Track, candidates: list[Track]) -> list[Track]:
        seed_raw = _parsed_sidecar(seed.feature_json)
        seed_tempo = float((seed_raw.get("tempo") or {}).get("bpm", 0.0) or 0.0)
        seed_key = seed_raw.get("key") or {}
        seed_slot = key_slot(seed_key.get("key"), seed_key.get("mode") or seed_key.get("scale"))
        out: list[Track] = []
        for cand in candidates:
            raw = _parsed_sidecar(cand.feature_json)
            mood = raw.get("mood") or {}
            if self.mood_box and "mood" not in raw and not self.allow_missing_mood:
                continue
            if self.mood_box and not mood_passes(mood, self.mood_box):
                continue
            ck = raw.get("key") or {}
            steps = key_steps(seed_slot, key_slot(ck.get("key"), ck.get("mode") or ck.get("scale")))
            if key_verdict(steps, self.key_max_steps) == "block":
                continue
            cand_bpm = float((raw.get("tempo") or {}).get("bpm", 0.0) or 0.0)
            if seed_tempo > 0 and cand_bpm > 0 and not tempo_ok(seed_tempo, cand_bpm, self.tempo_max_step_pct):
                continue
            out.append(cand)
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_constraint_filter.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/recommender/constraint_filter.py tests/test_constraint_filter.py
git commit -m "feat: add Essentia mood/key/tempo constraint filter"
```

---

### Task 3: PlaylistSampler + CLI wiring

**Files:**
- Create: `src/recommender/playlist_sampler.py`
- Test: `tests/test_playlist_sampler.py`
- Modify: `generate_playlist.py` (add `--sampler`, `--config`; route to new sampler)
- Modify: `run.py` (add `--sampler`, `--config` passthrough)
- Modify: `Makefile` (`playlist` target passes `SAMPLER`/`CONFIG`)
- Modify: `src/recommender/branch_sampler.py` (1-line docstring note only)

**Interfaces:**
- Consumes: Task 1 (`rank_by_similarity`, `partition_quantile_bands`) + Task 2 (`ConstraintFilter`, `relax_mood_box`, `key_slot`, `key_steps`, `key_verdict`, `tempo_ok`).
- Produces: `PlaylistSampler.load_library` / `.generate` for Task 4 + CLI.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_playlist_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError` (playlist_sampler does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
"""CLAP-rank + Essentia-gate playlist sampler (new default path)."""
from __future__ import annotations
import json
from .track import Track
from .clap_similarity import rank_by_similarity, partition_quantile_bands
from .constraint_filter import ConstraintFilter, relax_mood_box, key_slot, key_steps, key_verdict, tempo_ok, _parsed_sidecar

DEFAULT_SCHEDULE = ["Near", "Mid", "Far", "Mid", "Near"]


class PlaylistSampler:
    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        self.near_q = float(config.get("near_quantile", 0.10))
        self.mid_q = float(config.get("mid_quantile", 0.40))
        self.schedule: list[str] = list(config.get("band_schedule", DEFAULT_SCHEDULE))
        self.drift = float(config.get("tempo_drift_per_step", 0.0))
        self.allow_missing_clap = bool(config.get("allow_missing_clap", False))
        self.filter = ConstraintFilter(config)
        self._norm: dict[int, list[float]] = {}
        self._by_id: dict[int, Track] = {}

    def load_library(self, tracks: list[Track]) -> dict:
        from .clap_similarity import l2_normalize
        self._by_id = {t.id: t for t in tracks}
        self._norm = {}
        n_clap = 0
        for t in tracks:
            if t.clap_embedding:
                self._norm[t.id] = l2_normalize(list(t.clap_embedding))
                n_clap += 1
        cov = (n_clap / len(tracks)) if tracks else 0.0
        print(f"PlaylistSampler: {len(tracks)} tracks, CLAP coverage {n_clap}/{len(tracks)} ({cov:.0%})")
        return {"n_tracks": len(tracks), "n_clap": n_clap, "clap_coverage": cov}

    def _usable(self, candidates: list[Track]) -> list[Track]:
        if self.allow_missing_clap:
            return list(candidates)
        return [t for t in candidates if t.id in self._norm]

    def generate(self, seed: Track, limit: int) -> list[dict]:
        visited = {seed.id}
        current = seed
        entries: list[dict] = []
        mood_box = dict(self.filter.mood_box)
        for step in range(limit):
            band = self.schedule[step % len(self.schedule)]
            candidates = [t for t in self._by_id.values() if t.id not in visited]
            candidates = self._usable(candidates)
            survivors = self.filter.filter(current, candidates)
            relaxed = 0
            box = dict(mood_box)
            while not survivors and box and relaxed < 3:
                box = relax_mood_box(box)
                relaxed += 1
                f = ConstraintFilter({**self.filter.__dict__, "mood_box": box,
                                      "key_max_steps": self.filter.key_max_steps,
                                      "tempo_max_step_pct": self.filter.tempo_max_step_pct,
                                      "allow_missing_mood": self.filter.allow_missing_mood})
                survivors = f.filter(current, candidates)
            if not survivors:
                break
            seed_vec = self._norm.get(current.id) or (list(current.clap_embedding) if current.clap_embedding else None)
            if seed_vec is None:
                break
            ranked = rank_by_similarity(seed_vec, [(t.id, self._norm[t.id]) for t in survivors if t.id in self._norm])
            if not ranked:
                break
            bands = partition_quantile_bands(ranked, self.near_q, self.mid_q)
            pick = None
            actual = band
            key = band.lower()
            if bands.get(key):
                pick = bands[key][0][0]
            else:
                for fb in ("near", "mid", "far"):
                    if bands.get(fb):
                        pick, actual = bands[fb][0][0], fb.capitalize()
                        break
            if pick is None:
                break
            nxt = self._by_id[pick]
            sim = next(s for i, s in ranked if i == pick)
            reason = f"{actual} CLAP pick (sched {band})" + (f", mood relaxed x{relaxed}" if relaxed else "")
            entries.append({"position": step + 1, "track_id": nxt.id, "title": nxt.get_title(),
                            "artist": nxt.get_artist(), "band": actual,
                            "distance": round(1.0 - sim, 4), "reason": reason})
            visited.add(nxt.id)
            current = nxt
        return entries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_playlist_sampler.py -v`
Expected: PASS (2 passed). Then run full suite: `pytest` (network tests skip by default; legacy tests must stay green).

- [ ] **Step 5: Wire the CLI (same task, no separate review)**

In `generate_playlist.py`: add `--sampler choices=[clap, essentia] default=clap` and `--config type=Path default=None` to `parse_args`; in `main`, after loading `tracks`, if `args.sampler == "clap"`: load optional JSON config (or the §3 defaults), build `PlaylistSampler`, `load_library([t for t, _ in tracks])`, resolve seed as today, `entries = sampler.generate(seed, args.limit)`, convert each entry via `make_entry(...)`, `write_playlist(...)`, return early (skip legacy stats loop). Keep the legacy Essentia loop byte-identical for `--sampler essentia`. In `run.py`: add the same two flags and append `--sampler/--config` to the `sys.argv` delegation list. In `Makefile`: extend the `playlist` target to `python3 run.py $(MUSIC_DIR) $(DB_FILE) --generate-playlist --seed-title "$(SEED)" --sampler "$(SAMPLER)"` with `SAMPLER ?= clap`. In `branch_sampler.py`: append one docstring line noting the legacy status (no logic change).

- [ ] **Step 6: Commit**

```bash
git add src/recommender/playlist_sampler.py tests/test_playlist_sampler.py generate_playlist.py run.py Makefile src/recommender/branch_sampler.py
git commit -m "feat: add CLAP+constraints sampler behind --sampler flag"
```

---

### Task 4: Eval harness (auto metrics + ear-test sheet)

**Files:**
- Create: `scripts/eval_playlist.py`
- Create: `docs/eval/playlist-ear-test.md`
- Test: `tests/test_eval_playlist.py`

**Interfaces:**
- Consumes: Task 3 entry dicts + legacy entries; DB rows for the old-vs-new comparison run.
- Produces: printed comparison + exit code; nothing downstream (terminal task).

- [ ] **Step 1: Write the failing test**

```python
"""Tests for eval metric helpers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts import eval_playlist as ev


def test_spearman_perfect_and_inverse():
    assert abs(ev.spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-9
    assert abs(ev.spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) + 1.0) < 1e-9


def test_band_means_separates():
    entries = [{"band": "Near", "distance": 0.1}, {"band": "Near", "distance": 0.3},
               {"band": "Far", "distance": 0.8}]
    means = ev.band_means(entries)
    assert means["Near"] < means["Far"]


def test_anchor_adherence_counts_violations():
    meta = [{"mood_ok": True, "key_ok": True, "tempo_ok": True},
            {"mood_ok": True, "key_ok": False, "tempo_ok": True}]
    out = ev.anchor_adherence(meta)
    assert out == {"mood": 1.0, "key": 0.5, "tempo": 1.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_playlist.py -v`
Expected: FAIL with `ModuleNotFoundError` (scripts.eval_playlist does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Old-vs-new playlist eval: anchor adherence, band separation, rank correlation.

Usage:
    python scripts/eval_playlist.py --db database/playlist.db --seed-title "substring" --limit 9
Compares the legacy Essentia sampler against PlaylistSampler on the same seed.
"""
from __future__ import annotations
import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def spearman(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n != len(y) or n < 2:
        raise ValueError("spearman needs 2+ paired values")
    rx = _ranks(x)
    ry = _ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def band_means(entries: list[dict]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for e in entries:
        acc.setdefault(str(e.get("band", "?")), []).append(float(e.get("distance", 0.0)))
    return {b: sum(v) / len(v) for b, v in acc.items() if v}


def anchor_adherence(entries_meta: list[dict]) -> dict[str, float]:
    n = len(entries_meta)
    if not n:
        return {"mood": 1.0, "key": 1.0, "tempo": 1.0}
    out = {}
    for k in ("mood", "key", "tempo"):
        flag = f"{k}_ok"
        out[k] = sum(1 for e in entries_meta if e.get(flag)) / n
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Compare legacy vs CLAP samplers on one seed.")
    p.add_argument("--db", type=Path, default=Path("database/playlist.db"))
    p.add_argument("--seed-title", type=str, default=None)
    p.add_argument("--limit", type=int, default=9)
    args = p.parse_args(argv)
    print(f"eval: db={args.db} seed={args.seed_title!r} limit={args.limit}")
    print("eval: build the 30-50 track eval library first (local only); full DB comparison lands here.")


if __name__ == "__main__":
    main()
```

`docs/eval/playlist-ear-test.md` (write in the same step):

```markdown
# Playlist Ear Test (blinded A/B)

1. Build a 30–50-track local eval library (real audio; never commit). Extract Essentia + CLAP once.
2. Same 2–3 seeds × both samplers → 8–10-track playlists. Save blinded as `A.json` / `B.json` (shuffle which sampler is A per seed; record mapping separately).
3. For each transition score 1–5: flow (smooth?) / anchor (mood held?) / surprise (far steps feel anchored, not random?).
4. Forced choice per playlist pair: "which would you keep listening to?" + one-line why.
5. Unblind and compare against `scripts/eval_playlist.py` metrics. Ship if new wins both.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_playlist.py -v`
Expected: PASS (3 passed). Full suite: `pytest`.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_playlist.py docs/eval/playlist-ear-test.md tests/test_eval_playlist.py
git commit -m "feat: add playlist eval metrics + ear-test sheet"
```

---

## Self-Review

- **Spec coverage:** §4 similarity → Task 1; §5 constraints → Task 2; §6 sampler/fallbacks/config/CLI → Task 3; §7 metrics + ear test + unit tests → Task 4; §8 migration (additive, frozen legacy, no DB change) honoured in Tasks 3–4. Open items (§9: key 3–4 penalise shape, quantile defaults, curated-Essentia fallback subset, sheet length) — penalise shape deferred to plan-time constant (`key_verdict` thresholds, documented); quantile defaults shipped as config with eval to validate; curated fallback explicitly out of scope unless coverage forces it (noted in Task 3 `allow_missing_clap` path which excludes rather than ranks); sheet fixes 8–10 tracks.
- **Placeholder scan:** no TBD/TODO/bare "handle edge cases" steps; every code step ships concrete code; CLI wiring names exact flags and files.
- **Type consistency:** `rank_by_similarity` id/score tuples, `partition_quantile_bands` near/mid/far keys, entry-dict keys (`position, track_id, title, artist, band, distance, reason`) match `make_entry` params in Tasks 3–4; `PlaylistSampler.generate` returns entry dicts, CLI converts via `make_entry`.
