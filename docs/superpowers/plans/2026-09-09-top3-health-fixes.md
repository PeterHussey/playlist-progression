# Top-3 Health Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 3 highest-value health gaps: M1 docs drift, ruff+CI hygiene, red test + eval metric bugs.

**Architecture:** Docs-only rewrite for M1 (no code behavior change); config-only hygiene for ruff/CI (stdlib + yaml, no new deps); minimal targeted code fix for eval/test (small pure-function change + skip guard).

**Tech Stack:** Python 3.12, pytest, ruff (via pip, pinned `ruff==0.14.*`), GitHub Actions, SQLite (test-only tmp DBs).

**Spec:** Health audit 2026-09-09 (in-chat scorecard): findings #1,2,6 (M1 drift), #4,5,13,14 (lint/CI/ignore), #3,8,9 (red test + eval Spearman/scale). ROADMAP M1 acceptance: "no mention of Java orchestrator, three-table schema, or CLAP unused". ROADMAP M2 acceptance: "metric defined on N≥20 shared candidates or removed" + "eval output states scales differ or normalises".

## Global Constraints

- Python 3.10+ syntax only (uses `list[Type]`).
- Never commit secrets; no network calls in tests (network marker skipped by default in `tests/conftest.py`).
- Essentia/CLAP model downloads are out of scope; tests use tmp SQLite DBs and synthetic sidecars only.
- Keep diffs reviewable; format only new/touched files.

---

### Task 1: Rewrite docs/ARCHITECTURE.md to Python reality

**Files:**
- Modify: `docs/ARCHITECTURE.md` (full rewrite of Pipeline, Integration, Schema Overview, Branching, Scope, Decisions sections)
- Reference (read-only): `src/recommender/ingest_pipeline.py:64-89` (single `tracks` table DDL), `run.py:32-132` (CLI flags), `src/recommender/playlist_sampler.py:1-80` (CLAP-primary path), `src/recommender/branch_sampler.py` (deprecated path), `docs/CHARTER.md`, `docs/SCHEMA.md`

**Interfaces:**
- Consumes: `init_database()` DDL as schema source of truth; `run_pipeline()` signature as pipeline source of truth.
- Produces: Corrected architecture narrative that ROADMAP M1 acceptance can check with `grep -ri "ProcessBuilder\|Java orchestrator\|three-table\|CLAP.*unused\|CLAP.*optional secondary" docs/ARCHITECTURE.md` returning nothing.

- [ ] **Step 1: Verify drift strings exist**

Run: `grep -rn "ProcessBuilder\|Java orchestrator\|three-table\|features.*table\|runs.*table" docs/ARCHITECTURE.md | head -20`
Expected: multiple hits proving drift.

- [ ] **Step 2: Rewrite ARCHITECTURE.md sections**

Replace Java/ProcessBuilder/three-table content with: 5-stage Python pipeline (Scan → Extract DSP → Extract mood → Store → Sample → Output), subprocess via `subprocess.run()` in `src/recommender/feature_extractor.py`, single `tracks` table DDL (copy from `ingest_pipeline.py:76-86`), CLAP-primary sampler + Essentia gates + deprecated Essentia path, batch worker notes, scope boundaries matching CHARTER.

- [ ] **Step 3: Verify acceptance**

Run: `grep -ri "ProcessBuilder\|Java orchestrator" docs/ARCHITECTURE.md || echo CLEAN`
Expected: CLEAN.

- [ ] **Step 4: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: rewrite ARCHITECTURE.md to Python CLAP-primary reality"
```

### Task 2: Update AGENTS.md branching + structure sections

**Files:**
- Modify: `AGENTS.md`
- Reference: `src/recommender/feature_converter.py` (AXIS_NAMES), `src/recommender/playlist_sampler.py`, `src/recommender/clap_similarity.py`, `src/recommender/constraint_filter.py`, `README.md:59-74` (CLAP-primary description)

**Interfaces:**
- Consumes: Task 1 corrected narrative.
- Produces: AGENTS.md where `grep -n "Not used for distance" AGENTS.md` returns nothing and structure lists `playlist_sampler.py`, `clap_similarity.py`, `constraint_filter.py`.

- [ ] **Step 1: Edit Project Structure block**

Replace old tree (branch_sampler as primary, missing clap modules) with:

```
│   ├── playlist_sampler.py   # Primary sampler: CLAP rank + Essentia gates (default)
│   ├── clap_similarity.py    # CLAP cosine + quantile bands
│   ├── constraint_filter.py  # Essentia tempo/key/mood gates
│   ├── branch_sampler.py     # Deprecated: Essentia Euclidean (behind --sampler essentia)
```

- [ ] **Step 2: Edit Branching Algorithm section**

Replace Essentia z-Euclidean + σ bands primary description with CLAP cosine + quantile bands primary (near 0.10 / mid 0.40), Essentia gates, legacy Essentia path noted as deprecated behind `--sampler essentia`.

- [ ] **Step 3: Edit Feature Axes CLAP note**

Replace `**CLAP embeddings** (optional): ... **Not used for distance** in this prototype.` with `**CLAP embeddings** (default path): 512-dim ... Primary similarity via cosine in playlist_sampler.py; set --sampler essentia for legacy path.`

- [ ] **Step 4: Verify + commit**

Run: `grep -n "Not used for distance" AGENTS.md || echo CLEAN`
```bash
git add AGENTS.md
git commit -m "docs: update AGENTS.md to CLAP-primary reality"
```

### Task 3: Triage docs/BACKLOG.md overlap

**Files:**
- Modify: `docs/BACKLOG.md`
- Reference: `docs/ROADMAP.md`, `docs/CHARTER.md`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: BACKLOG with a header note pointing to ROADMAP as the only tracker and §8/§10/§11 marked superseded-or-linked.

- [ ] **Step 1: Add supersede header + section notes**

Prepend to `docs/BACKLOG.md`:

```markdown
> **Status (2026-09-09):** Historical log. ROADMAP.md is the only tracker;
> CHARTER.md holds decisions. §8 CLAP-for-distance and §10 comparison are done;
> §11 open items are folded into ROADMAP M2/M3 (eval hardening, subset test).
```

- [ ] **Step 2: Verify + commit**

Run: `grep -n "only tracker" docs/BACKLOG.md`
```bash
git add docs/BACKLOG.md
git commit -m "docs: triage BACKLOG.md, ROADMAP is the only tracker"
```

### Task 4: .gitignore hygiene (.DS_Store, .pytest_cache)

**Files:**
- Modify: `.gitignore`
- Test: `git check-ignore -v .DS_Store .pytest_cache branch_playlist.json` shows ignores

- [ ] **Step 1: Write the failing check**

Run: `git check-ignore -v .DS_Store .pytest_cache 2>&1 | head -5`
Expected: FAIL (no ignore, exit non-zero / empty output).

- [ ] **Step 2: Append ignores**

```
# macOS + pytest cache
.DS_Store
.pytest_cache/
```

- [ ] **Step 3: Verify passes**

Run: `git check-ignore -v .DS_Store .pytest_cache branch_playlist.json`
Expected: three lines showing `.gitignore` mappings.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore .DS_Store and .pytest_cache"
```

### Task 5: Wire ruff + document QA command

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `requirements.txt` (add `ruff==0.14.*` comment/pin) — actually keep runtime clean: pin ruff in workflow pip install, and add QA doc line in `AGENTS.md` Testing section.
- Test: `.venv/bin/python -m ruff check src scripts run.py generate_playlist.py tests` clean on touched files

**Interfaces:**
- Consumes: none.
- Produces: `ci.yml` running `pytest -q` + `ruff check`.

- [ ] **Step 1: Install ruff in venv (tooling only)**

Run: `.venv/bin/pip install "ruff==0.14.*" 2>&1 | tail -2`
Expected: installs cleanly.

- [ ] **Step 2: Baseline ruff on touched files**

Run: `.venv/bin/python -m ruff check src/recommender/ scripts/eval_playlist.py tests/test_eval_playlist.py tests/test_key_extraction.py 2>&1 | tail -10`
Expected: note violations to fix only in touched files (do not wholesale-format legacy).

- [ ] **Step 3: Create .github/workflows/ci.yml**

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt "ruff==0.14.*"
      - run: ruff check src scripts run.py generate_playlist.py tests
      - run: python -m pytest -q
```

- [ ] **Step 4: Document QA command in AGENTS.md Testing section**

Append under Testing/QA:

```markdown
QA: `.venv/bin/python -m pytest -q` and `.venv/bin/python -m ruff check <touched-paths>`
```

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml AGENTS.md
git commit -m "chore: add CI (pytest + ruff) and document QA command"
```

### Task 6: Fix test_key_extraction.py empty-fixture failure (TDD)

**Files:**
- Modify: `tests/test_key_extraction.py:24-27`
- Test: `tests/test_key_extraction.py`

**Interfaces:**
- Consumes: `test-playlist-music/` dir listing (has subdirs, no top-level `*.mp3`).
- Produces: `test_key_is_not_constant` skips (not fails) when <2 files found.

- [ ] **Step 1: Write the failing test expectation (RED)**

New test in `tests/test_key_extraction.py`:

```python
def test_key_missing_fixtures_skips(tmp_path, monkeypatch):
    import tests.test_key_extraction as kex
    monkeypatch.setattr(kex, "MUSIC_DIR", tmp_path)
    import pytest
    with pytest.raises(pytest.skip.Exception):
        kex.test_key_is_not_constant()
```

Run: `.venv/bin/python -m pytest tests/test_key_extraction.py::test_key_missing_fixtures_skips -v`
Expected: FAIL (function does not raise skip; raises AssertionError instead).

- [ ] **Step 2: Minimal implementation (GREEN)**

Replace:

```python
assert len(files) >= 2, f"Need >= 2 test files, found {len(files)}"
```

with:

```python
import pytest
if len(files) < 2:
    pytest.skip(f"Need >= 2 real-audio files, found {len(files)} in {MUSIC_DIR}")
```

- [ ] **Step 3: Verify green**

Run: `.venv/bin/python -m pytest tests/test_key_extraction.py -v`
Expected: 1 skipped + 1 skipped (new test passes, old test skips), no failures.

- [ ] **Step 4: Commit**

```bash
git add tests/test_key_extraction.py
git commit -m "fix: skip key-extraction test when real-audio fixtures absent"
```

### Task 7: Fix eval Spearman disjoint-set + scale annotation (TDD)

**Files:**
- Modify: `scripts/eval_playlist.py:249-277`
- Modify: `tests/test_eval_playlist.py` (add regression tests)
- Test: `.venv/bin/python -m pytest tests/test_eval_playlist.py -v`

**Interfaces:**
- Consumes: `spearman()`, `band_means()`, `_run_legacy()`, `_run_clap()`.
- Produces: `candidate_pool_spearman()` + scale-difference annotation in CLI output.

- [ ] **Step 1: Write failing regression tests (RED)**

```python
def test_candidate_pool_spearman_uses_shared_pool():
    from scripts import eval_playlist as ev
    ess = {"a": 0.1, "b": 0.5, "c": 0.9, "d": 0.4, "e": 0.7}
    clap = {"a": 0.2, "b": 0.4, "c": 0.8, "d": 0.6, "e": 0.3}
    rho, n = ev.candidate_pool_spearman(ess, clap)
    assert n == 5
    assert -1.0 <= rho <= 1.0


def test_candidate_pool_spearman_needs_overlap():
    from scripts import eval_playlist as ev
    rho, n = ev.candidate_pool_spearman({"a": 0.1}, {"b": 0.9})
    assert (rho, n) == (None, 0)
```

Run: `.venv/bin/python -m pytest tests/test_eval_playlist.py -v`
Expected: FAIL with `AttributeError: candidate_pool_spearman`.

- [ ] **Step 2: Minimal implementation (GREEN)**

In `scripts/eval_playlist.py` add:

```python
def candidate_pool_spearman(essentia_by_id: dict, clap_by_id: dict) -> tuple[float | None, int]:
    shared = sorted(set(essentia_by_id) & set(clap_by_id))
    if len(shared) < 2:
        return None, len(shared)
    x = [float(essentia_by_id[i]) for i in shared]
    y = [float(clap_by_id[i]) for i in shared]
    return spearman(x, y), len(shared)
```

And in `main()`, after computing the legacy-only intersection block, also compute pool-level correlation over all candidate tracks present in both distance maps and print `Note: CLAP cosine and Essentia z-Euclidean are on different scales; cross-sampler distances are not directly comparable.` plus extend `test_main_runs_both_paths` assertion to check `"different scales" in out`.

- [ ] **Step 3: Verify green**

Run: `.venv/bin/python -m pytest tests/test_eval_playlist.py -v`
Expected: PASS (all incl. 2 new tests).

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_playlist.py tests/test_eval_playlist.py
git commit -m "fix: pool-level eval Spearman plus scale annotation"
```
