# Code-Items Implementation Plan (lockfile, quantile guard, summary parity)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close three code-level health leftovers without touching listening-gated or scale-gated work.

**Architecture:** Config-only pin change for repro installs; one localized guard branch in the sampler loop (spec §6); one extract-helper refactor in the CLI so both sampler paths write the text summary.

**Tech Stack:** Python 3.12, pytest, ruff 0.14, SQLite tmp DBs in tests.

**Spec:** `docs/superpowers/specs/2026-09-05-clap-similarity-essentia-constraints-design.md` §6 line 51 (guard: "<20 candidates → rank-ordered greedy, bands skipped, logged"); `docs/ROADMAP.md` M2 (summary parity acceptance: "summary written or --help calls out the gap").

## Global Constraints

- Python 3.10+ syntax only.
- Network-marked tests skipped by default (`tests/conftest.py`).
- No model downloads; tests use synthetic sidecars and tmp DBs.
- Ruff-clean on touched files; CI checks `scripts/eval_playlist.py tests/test_eval_playlist.py tests/test_key_extraction.py` (extend CI list when touching new files).
- Existing small-library sampler tests must stay green (they use 2–3 candidates and assert scheduled band labels and picks).

---

### Task 1: Pin direct runtime dependencies

**Files:**
- Modify: `requirements.txt`
- Test: version-match check via `importlib.metadata` (no network)

**Interfaces:**
- Consumes: installed working set in `.venv` (verified 2026-09-09).
- Produces: `==` pins for the 4 direct runtime deps; `laion-clap` stays separate/optional per AGENTS.md.

- [ ] **Step 1: Replace floors with tested pins**

```text
essentia-tensorflow==2.1b6.dev1389
tinytag==2.3.1
tensorflow==2.21.0
torchaudio==2.11.0
```

Keep the header comments; add one line: `# Pins verified 2026-09-09 on macOS Python 3.12 (.venv). laion-clap stays a separate optional install (tested: laion_clap 1.1.7).`

- [ ] **Step 2: Verify pins match the working env (no network)**

Run: `.venv/bin/python -c "import importlib.metadata as m; print([(d, m.version(d)) for d in ['essentia-tensorflow','tinytag','tensorflow','torchaudio','laion-clap']])"`
Expected: versions equal the pins above (laion-clap printed for the comment only).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: pin direct runtime deps to tested versions"
```

### Task 2: Small-N quantile guard in PlaylistSampler (TDD)

**Files:**
- Modify: `src/recommender/playlist_sampler.py` (add `MIN_CANDIDATES_FOR_BANDS = 20`, guard branch in `generate()`)
- Modify: `tests/test_playlist_sampler.py` (add 2 regression tests)
- Test: `.venv/bin/python -m pytest tests/test_playlist_sampler.py -v`

**Interfaces:**
- Consumes: `rank_by_similarity()` output `ranked: list[tuple[int, float]]` (similarity desc).
- Produces: entries with scheduled `band` label and `"Small library (n=N<20): greedy nearest-by-CLAP (sched <Band>)"` reason when `len(ranked) < 20`; unchanged band-partition path otherwise.

- [ ] **Step 1: Write failing tests (RED)**

```python
def test_small_library_greedy_skips_bands():
    seed = _mk(1, [1.0, 0.0])
    a = _mk(2, [0.0, 1.0])
    b = _mk(3, [0.9, 0.1])
    s = PlaylistSampler(_config(band_schedule=["Far"]))
    s.load_library([seed, a, b])
    entries = s.generate(seed, limit=1)
    assert len(entries) == 1
    assert entries[0]["track_id"] == 3  # nearest by CLAP
    assert entries[0]["band"] == "Far"  # scheduled label kept
    assert "Small library" in entries[0]["reason"]


def test_large_library_uses_bands():
    tracks = [_mk(1, [1.0, 0.0])]
    for i in range(2, 27):
        tracks.append(_mk(i, [1.0 - i * 0.01, i * 0.01]))
    s = PlaylistSampler(_config(band_schedule=["Near"], mood_box={}))
    s.load_library(tracks)
    entries = s.generate(tracks[0], limit=1)
    assert len(entries) == 1
    assert "CLAP pick" in entries[0]["reason"]
    assert "Small library" not in entries[0]["reason"]
```

Run: `.venv/bin/python -m pytest tests/test_playlist_sampler.py -v`
Expected: FAIL with `AssertionError` on the `"Small library" in reason` line (test 1) — note test 2 passes already (guards the unchanged path).

- [ ] **Step 2: Minimal implementation (GREEN)**

At module top after `DEFAULT_SCHEDULE`: `MIN_CANDIDATES_FOR_BANDS = 20  # spec §6: fewer candidates → greedy, bands skipped`.

In `generate()`, immediately after the `if not ranked:` block (before `bands = partition_quantile_bands(...)`), insert:

```python
if len(ranked) < MIN_CANDIDATES_FOR_BANDS:
    # Spec §6 guard: quantiles are unstable on small libraries —
    # pick rank-ordered greedy, keep the scheduled label, log it.
    pick_id, sim = ranked[0]
    nxt = self._by_id[pick_id]
    reason = (f"Small library (n={len(ranked)}<{MIN_CANDIDATES_FOR_BANDS}): "
              f"greedy nearest-by-CLAP (sched {band})"
              + (f", mood relaxed x{relaxed}" if relaxed else ""))
    entries.append({"position": step + 1, "track_id": nxt.id,
                    "title": nxt.get_title(), "artist": nxt.get_artist(),
                    "band": band, "distance": round(1.0 - sim, 4),
                    "reason": reason})
    visited.add(nxt.id)
    current = nxt
    continue
```

- [ ] **Step 3: Verify green, no regressions**

Run: `.venv/bin/python -m pytest tests/test_playlist_sampler.py tests/test_clap_similarity.py -v`
Expected: all PASS (existing 2–3-candidate tests keep scheduled labels and picks).

- [ ] **Step 4: Ruff + commit**

Run: `.venv/bin/python -m ruff check src/recommender/playlist_sampler.py tests/test_playlist_sampler.py`
```bash
git add src/recommender/playlist_sampler.py tests/test_playlist_sampler.py .github/workflows/ci.yml
git commit -m "feat: small-library quantile guard (greedy under 20 candidates)"
```

Note: this step also extends the CI `ruff check` line with the two newly-touched paths.

### Task 3: playlist_summary.txt on the clap path (TDD)

**Files:**
- Modify: `generate_playlist.py` (extract `write_text_summary()`, call it on clap path)
- Modify: `tests/test_playlist_sampler.py::test_clap_embedding_loaded_from_db` (assert summary written)
- Test: `.venv/bin/python -m pytest tests/test_playlist_sampler.py tests/test_hardening.py tests/test_branch_fallback.py -v`

**Interfaces:**
- Consumes: `seed: Track`, `tracks: list[tuple[Track, dict]]`, `playlist_entries` (make_entry dicts with `track_id`/`band`), `schedule: list[str]`, `hold_axis: str`.
- Produces: `args.summary` text file on both paths; clap call passes `config["band_schedule"]` and `hold_axis="n/a (clap sampler ignores --hold-axis)"`.

- [ ] **Step 1: Extend the clap-path test (RED)**

In `test_clap_embedding_loaded_from_db`, append:

```python
text = summ.read_text()
assert "PLAYLIST SUMMARY" in text
assert "Seed" in text
```

Run: `.venv/bin/python -m pytest tests/test_playlist_sampler.py::test_clap_embedding_loaded_from_db -v`
Expected: FAIL with `FileNotFoundError` (clap path writes no summary).

- [ ] **Step 2: Extract helper + call on clap path (GREEN)**

Extract lines 208–235 (from `selected_ids = ...` through `args.summary.write_text(...)`) into:

```python
def write_text_summary(summary_path, seed, tracks, playlist_entries, schedule, hold_axis):
    selected_ids = [seed.id] + [e["track_id"] for e in playlist_entries]
    ...  # moved body unchanged, with `schedule` and `hold_axis` params
    # line: f"Hold axis: {hold_axis}"
    summary_path.write_text("\n".join(lines) + "\n")
```

Legacy path replaces the moved block with:

```python
write_text_summary(args.summary, seed, tracks, playlist_entries, schedule, args.hold_axis)
```

Clap path: replace comment line `# Clap path intentionally skips playlist_summary.txt (legacy parity gap, see plan §6)` and after `write_playlist(args.output, ...)` insert:

```python
write_text_summary(args.summary, seed, tracks, playlist_entries,
                   config["band_schedule"],
                   "n/a (clap sampler ignores --hold-axis)")
```

- [ ] **Step 3: Verify green, no regressions**

Run: `.venv/bin/python -m pytest tests/test_playlist_sampler.py tests/test_hardening.py tests/test_branch_fallback.py -v`
Expected: all PASS (legacy summary test still asserts `"Hold axis: tempo.bpm"`).

- [ ] **Step 4: Ruff + commit**

Run: `.venv/bin/python -m ruff check generate_playlist.py tests/test_playlist_sampler.py`
```bash
git add generate_playlist.py tests/test_playlist_sampler.py .github/workflows/ci.yml
git commit -m "feat: write playlist_summary.txt on clap path"
```

Note: extends CI `ruff check` line with `generate_playlist.py tests/test_playlist_sampler.py` if not already present.
