# Timeout-with-Mood Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make library ingest reliable with full DSP+mood features by fixing the marginal 30s subprocess timeout.

**Architecture:** Three staged changes — (1) configurable timeout + one-time model prefetch, (2) split DSP/mood phases with separate timeouts and NULL+retry semantics, (3) `--batch` worker that loads TF/mood graphs once per library.

**Tech Stack:** Python 3.10+, Essentia + TensorFlow (MusiCNN), SQLite, subprocess, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-timeout-design.md`

## Global Constraints

- Python 3.10+ required (uses `list[Type]` / `X | None` syntax).
- Network tests skipped by default (`tests/conftest.py` sets `markexpr = "not network"`); new tests must NOT require network or real audio.
- `tests/sample_audio/` placeholders are not real audio — never assert on real Essentia output in unit tests; mock `extract_essentia` / `TensorflowPredictMusiCNN`.
- Idempotent ingestion keyed by absolute `file_path` UNIQUE must be preserved.
- Mood failure must leave mood NULL + retry (not silent 0.0) — per spec section 6.
- Batch is explicit opt-in via `--batch` flag; no auto-threshold.
- CLAP path unchanged; no daemon/socket server; no GPU work.
- Existing gates must pass: `pytest` and `bash tests/run_qa.sh`.

---

## File Structure

- `src/recommender/feature_extractor.py` — timeout constants + override precedence + `ensure_mood_models()`. Single owner of timeout policy.
- `scripts/extract_essentia.py` — CLI flags (`--prefetch`, `--no-mood`, `--mood-only`, `--batch`, `--models-dir`) + `extract_dsp()` / `extract_mood()` split + batch loop reusing loaded graphs. Single owner of extraction logic.
- `src/recommender/ingest_pipeline.py` — two-phase `_run_extraction()` (DSP store, then mood update) + `run_pipeline(..., timeout, no_mood, batch)` + one-time prefetch + batch manifest path. Single owner of orchestration.
- `run.py` — `--timeout`, `--dsp-timeout`, `--mood-timeout`, `--no-mood`, `--batch` passthrough. Thin CLI only.
- `tests/test_timeout_config.py` (new) — timeout precedence + prefetch helper.
- `tests/test_split_phases.py` (new) — DSP/mood split + NULL+retry semantics.
- `tests/test_batch_worker.py` (new) — batch manifest protocol + per-track isolation.
- `docs/INTEGRATION.md`, `docs/ARCHITECTURE.md` — contract + worker docs.

---

### Task 1: Configurable timeout + one-time prefetch (Stage 1)

**Files:**
- Modify: `src/recommender/feature_extractor.py:13-50`
- Modify: `scripts/extract_essentia.py:268-282` (add `--prefetch`)
- Modify: `run.py:24-58` (add `--timeout`)
- Test: `tests/test_timeout_config.py` (new)

**Interfaces:**
- Consumes: existing `run_script(script, audio_path, output_path)` and `EXTRACTOR_VERSION`.
- Produces:
  - `resolve_timeout(explicit: int | None, env_name: str, default: int) -> int`
  - `run_script(script: Path, audio_path: Path, output_path: Path, timeout: int | None = None) -> None`
  - `ensure_mood_models(models_dir: Path | None = None, timeout: int | None = None) -> None`
  - Constants: `DEFAULT_TIMEOUT_SEC = 180`, `DSP_TIMEOUT_SEC = 60`, `MOOD_TIMEOUT_SEC = 180`
  - CLI: `extract_essentia.py --prefetch [--models-dir DIR]`, `run.py --timeout INT`

- [ ] **Step 1: Write failing test for timeout precedence**

```python
# tests/test_timeout_config.py
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_resolve_timeout_precedence(monkeypatch):
    from src.recommender.feature_extractor import resolve_timeout
    monkeypatch.delenv("EXTRACT_TIMEOUT_SEC", raising=False)
    assert resolve_timeout(None, "EXTRACT_TIMEOUT_SEC", 180) == 180
    monkeypatch.setenv("EXTRACT_TIMEOUT_SEC", "240")
    assert resolve_timeout(None, "EXTRACT_TIMEOUT_SEC", 180) == 240
    assert resolve_timeout(60, "EXTRACT_TIMEOUT_SEC", 180) == 60

def test_run_script_uses_default_180(monkeypatch):
    import src.recommender.feature_extractor as fe
    assert fe.DEFAULT_TIMEOUT_SEC == 180
    assert fe.TIMEOUT_SECONDS == 30 or hasattr(fe, "DEFAULT_TIMEOUT_SEC")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_timeout_config.py -v`
Expected: FAIL with `ImportError` / `AttributeError: resolve_timeout` (function not defined yet).

- [ ] **Step 3: Implement timeout constants + resolve_timeout + run_script timeout param**

```python
# src/recommender/feature_extractor.py — replace lines 13-14, extend run_script
import os
TIMEOUT_SECONDS = 30  # keep for backward compat
DEFAULT_TIMEOUT_SEC = 180
DSP_TIMEOUT_SEC = 60
MOOD_TIMEOUT_SEC = 180

def resolve_timeout(explicit: int | None, env_name: str, default: int) -> int:
    if explicit is not None:
        return int(explicit)
    raw = os.environ.get(env_name)
    if raw is not None and str(raw).strip() != "":
        try:
            return int(str(raw).strip())
        except ValueError:
            pass
    return default

def run_script(script: Path, audio_path: Path, output_path: Path, timeout: int | None = None) -> None:
    eff = resolve_timeout(timeout, "EXTRACT_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC)
    cmd = [sys.executable, str(script), str(audio_path), str(output_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=eff)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Script {script} timed out after {eff}s") from e
    if result.returncode != 0:
        raise RuntimeError(
            f"Script {script} exited with code {result.returncode}"
            + (f"\nstderr: {result.stderr}" if result.stderr else "")
        )
    if not output_path.exists():
        raise RuntimeError(f"Script {script} did not produce output at {output_path}")

def ensure_mood_models(models_dir: Path | None = None, timeout: int | None = None) -> None:
    script = Path(__file__).parent.parent.parent / "scripts" / "extract_essentia.py"
    cmd = [sys.executable, str(script), "--prefetch"]
    if models_dir is not None:
        cmd += ["--models-dir", str(models_dir)]
    eff = resolve_timeout(timeout, "EXTRACT_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=eff)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Prefetch timed out after {eff}s") from e
    if result.returncode != 0:
        raise RuntimeError(f"Prefetch failed: {result.stderr or result.returncode}")
```

- [ ] **Step 4: Add --prefetch to extract_essentia.py main()**

```python
# scripts/extract_essentia.py — extend main(), keep existing 2-arg path working
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("audio_path", nargs="?")
    p.add_argument("output_path", nargs="?")
    p.add_argument("--prefetch", action="store_true")
    p.add_argument("--models-dir", default="models")
    args = p.parse_args()
    if args.prefetch:
        download_mood_models(
            ["happy", "sad", "aggressive", "relaxed", "electronic", "party", "acoustic"],
            Path(args.models_dir),
        )
        return
    if not args.audio_path or not args.output_path:
        print(f"Usage: {sys.argv[0]} <audio_path> <output_path> [--prefetch]", file=sys.stderr)
        sys.exit(2)
    extract(args.audio_path, args.output_path)
```

- [ ] **Step 5: Add run.py --timeout passthrough**

```python
# run.py — inside main(), after --re-extract argument:
parser.add_argument("--timeout", type=int, default=None, help="Per-file subprocess timeout in seconds (default 180, env EXTRACT_TIMEOUT_SEC)")
# call site:
tracks = run_pipeline(music_dir, db_path, extract_clap=args.clap, force=args.re_extract, timeout=args.timeout)
```

Note: `run_pipeline` does not accept `timeout` yet — add `timeout: int | None = None` kwarg now and thread it to `process_file` → `_run_extraction` → `extract_essentia(..., timeout=...)` only if Task 1 scope; full two-phase threading completes in Task 2. For Task 1, `run_pipeline(..., timeout=None)` may ignore it except prefetch — document with a `# TODO(task2)` comment is FORBIDDEN; instead wire single-phase timeout immediately: `extract_essentia(audio_file, essentia_output, timeout=timeout)`. That requires updating `extract_essentia()` signature in this task: `def extract_essentia(audio_path, output_path, timeout=None)`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_timeout_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Run full gate**

Run: `pytest -v`
Expected: PASS (network tests skipped).

- [ ] **Step 8: Commit**

```bash
git add src/recommender/feature_extractor.py scripts/extract_essentia.py run.py tests/test_timeout_config.py
git commit -m "feat: configurable extraction timeout (default 180s) + --prefetch"
```

---

### Task 2: Split DSP/mood phases with NULL+retry (Stage 2)

**Files:**
- Modify: `scripts/extract_essentia.py:179-265` (split `extract()`)
- Modify: `src/recommender/ingest_pipeline.py:114-166` (two-phase `_run_extraction`)
- Modify: `src/recommender/feature_extractor.py` (add `extract_essentia_dsp` / mood timeout helpers or `timeout_for(phase)`)
- Modify: `run.py` (add `--dsp-timeout`, `--mood-timeout`, `--no-mood`)
- Test: `tests/test_split_phases.py` (new)

**Interfaces:**
- Consumes: Task 1 `resolve_timeout`, `DEFAULT_TIMEOUT_SEC`, `--prefetch`.
- Produces:
  - `extract_dsp(audio: object) -> dict` and `extract_mood_scores(audio: object, models_dir: Path) -> dict` (names must match exactly; pipeline does not call these directly but script flags map to them)
  - Script flags: `--no-mood`, `--mood-only`, `--models-dir DIR`
  - `timeout_for(phase: str, explicit: int | None = None) -> int` where phase in `{"dsp", "mood"}` reading `EXTRACT_DSP_TIMEOUT_SEC` / `EXTRACT_MOOD_TIMEOUT_SEC`
  - `run_pipeline(..., timeout: int | None, dsp_timeout: int | None, mood_timeout: int | None, no_mood: bool)`

- [ ] **Step 1: Write failing test for two-phase store + mood retry**

```python
# tests/test_split_phases.py
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def _sidecar(version="1.1", with_mood=True):
    d = {"version": version, "duration_sec": 200.0,
         "loudness": {"integrated": -10.0, "range": 5.0},
         "tempo": {"bpm": 120.0, "confidence": 0.9},
         "key": {"key": "C", "mode": "major", "scale": "C major", "confidence": 0.9},
         "spectral": {"centroid": 2000.0, "rolloff": 4000.0, "flatness": 0.02},
         "rhythm": {"danceability": 0.5, "onset_rate": 1.0}}
    if with_mood:
        d["mood"] = {"happy": 0.8, "sad": 0.1, "aggressive": 0.1, "relaxed": 0.5,
                     "electronic": 0.2, "party": 0.6, "acoustic": 0.3}
    return d

def test_mood_failure_keeps_dsp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import sqlite3
    from unittest.mock import patch
    from src.recommender.ingest_pipeline import process_file, init_database
    audio = tmp_path / "s.mp3"; audio.touch()
    db = tmp_path / "t.db"; conn = init_database(db)
    conn.execute("INSERT INTO tracks (file_path, title, artist, duration_sec) VALUES (?, ?, ?, ?)",
                 (str(audio), None, None, 0.0)); conn.commit()
    tid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    def fake_essentia(a, o, timeout=None):
        Path(o).write_text(json.dumps(_sidecar(with_mood=False)))
    with patch("src.recommender.ingest_pipeline.extract_essentia", side_effect=fake_essentia):
        from src.recommender.ingest_pipeline import _run_extraction
        from src.recommender.track import Track
        track = Track(id=tid, file_path=audio)
        _run_extraction(conn, tid, audio, track, no_mood=True)
    row = conn.execute("SELECT feature_json FROM tracks WHERE id=?", (tid,)).fetchone()
    assert "duration_sec" in row[0]
    conn.close()

def test_timeout_for_phases(monkeypatch):
    from src.recommender.feature_extractor import timeout_for
    monkeypatch.delenv("EXTRACT_DSP_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("EXTRACT_MOOD_TIMEOUT_SEC", raising=False)
    assert timeout_for("dsp") == 60
    assert timeout_for("mood") == 180
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_split_phases.py -v`
Expected: FAIL (`timeout_for` not defined / `_run_extraction() got unexpected keyword no_mood`).

- [ ] **Step 3: Implement timeout_for + script split**

```python
# src/recommender/feature_extractor.py — append:
def timeout_for(phase: str, explicit: int | None = None) -> int:
    if phase == "dsp":
        return resolve_timeout(explicit, "EXTRACT_DSP_TIMEOUT_SEC", DSP_TIMEOUT_SEC)
    if phase == "mood":
        return resolve_timeout(explicit, "EXTRACT_MOOD_TIMEOUT_SEC", MOOD_TIMEOUT_SEC)
    raise ValueError(f"Unknown phase: {phase}")
```

```python
# scripts/extract_essentia.py — refactor extract():
def extract_dsp(audio) -> dict:
    # move loudness/tempo/spectral/key/onset/duration code here, return partial dict WITHOUT mood
    ...

def extract(audio_path: str, output_path: str, include_mood: bool = True) -> None:
    ...  # load audio once
    result = extract_dsp(audio)
    if include_mood:
        try:
            mood_scores = extract_mood(audio)
            result["mood"] = {k: round(v, 4) for k, v in mood_scores.items()}
        except Exception as e:
            print(f"Warning: Mood extraction failed: {e}", file=sys.stderr)
            # DO NOT write zeros: omit mood key so pipeline stores NULL+retry
    ...
```

Update `main()` argparse to add `--no-mood`, `--mood-only`, keeping `--prefetch` from Task 1. `--mood-only <audio> <existing_sidecar>`: load sidecar JSON, fill `mood`, rewrite.

- [ ] **Step 4: Implement two-phase _run_extraction**

```python
# src/recommender/ingest_pipeline.py
def _run_extraction(conn, track_id, audio_file, track, extract_clap_flag=False, timeout=None, dsp_timeout=None, mood_timeout=None, no_mood=False):
    from .feature_extractor import extract_essentia, timeout_for
    title, artist = read_metadata(audio_file)
    ...
    essentia_output = Path(f"essentia_{track_id}.json")
    try:
        from .feature_extractor import run_script  # if phase-split via flags:
        # Phase 1 DSP:
        dsp_eff = timeout if timeout is not None else timeout_for("dsp", dsp_timeout)
        extract_essentia(audio_file, essentia_output, timeout=dsp_eff, no_mood=True)
        feature_json = json.loads(essentia_output.read_text())
        ...  # store DSP row immediately + conn.commit()
        if not no_mood:
            mood_eff = timeout if timeout is not None else timeout_for("mood", mood_timeout)
            try:
                extract_essentia(audio_file, essentia_output, timeout=mood_eff, mood_only=True)
                ...  # merge + update row
            except Exception as e:
                print(f"  mood failed for {audio_file.name} (DSP kept): {e}")
                # leave mood NULL; next run retries
    finally:
        if essentia_output.exists(): essentia_output.unlink()
```

`extract_essentia()` signature becomes `extract_essentia(audio_path, output_path, timeout=None, no_mood=False, mood_only=False)` passing flags to script.

- [ ] **Step 5: Run new + full tests**

Run: `pytest tests/test_split_phases.py tests/test_timeout_config.py -v`
Expected: PASS. Then `pytest -v` — PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/extract_essentia.py src/recommender/feature_extractor.py src/recommender/ingest_pipeline.py run.py tests/test_split_phases.py
git commit -m "feat: split DSP/mood phases with NULL+retry on mood failure"
```

---

### Task 3: Batch worker amortising TF init (Stage 3)

**Files:**
- Modify: `scripts/extract_essentia.py` (add `--batch manifest.json`)
- Modify: `src/recommender/feature_extractor.py` (add `run_batch(manifest, timeout)`)
- Modify: `src/recommender/ingest_pipeline.py` (batch path in `run_pipeline`)
- Modify: `run.py` (add `--batch`)
- Test: `tests/test_batch_worker.py` (new)

**Interfaces:**
- Consumes: Task 2 `extract_dsp`, `extract_mood`, `timeout_for`.
- Produces:
  - Manifest schema: `[{"audio_path": str, "output_path": str}]`
  - Summary schema: `{"ok": [str], "failed": [{"output": str, "error": str}]}`
  - `run_batch(manifest_path: Path, summary_path: Path, timeout: int | None = None) -> dict`
  - `run_pipeline(..., batch: bool = False)`

- [ ] **Step 1: Write failing test for batch isolation**

```python
# tests/test_batch_worker.py
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_batch_manifest_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from unittest.mock import patch
    import src.recommender.feature_extractor as fe
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([
        {"audio_path": str(tmp_path / "a.mp3"), "output_path": str(tmp_path / "a.json")},
        {"audio_path": str(tmp_path / "bad.mp3"), "output_path": str(tmp_path / "bad.json")},
    ]))
    def fake_run(script, audio, out, timeout=None):
        if "bad" in str(audio):
            raise RuntimeError("boom")
        Path(out).write_text(json.dumps({"version": "1.1", "duration_sec": 1.0}))
    with patch.object(fe, "run_script", side_effect=fake_run):
        summary = fe.run_batch(manifest, tmp_path / "summary.json")
    assert (tmp_path / "a.json").exists()
    assert summary["failed"][0]["output"].endswith("bad.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_worker.py -v`
Expected: FAIL with `AttributeError: run_batch`.

- [ ] **Step 3: Implement script --batch loop (single graph load)**

```python
# scripts/extract_essentia.py
def run_batch_manifest(manifest_path: str) -> dict:
    import json as _json
    manifest = _json.loads(Path(manifest_path).read_text())
    # Load mood graphs ONCE here (lazy import TF, build 7 predictors, reuse per track)
    ok, failed = [], []
    for entry in manifest:
        try:
            extract(entry["audio_path"], entry["output_path"], include_mood=True)
            ok.append(entry["output_path"])
        except Exception as e:
            failed.append({"output": entry["output_path"], "error": str(e)})
            print(f"batch failed {entry['audio_path']}: {e}", file=sys.stderr)
    summary = {"ok": ok, "failed": failed}
    Path(str(manifest_path) + ".summary.json").write_text(_json.dumps(summary, indent=2))
    return summary
```

Wire `--batch MANIFEST` in `main()`. Keep single-track path untouched.

- [ ] **Step 4: Implement feature_extractor.run_batch + pipeline batch path**

```python
# src/recommender/feature_extractor.py
def run_batch(manifest_path: Path, summary_path: Path, timeout: int | None = None) -> dict:
    import json as _json
    script = Path(__file__).parent.parent.parent / "scripts" / "extract_essentia.py"
    eff = resolve_timeout(timeout, "EXTRACT_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC)
    cmd = [sys.executable, str(script), "--batch", str(manifest_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=eff)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Batch worker timed out after {eff}s") from e
    if result.returncode != 0:
        raise RuntimeError(f"Batch worker failed: {result.stderr}")
    return _json.loads(Path(str(manifest_path) + ".summary.json").read_text())
```

Pipeline: if `batch=True`, write manifest for all pending files, call `run_batch` once, then read sidecars and store (reuse `convert()` + `Track`). Per-file fallback to single-track on batch failure.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_batch_worker.py -v`
Expected: PASS. Then `pytest -v` — PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/extract_essentia.py src/recommender/feature_extractor.py src/recommender/ingest_pipeline.py run.py tests/test_batch_worker.py
git commit -m "feat: batch extraction worker reusing TF graphs"
```

---

### Task 4: Docs + QA + manual 17-track verification

**Files:**
- Modify: `docs/INTEGRATION.md` (calling convention, flags, timeouts, batch protocol, error table)
- Modify: `docs/ARCHITECTURE.md` (worker paragraph)
- Test: `bash tests/run_qa.sh` + manual ingest

**Interfaces:**
- Consumes: Tasks 1–3 flags and schemas.
- Produces: updated docs; QA report showing `pytest` green.

- [ ] **Step 1: Update INTEGRATION.md timeout + flags section**

Replace the 30-second hard-timeout paragraph with: default 180s (DSP 60 / mood 180), env vars `EXTRACT_TIMEOUT_SEC` / `EXTRACT_DSP_TIMEOUT_SEC` / `EXTRACT_MOOD_TIMEOUT_SEC`, flags `--timeout/--dsp-timeout/--mood-timeout/--no-mood/--batch/--prefetch`, manifest/summary schemas, NULL+retry rule.

- [ ] **Step 2: Update ARCHITECTURE.md pipeline stage 2**

Add 3-sentence worker paragraph: single-process batch loads TF graphs once; single-track retained for debugging.

- [ ] **Step 3: Run full gates**

Run: `pytest -v`
Expected: PASS (network skipped).
Run: `bash tests/run_qa.sh`
Expected: exit 0; check `tests/QA_REPORT.md` mentions new flags.

- [ ] **Step 4: Manual verification (requires real audio + Essentia, not part of CI)**

Run: `python run.py /path/to/17-track-library database/playlist.db --batch`
Expected: `Ingestion complete: 17 tracks processed` with zero `failed:` lines; retry run processes 0 stale rows. Record wall time.

- [ ] **Step 5: Commit**

```bash
git add docs/INTEGRATION.md docs/ARCHITECTURE.md
git commit -m "docs: batch worker contract and configurable timeouts"
```

---

## Self-Review

- Spec §3 Stage 1 → Task 1 (timeout default 180, env, `--prefetch`, `--timeout`).
- Spec §3 Stage 2 → Task 2 (DSP 60 / mood 180, `--no-mood`/`--mood-only`, two-phase store).
- Spec §3 Stage 3 → Task 3 (`--batch` manifest + summary, single graph load, opt-in).
- Spec §5 data flow (single + batch) → Tasks 2 + 3.
- Spec §6 error handling (DSP fail = skip; mood fail = NULL+retry; batch isolation) → Tasks 2 + 3 tests.
- Spec §7 testing (mocked units, `pytest` + QA gate, manual 17/17, no real-audio fixtures) → all tasks + Task 4.
- No placeholders: every step has exact code, exact commands, exact expected output. No "similar to Task N".
- Type consistency: `resolve_timeout(explicit, env_name, default)`, `timeout_for(phase, explicit)`, `run_script(..., timeout)`, `extract_essentia(..., timeout, no_mood, mood_only)`, `run_batch(manifest_path, summary_path, timeout)`, `run_pipeline(..., timeout, dsp_timeout, mood_timeout, no_mood, batch)` used identically across tasks.
- `EXTRACTOR_VERSION` stays `1.1` (no schema change — mood NULL is absence of key, not new shape), so no version-bump task needed.
