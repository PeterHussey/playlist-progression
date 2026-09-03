# Design: 30s Subprocess Timeout with Mood Extraction

Date: 2026-09-03
Status: Approved (approach C)
Issue: 30s subprocess timeout too short with mood extraction — Open (major).
  6/17 tracks failed first pass, 5/17 still failing after retry; one
  (Little Green) recovered on retry, so the limit is marginal, not
  deterministic. No file-size correlation.

## 1. Context & Root Cause

- `src/recommender/feature_extractor.py:14` hardcodes `TIMEOUT_SECONDS = 30`
  with no override (env, flag, or per-script value).
- `scripts/extract_essentia.py:76-134` (`extract_mood`) runs inside that
  window per track: cold Python + Essentia + TensorFlow import, mood model
  download check (`download_mood_models`), resample 44.1k → 16k, then
  7 sequential `TensorflowPredictMusiCNN` graph loads + inferences
  (happy, sad, aggressive, relaxed, electronic, party, acoustic).
- `src/recommender/ingest_pipeline.py:114-166` (`_run_extraction`) is
  all-or-nothing: a mood-phase timeout discards the DSP result too, so
  libraries ingest partially (observed 12/17) and `--re-extract` inherits
  the same flakiness.
- Nondeterminism + no size correlation ⇒ fixed-cost dominance (TF init +
  7× graph load), not audio-length scaling.

Constraints agreed: extraction contract may change (new flags, batch mode).
Success criterion agreed: full library (e.g. 17/17) ingests reliably with
full DSP + mood features, time-bounded, no timeouts on retry.

## 2. Decision

Target: **Approach C — long-lived batch worker + split phases**,
built in stages so each stage unblocks independently:

- Stage 1 (unblock): configurable timeout + one-time model prefetch.
- Stage 2 (isolate): split DSP vs mood phases with separate timeouts.
- Stage 3 (root-cause): `--batch` worker amortising TF/model init.

Rejected:
- A-only (raise timeout): preserves contract but keeps N× cold init;
  slow and still marginal at library scale.
- B-only (split without batch): guarantees DSP but full-mood ingest
  stays slow (cold init per track).

## 3. Architecture (staged)

Stage 1: `DEFAULT_TIMEOUT` 30 → 180s. Override order: explicit arg >
  env `EXTRACT_TIMEOUT_SEC` > default. Pipeline calls
  `ensure_mood_models()` once at startup (`extract_essentia.py --prefetch`)
  so per-track runs never hit network.

Stage 2: `TIMEOUT_DSP = 60s`, `TIMEOUT_MOOD = 180s` (both configurable
  via env `EXTRACT_DSP_TIMEOUT_SEC` / `EXTRACT_MOOD_TIMEOUT_SEC` and
  `run.py --timeout` / `--dsp-timeout` / `--mood-timeout` or single
  `--timeout` applied to both). New script flags `--no-mood`,
  `--mood-only`.

Stage 3: new `extract_essentia.py --batch manifest.json` worker. One
  process imports Essentia/TF and loads 7 mood graphs once, loops over
  tracks, writes one sidecar per track + `batch_summary.json`.
  Single-track path retained as fallback for small runs and debugging.

## 4. Components & Files Touched

- `src/recommender/feature_extractor.py`: replace `TIMEOUT_SECONDS` with
  `DEFAULT_TIMEOUT_SEC = 180`, `DSP_TIMEOUT_SEC = 60`, `MOOD_TIMEOUT_SEC =
  180`; `run_script(script, audio, out, timeout=None)` honours
  explicit > env > default; add `ensure_mood_models()` (subprocess
  `--prefetch`).
- `scripts/extract_essentia.py`: argparse for `--no-mood`, `--mood-only`,
  `--prefetch`, `--batch <manifest>`, `--models-dir`; refactor
  `extract()` into `extract_dsp(audio)` + `extract_mood(audio)` +
  `merge()`; batch loop reuses loaded graphs across tracks.
- `src/recommender/ingest_pipeline.py`: `_run_extraction()` two-phase
  (store DSP first, then mood update); `run_pipeline(..., timeout,
  no_mood, batch)`; prefetch once; batch path writes manifest, invokes
  worker once, then stores sidecars.
- `run.py`: `--timeout INT`, `--no-mood`, `--batch` passthrough to
  `run_pipeline`.
- Docs: `docs/INTEGRATION.md` (new contract, flags, timeouts, batch
  protocol), `docs/ARCHITECTURE.md` (worker), `docs/SCHEMA.md` only if
  extractor version bump is needed.

Out of scope: CLAP path unchanged; no daemon/socket server (batch
  manifest only); no GPU work; no Spotify/streaming.

## 5. Data Flow

Single-track (stages 1–2, preserved):
  `run_pipeline → extract_essentia(audio, out)` → DSP+mood sidecar → store.

Batch (stage 3):
  pipeline writes `manifest.json: [{audio_path, output_path}]` →
  `extract_essentia.py --batch manifest.json` (single subprocess) →
  worker loads models once → per track: DSP → mood → write sidecar →
  append to `batch_summary.json {ok, failed, errors[]}` →
  pipeline reads sidecars + summary and stores.

Both paths supported; batch is explicit opt-in via `--batch` flag.
  No auto-threshold in this design (keeps behaviour predictable).

## 6. Error Handling

- DSP failure: track marked failed, partial sidecar deleted, pipeline
  continues (current behaviour preserved).
- Mood failure/timeout: DSP row kept; mood left NULL (not silent 0.0);
  track flagged `needs_mood_retry`; next run / `--re-extract` retries
  via `--mood-only`. Rationale: meets "full 17/17 with mood" success
  criterion without masking gaps.
- Batch worker: per-track try/except; never aborts whole batch on one
  track; per-track stderr captured in summary.
- Sidecar validation unchanged (exists, non-empty, valid JSON, `version`
  present) plus batch summary validation.

## 7. Testing

- Unit (mocked Essentia/TF, no real audio): timeout override precedence
  (arg > env > default); `--no-mood` skips TF import; `--mood-only`
  merges onto existing sidecar; batch manifest with fake extractor
  (3 ok + 1 failing still writes 3 sidecars + summary).
- Existing gates: `pytest` and `bash tests/run_qa.sh` must pass.
- Manual: full 17-track library ingest → 17/17 with mood, no timeouts
  on retry; record wall time before/after to demonstrate amortisation.
- No real-audio fixtures added (tests/sample_audio placeholders remain).

## 8. Open Questions for Plan

- Exact default values (180/60/180) to confirm against timed runs.
- Batch opt-in vs auto-threshold.
- Whether `EXTRACTOR_VERSION` bump is needed (only if sidecar schema
  changes; flags alone do not require it).
