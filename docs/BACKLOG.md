# Backlog — playlist-progression next steps

Session basis: 17-track test playlist run (real audio), key-extraction fixed
(frame-wise KeyExtractor), playlist regenerated with real keys. Mood description
(7 axes) implemented; metadata extraction done; QA suite modernized. Full list of
development next steps, ordered by impact.

Status legend: ✅ done · 🔶 done in part · ⬜ open

## 1. Fix the distance-band math (correctness — do first) ✅

**Problem:** `BranchSampler` band thresholds are per-axis σ units (near=0.3,
mid=0.7), but `compute_distance` previously returned distance summed across all
axes (`sqrt(Σ w·dz²)`) — ≈ 3.3σ for typical tracks — making all band thresholds
impossible to satisfy.

**Fix:** RMS-normalize distance — `d = sqrt(Σ w_i·dz_i² / n_axes)` — so 0.3/0.7
mean per-axis σ. Changes: `branch_sampler.py` (`total/n` division), fallback path
(`diff_sq / n`), docs (`BRANCHING.md` distance formula updated).

**Status:** ✅ Done (commit `843c4cf`, 2026-09-02). Verified: smoke test distance ≈ 1.0
(11 axes, z-diff=1.0); QA 25/25; `branch_playlist.json` distances fall within
expected 0–1.5 range; `select_directed_jump` holds `tempo.bpm` correctly.

## 2. Mood descriptors (7 axes) ✅

SCHEMA.md specifies `highlevel.mood_{happy,sad,aggressive,relaxed,electronic,
party,acoustic}` (0–1). **Constraint (verified):** the installed Essentia pip
build has none of these (MoodHappy etc. all absent). Requires a model-source
decision first — `essentia-models` pretrained PNNs, a standalone classifier, or
a lighter alternative. Then: extend `scripts/extract_essentia.py`, add axes to
`generate_playlist.py` AXIS_NAMES, update SCHEMA.md.

**Status:** ✅ Done (2026-09-02) — resolved via Essentia pre-trained **MusiCNN
TensorFlow** classifiers (one binary model per mood; 7 axes: happy, sad,
aggressive, relaxed, electronic, party, acoustic). `scripts/extract_essentia.py`
downloads models to `models/` on first use and selects each mood's activation
column from its `classes` metadata; `generate_playlist.py` appends the 7 axes to
AXIS_NAMES; SCHEMA.md and README updated; QA includes a mood check (25/25).
Requires `essentia-tensorflow` + TensorFlow; falls back to 0.0 if unavailable.
Minor follow-up: doc uses `highlevel.mood_*` while code/JSON use `mood.<key>`
dot notation — cosmetic, unify if desired.

## 3. Deferred — float-vector conversion gap ✅

`Track.features` was never populated by the pipeline; only `generate_playlist.py`
knows the axis layout (`AXIS_NAMES` + hand-rolled `extract_features`). Resolved via
shared query-time converter (`feature_converter.py`) — removes duplicated axis
knowledge and populates `Track.features` at ingest time.

**Status:** ✅ Done (commit `3bed92c`). `feature_converter.py` holds `AXIS_NAMES`
and `convert()`; `ingest_pipeline.py` sets `track.features` for new/existing rows;
`generate_playlist.py` imports from converter and deletes its duplicate. TDD test
`tests/test_feature_converter.py` passes (3/3).

## 4. Metadata extraction (title/artist) ✅

All 17 rows have NULL title/artist; playlist falls back to filenames. Add
ID3/Vorbis tag reading (tinytag or mutagen) into `ingest_pipeline.py`.

**Status:** ✅ Done — `ingest_pipeline.py` uses **tinytag** (`read_metadata`)
to populate `title`/`artist` (added to requirements.txt); all 17 tracks now
have real title/artist shown in playlists and `playlist_summary.txt`.

## 5. Key value as a similarity axis ✅

Only `key.confidence` feeds distance today — a tritone apart scores equal.
Add circle-of-fifths key distance (harmonically meaningful) as an axis.

**Status:** ✅ Done — `feature_converter.py` maps key+mode to 2D
circle-of-fifths coordinates (`key.fifths_x`, `key.fifths_y`) on a 24-slot
circle where relative major/minor are adjacent (C–Am = 1 step, C–G = 2 steps,
C–F# = 12 steps; enharmonics normalised; unknown → (0, 0)).
`key.confidence` retained as a reliability axis. AXIS_NAMES 18→20;
`tests/test_feature_converter.py` covers neighbours-vs-tritone, relative-minor
adjacency, enharmonics, scale-string fallback, and unknown-key origin.
SCHEMA.md and README updated.

## 6. Spectral descriptors frame-wise ✅

`scripts/extract_essentia.py:69` `es.Spectrum()(audio)` is the same full-track
FFT flaw as the key bug — centroid/rolloff/flatness from one global spectrum
instead of frame-wise mean/std. Same class of fix as #1 of the key work.

**Status:** ✅ Done (commit `382725d`, 2026-09-03). Fixed via new
`compute_spectral_descriptors()` — `FrameGenerator(2048/1024)` → `Windowing`
→ `Spectrum` per frame, with `Centroid`/`RollOff`/`Flatness` means aggregated
via numpy. JSON keys unchanged (backward-compatible with `feature_converter.py`,
20 axes). TDD regression test `tests/test_spectral_framewise.py` (frame-wise
pipeline + no-full-track-Spectrum source guard). Verified: new tests 2/2;
full suite 12 passed, 1 deselected (network).

## 7. QA suite modernization ✅

`tests/run_qa.sh` is Java-era (checks Track.java, javac, sample placeholders
that Essentia fails on). Rewrite to verify the Python pipeline: ingest idempotency,
sampler bands, JSON output, extraction test.

**Status:** ✅ Done — rewritten for the Python pipeline (DB init, imports,
BranchSampler, pipeline components, JSON output, git status, mood extraction);
QA runs 25/25.

## 8. Lower-priority hardening 🔶

- `--re-extract` / version-checked re-ingest ✅ Done (2026-09-03) —
  `scripts/extract_essentia.py` owns `EXTRACTOR_VERSION` (bumped 1.0→1.1 for
  the frame-wise spectral fix); `process_file()`/`run_pipeline()` accept
  `force`, auto-refresh rows whose stored version differs, and now also
  populate the `duration_sec` column (previously left at 0.0) plus
  title/artist on re-extract. `run.py` gains `--re-extract`. Covered by
  `tests/test_hardening.py` (skip-current / stale-version / force paths).
- Seed selection as CLI arg ✅ Done (2026-09-03) — `generate_playlist.py`
  now takes `--db`, `--seed-id`, `--limit`, `--hold-axis`, `--output`,
  `--summary` (defaults reproduce legacy behaviour: seed id 17 else highest
  id, 9 entries, hold `tempo.bpm`). Unknown seed exits 1, unknown axis or
  bad limit exits 2; empty database exits 1 instead of crashing in `max()`.
  Covered by `tests/test_hardening.py`.
- Use CLAP embeddings for distance (currently stored, never sampled) — ⏸️
  **Deferred (design, not hardening).** Fusing a 512-dim CLAP vector with the
  20-dim Essentia axes needs a weighting/normalisation design (raw fusion
  would let CLAP dominate the RMS distance and silently change all band
  behaviour). Parked until a fusion proposal exists; no code change.

## 9. Emerging issues (E2E 2026-09-03) ⬜

Full-pipeline user test on `test-playlist-music` (17 files): fresh ingest →
idempotent rerun (11 rows skipped, 6 NULL rows auto-refreshed, 1 recovered)
→ `generate_playlist.py --seed-id 17` (Orphan Girl) → 10-track playlist +
summary. Frame-wise spectral fix validated live (flatness 0.02–0.10, never
≈0.000; version 1.1 stored; `duration_sec` populated; real tinytag
title/artist throughout).

- **30s subprocess timeout too short with mood extraction — ✅ Done (2026-09-03).**
  6/17 tracks failed first pass, 5/17 still failing after retry (one
  recovered on retry — marginal, not deterministic; no file-size
  correlation). Cause: 7 sequential MusiCNN TF models + cold TF init inside
  every 30s subprocess. Fixed in three stages (`774b126..2f19a1a`, 59 tests
  green): configurable timeout (default 180s overall; DSP 60s / mood 180s
  via `--timeout`/`--dsp-timeout`/`--mood-timeout` or `EXTRACT_*_TIMEOUT_SEC`
  env); `--batch` worker loading the 7 TF graphs once per library;
  two-phase ingest (DSP stored first, mood NULL+retry via `--mood-only`,
  `--no-mood` escape hatch, one-time `--prefetch`). Design:
  `docs/superpowers/specs/2026-09-03-timeout-design.md`; plan:
  `docs/superpowers/plans/2026-09-03-timeout-fix.md`. Manual 17-track
  full-mood verification still to be run on real audio.
- **Fallback picks mislabelled with the scheduled band — ✅ Done (2026-09-04).**
  E2E entries read `"Near" d=0.64`, `"Mid" d=0.78`, `"Far" d=0.38` — impossible
  under the 0.3/0.7 thresholds. When a band was empty the generator fell back
  to global-nearest but kept the scheduled band label/reason. Fixed:
  `generate_playlist.py` labels the actual band (by distance thresholds) and
  records the fallback in `reason` (`"Fallback: scheduled Near empty,
  global-nearest (actual Mid) from seed"`); `branch_sampler.py`
  (`_filter_by_band`, `select_directed_jump`, `select_near_mid_far` far
  branch) sorts matches nearest-first per BRANCHING.md ("preferring the
  nearest candidate within that band"). Covered by
  `tests/test_branch_fallback.py` (3/3); full suite 62 passed, 1 deselected.
- **Descriptor ranges in docs are wrong — ✅ Done (docs).** SCHEMA.md claimed
  danceability 0–1 and confidence 0–1; E2E showed `danceability=1.04`
  (Essentia's range is 0–~3 by design) and `tempo.confidence=2.27`
  (beat confidence is 0–5.32). Fixed SCHEMA.md/INTEGRATION.md (+ README.md,
  BRANCHING.md for consistency) to document raw ranges. Clamping/normalising
  at extract time considered and deliberately rejected — stored raw, distance
  is z-scored.
- **`clap_embedding` TEXT vs BLOB — ✅ Done (docs/schema).** Kept TEXT JSON
  (`ingest_pipeline.py` `init_database` TEXT, `json.dumps`/`json.loads`
  round-trip); aligned `database/init.db`, SCHEMA.md, ARCHITECTURE.md,
  INTEGRATION.md and AGENTS.md to TEXT.
- **`tests/run_qa.sh` is destructive — ✅ Done.** Fixed by pointing QA at a temp DB
  (`database/qa_temp_playlist.db`) instead of wiping the real library DB.
  All 25 QA checks now pass without touching `database/playlist.db`.
- **Rerun log wording — ✅ Done (2026-09-03).** Added status indicators: `skipped:` (existing, not stale), `re-extracted:` (stale/force), `processed:` (new). Updated `process_file()` → `tuple[Track, str]` return, `run_pipeline()` → `list[tuple[Track, str]]`, updated tests and `run.py` accordingly.
- **No single ingest→playlist command; seed-by-id friction — ⬜ Open
  (usability).** E2E needed three manual steps plus a sqlite lookup to find
  Orphan Girl's id. Consider a Makefile target and `--seed-title` substring
  match.
