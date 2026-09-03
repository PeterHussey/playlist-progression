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

## 6. Spectral descriptors frame-wise ⬜

`scripts/extract_essentia.py:69` `es.Spectrum()(audio)` is the same full-track
FFT flaw as the key bug — centroid/rolloff/flatness from one global spectrum
instead of frame-wise mean/std. Same class of fix as #1 of the key work.

**Status:** ⬜ Open. Diagnostics still show over-aggregated spectral values
(e.g. flatness≈0.000, danceability>1.0, tempo.confidence>1.0), consistent with
full-track FFT artifacts. Needs the frame-wise fix.

## 7. QA suite modernization ✅

`tests/run_qa.sh` is Java-era (checks Track.java, javac, sample placeholders
that Essentia fails on). Rewrite to verify the Python pipeline: ingest idempotency,
sampler bands, JSON output, extraction test.

**Status:** ✅ Done — rewritten for the Python pipeline (DB init, imports,
BranchSampler, pipeline components, JSON output, git status, mood extraction);
QA runs 25/25.

## 8. Lower-priority hardening 🔶

- `--re-extract` / version-checked re-ingest (reruns skip existing rows → stale features) — ⬜ Open
- Seed selection as CLI arg (currently hardcoded `id == 17` in generate_playlist.py) — 🔶 partial: fallback to highest-id seed added (never crashes); CLI arg still open
- Use CLAP embeddings for distance (currently stored, never sampled) — ⬜ Open
