# Backlog — playlist-progression next steps

Session basis: 17-track test playlist run (real audio), key-extraction fixed
(frame-wise KeyExtractor), playlist regenerated with real keys. Full list of
development next steps, ordered by impact.

## 1. Fix the distance-band math (correctness — do first)

**Problem:** `BranchSampler` band thresholds are per-axis σ units (near=0.3,
mid=0.7), but `compute_distance` returns distance summed across all 11 axes
(sqrt(Σ w·dz²) ≈ 3.3σ for typical tracks). Nothing ever falls ≤0.3σ, so every
Near/Mid/Far selection falls through to the nearest-unvisited fallback in
`generate_playlist.py:84-89`. Evidence: generated playlist "Near" entries have
distances 2.378–3.334 — impossible for a ≤0.3 threshold. `select_directed_jump`
qualifies the same way (non-hold > 0.7 trivially true, hold ≤ 0.3 rarely true),
so Far/hold-axis anchoring is also broken.

**Fix:** RMS-normalize distance — `d = sqrt(Σ w_i·dz_i² / n_axes)` — so 0.3/0.7
again mean per-axis σ. Keep thresholds as-is. Verify bands genuinely select
(distances now fall in 0–~1.5 range) and directed jumps hold `tempo.bpm`.

**Files:** `src/recommender/branch_sampler.py` (compute_distance + helpers),
`generate_playlist.py` (re-run), docs/BRANCHING.md (distance formula).

## 2. Mood descriptors (7 axes)

SCHEMA.md specifies `highlevel.mood_{happy,sad,aggressive,relaxed,electronic,
party,acoustic}` (0–1). **Constraint (verified):** the installed Essentia pip
build has none of these (MoodHappy etc. all absent). Requires a model-source
decision first — `essentia-models` pretrained PNNs, a standalone classifier, or
a lighter alternative. Then: extend `scripts/extract_essentia.py`, add axes to
`generate_playlist.py` AXIS_NAMES, update SCHEMA.md.

## 3. Deferred — float-vector conversion gap

`Track.features` is never populated by the pipeline; only `generate_playlist.py`
knows the axis layout (AXIS_NAMES + hand-rolled extract_features). Decision
deferred earlier: ingest-time parser vs shared query-time converter vs new DB
column. Whatever is chosen removes the duplicated axis knowledge.

## 4. Metadata extraction (title/artist)

All 17 rows have NULL title/artist; playlist falls back to filenames. Add
ID3/Vorbis tag reading (tinytag or mutagen) into `ingest_pipeline.py`.

## 5. Key value as a similarity axis

Only `key.confidence` feeds distance today — a tritone apart scores equal.
Add circle-of-fifths key distance (harmonically meaningful) as an axis.

## 6. Spectral descriptors frame-wise

`scripts/extract_essentia.py:69` `es.Spectrum()(audio)` is the same full-track
FFT flaw as the key bug — centroid/rolloff/flatness from one global spectrum
instead of frame-wise mean/std. Same class of fix as #1 of the key work.

## 7. QA suite modernization

`tests/run_qa.sh` is Java-era (checks Track.java, javac, sample placeholders
that Essentia fails on). Rewrite to verify the Python pipeline: ingest idempotency,
sampler bands, JSON output, extraction test.

## 8. Lower-priority hardening

- `--re-extract` / version-checked re-ingest (reruns skip existing rows → stale features)
- Seed selection as CLI arg (currently hardcoded `id == 17` in generate_playlist.py)
- Use CLAP embeddings for distance (currently stored, never sampled)
