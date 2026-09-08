# Project Charter — playlist-progression

> Status: active. This is the source of truth for objective, scope, and decisions.
> Task tracking lives in `ROADMAP.md`. Deferred ideas live in `FEATURE_IDEAS.md`.

## 1. Objective

Build a reliable **CLAP-primary** playlist engine that turns a local library of
**thousands of tracks** into deliberate Near/Mid/Far progressions.

- Primary similarity: LAION-CLAP 512-dim embeddings, L2-normalized cosine
  (`src/recommender/clap_similarity.py`), quantile bands
  (`partition_quantile_bands`, defaults near 0.10 / mid 0.40).
- Essentia DSP + mood features are **constraint gates only** (tempo window, key
  compatibility, mood box + relax in `src/recommender/constraint_filter.py`) —
  not a distance measure.
- Path: validate on a **mid-size subset (~150–250 tracks)** before the full-library run.

## 2. Scope boundaries

- ✅ Local music library only (owned files, recursive scan).
- ✅ Python pipeline, subprocess-only extraction (no JNI, no service).
- ✅ SQLite storage, JSON playlist output.
- ❌ No Spotify/streaming API integration.
- ❌ No production auth, multi-user, or server deployment.
- ❌ No MiMo-Audio natural-language layer.
- ❌ No real-time playback or streaming server.
- ❌ No GPU requirement for core similarity (GPU helps the CLAP extraction pass only).

## 3. Decisions log

| Decision | Rationale |
|---|---|
| CLAP cosine + quantile bands as default (`PlaylistSampler`, `--sampler clap`) | 49-track blinded A/B (`docs/eval/blinded-2026-09-05/REPORT.md`): tempo adherence 0.33–0.44 → 0.89–1.00, ear test 3/3 to clap, full Near/Mid/Far schedule with 0 fallbacks vs Essentia Far-only fallback chain. |
| Essentia distance **deprecated** (`branch_sampler.py`, `--sampler essentia` retained only as fallback) | 17-track comparison (`docs/comparison-clap-vs-essentia.md`): Spearman ρ = 0.078, NN match 5.9% — spaces are uncorrelated; Essentia z-space puts every candidate at d > 0.7 from typical seeds. Removal waits until subset test passes (ROADMAP M5). |
| Subset-before-full-scale rule | Thousands-track target is untested; 17- and 49-track results do not prove quantile stability or batch throughput at scale. |
| Batch worker as the scale path (`--batch`, TF graphs / CLAP model loaded once) | Amortises ~5 s model-init cost; single-track subprocess per file does not scale. |

## 4. Success criteria

- Subset eval (~150–250 tracks): 100% CLAP coverage (`PlaylistSampler.load_library`
  report), scheduled bands exercised with minimal fallbacks, ear-test forced-choice recorded.
- Full-library run: `--batch --clap` completes with idempotent re-runs (UNIQUE `file_path`, version-checked re-extract).
- Docs truth: README describes the CLAP-primary system; this charter + ROADMAP + FEATURE_IDEAS are linked from README.

## 5. What this charter is NOT

- No task checklists or timelines — see `ROADMAP.md`.
- No feature proposals — see `FEATURE_IDEAS.md` (append-only, promotion requires explicit approval).
