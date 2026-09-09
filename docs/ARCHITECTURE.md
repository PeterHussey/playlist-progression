# Architecture Design — Playlist Progression Prototype

> **Status:** Weekend-scale Python prototype, CLAP-primary similarity. See `PROJECT.md` (objective/criteria/scope/decisions), `ROADMAP.md` (milestones), `SCHEMA.md` (exact DDL).

## Overview

This document describes the architecture for a local music similarity and discovery
prototype. The system reads a user's local music library, extracts acoustic and
neural features, stores them in a lightweight database, and generates a playlist
that follows a deliberate progression through similarity space. The entire pipeline
runs on a single machine with no cloud dependencies and no external API calls.

Implementation is **Python** (`src/recommender/` + `run.py` + `generate_playlist.py`).

---

## Pipeline

The pipeline has six sequential stages: **Scan → Extract DSP → Extract mood → Store → Sample → Output**.

1. **Scan.** `ingest_pipeline.scan_directory()` recursively walks the configured music directory for audio files (`.mp3`, `.flac`, `.ogg`, `.wav`, `.m4a`). It resolves absolute paths, reads ID3/Vorbis tags (`title`, `artist`) via `tinytag`, and inserts a metadata row into SQLite. Re-runs skip rows whose absolute `file_path` already exists (UNIQUE constraint) unless the stored extractor version is stale or `--re-extract` is passed. No audio conversion happens — raw files are left untouched.

2. **Extract DSP.** For each track needing extraction, `feature_extractor.extract_essentia()` shells out via `subprocess.run()` to `scripts/extract_essentia.py` (`--no-mood` phase). Produces a JSON sidecar with duration, loudness (EBU R128), tempo/BPM, key/scale, danceability, onset rate, and frame-wise spectral centroid/rolloff/flatness. DSP timeout defaults to 60 s (`--dsp-timeout` / `EXTRACT_DSP_TIMEOUT_SEC`).

3. **Extract mood.** A second `subprocess.run()` call to `scripts/extract_essentia.py` (`--mood-only` phase) runs the 7 pre-trained Essentia MusiCNN TensorFlow classifiers (happy, sad, aggressive, relaxed, electronic, party, acoustic). Mood models download automatically to `models/` on first use and require `essentia-tensorflow` + TensorFlow. Mood timeout defaults to 180 s (`--mood-timeout` / `EXTRACT_MOOD_TIMEOUT_SEC`). If mood fails, the DSP row is kept with mood NULL and retried on the next run (`--mood-only`); `--no-mood` skips mood entirely.

4. **Extract CLAP (default path).** When `--clap` is passed, `feature_extractor.extract_clap()` shells out via `subprocess.run()` to `scripts/extract_clap.py` (LAION-CLAP, `laion-clap` package), producing a 512-dim embedding stored as a JSON array in `tracks.clap_embedding`. CLAP is **required for the default sampler** — run ingestion with `--clap`.

5. **Store.** `ingest_pipeline.init_database()` persists everything into a **single `tracks` table** in SQLite (`database/playlist.db`). Essentia output is stored as `feature_json` TEXT; CLAP as `clap_embedding` TEXT (JSON array, NULL when not extracted). `duration_sec`, `title`, `artist` are populated at ingest/re-extract time. Each subprocess writes its JSON output to a temporary sidecar file (`essentia_<id>.json` / `clap_<id>.json`), which the pipeline reads back and deletes.

Worker notes: `--batch` runs a batch worker that loads the 7 TF mood graphs (and the CLAP model for the CLAP pass) **once per library** instead of once per track, amortising the ~5 s model-init cost. Single-track extraction remains for debugging and interactive use. On batch failure the pipeline falls back to single-track per file with a logged notice.

6. **Sample.** The default `PlaylistSampler` (`src/recommender/playlist_sampler.py`) gates candidates with the Essentia `ConstraintFilter` (tempo window, key compatibility, mood box + up to 3 relaxation steps), then ranks survivors by CLAP cosine similarity and partitions the ranking into quantile bands (Near top 10%, Mid next 30%, Far rest; configurable via `near_quantile` / `mid_quantile`). Empty gates fall back to global-nearest-by-CLAP with the fallback recorded in `reason`. The legacy `BranchSampler` (standardised Essentia Euclidean, σ bands) is retained only behind `--sampler essentia` and is deprecated (see CHARTER decisions log).

7. **Output.** The sampler produces an ordered sequence of track IDs and `playlist_writer` writes them to `branch_playlist.json` (seed info, per-track band, CLAP distance, reason string). This file can be loaded by any local player or inspected manually.

---

## Integration Approach

The system uses a **Python-to-Python subprocess boundary**. `src/recommender/` owns the main loop, file I/O, database access, and playlist generation logic. Extraction lives in standalone CLI scripts, which keeps the mature Essentia and CLAP dependencies isolated and independently testable — no JNI, no native bindings, no HTTP service.

Concrete subprocess calls (`feature_extractor.run_script()`):

```
sys.executable scripts/extract_essentia.py <audioPath> <outputPath> [--no-mood | --mood-only]
sys.executable scripts/extract_clap.py <audioPath> <outputPath>
```

Each script is a standalone CLI tool that reads one audio file and writes
one JSON sidecar. The scripts are intentionally stateless — no shared memory,
no daemon process. This makes them easy to test, debug, and replace independently.

CLAP extraction is gated behind `--clap`. When disabled, the pipeline
runs Essentia-only and the `clap_embedding` column remains `NULL`; the default
CLAP sampler then reports 0% coverage and yields no picks (use
`--sampler essentia` only for the deprecated legacy comparison). This keeps the
core pipeline functional on machines where the neural embedding model is too
large, while the similarity engine expects CLAP for real playlists.

---

## Schema Overview

The prototype uses a single SQLite database file (`database/playlist.db`). SQLite is
chosen for zero-configuration deployment and straightforward file-based backup.

### Tables

| Table | Purpose |
|---|---|
| `tracks` | One row per audio file. Columns: `id` (INTEGER PK), `file_path` (TEXT UNIQUE), `title`, `artist`, `duration_sec` (REAL), `feature_json` (TEXT — full Essentia output), `clap_embedding` (TEXT — nullable 512-dim float array as JSON), `created_at`. Runtime source of truth: `init_database()` in `src/recommender/ingest_pipeline.py`. Exact DDL: `docs/SCHEMA.md`. |

Feature data is stored in its native JSON shape rather than being decomposed into
individual columns. This avoids schema migrations every time Essentia adds a new
descriptor and keeps the extract scripts decoupled from database schema changes.
CLAP embeddings use `TEXT` (JSON array) storage for consistency with `feature_json`
— human-readable and directly matching what `ingest_pipeline.py` writes via `json.dumps`.
There are no `features` or `runs` tables.

---

## Branching Design

The playlist progression engine moves through similarity space using **distance
bands** — discrete zones that control how far each successive track can deviate
from the current one.

### Distance Bands (primary: CLAP quantiles)

| Band | Rank quantile (defaults) | Behaviour |
|---|---|---|
| **Near** | top 10% (`near_quantile=0.10`) | Close neighbours; minimal perceptual change. Good for establishing a mood. |
| **Mid** | next 30% (up to `mid_quantile=0.40`) | Moderate jumps; noticeable shift but still related. Good for transition. |
| **Far** | rest | Large leaps after Essentia gating; contrast and surprise. Good for discovery moments. |

Band boundaries are quantile thresholds (`near_quantile` / `mid_quantile`) and can be tuned per run via `--config` JSON.

Legacy σ bands (`d ≤ 0.3σ` / `0.3σ < d ≤ 0.7σ` / far-but-directed with `hold_axis`) apply to the deprecated Essentia path only (`--sampler essentia`, `BranchSampler`). Full legacy design: `docs/BRANCHING.md`.

### Essentia gating (not distance)

Before CLAP ranking, `ConstraintFilter` gates candidates: tempo window around the current (optionally drifted) BPM, key compatibility (circle-of-fifths steps), and a mood box with up to 3 relaxation steps. Empty gates → global-nearest-by-CLAP fallback, recorded in `reason`.

### Sampling Strategy

The sampler maintains a candidate list from the CLAP-ranked, Essentia-gated survivors in the target band. Within a band the most similar candidate is picked first. When gates are empty it relaxes the mood box (≤3 steps), then falls back to global-nearest-by-CLAP with a logged reason. The progression defaults to Near → Mid → Far → Mid → Near until the limit is reached or candidates are exhausted.

---

## Scope Boundaries

This project is deliberately scoped as a **weekend-scale prototype**, not a
production system. The following boundaries apply (see `PROJECT.md` Out of scope):

- **Local library only.** The system reads files from a local directory tree. There
  is no Spotify API integration, no streaming-service authentication, and no
  remote file access. The user's own audio files are the sole data source.
- **No production auth.** There is no user authentication, no multi-tenancy, and
  no access control. The SQLite file is open to whoever has filesystem access.
- **No JNI or native bindings.** The Python-to-Python extraction interface is strictly
  subprocess-based via `subprocess.run()`. There are no JNI wrappers, no
  shared-memory mechanisms, and no in-process embedding.
- **No neural audio generation.** While the architecture notes mention CLAP
  embeddings and MiMo-Audio was evaluated during research, this prototype does not
  implement any natural-language or generative audio layer. CLAP is used solely as
  a similarity feature, not for generation.
- **Single-machine deployment.** There are no Docker containers, no cloud hosting. The system runs as local Python (`run.py` → `ingest_pipeline.py` → `generate_playlist.py`). The SQLite database file and JSON playlist output live alongside the source tree. (CI runs pytest + ruff only.)

---

## Decisions Log

| Decision | Rationale |
|---|---|
| CLAP cosine + quantile bands as default (`PlaylistSampler`, `--sampler clap`) | 49-track blinded A/B (`docs/eval/blinded-2026-09-05/REPORT.md`): tempo adherence 0.33–0.44 → 0.89–1.00, ear test 3/3 to clap, full Near/Mid/Far schedule with 0 fallbacks vs Essentia Far-only fallback chain. |
| Essentia DSP + mood as constraint gates only (`constraint_filter.py`) | The 20 DSP/mood axes gate candidates (tempo/key/mood) instead of measuring distance; 17-track comparison showed Essentia z-space is uncorrelated with CLAP (Spearman ρ = 0.078). |
| SQLite over PostgreSQL | Zero-install, single-file database ideal for a local prototype. No server process. |
| JSON feature storage over relational decomposition | Avoids schema churn; keeps extract scripts decoupled from DB migrations. |
| Subprocess (`subprocess.run()`) over in-process import | Simpler debugging, isolates TensorFlow/CLAP imports, scripts are independently testable; batch worker amortises model-load cost. |
| Quantile bands over continuous similarity scoring | Discrete bands give the user predictable playlist behaviour; quantiles adapt to library size where fixed σ thresholds did not. |
