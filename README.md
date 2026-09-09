# playlist-progression

A weekend-scale Python prototype for music similarity and playlist generation from local audio files.

> Direction: **CLAP-primary similarity** — see `PROJECT.md` (objective/criteria/scope/decisions),
> `docs/ROADMAP.md` (milestones + progress tracker), `docs/FEATURE_IDEAS.md` (deferred ideas).

## Overview

Ingests local audio files → extracts CLAP embeddings (primary similarity) + Essentia DSP/mood features (constraint gates) via subprocess → stores in SQLite → generates a JSON playlist using CLAP cosine similarity with quantile Near/Mid/Far bands + Essentia-gated progression.

## Branching Methodology

### Feature Extraction Pipeline

**Essentia (DSP descriptors)** — `scripts/extract_essentia.py` calls the Essentia CLI to compute a fixed descriptor set per track:

| Axis Group | Descriptors | Dimensions |
|------------|-------------|------------|
| **Duration** | track length (sec) | 1 |
| **Loudness** | integrated loudness (LUFS), loudness range (LU) | 2 |
| **Tonal** | key confidence, key circle-of-fifths (x, y) — derived from the key + scale string | 3 |
| **Rhythm** | BPM (tempo), beat confidence, danceability, onset rate | 4 |
| **Spectral** | spectral centroid, spectral rolloff, spectral flatness | 3 |
| **Mood** | happy, sad, aggressive, relaxed, electronic, party, acoustic | 7 |

Total: **20 dimensions** per track, stored as JSON in `tracks.feature_json`.

### Essentia Descriptor Definitions

| Descriptor | Output Key | Range | What It Measures |
|------------|------------|-------|------------------|
| **Integrated Loudness** | `loudness.integrated` | dB (LUFS) | Overall perceived loudness (EBU R128). |
| **Loudness Range** | `loudness.range` | dB (LU) | Dynamic range — difference between quiet and loud sections. |
| **Key + Scale** | `key.scale` | e.g. "A minor" | Estimated musical key and mode via HPCP chroma. |
| **Key Confidence** | `key.confidence` | unbounded (KeyExtractor strength, stored raw) | Strength of the key estimate. |
| **Key Fifths X/Y** | `key.fifths_x`, `key.fifths_y` | −1–1 | 2D circle-of-fifths coordinates (cos/sin) on a 24-slot circle where relative major/minor are adjacent (C–Am = 1 step, C–G = 2 steps, C–F# = 12 steps). Unknown key/mode → (0, 0). |
| **BPM (Tempo)** | `tempo.bpm` | ~40–200 | Beats per minute. Primary rhythmic anchor. |
| **Beat Confidence** | `tempo.confidence` | 0–5.32 (BeatTrackerMultiFeature scale; stored raw, not 0–1) | Reliability of the beat tracking. |
| **Danceability** | `rhythm.danceability` | 0–~3 (Essentia Danceability; stored raw, not 0–1) | How suitable for dancing — beat strength, regularity, tempo. |
| **Onset Rate** | `rhythm.onset_rate` | onsets/sec | Rate of note onsets — proxy for rhythmic density. |
| **Spectral Centroid** | `spectral.centroid` | Hz | "Brightness" — centre of mass of the spectrum. Higher = brighter. |
| **Spectral Rolloff** | `spectral.rolloff` | Hz | Frequency below which 85% of energy lies. |
| **Spectral Flatness** | `spectral.flatness` | 0–1 | How tone-like (0) vs noise-like (1) the spectrum is. |
| **Mood: Happy** | `mood.happy` | 0–1 | Upbeat, major-key, optimistic confidence score. |
| **Mood: Sad** | `mood.sad` | 0–1 | Melancholy, minor-key, emotionally heavy confidence score. |
| **Mood: Aggressive** | `mood.aggressive` | 0–1 | Loud, fast, distorted intensity confidence score. |
| **Mood: Relaxed** | `mood.relaxed` | 0–1 | Gentle, slow, low-energy calm confidence score. |
| **Mood: Electronic** | `mood.electronic` | 0–1 | Synthesised, produced, electronic confidence score. |
| **Mood: Party** | `mood.party` | 0–1 | High-energy, social, danceable celebration score. |
| **Mood: Acoustic** | `mood.acoustic` | 0–1 | Naturally recorded, unplugged, instrumental organic score. |

**Notes:**
- All spectral descriptors are computed frame-wise then aggregated to a single value per track.
- Tonal analysis uses `SpectralPeaks` → `HPCP` (Harmonic Pitch Class Profile) → `Key` estimation.
- Loudness uses EBU R128 (`LoudnessEBUR128`), which requires stereo input (mono is duplicated).
- **Mood descriptors** come from pre-trained Essentia MusiCNN TensorFlow classifiers (one binary model per mood). Each model's activation column is selected from its `classes` metadata and averaged over time frames. Models download automatically to `models/` on first use and require `essentia-tensorflow` + TensorFlow. If mood prediction fails for a track, the DSP features are still stored and the `mood` key is left NULL (not zero-filled); the next run retries mood via `--mood-only`. The 7 mood axes participate in the playlist distance computation.

**CLAP (primary similarity)** — `scripts/extract_clap.py` runs LAION-CLAP (via `laion-clap` Python package) to produce a 512-dim embedding vector per track. Stored as JSON array in `tracks.clap_embedding`. Distance is L2-normalized cosine (`src/recommender/clap_similarity.py`); bands are rank quantiles (`partition_quantile_bands`, defaults near 0.10 / mid 0.40). Required for the default sampler — run with `--clap --batch`.

**Essentia (constraint gates)** — the 20 DSP/mood axes above feed `src/recommender/constraint_filter.py` (tempo window, key compatibility, mood box + relax), not a distance measure.

### Distance Computation (primary: CLAP)

`PlaylistSampler` ranks candidates by CLAP cosine similarity to the current track and partitions the ranking into quantile bands:

```
similarity = cosine(l2_norm(seed), l2_norm(candidate))
distance   = 1 - similarity   # ∈ [0, 2], typically [0, 1]
```

Near = top 10% most similar, Mid = next 30%, Far = rest (quantile thresholds configurable via `near_quantile` / `mid_quantile`). Essentia gates filter candidates *before* ranking; mood-box relaxation (up to 3 steps) and global-nearest-by-CLAP fallback apply when gates are empty.

Legacy path (deprecated): `BranchSampler.compute_distance()` used standardised weighted Euclidean distance across the 20 Essentia axes (`sqrt( Σ weight[i] * (z_a[i] - z_b[i])² )`, z-scored per axis). Retained only behind `--sampler essentia`; removal waits on the subset scale test (see `docs/ROADMAP.md` M5). Evidence for deprecation: 17-track Spearman ρ = 0.078, NN match 5.9% (`docs/comparison-clap-vs-essentia.md`); 49-track blinded A/B verdict ship-clap-as-default (`docs/eval/blinded-2026-09-05/REPORT.md`).

### Three Distance Bands (quantile-based, CLAP)

Given the current track, CLAP-ranked candidates are partitioned by rank quantile:

| Band | Quantile (defaults) | Purpose |
|------|---------------------|---------|
| **Near** | top 10% (`near_quantile=0.10`) | Close neighbours — minimal perceptual change. Establishes mood/groove. |
| **Mid** | next 30% (up to `mid_quantile=0.40`) | Moderate jumps — noticeable shift but related. Drives transitions. |
| **Far** | rest | Large leaps after Essentia gating — contrast/surprise. |

Legacy σ bands (`d ≤ 0.3σ` / `0.3σ < d ≤ 0.7σ` / far-but-directed) apply to the deprecated Essentia path only.

### Essentia gating (not distance)

Before CLAP ranking, `ConstraintFilter` gates candidates: tempo window around the current (optionally drifted) BPM, key compatibility, and a mood box with up to 3 relaxation steps. Empty gates → global-nearest-by-CLAP fallback, recorded in `reason`.

### Playlist Generation Flow (primary)

```
seed track
    │
    ├─► gate (Essentia) ──► rank (CLAP) ──► Near pick ──► next seed
    │
    ├─► gate (Essentia) ──► rank (CLAP) ──► Mid pick  ──► next seed
    │
    └─► gate (Essentia) ──► rank (CLAP) ──► Far pick  ──► next seed
    │
    └─► repeat per band schedule (default Near→Mid→Far→Mid→Near) until limit or candidates exhausted
```

Fallback: if gates are empty, pick global-nearest-by-CLAP ignoring gates and record it in `reason` (e.g. `"Fallback: gates empty, global-nearest-by-CLAP (sched Near)"`). Within a band the most similar candidate is picked first.

### Output

`branch_playlist.json`:
```json
{
  "seed": { "id": 1, "title": "...", "artist": "..." },
  "playlist": [
    { "position": 1, "id": 5, "title": "...", "artist": "...", "band": "Near", "distance": 0.12, "reason": "Close timbral neighbour, minimal shift" },
    { "position": 2, "id": 12, "title": "...", "artist": "...", "band": "Mid", "distance": 0.45, "reason": "Moderate shift in mood, related rhythm" },
    { "position": 3, "id": 27, "title": "...", "artist": "...", "band": "Far", "distance": 0.89, "reason": "Directed jump: same tempo, far mood/timbre" }
  ]
}
```

## Features

- **Subprocess integration**: Java-free — calls Essentia CLI and CLAP Python scripts via `subprocess.run()`
- **Essentia DSP features**: loudness (EBU R128), tempo/BPM, key/scale + circle-of-fifths key distance, danceability, spectral centroid/rolloff/flatness
- **Mood descriptors**: 7 mood scores (happy, sad, aggressive, relaxed, electronic, party, acoustic) via pre-trained Essentia MusiCNN TensorFlow classifiers, included in similarity distance
- **SQLite storage**: `tracks` table with feature JSON and CLAP embeddings
- **CLAP-primary recommender**: quantile bands over cosine similarity (Near top 10%, Mid next 30%, Far rest) with Essentia tempo/key/mood gates; legacy Essentia-Euclidean path deprecated behind `--sampler essentia`
- **JSON output**: `branch_playlist.json` with seed info, distance band per track, reason string

## Quick Start

```bash
# 1. Create venv and install Essentia (incl. TensorFlow build for mood)
python3 -m venv .venv
source .venv/bin/activate
pip install essentia-tensorflow  # include for mood extraction (Essentia + TensorFlow)
# or plain: pip install essentia  (mood extraction will be unavailable)

# 2. Run the pipeline
make run MUSIC_DIR=/path/to/your/music
# or directly:
python3 run.py /path/to/your/music database/playlist.db
# useful flags:
#   --clap              extract CLAP embeddings (required for default --sampler clap; needs laion-clap)
#   --sampler {clap,essentia}  playlist backend (default: clap; essentia is deprecated legacy)
#   --config PATH       JSON config for clap sampler (quantiles, gates, schedule, drift)
#   --re-extract        re-extract tracks already in the database
#   --timeout SEC       overall extraction timeout (default 180s, env EXTRACT_TIMEOUT_SEC)
#   --dsp-timeout SEC   DSP-phase timeout (default 60s)
#   --mood-timeout SEC  mood-phase timeout (default 180s)
#   --no-mood           skip mood extraction (DSP only)
#   --batch             batch mode: one worker loads the 7 TF mood models once
#                       for the whole library instead of once per track
#   --models-dir DIR    where mood models live (default models/)

# 3. Generate a playlist (single step: ingest + playlist)
make playlist SEED="Orphan Girl"
# or directly:
python3 run.py /path/to/music database/playlist.db --generate-playlist --seed-title "Orphan Girl"
# playlist flags:
#   --seed-title STR    substring to match in track title (case-insensitive)
#   --limit N           playlist entries after seed (default 9)
#   --output PATH       output JSON path (default: branch_playlist.json)
#   --hold-axis AXIS    legacy Essentia directed-jump axis (default: tempo.bpm; ignored by clap sampler)
#   --sampler {clap,essentia}  playlist backend (default: clap)
#   --config PATH       JSON config for clap sampler
```

Extraction runs in two phases per track (DSP first, then mood); a mood
failure keeps the DSP row and retries mood on the next run. See
`docs/INTEGRATION.md` for the full contract.
```

## Input Processing

### Directory Scanning

The pipeline **recursively scans** the supplied music directory — no flat structure required. It walks all subdirectories and finds any file with a supported extension:

```python
# From src/recommender/ingest_pipeline.py
AUDIO_EXTENSIONS = {".mp3", ".flac", ".ogg", ".wav", ".m4a"}
```

**No playlist file is needed** — just point at your music root:

```bash
python run.py /path/to/music database/playlist.db
```

This will find:
```
/path/to/music/
├── Artist A/
│   ├── Album 1/
│   │   ├── 01 Track.mp3
│   │   ├── 02 Track.flac
│   │   └── 03 Track.m4a
│   └── Album 2/
│       └── 01 Track.ogg
└── Artist B/
    └── Single.wav
```

All 5 files above would be discovered and processed.

### Re-ingestion Guard

The pipeline is **idempotent** — it tracks processed files by absolute path in SQLite. Re-running on the same directory:

- Skips files already in the database (by `file_path` UNIQUE constraint)
- Only processes new/changed files
- Safe to run incrementally as you add music

### Seed Selection

The first track in the generated playlist is the **seed**. Pick it with `generate_playlist.py --db <db> --seed-title <title>` (or `--seed-id <id>`). Additional flags: `--limit`, `--hold-axis`, `--output`, `--summary`. If multiple tracks match the title substring, the script exits with a list of matching IDs for disambiguation.

## Project Structure

```
playlist-progression/
├── src/recommender/          # Core pipeline
│   ├── track.py              # Track dataclass
│   ├── feature_extractor.py  # Subprocess wrapper for extraction scripts
│   ├── feature_converter.py  # Essentia axis layout (AXIS_NAMES) + JSON→vector convert() for gates
│   ├── ingest_pipeline.py    # Main entry: scan → extract → store
│   ├── playlist_sampler.py   # Primary sampler: CLAP rank + Essentia gates (default)
│   ├── clap_similarity.py    # CLAP cosine + quantile bands
│   ├── constraint_filter.py  # Essentia tempo/key/mood gates
│   ├── branch_sampler.py     # Deprecated: Essentia Euclidean distance (behind --sampler essentia)
│   └── playlist_writer.py    # JSON output
├── scripts/
│   ├── extract_essentia.py   # Essentia CLI wrapper (gates)
│   └── extract_clap.py       # CLAP embedding wrapper (primary similarity)
├── docs/                     # CHARTER, ROADMAP, FEATURE_IDEAS + architecture/schema/integration/branching/eval
├── database/init.db          # SQLite schema
├── run.py                    # CLI entry point
├── Makefile                  # run, init-db, clean targets
└── requirements.txt          # essentia-tensorflow, tinytag, tensorflow (laion-clap for CLAP, installed separately)
```

## Scope Boundaries

- ✅ Local music library only (owned files)
- ✅ Subprocess integration (no JNI, no HTTP service)
- ❌ No Spotify/streaming API integration
- ❌ No production auth, multi-user, or server deployment
- ❌ No MiMo-Audio natural-language layer
- ❌ No real-time playback or streaming server
- ❌ No GPU dependency for core similarity

## License

Personal exploration tool — not a commercial product.