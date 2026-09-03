# playlist-progression

A weekend-scale Python prototype for music similarity and playlist generation from local audio files.

## Overview

Ingests local audio files → extracts Essentia DSP features + optional CLAP embeddings via subprocess → stores in SQLite → generates a JSON playlist using a branching recommender (near/mid/far distance bands + axis-controlled jumps).

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

**CLAP (semantic embeddings)** — `scripts/extract_clap.py` runs LAION-CLAP (via `laion-clap` Python package) to produce a 512-dim embedding vector per track. Stored as JSON array in `tracks.clap_embedding`. Optional — pipeline works without it.

### Distance Computation

`BranchSampler.compute_distance()` uses **standardised weighted Euclidean distance** across the feature axes:

```
distance = sqrt( Σ weight[i] * (z_a[i] - z_b[i])² )
```

Where `z = (value - population_mean) / population_stddev` per axis. Axis weights are configurable (default 1.0). Population statistics are computed across the full library at sampler initialisation.

CLAP embeddings are **not used for distance** in this prototype — they're stored for future semantic similarity experiments.

### Three Distance Bands

Given a seed track, candidates are partitioned by standardised distance:

| Band | Threshold | Purpose |
|------|-----------|---------|
| **Near** | `d ≤ 0.3σ` | Close neighbours — minimal perceptual change. Establishes mood/groove. |
| **Mid** | `0.3σ < d ≤ 0.7σ` | Moderate jumps — noticeable shift but related. Drives transitions. |
| **Far-but-directed** | `d > 0.7σ` on all axes **except** `hold_axis`, where `d ≤ 0.3σ` | Large leaps anchored by one constant quality (e.g., same tempo, different mood). Creates contrast/surprise. |

### Directed Jump Algorithm

`select_directed_jump(seed, candidates, hold_axis)`:

1. Compute full distance on all feature axes
2. Compute distance on all axes **except** `hold_axis` (zero out that dimension)
3. Compute distance on `hold_axis` only
4. Candidate qualifies if: `non_hold_distance > 0.7σ` **AND** `hold_distance ≤ 0.3σ`

Example: `hold_axis="tempo.bpm"` → playlist jumps far in timbre/mood/key while keeping tempo constant.

### Playlist Generation Flow

```
seed track
    │
    ├─► select_near()  ──► pick 1 ──► next seed
    │
    ├─► select_mid()   ──► pick 1 ──► next seed
    │
    └─► select_directed_jump(hold_axis="tempo.bpm") ──► pick 1 ──► next seed
    │
    └─► repeat until playlist length reached or candidates exhausted
```

Fallback: if a band is empty, pick the globally nearest unvisited track, label it with its actual band (by distance thresholds), and record the fallback in `reason` (e.g. `"Fallback: scheduled Near empty, global-nearest (actual Mid) from seed"`). Band matches are sorted nearest-first, so the nearest candidate within a band is always picked.

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
- **SQLite storage**: `tracks` table with feature JSON and optional CLAP embeddings
- **Branching recommender**: 3 distance bands — near (≤0.3σ), mid (0.3–0.7σ), far-but-directed (≥0.7σ along specific axis, holding one descriptor constant)
- **JSON output**: `branch_playlist.json` with seed info, distance band per track, reason string

## Quick Start

```bash
# 1. Create venv and install Essentia (incl. TensorFlow build for mood)
python -m venv .venv
source .venv/bin/activate
pip install essentia-tensorflow  # include for mood extraction (Essentia + TensorFlow)
# or plain: pip install essentia  (mood extraction will be unavailable, scores default to 0.0)

# 2. Run the pipeline
make run MUSIC_DIR=/path/to/your/music
# or directly:
python run.py /path/to/your/music database/playlist.db
# useful flags:
#   --clap              also extract CLAP embeddings (requires laion-clap)
#   --re-extract        re-extract tracks already in the database
#   --timeout SEC       overall extraction timeout (default 180s, env EXTRACT_TIMEOUT_SEC)
#   --dsp-timeout SEC   DSP-phase timeout (default 60s)
#   --mood-timeout SEC  mood-phase timeout (default 180s)
#   --no-mood           skip mood extraction (DSP only)
#   --batch             batch mode: one worker loads the 7 TF mood models once
#                       for the whole library instead of once per track
#   --models-dir DIR    where mood models live (default models/)
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

The first track in the generated playlist is the **seed**. Pick it with `generate_playlist.py --db <db> --seed-id <id>` (plus `--limit`, `--hold-axis`, `--output`, `--summary`); defaults reproduce the legacy behaviour (seed id 17 if present else highest id, 9 entries, hold `tempo.bpm`).

## Project Structure

```
playlist-progression/
├── src/recommender/          # Core pipeline
│   ├── track.py              # Track dataclass
│   ├── feature_extractor.py  # Subprocess wrapper for extraction scripts
│   ├── feature_converter.py  # Axis layout (AXIS_NAMES) + JSON→vector convert()
│   ├── ingest_pipeline.py    # Main entry: scan → extract → store
│   ├── branch_sampler.py     # Distance bands + directed jumps
│   └── playlist_writer.py    # JSON output
├── scripts/
│   ├── extract_essentia.py   # Essentia CLI wrapper
│   └── extract_clap.py       # CLAP embedding wrapper
├── docs/                     # Architecture, schema, integration, branching design
├── database/init.db          # SQLite schema
├── run.py                    # CLI entry point
├── Makefile                  # run, init-db, clean targets
└── requirements.txt          # essentia-tensorflow, tinytag, tensorflow (laion-clap optional, installed separately)
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