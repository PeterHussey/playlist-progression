# Schema Design — Playlist Progression Prototype

> **Status:** Weekend-scale prototype, not a production system.

## Overview

This document defines the SQLite schema used by the playlist progression prototype.
The database stores track metadata, Essentia feature output, and optional CLAP
embeddings in a lightweight, single-file format. Schema design prioritises
simplicity — feature data is kept in its native JSON shape rather than decomposed
into individual columns, avoiding schema churn when Essentia adds or renames
descriptors. For the broader system context, see [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## Tables

### `tracks`

One row per audio file in the user's library. Metadata is extracted from ID3/Vorbis
tags during ingestion; feature data is written after extraction completes.

```sql
CREATE TABLE tracks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT    UNIQUE NOT NULL,
    title       TEXT,
    artist      TEXT,
    duration_sec REAL,
    feature_json TEXT,                -- full Essentia JSON output
    clap_embedding BLOB,             -- nullable 512-dim float array (CLAP)
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | Auto-incremented surrogate key. |
| `file_path` | TEXT UNIQUE | Normalised absolute path to the audio file. Uniqueness constraint prevents duplicate ingestion. |
| `title` | TEXT | From ID3/Vorbis tags. May be NULL for files without tags. |
| `artist` | TEXT | From ID3/Vorbis tags. May be NULL. |
| `duration_sec` | REAL | Track length in seconds. |
| `feature_json` | TEXT | The complete Essentia JSON output stored as a text blob. Keeps extract scripts decoupled from schema — no migration needed when Essentia adds descriptors. |
| `clap_embedding` | BLOB | Nullable. Populated only when CLAP extraction is enabled. Stored as raw bytes (512 × 4-byte floats). Remains NULL when CLAP is disabled. |
| `created_at` | TIMESTAMP | Set automatically on row creation. |

> **Design note:** Storing `feature_json` as a single TEXT column rather than
> decomposing Essentia descriptors into individual columns avoids schema
> migrations every time Essentia adds or renames a descriptor. The sampler
> parses the JSON at query time. See the [Essentia Descriptors](#essentia-descriptors)
> section below for the specific fields the branching engine reads from this blob.

---

## Essentia Descriptors

The following Essentia descriptors are extracted for every track and stored inside
`feature_json`. Each descriptor group serves a specific role as a **branching axis**
— a dimension along which the playlist can deliberately progress or jump.

### lowlevel.timbre

Timbral descriptors capture the tonal "colour" or texture of a track. They are
extracted from the spectral envelope and describe how bright, rough, or complex
a sound is.

| Descriptor | Unit | Branching Role |
|---|---|---|
| `lowlevel.spectral_centroid.mean` | Hz | **Brightness axis.** Higher values indicate brighter, more treble-heavy tracks. Useful for transitioning from dark/mellow to bright/energetic progressions. |
| `lowlevel.spectral_centroid.stdev` | Hz | **Brightness consistency.** High std dev means the track varies in brightness over time (e.g., quiet intro to loud chorus). Low std dev means uniform timbre. |
| `lowlevel.spectral_complexity.mean` | 0–1 | **Roughness axis.** Higher values indicate noisier, more complex spectra (distorted guitars, dense mixes). Lower values indicate cleaner, simpler tones (solo piano, acoustic guitar). Drives transitions between polished and raw textures. |
| `lowlevel.spectral_complexity.stdev` | 0–1 | **Texture variability.** Tracks with high std dev shift between clean and rough sections. |
| `lowlevel.spectral_rolloff.mean` | Hz | **Energy concentration.** The frequency below which 85% of energy is concentrated. Low rolloff = bass-heavy; high rolloff = treble-heavy. Complements centroid for finer brightness control. |
| `lowlevel.spectral_rolloff.stdev` | Hz | **Energy shift.** High std dev indicates the energy centre moves over time, typical of dynamic arrangements. |

### lowlevel.tonal

Tonal descriptors capture key and harmonic content. They drive transitions
through musical key and chord colour.

| Descriptor | Values | Branching Role |
|---|---|---|
| `lowlevel.tonal.hkey_scale` | string (e.g., "C", "F#") | **Key axis.** The detected musical key of the track. Converted to 2D circle-of-fifths coordinates (`key.fifths_x`, `key.fifths_y`) on a 24-slot circle where relative major/minor are adjacent (e.g. C–Am = 1 step, C–G = 2 steps, C–F# = 12 steps). Progressive playlists can circle through the circle of fifths or make deliberate key jumps for emotional effect. `key.confidence` is retained as a separate reliability axis. |
| `lowlevel.tonal.chord` | string (e.g., "C major", "G minor") | **Harmonic colour.** The detected chord or chord progression root. Useful for harmonic continuity — grouping tracks by shared chord quality for smooth transitions. |

### rhythm.tempo

Tempo describes the speed of the beat. It is the single most effective axis for
controlling playlist energy and momentum.

| Descriptor | Unit | Branching Role |
|---|---|---|
| `rhythm.tempo` | BPM | **Speed axis.** The primary axis for acceleration and deceleration progressions. A playlist can gradually climb from 90 BPM to 140 BPM, or drop from 128 BPM to 70 BPM for a dramatic energy shift. The sampler can weight this dimension heavily to force tempo-driven journeys even when other qualities stay within the Near band. |

### rhythm.danceability

Danceability measures how rhythmically regular and "groovy" a track is, combining
tempo, beat strength, and rhythmic stability into a single score.

| Descriptor | Range | Branching Role |
|---|---|---|
| `rhythm.danceability` | 0–~3 (Essentia `Danceability`; higher = more danceable) | **Groove axis.** High danceability (~2.0+, strong regular beats) suits energetic playlists. Low danceability (<1.0) indicates ambient, free-form, or irregular rhythms. Useful for transitioning between danceable and introspective sections of a playlist. |

> **Value ranges — stored raw, never clamped.** `rhythm.danceability` spans
> 0–~3 by design (E2E observed 1.04). `tempo.confidence` is the
> `RhythmExtractor2013` (multifeature) beat-tracking confidence on a 0–5.32
> scale (E2E observed 2.27 = "good confidence"); it is **not** 0–1.
> `key.confidence` is the `KeyExtractor` strength value (Essentia publishes no
> bounded range — stored raw). Clamping/normalising at extract time was
> considered and deliberately rejected: it would discard real signal (e.g.
> danceability > 1) for no benefit, since the sampler z-scores every axis
> before distance computation (see [BRANCHING.md](./BRANCHING.md)).

### highlevel.mood

Mood descriptors are neural-network-derived probability scores extracted using
Essentia's pre-trained MusiCNN TensorFlow classifiers. Each value represents
the model's confidence that the track belongs to a particular mood category.
Together they form a multi-dimensional mood space.

| Descriptor | Range | Branching Role |
|---|---|---|
| `highlevel.mood_happy` | 0–1 | **Joy axis.** High values indicate upbeat, major-key, optimistic tracks. Useful for building a "feel-good" progression or deliberately descending from joy into melancholy. |
| `highlevel.mood_sad` | 0–1 | **Melancholy axis.** High values indicate minor-key, slow, emotionally heavy tracks. Pairs with happy for emotional contrast transitions. |
| `highlevel.mood_aggressive` | 0–1 | **Intensity axis.** High values indicate loud, fast, distorted, or confrontational tracks. Drives transitions between calm and aggressive energy states. |
| `highlevel.mood_relaxed` | 0–1 | **Calm axis.** High values indicate gentle, slow, low-energy tracks. The complement to aggressive — useful for wind-down progressions at the end of a playlist. |
| `highlevel.mood_electronic` | 0–1 | **Synthetic axis.** High values indicate synthesised, produced, or electronically-generated sounds. Useful for genre-blending transitions between organic and electronic sections. |
| `highlevel.mood_party` | 0–1 | **Celebration axis.** High values indicate high-energy, social, danceable tracks. Overlaps with danceability but captures a broader social/energetic vibe rather than just rhythmic regularity. |
| `highlevel.mood_acoustic` | 0–1 | **Organic axis.** High values indicate naturally recorded, unplugged, or instrument-driven tracks. The complement to electronic — drives transitions between synthetic and acoustic textures. |

**Implementation Notes:**
- Requires TensorFlow >= 2.10.0
- Models download automatically from Essentia servers on first use
- Audio is resampled to 16kHz for MusiCNN models
- Extraction adds ~7-14 seconds per track
- Graceful fallback to 0.0 if extraction fails for any mood

---

## How Axes Drive Branching

The playlist sampler computes pairwise distance across all descriptor axes. By
default, all axes contribute equally to the distance metric. Axis weights can be
configured per run to bias the progression toward specific musical qualities:

- **Tempo progression:** Set a high weight on `rhythm.tempo`. The playlist will
  accelerate or decelerate even when staying within the Near band.
- **Energy arc:** Weight `rhythm.danceability` and `highlevel.mood_aggressive`
  heavily to build from calm to energetic over the course of a playlist.
- **Emotional journey:** Weight `highlevel.mood_happy`, `highlevel.mood_sad`, and
  `highlevel.mood_relaxed` to create deliberate mood swings.
- **Genre blending:** Weight `highlevel.mood_electronic` and
  `highlevel.mood_acoustic` to transition between synthetic and organic sounds.
- **Timbral shift:** Weight the `lowlevel.timbre` descriptors to move between
  bright/dark and clean/rough textures.

Axis weights are passed as a JSON configuration object and modify the Euclidean
distance function at sampling time. See the [Branching Design](./ARCHITECTURE.md#branching-design)
section in ARCHITECTURE.md for the distance band thresholds.

---

## Optional: CLAP Embeddings

When CLAP extraction is enabled, the 512-dimensional embedding vector is stored
in the `clap_embedding` BLOB column on the `tracks` table. The embedding captures
high-level semantic similarity — two tracks with similar mood, instrumentation,
or lyrical theme will have similar CLAP vectors even if their low-level DSP
features differ.

CLAP embeddings are **optional**. When disabled, the column remains NULL and the
similarity engine operates purely on Essentia descriptors. The prototype is fully
functional without CLAP.

A dedicated `clap_embeddings` table is not needed — the BLOB column on `tracks`
keeps the schema simple and avoids an extra JOIN during sampling.
