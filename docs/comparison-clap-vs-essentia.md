# CLAP vs Essentia Comparison Results

**Date:** 2026-09-04
**Library:** 17-track test playlist (real audio)
**Seed:** Orphan Girl (Gillian Welch, id 17, row index 16 under `ORDER BY id`)

## Setup

### Step 1: Install dependency + smoke test

```bash
.venv/bin/pip install laion-clap
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python scripts/extract_clap.py "test-playlist-music/01 Orphan Girl.m4a" /tmp/clap_smoke.json
.venv/bin/python -c "import json; e=json.load(open('/tmp/clap_smoke.json'))['embedding']; print(len(e), e[:3])"
```

Result: `512 [0.0351426862180233, 0.0006713420152664185, 0.011621251702308655]` -- smoke passed.

**Note:** `scripts/extract_clap.py` required a fix: the original code called `model.load_audio()` which does not exist in laion-clap 1.1.7. The correct API is `model.get_audio_embedding_from_filelist()`. This was fixed before the smoke test succeeded.

**Note:** `clap_compare.py` `pairwise_report` required a fix: the original code passed Essentia means/stds (20-dim) to `_rms_zdist` for CLAP vectors (512-dim), causing an IndexError. Fixed by computing CLAP population means/stds independently within `pairwise_report`.

### Step 2: Populate embeddings

```bash
.venv/bin/python run.py test-playlist-music database/playlist.db --clap --batch --timeout 1800 --no-mood
```

Result: `17 tracks processed`

### Verification gates

```bash
sqlite3 database/playlist.db "SELECT COUNT(*) FROM tracks WHERE clap_embedding IS NOT NULL;"
```
Result: `17`

```bash
.venv/bin/python -c "import sqlite3,json; c=sqlite3.connect('database/playlist.db'); print(set(len(json.loads(r[0])) for r in c.execute('SELECT clap_embedding FROM tracks')))"
```
Result: `{512}`

Mood preservation:
```bash
sqlite3 database/playlist.db "SELECT COUNT(*) FROM tracks WHERE feature_json LIKE '%"mood"%';"
```
Result: `17` (all mood data intact, no Essentia re-extraction)

## Comparison Metrics

### Step 3: Pairwise distance comparison

```python
import sqlite3, json, sys
sys.path.insert(0, '.')
from src.recommender.feature_converter import AXIS_NAMES, convert
from src.recommender import clap_compare as cc
conn = sqlite3.connect('database/playlist.db')
rows = conn.execute("SELECT feature_json, clap_embedding FROM tracks ORDER BY id").fetchall()
ess = [convert(r[0]) for r in rows]
clap = [json.loads(r[1]) for r in rows]
n = len(AXIS_NAMES)
means = [sum(v[i] for v in ess)/len(ess) for i in range(n)]
stds = [(sum((v[i]-means[i])**2 for v in ess)/len(ess))**0.5 or 1.0 for i in range(n)]
rep = cc.pairwise_report(ess, clap, means, stds)
agr = cc.nn_agreement(rep["essentia_dists"], rep["clap_dists"], len(ess))
walk = cc.clap_walk(seed_idx=16, clap_vecs=clap, limit=9)
print("pairs:", rep["n_pairs"], "spearman:", round(rep["spearman"], 3))
print("nn_exact:", round(agr["exact_match_rate"], 3), "top3:", round(agr["top3_overlap_mean"], 3))
print("clap walk:", [(e["track_idx"], e["band"], e["distance"]) for e in walk])
```

| Metric | Value |
|--------|-------|
| Pairs compared | 136 (C(17,2)) |
| Spearman rank correlation | **0.078** |
| Nearest-neighbor exact match rate | **0.059** (1/17) |
| Top-3 overlap mean | **0.176** |

### Distance matrix approach

- **Essentia distances**: RMS z-distance across 20 DSP/mood axes (population z-scored)
- **CLAP distances**: RMS z-distance across 512 CLAP embedding dimensions (population z-scored, means/stds computed independently from Essentia ones)

## Playlist Orders

### Essentia 9-track order (from `branch_playlist.json`)

Seed: Orphan Girl (Gillian Welch)

| Pos | ID | Title | Artist | Band | Distance |
|-----|----|-------|--------|------|----------|
| 1 | 3 | Hey, That's No Way to Say Goodbye | Leonard Cohen | Far | 0.8289 |
| 2 | 2 | Sovay | Andrew Bird | Far | 0.9686 |
| 3 | 9 | Wonderwall | Ryan Adams | Far | 0.8364 |
| 4 | 6 | Little Green | Joni Mitchell | Far | 0.9380 |
| 5 | 4 | Naked If I Want To | Cat Power | Far | 0.8915 |
| 6 | 12 | Hallelujah | John Cale | Far | 1.0061 |
| 7 | 16 | Girl From The North Country | Bob Dylan | Far | 0.7641 |
| 8 | 13 | Farewell, Angelina | Joan Baez | Far | 0.9511 |
| 9 | 11 | Challengers | The New Pornographers | Far | 0.9332 |

Note: All 9 transitions fell back to "global-nearest (actual Far)" because the 20-axis z-scored distance space has no Near or Mid candidates from Orphan Girl -- every other track is far away in that space.

### CLAP 9-track order (greedy walk, same schedule)

Seed: Orphan Girl (Gillian Welch, row index 16)

| Pos | Row Idx | DB ID | Title | Artist | Band | Distance |
|-----|---------|-------|-------|--------|------|----------|
| 1 | 15 | 16 | Girl From The North Country | Bob Dylan | Near | 0.1299 |
| 2 | 9 | 10 | The Boxer | Simon & Garfunkel | Mid | 0.3050 |
| 3 | 6 | 7 | Casimir Pulaski Day | Sufjan Stevens | Far | 0.5185 |
| 4 | 10 | 11 | Challengers | The New Pornographers | Mid | 0.3449 |
| 5 | 0 | 1 | Surf Song | James Yorkston | Near | 0.3168 |
| 6 | 13 | 14 | Revelator | Gillian Welch | Near | 0.3877 |
| 7 | 11 | 12 | Hallelujah | John Cale | Mid | 0.3404 |
| 8 | 7 | 8 | Angel From Montgomery | Bonnie Raitt | Far | 0.6557 |
| 9 | 5 | 6 | Little Green | Joni Mitchell | Mid | 0.4903 |

Note: CLAP walk exercises Near/Mid/Far transitions as designed -- 3 Near, 4 Mid, 2 Far selections with zero fallbacks.

## Interpretation

The Spearman rank correlation of **0.078** across 136 pairs indicates the two distance spaces are nearly uncorrelated. Essentia's 20-axis DSP/mood feature space and CLAP's 512-dim learned audio embedding capture fundamentally different aspects of musical similarity. The nearest-neighbor exact match rate of **5.9%** (only 1 out of 17 tracks agrees on the closest neighbor) confirms this: the two metrics identify different "nearest" tracks for almost every song.

The practical consequence is most visible in the playlist walks. The Essentia walk from Orphan Girl produces a homogeneous Far-only sequence -- the z-scored 20-axis space places every other track at d > 0.7 from the seed, so the generator falls back to global-nearest at every step and never finds Near or Mid candidates. By contrast, CLAP successfully finds Near transitions (Girl From The North Country at d=0.13) and Mid transitions (The Boxer at d=0.31) that Essentia completely misses. CLAP's ability to exercise the full Near/Mid/Far band schedule -- including a meaningful Near pairing between Orphan Girl and Girl From The North Country, two folk songs with similar vocal timbre -- suggests it captures perceptual/audio-content similarity that the hand-crafted DSP axes do not.

This does not mean CLAP is "better" in an absolute sense. The low correlation means they encode complementary information: Essentia's axes are interpretable and axis-specific (tempo, key, mood), while CLAP's 512 dimensions are opaque but capture holistic audio similarity. A fusion approach that weights both spaces could combine interpretability with perceptual accuracy, but would require a design for normalisation and weighting (currently parked in BACKLOG #8).
