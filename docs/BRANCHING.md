# Branching Sampling Algorithm — Playlist Progression Prototype

> **Status:** Design document for the similarity-based playlist sampler.

## Overview

The branching sampler controls how each successive track is selected from the
library relative to the current track. Rather than always picking the nearest
neighbour (which produces a monotonous, barely-changing playlist), the sampler
uses **distance bands** — discrete zones that determine how much perceptual
change each transition introduces. Band selection follows a configurable schedule
that cycles through Near, Mid, and Far zones to balance coherence with
discovery.

For the system context, see [ARCHITECTURE.md](./ARCHITECTURE.md#branching-design).
For the feature axes and their branching roles, see [SCHEMA.md](./SCHEMA.md#how-axes-drive-branching).

---

## Distance Bands

All distances are computed as **standardised Euclidean distance** across
descriptor axes. Before distance calculation, each axis value is normalised to
z-scores using the population mean and standard deviation of that axis across
all tracks in the library. This ensures axes with naturally large ranges (e.g.
spectral_centroid in Hz) don't dominate axes with small ranges (e.g.
danceability 0–1).

| Band | Normalised Distance | Behaviour |
|---|---|---|
| **Near** | distance ≤ 0.3σ | Close neighbours; minimal perceptual change. Establishes mood and groove. |
| **Mid** | 0.3σ < distance ≤ 0.7σ | Moderate jumps; noticeable shift but still related. Drives transitions between sections. |
| **Far-but-directed** | distance > 0.7σ on all axes EXCEPT holdAxis | Large leaps along most dimensions, but one axis stays close. Creates contrast and surprise while maintaining an anchor (e.g. keep tempo steady but change genre). |

### Fallback Rule

When a band runs out of unvisited candidates, the sampler falls back to the next
wider band with a logged notice. Fallback order: Near → Mid → Far. If all bands
are exhausted, the sampler selects the globally nearest unvisited track.

---

## Axis Weights

Axis weights are a JSON configuration object that modifies the distance
function. Each axis receives a multiplicative weight (default: 1.0). Higher
weights make the sampler treat differences on that axis as more significant,
which can force the playlist to drift along that dimension even within a single
band.

Example config:
```json
{
  "rhythm.tempo": 2.0,
  "highlevel.mood_happy": 1.5,
  "default": 1.0
}
```

With `rhythm.tempo` weighted 2.0, the playlist will tend to accelerate or
decelerate across successive tracks, because the distance function amplifies
tempo differences. Other axes still contribute but at their base weight.

---

## Directed Jump (Far-but-directed)

A directed jump picks a single **hold axis** and finds tracks that are far
(> 0.7σ) on every other axis while staying close (≤ 0.3σ) on the hold axis.
This creates a large perceptual shift anchored by one constant quality.

Example: hold `rhythm.tempo` constant, jump far on everything else → the playlist
lands on a track with a completely different mood, timbre, and key, but the same
tempo. The listener feels surprise while the rhythmic anchor keeps the flow
from breaking.

### Directed Jump Pseudocode

```
function selectDirectedJump(seed, candidates, holdAxis):
    holdIdx = axisIndex[holdAxis]
    results = []
    for track in candidates:
        // Compute full distance on all axes
        allDist = computeDistance(seed.features, track.features)
        // Compute distance on all axes EXCEPT holdAxis (weight holdAxis = 0)
        nonHoldFeatures = zeroOutAxis(seed.features, holdIdx)
        nonHoldTrack    = zeroOutAxis(track.features, holdIdx)
        nonHoldDist     = computeDistance(nonHoldFeatures, nonHoldTrack)
        // Compute distance on holdAxis only
        holdDist = singleAxisDistance(seed.features[holdIdx], track.features[holdIdx], axisWeights[holdIdx])

        if nonHoldDist > 0.7 AND holdDist ≤ 0.3:
            results.append(track)
    return results
```

---

## Sampling Strategy

The sampler maintains a position in the library (current track) and a target
band. It selects the next track from the target band's candidates, preferring
the nearest candidate within that band.

### Band Schedule

The default schedule is a repeating cycle: **Near → Mid → Far → Mid → Near**.
This produces an arc: establish mood (Near), transition (Mid), discovery peak
(Far), ease back (Mid), re-anchor (Near). The cycle length and band sequence
are configurable.

### Full Sampling Pseudocode

```
function generatePlaylist(seedTrack, library, config):
    schedule = config.bandSchedule  // default: [Near, Mid, Far, Mid, Near]
    scheduleIdx = 0
    visited = {seedTrack.id}
    current = seedTrack
    playlist = [seedTrack]

    while playlist.length < config.targetLength:
        targetBand = schedule[scheduleIdx % schedule.length]
        candidates = library.filter(t -> t.id not in visited)

        switch targetBand:
            case Near:
                matches = selectNear(current, candidates)
            case Mid:
                matches = selectMid(current, candidates)
            case Far:
                matches = selectDirectedJump(current, candidates, config.holdAxis)

        // Fallback: if band is empty, try next wider band
        if matches.isEmpty():
            matches = selectMid(current, candidates)  // fallback from Near
            if matches.isEmpty():
                matches = selectDirectedJump(current, candidates, config.holdAxis)  // fallback from Mid
                if matches.isEmpty():
                    matches = [globalNearest(current, candidates)]  // ultimate fallback

        // Pick nearest candidate within the matched band
        next = matches.minBy(t -> computeDistance(current.features, t.features))
        visited.add(next.id)
        playlist.append(next)
        current = next
        scheduleIdx += 1

    return playlist
```

---

## ComputeDistance — Standardised Euclidean

```
function computeDistance(trackA, trackB):
    sum = 0
    for i in 0..numAxes:
        diff = (trackA[i] - mean[i]) / stddev[i] - (trackB[i] - mean[i]) / stddev[i]
        weight = axisWeights[i]  // default 1.0
        sum += weight * diff * diff
    return sqrt(sum / numAxes)
```

Where `mean[i]` and `stddev[i]` are the population mean and standard deviation
of axis `i` across all tracks in the library. These are precomputed once when
the library is loaded.

### selectNear

```
function selectNear(seed, candidates):
    return candidates.filter(t -> computeDistance(seed.features, t.features) ≤ 0.3)
```

### selectMid

```
function selectMid(seed, candidates):
    return candidates.filter(t -> 0.3 < computeDistance(seed.features, t.features) AND
                                       computeDistance(seed.features, t.features) ≤ 0.7)
```

---

## Configuration Object

```json
{
  "axisWeights": {
    "rhythm.tempo": 1.0,
    "highlevel.mood_happy": 1.0,
    "highlevel.mood_sad": 1.0,
    "highlevel.mood_aggressive": 1.0,
    "highlevel.mood_relaxed": 1.0,
    "highlevel.mood_electronic": 1.0,
    "highlevel.mood_party": 1.0,
    "highlevel.mood_acoustic": 1.0,
    "rhythm.danceability": 1.0,
    "lowlevel.spectral_centroid.mean": 1.0,
    "lowlevel.spectral_complexity.mean": 1.0,
    "lowlevel.spectral_rolloff.mean": 1.0,
    "default": 1.0
  },
  "bandThresholds": {
    "near": 0.3,
    "mid": 0.7
  },
  "holdAxis": "rhythm.tempo",
  "targetLength": 20,
  "bandSchedule": ["Near", "Mid", "Far", "Mid", "Near"]
}
```
