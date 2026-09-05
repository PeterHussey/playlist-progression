# Playlist Sampler v2 — CLAP Similarity + Essentia Constraints

**Date:** 2026-09-05
**Status:** Approved design (sections 1–5 signed off in chat)
**Context:** Essentia + CLAP extraction working; Essentia 20-axis Euclidean distance not trusted. Goal: discovery-with-anchor playlists. Decision: Approach A (CLAP rank + Essentia gate). Approach B rejected. Approach C (beam-search arc planner) parked for later.

## 1. Problem

The current `BranchSampler.compute_distance` (`src/recommender/branch_sampler.py:44-79`) computes standardised weighted Euclidean over 20 mixed axes (`src/recommender/feature_converter.py:6-14`): `duration_sec`, `loudness.*`, `tempo.confidence`, `key.confidence`, spectral descriptors, danceability, 7 moods, key circle coords. Concrete defects:

- Non-perceptual axes (duration, loudness, confidences) enter with equal weight — confidence ≠ content.
- Correlated axes double-count (centroid + rolloff = brightness; moods + danceability overlap).
- Z-scoring needs stable population stats; unstable on small/unmeasured libraries.
- Directed-jump implementation zeroes the hold axis asymmetrically (`branch_sampler.py:132-159`) and compares raw abs-diff hold distance against σ-unit thresholds — incomparable units.

CLAP 512-dim embeddings (`tracks.clap_embedding`, cosine space) capture semantic/instrumental/mood similarity and are the better ranking signal. Essentia's value is interpretability and control: BPM (float), key (24-slot circle), mood classifiers (noisy but useful as bounding boxes), danceability.

## 2. Goal & Non-Goals

**Goal:** Discovery-with-anchor playlists — surprising picks anchored by a held quality, mood-box first. Success = wins on both auto metrics and a blinded ear test vs the legacy sampler (see §6).

**Non-goals:** No hybrid α-weighted score (Approach B, rejected). No beam-search arc planner (Approach C, parked — this design must not preclude it). No DB migration, no new extraction descriptors, no GPU work, no player integration.

## 3. Architecture

```
Library load → normalise CLAP (L2) + parse Essentia JSON + coverage log
  → per step: ConstraintFilter.filter(candidates)   [Essentia gates]
  → ClapSimilarity.rank(seed, survivors)            [cosine]
  → pick nearest-within-target-band per schedule
  → fallback chain (band relax → gate relax → global nearest)
  → JSON playlist (existing writer format)
```

New modules are additive; legacy path frozen:

| Module | Role | Status |
|---|---|---|
| `src/recommender/clap_similarity.py` | L2-normalise, cosine sim/dist, per-seed quantile bands | NEW |
| `src/recommender/constraint_filter.py` | `MoodBox`, `key_steps()`, `tempo_ok()` | NEW |
| `src/recommender/playlist_sampler.py` | Orchestrating loop (`PlaylistSampler`) | NEW |
| `scripts/eval_playlist.py` | Old-vs-new auto metrics | NEW |
| `docs/eval/playlist-ear-test.md` | Blinded ear-test sheet | NEW |
| `src/recommender/branch_sampler.py` | Legacy σ-band sampler | FROZEN (docstring note, no logic change) |
| `src/recommender/feature_converter.py` | 20-axis `convert()` + `key_to_circle` | READ-ONLY (key mapping reused) |

## 4. Similarity Core (CLAP cosine + quantile bands)

- Primary ranking = cosine similarity on 512-dim CLAP embeddings. `sim = dot(a,b)/(‖a‖‖b‖)`, `dist = 1 - sim`. Embeddings L2-normalised once at library load; dot = cosine thereafter.
- Bands = **per-seed quantiles** over unvisited candidates: Near = top 10% most similar (above the 90th similarity percentile), Mid = 10–40% (60th–90th percentile), Far = below the 60th percentile but inside the mood box ("far yet anchored"). Configurable via `near_quantile: 0.10`, `mid_quantile: 0.40`.
- Rationale: σ thresholds collapse on small libraries; quantiles always yield non-empty bands. Minimum-library guard: <20 candidates → rank-ordered greedy, bands skipped, logged.
- Legacy Essentia distance frozen, kept importable for the eval baseline and existing tests.

## 5. Essentia Constraints (filter, not distance)

Order per transition: **mood box → key + tempo gates → CLAP rank within survivors.**

- **Mood box (primary anchor, hard filter):** config e.g. `{"mood.sad": {"min": 0.5}, "mood.aggressive": {"max": 0.4}}`. Missing `mood` key → excluded by default (`allow_missing_mood: false` override). Empty-result fallback: relax each bound by 0.1 up to 3 steps, logging each relaxation.
- **Key gate (congruence, skippable):** reuse `key_to_circle` 24-slot mapping. Steps = circular slot distance 0–12 (C–Am = 1, C–G = 2, C–F♯ = 12). Allow ≤2, penalise 3–4 (rank demotion, not exclusion — exact penalty defined at plan time), block ≥5. Low `key.confidence` → skip gate entirely. "Low" = below library 25th percentile at load (avoids hardcoding Essentia's unbounded strength scale).
- **Tempo window (anchor, optional ramp):** default `tempo_max_step_pct: 0.10` (±10% BPM vs current track, BPM read raw from `tempo.bpm`). Optional `tempo_drift_per_step` (e.g. +0.02) slides the window centre for ramps while CLAP still picks *which* track inside it. `tempo.confidence` ignored.

No changes to `extract_essentia.py` or the extractor schema.

## 6. Sampler Integration, Fallbacks, Config

- `PlaylistSampler` owns the loop: filter → rank → pick-nearest-within-target-band per schedule. Default schedule reuses `[Near, Mid, Far, Mid, Near]` with quantile-band meanings.
- Fallback order (each logged with reason): relax band outward (Near→Mid→Far) → relax Essentia gates (mood-relax steps) → global nearest-by-CLAP.
- Missing-data policy (fail-closed defaults for unknown-coverage libraries): missing `clap_embedding` → excluded; `allow_missing_clap: false` override ranks those tracks by curated-Essentia fallback and logs it. Library load logs CLAP coverage %, mood-missing count, key-confidence p25.
- Config (single JSON to sampler / `run.py --config`):
  ```json
  {
    "near_quantile": 0.10, "mid_quantile": 0.40,
    "mood_box": {"mood.sad": {"min": 0.5}},
    "key_max_steps": 2, "tempo_max_step_pct": 0.10, "tempo_drift_per_step": 0.0,
    "band_schedule": ["Near", "Mid", "Far", "Mid", "Near"],
    "allow_missing_clap": false, "allow_missing_mood": false
  }
  ```
- CLI: `run.py --sampler {clap,essentia}` (default `clap`) + `--config playlist_config.json`; `make playlist` passes through. Ingest `--clap` behaviour unchanged.

## 7. Eval Harness

- **Auto metrics** (`scripts/eval_playlist.py`, same seed+library, old vs new): (1) anchor adherence — % transitions inside mood box / key ≤2 / tempo window; (2) band separation — mean CLAP cosine of Near vs Mid vs Far picks (new should separate, old should look flat); (3) old-vs-new agreement — rank correlation of Essentia distance vs CLAP distance over all pairs (near-zero confirms the suspicion).
- **Ear test** (blinded A/B; requires real audio — `tests/sample_audio/` placeholders insufficient): fix a 30–50-track local-only eval library, extract both feature types once; same 2–3 seeds × both samplers → 8–10-track playlists blinded as `A.json`/`B.json`; score each transition 1–5 on flow / anchor / surprise + forced choice "which would you keep listening to". N=1 (you) suffices for phase 1. Sheet: `docs/eval/playlist-ear-test.md`.
- **Unit tests:** `test_clap_similarity.py` (cosine, quantiles, small-library collapse); `test_constraint_filter.py` (mood relax steps, key steps C–Am=1/C–F♯=12, low-confidence skip, tempo window/drift).

## 8. Migration & Rollout

1. Assemble 30–50-track eval set + extract both feature types. 2. Implement similarity + filter + sampler + unit tests. 3. Run auto metrics old-vs-new. 4. Blinded ear test. 5. Keep or tune quantiles/gates. No DB migration (columns already exist). Old code frozen with a "superseded" docstring; existing tests untouched. C parked: quantile bands + constraint objects are reusable there, nothing here blocks it.

## 9. Open Items for the Plan

- Exact key 3–4-step penalty shape (demotion weight vs exclusion).
- Quantile defaults validation against the eval set (0.10/0.40 are starting points).
- Curated-Essentia fallback axis subset for `allow_missing_clap: true` (only needed if coverage is low).
- Ear-test sheet format and playlist length (8 vs 10).
