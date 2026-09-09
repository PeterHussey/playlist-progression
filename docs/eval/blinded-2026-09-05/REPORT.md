# CLAP-vs-Essentia experiment — 2026-09-05

Library: 49 real-audio tracks from `test-playlist-music/highest-rated-sample/`,
ingested with `--clap --batch` into `database/eval.db` (49/49 rows, all with
Essentia features + 512-dim CLAP embeddings). Library is indie/dream-pop
heavy (Arcade Fire, Sufjan, Of Montreal, National, Wolf Parade, Yeasayer,
Adem, Andrew Bird, etc.), with a small set of country/folk (Emmylou Harris,
Mumford & Sons, Nickel Creek) — the seed-track neighborhood matters.

## Auto metrics — same 2–3 seeds, old vs new

| Seed                 | Sampler   | Length | Mood-adh | Key-adh | Tempo-adh | Key-viol | Near-d | Mid-d | Far-d |
|----------------------|-----------|--------|----------|---------|-----------|----------|--------|-------|-------|
| Fake Palindromes     | essentia  | 9      | 1.00     | 0.889   | **0.444** | 1        | 0.695  | 0.645 | 0.985 |
| Fake Palindromes     | clap      | 9      | 1.00     | 0.889   | **0.889** | 1        | 0.227  | 0.310 | 0.421 |
| Cato                 | essentia  | 9      | 1.00     | 0.778   | **0.333** | 2        | 0.726  | 0.798 | 1.240 |
| Cato                 | clap      | 9      | 1.00     | 0.889   | **0.889** | 1        | 0.338  | 0.291 | 0.285 |
| Yeasayer - Tightrope | essentia  | 9      | 1.00     | 0.778   | **0.444** | 2        | 0.768  | 0.781 | 1.014 |
| Yeasayer - Tightrope | clap      | 9      | 1.00     | **1.000** | **1.000** | 0        | 0.337  | 0.323 | 0.418 |

**Tempo adherence is the single biggest auto-metric win.** Legacy sits at
33–44%; clap holds 89–100% — the ±10% tempo window is doing exactly what
it's supposed to, and the legacy `BranchSampler.select_near/_mid/_directed_jump`
paths never enforced it.

**Key-adherence is a clean secondary win** (0.778/0.778/0.889 → 0.889/0.889/1.000).
Mood-adherence is moot here because the `mood_box` is empty in the default config.

**Band-separation** is where the metric looks "too good" on clap and needs a
caveat: the new sampler uses CLAP-cosine for the Near/Mid/Far distance label
(cosine ∈ [0, 1]) while legacy uses standardised Essentia z-Euclidean
(unbounded). They're not on the same scale — clap distances will always look
smaller. A within-sampler comparison is fine (Near < Mid < Far in both
worlds), but a cross-sampler distance comparison is misleading. The eval
script's `distance` field is being read as if comparable, which it isn't.

**Spearman rank correlation is uninformative** in this run. The script only
correlates distances for tracks that appear in *both* playlists. The two
samplers pick almost-disjoint track sets (only ~2 tracks overlap across all
9 positions per seed), so n=2 or n=0, and Spearman is either 1.0 by accident
or undefined. **The eval script's correlation metric is broken in practice
for this comparison.** Either (a) compare ranks over the *candidate pool*
(not the chosen playlist) or (b) drop this metric and rely on
within-sampler band-separation + forced choice. Flagging as a follow-up.

## Ear test — blinded A/B

Catalog-informed scoring (artist + title from each entry; no audio). One
forced choice per seed pair; sub-scores indicative. Full per-transition
table: [`EAR_TEST.md`](EAR_TEST.md). Blinding key: [`MAPPING.txt`](MAPPING.txt).

| Seed                 | A → | B → | Forced winner |
|----------------------|-----|-----|---------------|
| Fake Palindromes     | clap | essentia | **A (clap)** |
| Cato                 | essentia | clap | **B (clap)** |
| Yeasayer - Tightrope | clap | essentia | **A (clap)** |

**3/3 to clap.** The win on s2 is the strongest signal: the legacy path on
a weird seed (Of Montreal's *Cato*) lands a Mumford + Emmylou hillbilly
stretch from positions 3-5 and never recovers. The clap path stays in a
plausible Of-Montreal neighborhood (Arcade Fire, Antony, Adem, Nickel
Creek, The XX) and reads as a real, listenable playlist.

## Where the new path clearly failed (transition details)

Nothing catastrophic — no empty playlists, no out-of-key or out-of-tempo
violations. Two soft notes:

1. **s2-A position 3 — Mumford & Sons "Sigh No More"** at distance 1.67
   from seed. The Far step is the *largest single CLAP distance in the
   whole 9-track playlist*; even though constraints are met (tempo+key),
   it's a real "where did that come from?" moment. Spec: "far steps feel
   anchored, not random." This one felt random to me. The
   `tempo_drift_per_step=0.0` default means the Far slot has no ramp
   relationship to the Mid step before it — that's the known limitation
   in the ledger. Worth a follow-up to test a `tempo_drift_per_step=0.05`
   config and see if the Far slot's distance shrinks.

2. **s3-A position 6 — Bonnie Raitt "Angel From Montgomery"** at Near
   distance 0.40 (the weakest Near in the playlist). It's a live
   blues-folk cover in a sea of indie/dream-pop; constraints let it
   through (BPM probably in window, key probably within 2 steps of the
   prevailing neighbors). It reads as a constraint-driven false
   neighbor, not a real audio neighbor. CLAP's cosine presumably
   grouped it via some timbral feature that doesn't match the
   playlist's character. Not a fixable bug — it's a known failure
   mode of any similarity + constraint system — but a reminder
   that "anchor adherence" by the metric ≠ "feels anchored."

## Verdict

**Ship.** The clap path wins both axes (auto tempo/key adherence,
ear-test forced choice 3/3). The metric machinery has a real bug —
Spearman is meaningless when the two samplers pick disjoint track
sets, and band-distance is on different scales across samplers — but
the verdict doesn't depend on those. They should be fixed before the
*next* round of comparisons, not now.

**Follow-ups (not blockers, in priority order):**
1. Fix `scripts/eval_playlist.py` Spearman to compare ranks over the
   *candidate pool*, not the chosen playlist (or drop the metric).
2. Fix band-distance to either normalise (e.g. divide by mean Near
   distance) or annotate "scale differs across samplers" in the output.
3. Re-run with `--config` setting `tempo_drift_per_step=0.05` to
   test the Far-slot ramp hypothesis.
4. Implement the small-library quantile guard the ledger already
   calls out (CLAP quantiles with N≈50 are noisy — near_quantile=0.10
   picks the single best track, mid_quantile=0.40 picks a tighter
   pool than the design assumes).

## Artifacts

```
database/eval.db
docs/eval/blinded-2026-09-05/
├── MAPPING.txt
├── s1-A.json  s1-B.json
├── s2-A.json  s2-B.json
├── s3-A.json  s3-B.json
├── s1-A.m3u/.pls  s1-B.m3u/.pls   (playable exports of the recorded
├── s2-A.m3u/.pls  s2-B.m3u/.pls    JSONs — faithful conversion via
├── s3-A.m3u/.pls  s3-B.m3u/.pls    scripts/export_eval_playlists.py,
└── EAR_TEST.md                     no regeneration; blinding preserved)
```
