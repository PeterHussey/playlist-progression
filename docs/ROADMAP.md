# Roadmap — playlist-progression

> The only place with checkboxes and milestones. `PROJECT.md` holds
> objective/criteria/scope/decisions. Feature ideas (`FEATURE_IDEAS.md`) are deferred by default.

Status legend: ✅ done · 🔶 in progress · ⬜ open

## M0 — Evidence so far ✅

- ✅ 17-track CLAP vs Essentia comparison (Spearman ρ = 0.078, NN match 5.9%,
  CLAP full Near/Mid/Far schedule, 0 fallbacks) — `docs/comparison-clap-vs-essentia.md`.
- ✅ 49-track blinded A/B, verdict ship-clap-as-default — `docs/eval/blinded-2026-09-05/REPORT.md`.
- ✅ CLAP+constraints sampler behind `--sampler` flag (default `clap`) —
  `src/recommender/playlist_sampler.py`, `clap_similarity.py`, `constraint_filter.py`.

## M1 — Docs truth ✅

- ✅ README flipped to CLAP-primary (`a3f6a47`).
- ✅ `docs/CHARTER.md` (later demoted to `PROJECT.md` on 2026-09-09), `docs/ROADMAP.md` (this file), `docs/FEATURE_IDEAS.md` created (`a3f6a47`).
- ✅ `docs/ARCHITECTURE.md` rewrite (Java/`ProcessBuilder` → Python
  `ingest_pipeline.py` + subprocess; single-table `tracks` schema; batch worker notes)
  (`e8171a5`). Acceptance: no mention of Java orchestrator, three-table schema, or "CLAP unused".
- ✅ `docs/BACKLOG.md` triage (§8 CLAP-for-distance and §10 comparison marked done;
  §11 open items folded or linked here) (`e8171a5`). Acceptance: no contradiction
  with CHARTER decisions.

## M2 — Eval hardening ✅

- ✅ Fix `scripts/eval_playlist.py` Spearman on disjoint track sets — compare ranks over
  the candidate pool, not chosen playlists (`c54abc9`). Acceptance: metric defined
  on N ≥ 20 shared candidates or removed.
- ✅ Annotate / normalise cross-sampler distance scales (CLAP cosine ∈ [0,1] vs Essentia
  z-Euclidean unbounded) (`c54abc9`). Acceptance: eval output states "scales differ" or normalises.
- ✅ `tempo_drift_per_step=0.05` config test (Mumford Far-step outlier from blinded s2-A).
  Acceptance: recorded distance + listening note, no code change required.
- ✅ Small-N quantile guard (`near_quantile=0.10` on N≈50 picks a single track)
  (`9b1ed6c`). Acceptance: guard from spec §6 reused or ledger-accepted with note.
- ✅ `playlist_summary.txt` parity on clap path (currently JSON only)
  (`2ac8e4d`). Acceptance: summary written or `--help` calls out the gap.

## M3 — Subset scale test ⬜

- ⬜ Ingest ~150–250 tracks (larger than 49-track test, smaller than full library):
  `run.py <subset> <db> --clap --batch`. Acceptance: 100% CLAP coverage per
  `PlaylistSampler.load_library`, idempotent re-run skips clean.
- ⬜ Generate + ear-test 3 seeds × clap sampler, blinded where practical.
  Acceptance: band exercise + fallback counts recorded, forced-choice votes recorded.

## M4 — Full-library run ⬜

- ⬜ Run thousands-track library with `--batch --clap` (CUDA CLAP pass on Lenovo
  if CPU throughput is inadequate — runbook note only, no code change assumed).
  Acceptance: completed ingest + coverage report + runbook notes (timeouts, retries).

## M5 — Deprecation cleanup ⬜

- ⬜ Remove `--sampler essentia` + `src/recommender/branch_sampler.py` **only after M3 passes**.
  Acceptance: sampler flag + legacy module + docs references removed; full test suite green.

## Explicit non-goals for M1–M4

New playlist features (fusion weighting, schedules, playback/export) stay in
`FEATURE_IDEAS.md` and do not block milestones above.
