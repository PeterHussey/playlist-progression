# Feature Ideas — deferred ledger

> Append-only parking lot. Nothing here is committed work. Promotion to
> `ROADMAP.md` requires explicit approval. Discussion of these ideas never
> blocks ROADMAP M1–M4.

Entry format: title · 2–3 sentence description · why deferred · what would unblock it.

## Seed entries (recorded, not decided)

### Essentia+CLAP fusion weighting

Combine the 512-dim CLAP vector with the 20-dim Essentia axes under a
normalisation/weighting design (raw fusion would let CLAP dominate RMS distance).
Deferred: needs a fusion proposal; CLAP-primary already ships without it.
Unblocks: a weighting/normalisation design + subset evidence it beats CLAP-alone.

### New band schedules / hold-axis variants

E.g. alternative Near/Mid/Far orders, `tempo_drift_per_step` ramps, key-based
directed jumps beyond the current tempo/key/mood gates.
Deferred: current default schedule (Near→Mid→Far→Mid→Near) is unevaluated at subset scale.
Unblocks: M3 subset results showing a schedule-shaped failure.

### Single ingest→playlist UX + `--seed-title` polish

One command from music dir to playlist JSON, friendlier seed picking, Makefile targets.
Deferred: usability, not correctness at scale.
Unblocks: M3/M4 operator pain notes.

### Playback / export integrations

M3U/PLS export, local player handoff.
Deferred: output contract is JSON today; no consumer requirement stated.
Unblocks: a stated consumer + format decision.

### GPU/CUDA CLAP runbook for Lenovo

Documented CUDA torch setup + throughput numbers for the thousand-track CLAP pass.
Deferred: unknown whether CPU batch throughput is actually inadequate.
Unblocks: M3 timing data showing the CLAP pass is the bottleneck.
