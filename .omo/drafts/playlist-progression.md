---
slug: playlist-progression
status: awaiting-approval
intent: clear
review_required: true
approach: Prototype a Java-based local music similarity/discovery prototype using Essentia (subprocess CLI) for structured DSP features + optional CLAP embeddings; SQLite feature store; branching recommender (near/mid/far bands + axis-controlled jumps); JSON playlist output; full GitHub init included.
---

# Draft: playlist-progression

## Components (topology ledger)
- C1: Architecture + Integration (Java ↔ Python/native) - active
- C2: Feature/Embedding Schema (Essentia descriptors + SQLite) - active
- C3: Branching Sampling Algorithm - active
- C4: Prototype Scaffold (ingest → extract → sample → JSON output) - active
- C5: GitHub Repo Init + Remote Push - active

## Open assumptions (announced defaults)
- Integration approach: Subprocess (ProcessBuilder) from Java to Essentia CLI + CLAP Python script. Recommended; user selected.
- Descriptor set: Full (timbre/tonal/rhythm/mood/sfx harmonic). Recommended; user selected.
- Output format: JSON playlist file (structured with seed + distance bands + reason). User selected.
- GitHub init: Full init + push included in plan. User selected.

## Findings (cited - path:lines)
- Workspace: /Users/hussey/Documents/GitHub/playlist-progression/ — empty, no .git, no source files.
- User research: Evaluated Essentia (AGPLv3, CPU-friendly, interpretable), CLAP (open-source neural embeddings), Last.fm (tertiary signal only), MiMo-Audio (ruled out as core). Decision to use Essentia primary + CLAP secondary is sound.
- Subprocess approach avoids JNI complexity; compatible with prototype scope. CLAP Python script can be a single-file wrapper around laion-CLAP or msclap.

## Decisions (with rationale)
- Subprocess integration: simplest viable for weekend prototype; avoids extra HTTP service, no JNI, Python dependency explicit.
- SQLite schema: `tracks` table (path, title, artist, duration, feature_json, clap_embedding_blob); `embeddings` optional separate table.
- Branching algorithm: 3 distance bands (near ≤ 0.3σ, mid 0.3-0.7σ, far-but-directed ≥ 0.7σ along a specific axis) to avoid pure nearest-neighbor.
- Full descriptor set: gives maximum axes for directed jumps (e.g., same tempo/different mood, same mood/different instrumentation).

## Scope IN
- Java prototype with subprocess integration to Essentia CLI + CLAP Python script
- SQLite feature/embedding storage
- Ingest pipeline: scan audio folder, extract descriptors, store
- Branching recommender: near/mid/far sampling + directed axis jumps
- JSON playlist output with seed info + per-track reason
- GitHub repo init with remote push steps included

## Scope OUT (Must NOT have)
- No Spotify/streaming service integration (local files only)
- No real-time audio playback or streaming server
- No production-grade user authentication or multi-user support
- No MiMo-Audio natural-language explanation layer (deferred)
- No GPU-required heavy models at core (CLAP embedding optional, can fall back to CPU)

## Open questions
- None — all forks resolved through user selections.

## Review receipts (dual review — round-playlist-progression-01)
- momus (bg_f477c9a4): APPROVE — all 9 checks pass (References/Acceptance/QA/Commit per todo; F1–F4 present; no prose-as-tasks; IN/OUT explicit; zero judgment calls; branching concrete with 3 distance bands; subprocess defensible; exclusions complete; TL;DR full). Session: ses_fa54cb6acffefg6rN5E2DpXCjI.
- independent/oracle (bg_9f306265): APPROVE — digest `322cc7b973aac1ce601e5a9897bd253525526f21cc61dde1744f51653c45ed3d` (12645 bytes, regular file, not symlink); all 8 checks pass; header order, 6 column-zero todos, F1–F4, scope IN/OUT, no forbidden items, integration defensible, dependency matrix consistent. Session: ses_fa54c8d88ffeL4fgOTLQBqMbCx.

## Approval gate
status: approved (user opted for review; both lanes APPROVE, no fixes required)
next_action: execute via `$start-work playlist-progression`
