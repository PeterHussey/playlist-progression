# Playlist Ear Test (blinded A/B)

1. Build a 30–50-track local eval library (real audio; never commit). Extract Essentia + CLAP once.
2. Same 2–3 seeds × both samplers → 8–10-track playlists. Save blinded as `A.json` / `B.json` (shuffle which sampler is A per seed; record mapping separately).
3. For each transition score 1–5: flow (smooth?) / anchor (mood held?) / surprise (far steps feel anchored, not random?).
4. Forced choice per playlist pair: "which would you keep listening to?" + one-line why.
5. Unblind and compare against `scripts/eval_playlist.py` metrics. Ship if new wins both.
