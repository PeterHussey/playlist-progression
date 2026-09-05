"""CLAP-rank + Essentia-gate playlist sampler (new default path)."""
from __future__ import annotations
import json
from .track import Track
from .clap_similarity import rank_by_similarity, partition_quantile_bands
from .constraint_filter import ConstraintFilter, relax_mood_box, key_slot, key_steps, key_verdict, tempo_ok, _parsed_sidecar

DEFAULT_SCHEDULE = ["Near", "Mid", "Far", "Mid", "Near"]


class PlaylistSampler:
    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        self.near_q = float(config.get("near_quantile", 0.10))
        self.mid_q = float(config.get("mid_quantile", 0.40))
        self.schedule: list[str] = list(config.get("band_schedule", DEFAULT_SCHEDULE))
        self.drift = float(config.get("tempo_drift_per_step", 0.0))
        self.allow_missing_clap = bool(config.get("allow_missing_clap", False))
        self.filter = ConstraintFilter(config)
        self._norm: dict[int, list[float]] = {}
        self._by_id: dict[int, Track] = {}

    def load_library(self, tracks: list[Track]) -> dict:
        from .clap_similarity import l2_normalize
        self._by_id = {t.id: t for t in tracks}
        self._norm = {}
        n_clap = 0
        for t in tracks:
            if t.clap_embedding:
                self._norm[t.id] = l2_normalize(list(t.clap_embedding))
                n_clap += 1
        cov = (n_clap / len(tracks)) if tracks else 0.0
        print(f"PlaylistSampler: {len(tracks)} tracks, CLAP coverage {n_clap}/{len(tracks)} ({cov:.0%})")
        return {"n_tracks": len(tracks), "n_clap": n_clap, "clap_coverage": cov}

    def _usable(self, candidates: list[Track]) -> list[Track]:
        if self.allow_missing_clap:
            return list(candidates)
        return [t for t in candidates if t.id in self._norm]

    def generate(self, seed: Track, limit: int) -> list[dict]:
        visited = {seed.id}
        current = seed
        entries: list[dict] = []
        mood_box = dict(self.filter.mood_box)
        for step in range(limit):
            band = self.schedule[step % len(self.schedule)]
            candidates = [t for t in self._by_id.values() if t.id not in visited]
            candidates = self._usable(candidates)
            survivors = self.filter.filter(current, candidates)
            relaxed = 0
            box = dict(mood_box)
            while not survivors and box and relaxed < 3:
                box = relax_mood_box(box)
                relaxed += 1
                f = ConstraintFilter({**self.filter.__dict__, "mood_box": box,
                                       "key_max_steps": self.filter.key_max_steps,
                                       "tempo_max_step_pct": self.filter.tempo_max_step_pct,
                                       "allow_missing_mood": self.filter.allow_missing_mood})
                survivors = f.filter(current, candidates)
            if not survivors:
                break
            seed_vec = self._norm.get(current.id) or (list(current.clap_embedding) if current.clap_embedding else None)
            if seed_vec is None:
                break
            ranked = rank_by_similarity(seed_vec, [(t.id, self._norm[t.id]) for t in survivors if t.id in self._norm])
            if not ranked:
                break
            bands = partition_quantile_bands(ranked, self.near_q, self.mid_q)
            pick = None
            actual = band
            key = band.lower()
            if bands.get(key):
                pick = bands[key][0][0]
            else:
                for fb in ("near", "mid", "far"):
                    if bands.get(fb):
                        pick, actual = bands[fb][0][0], fb.capitalize()
                        break
            if pick is None:
                break
            nxt = self._by_id[pick]
            sim = next(s for i, s in ranked if i == pick)
            reason = f"{actual} CLAP pick (sched {band})" + (f", mood relaxed x{relaxed}" if relaxed else "")
            entries.append({"position": step + 1, "track_id": nxt.id, "title": nxt.get_title(),
                            "artist": nxt.get_artist(), "band": actual,
                            "distance": round(1.0 - sim, 4), "reason": reason})
            visited.add(nxt.id)
            current = nxt
        return entries
