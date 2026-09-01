"""Branch sampler — distance-band playlist selection.

Selects successive tracks from an audio library using standardised Euclidean
distance across descriptor axes, with three bands (Near/Mid/Far-but-directed)
and a directed-jump mode.

Algorithm design follows BRANCHING.md:
  • Near: distance ≤ 0.3σ — close neighbours, minimal perceptual change
  • Mid: 0.3σ < distance ≤ 0.7σ — moderate jumps, noticeable but related
  • Far-but-directed: distance > 0.7σ on all axes except holdAxis, which stays
    close (≤ 0.3σ) — large leaps anchored by one constant quality
"""

from __future__ import annotations

from typing import Optional

from .track import Track


class BranchSampler:
    """Similarity-based playlist sampler using distance bands."""

    # ---- Default thresholds (in standardised σ units) ----
    DEFAULT_NEAR_THRESHOLD = 0.3
    DEFAULT_MID_THRESHOLD = 0.7

    def __init__(
        self,
        axis_weights: list[float],
        axis_index: dict[str, int],
        band_thresholds: Optional[dict[str, float]] = None,
        axis_means: Optional[list[float]] = None,
        axis_stddevs: Optional[list[float]] = None,
    ) -> None:
        self.axis_weights = axis_weights
        self.axis_index = axis_index
        self.band_thresholds = band_thresholds or {"near": self.DEFAULT_NEAR_THRESHOLD, "mid": self.DEFAULT_MID_THRESHOLD}
        self.axis_means = axis_means or []
        self.axis_stddevs = axis_stddevs or []

    # ---- Distance computation ----

    def compute_distance(self, track_a: Track, track_b: Track) -> float:
        """Compute standardised weighted Euclidean distance between two tracks.

        Formula: sqrt(Σ weight[i] * ((a[i] - mean[i]) / stddev[i] - (b[i] - mean[i]) / stddev[i])²)

        Returns:
            distance (≥ 0), or float('inf') if stddevs are not available
        """
        if track_a.features is None or track_b.features is None:
            return float("inf")

        features_a = track_a.features
        features_b = track_b.features

        if len(features_a) != len(features_b):
            raise ValueError(
                f"Descriptor vectors must have the same length: {len(features_a)} vs {len(features_b)}"
            )

        if not self.axis_stddevs:
            # No standardisation available — fall back to unweighted Euclidean
            diff_sq = sum((a - b) ** 2 for a, b in zip(features_a, features_b))
            return diff_sq ** 0.5

        total = 0.0
        n = min(len(features_a), len(self.axis_stddevs))
        for i in range(n):
            w = self.axis_weights[i] if i < len(self.axis_weights) else 1.0
            std_a = (self.axis_stddevs[i] != 0.0) and self.axis_stddevs[i] or 1.0
            std_b = (self.axis_stddevs[i] != 0.0) and self.axis_stddevs[i] or 1.0
            z_a = (features_a[i] - (self.axis_means[i] if self.axis_means and i < len(self.axis_means) else 0.0)) / std_a
            z_b = (features_b[i] - (self.axis_means[i] if self.axis_means and i < len(self.axis_means) else 0.0)) / std_b
            diff = z_a - z_b
            total += w * diff * diff
        return total ** 0.5

    # ---- Band selection ----

    def select_near(self, seed: Track, candidates: list[Track]) -> list[Track]:
        """Select tracks in the Near band: distance ≤ 0.3σ from seed.

        Near-band tracks are close neighbours with minimal perceptual change.
        Good for establishing a mood or groove at the start of a playlist section.
        """
        threshold = self.band_thresholds["near"]
        return self._filter_by_band(seed, candidates, 0.0, threshold)

    def select_mid(self, seed: Track, candidates: list[Track]) -> list[Track]:
        """Select tracks in the Mid band: 0.3σ < distance ≤ 0.7σ from seed.

        Mid-band tracks produce noticeable shifts while remaining related to the
        current track. Good for transitions between playlist sections.
        """
        near_threshold = self.band_thresholds["near"]
        mid_threshold = self.band_thresholds["mid"]
        return self._filter_by_band(seed, candidates, near_threshold, mid_threshold)

    def select_directed_jump(
        self, seed: Track, candidates: list[Track], hold_axis: str
    ) -> list[Track]:
        """Select tracks in the Far-but-directed band.

        Creates a large perceptual shift anchored by one constant quality
        (holdAxis). For example, holding rhythm.tempo constant while jumping
        far on mood, timbre, and key produces a track that surprises the listener
        while maintaining a rhythmic anchor.

        Algorithm:
          1. Compute full distance on all axes
          2. Compute distance on all axes EXCEPT holdAxis
          3. If non-hold distance > farThreshold AND hold distance ≤ holdNearThreshold → qualify

        Returns:
            tracks in the Far-but-directed band
        """
        far_threshold = self.band_thresholds["mid"]
        hold_near_threshold = self.band_thresholds["near"]

        hold_idx = self.axis_index.get(hold_axis)
        if hold_idx is None:
            raise ValueError(f"Unknown hold axis: {hold_axis}")

        seed_features = seed.features
        if seed_features is None:
            return []

        # Pre-compute seed non-hold vector (hold axis zeroed)
        seed_non_hold = list(seed_features)
        if hold_idx < len(seed_non_hold):
            seed_non_hold[hold_idx] = 0.0

        results: list[Track] = []
        for candidate in candidates:
            cand_features = candidate.features
            if cand_features is None:
                continue

            # Full distance on all axes
            full_dist = self.compute_distance(seed, candidate)

            # Distance on all axes EXCEPT holdAxis
            cand_non_hold = list(cand_features)
            if hold_idx < len(cand_non_hold):
                cand_non_hold[hold_idx] = 0.0
            non_hold_track = Track(
                id=candidate.id,
                file_path=candidate.file_path,
                title=candidate.title,
                artist=candidate.artist,
                duration_sec=candidate.duration_sec,
                features=cand_non_hold,
                feature_json=candidate.feature_json,
                clap_embedding=candidate.clap_embedding,
            )
            non_hold_dist = self.compute_distance(seed, non_hold_track)

            # Distance on hold axis only
            hold_val_a = seed_features[hold_idx] if hold_idx < len(seed_features) else 0.0
            hold_val_b = cand_features[hold_idx] if hold_idx < len(cand_features) else 0.0
            hold_dist = self._single_axis_distance(hold_val_a, hold_val_b, hold_idx)

            if non_hold_dist > far_threshold and hold_dist <= hold_near_threshold:
                results.append(candidate)

        return results

    # ---- Internal helpers ----

    def _filter_by_band(
        self, seed: Track, candidates: list[Track], low: float, high: float
    ) -> list[Track]:
        """Filter candidates whose distance from seed falls within (low, high].

        Args:
            seed: the current track to measure from
            candidates: unvisited tracks to filter
            low: lower threshold (exclusive)
            high: upper threshold (inclusive)

        Returns:
            tracks whose distance from seed is in (low, high]
        """
        results: list[Track] = []
        for candidate in candidates:
            if candidate.features is None:
                continue
            dist = self.compute_distance(seed, candidate)
            if dist > low and dist <= high:
                results.append(candidate)
        return results

    @staticmethod
    def _single_axis_distance(val_a: float, val_b: float, axis_idx: int, weight: float = 1.0) -> float:
        """Compute distance on a single axis with an optional weight.

        Args:
            val_a: value for track A
            val_b: value for track B
            axis_idx: axis index (for weight lookup — currently unused as static)
            weight: multiplicative weight for this axis

        Returns:
            absolute weighted distance on this axis (≥ 0)
        """
        diff = val_a - val_b
        return abs(diff) * weight

    def select_near_mid_far(
        self, seed: Track, candidates: list[Track]
    ) -> dict[str, list[Track]]:
        """Select candidates in all three bands and return a dict.

        Returns:
            {"near": [...], "mid": [...], "far": [...]} where each key maps to
            tracks in that band relative to seed. The "far" band uses the
            directed-jump logic (hold axis not applicable here, so all axes
            must exceed the far threshold).
        """
        near = self.select_near(seed, candidates)
        mid = self.select_mid(seed, candidates)

        # "Far" in this helper sense = distance > mid threshold on all axes
        far: list[Track] = []
        for candidate in candidates:
            if candidate.features is None:
                continue
            d = self.compute_distance(seed, candidate)
            if d > self.band_thresholds["mid"]:
                far.append(candidate)

        return {"near": near, "mid": mid, "far": far}