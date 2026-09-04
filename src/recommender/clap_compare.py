"""stdlib-only CLAP vs Essentia comparison module.

No external dependencies — uses only math, statistics from the standard library.
"""

from __future__ import annotations

import math


def cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute 1 − cosine_similarity(a, b).

    Returns:
        0.0 for two zero vectors,
        1.0 if exactly one vector is zero,
        otherwise 1 − (dot product / (|a| · |b|)), with cos clamped to [-1, 1].

    Formula: 1 − (Σaᵢbᵢ) / (√Σaᵢ² · √Σbᵢ²)
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")

    n = len(a)
    dot = sum(a[i] * b[i] for i in range(n))
    na_sq = sum(a[i] ** 2 for i in range(n))
    nb_sq = sum(b[i] ** 2 for i in range(n))
    na = math.sqrt(na_sq)
    nb = math.sqrt(nb_sq)

    if na == 0.0 and nb == 0.0:
        return 0.0
    if na == 0.0 or nb == 0.0:
        return 1.0

    cos = dot / (na * nb)
    # Clamp to [-1, 1] to guard against floating-point rounding
    if cos > 1.0:
        cos = 1.0
    elif cos < -1.0:
        cos = -1.0
    return 1.0 - cos


def rank_spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation coefficient (ρ).

    Uses average-tie ranks and Pearson-on-ranks.
    - Raises ValueError if lengths differ or are < 2.
    - Returns 0.0 if either rank vector has zero variance.

    Formula: ρ = cov(rx, ry) / (sx · sy) where rx, ry are average-ranked.
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError(f"Length mismatch: {n} vs {len(ys)}")
    if n < 2:
        raise ValueError(f"Need at least 2 elements, got {n}")

    # ---- Build average-rank vector ----
    def _average_ranks(x: list[float]) -> list[float]:
        """Return 1-indexed average ranks for ties in x."""
        n = len(x)
        sorted_idx = sorted(range(n), key=x.__getitem__)
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and x[sorted_idx[j]] == x[sorted_idx[i]]:
                j += 1
            avg = (i + j - 1) / 2.0 + 1  # 1-indexed average
            for k in range(i, j):
                ranks[sorted_idx[k]] = avg
            i = j
        return ranks

    rx = _average_ranks(xs)
    ry = _average_ranks(ys)

    # ---- Zero-variance guard ----
    var_x = sum((v - sum(rx) / n) ** 2 for v in rx) / n
    var_y = sum((v - sum(ry) / n) ** 2 for v in ry) / n
    if var_x == 0.0 or var_y == 0.0:
        return 0.0

    # ---- Pearson correlation of ranks ----
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    cov = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n)) / n
    sx = math.sqrt(sum((rx[i] - mean_x) ** 2 for i in range(n)) / n)
    sy = math.sqrt(sum((ry[i] - mean_y) ** 2 for i in range(n)) / n)

    if sx == 0.0 or sy == 0.0:
        return 0.0

    return cov / (sx * sy)


def _rms_zdist(u: list[float], v: list[float], means: list[float], stddevs: list[float]) -> float:
    """RMS z-distance between u and v using shared means/stddevs.

    Formula: sqrt( Σ ((uᵢ−μᵢ)/σᵢ − (vᵢ−μᵢ)/σᵢ)² / n )  with σᵢ or 1.0
    """
    total = 0.0
    n = len(u)
    for i in range(n):
        u_z = (u[i] - means[i]) / (stddevs[i] if stddevs[i] != 0 else 1.0)
        v_z = (v[i] - means[i]) / (stddevs[i] if stddevs[i] != 0 else 1.0)
        total += (u_z - v_z) ** 2
    return math.sqrt(total / n)


def pairwise_report(
    essentia_vecs: list[list[float]],
    clap_vecs: list[list[float]],
    means: list[float],
    stddevs: list[float],
) -> dict:
    """Compare Essentia and CLAP distance matrices over all unique pairs.

    Returns:
        {"n_pairs": int, "spearman": float, "essentia_dists": [...], "clap_dists": [...]}

    Pair order: (i, j) with i < j, enumerated sequentially.
    """
    n = len(essentia_vecs)
    essentia_dists: list[float] = []
    clap_dists: list[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            d_ess = _rms_zdist(essentia_vecs[i], essentia_vecs[j], means, stddevs)
            d_clap = _rms_zdist(clap_vecs[i], clap_vecs[j], means, stddevs)
            essentia_dists.append(d_ess)
            clap_dists.append(d_clap)

    n_pairs = len(essentia_dists)
    spearman = rank_spearman(clap_dists, essentia_dists)

    return {
        "n_pairs": n_pairs,
        "spearman": spearman,
        "essentia_dists": essentia_dists,
        "clap_dists": clap_dists,
    }


def _nearest_neighbor_track(idx: int, dist_pairs: list[float], n: int) -> int:
    """Find the track index (other than idx) nearest to idx in the pair-ordered distances.

    dist_pairs are ordered by pair (i, j) with i < j, enumerated 0..n*(n-1)/2-1.
    Returns the nearest other track to idx. If idx is not in any pair (should not happen),
    returns -1.
    """
    best_dist = float("inf")
    best_nbr = -1
    for k, d in enumerate(dist_pairs):
        # reconstruct pair (i, j) from pair order k
        # pairs are (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2, n-1)
        # find i such that sum_{r=0}^{i-1} (n-1-r) <= k < sum_{r=0}^{i} (n-1-r)
        # Within row i, j = i + 1 + (k - prefix_i)
        if n <= 1:
            return -1

        # binary search for i
        lo, hi = 0, n - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            # number of pairs before row mid: mid * (2*n - mid - 1) // 2
            prefix = mid * (2 * n - mid - 1) // 2
            if prefix <= k:
                lo = mid + 1
            else:
                hi = mid - 1
        i = lo - 1
        if i < 0 or i >= n - 1:
            continue
        prefix_i = i * (2 * n - i - 1) // 2
        if k < prefix_i or k >= prefix_i + (n - i - 1):
            continue
        j = i + 1 + (k - prefix_i)
        if i == idx:
            other = j
        elif j == idx:
            other = i
        else:
            continue
        if d < best_dist:
            best_dist = d
            best_nbr = other
    return best_nbr


def nn_agreement(
    essentia_dists: list[float],
    clap_dists: list[float],
    n: int,
) -> dict:
    """Compute nearest-neighbor agreement between two distance lists.

    Rebuilds per-item neighbour lists from the pair-ordered lists
    (pair k ↔ (i, j) enumeration with i < j).
    For each item finds nearest neighbour under each metric;
    returns exact_match_rate and top3_overlap_mean.

    Args:
        essentia_dists: list of pairwise distances (i < j order)
        clap_dists: list of pairwise distances (i < j order), same length
        n: number of tracks (must satisfy n*(n-1)/2 == len(dist_pairs))

    Returns:
        {"exact_match_rate": float, "top3_overlap_mean": float}
    """
    expected_len = n * (n - 1) // 2
    if len(essentia_dists) != expected_len or len(clap_dists) != expected_len:
        raise ValueError(
            f"Distance list length {len(essentia_dists)} vs expected {expected_len}"
        )

    exact_matches = 0
    top3_overlap_sum = 0.0

    for track in range(n):
        # nearest neighbor under essentia
        ess_nbr = _nearest_neighbor_track(track, essentia_dists, n)
        # nearest neighbor under clap
        clap_nbr = _nearest_neighbor_track(track, clap_dists, n)

        if ess_nbr == clap_nbr:
            exact_matches += 1

        # top-3 neighbors under each metric
        ess_top3: set[int] = set()
        clap_top3: set[int] = set()

        # collect all neighbors for this track under essentia
        ess_nbrs: list[tuple[float, int]] = []
        for k, d in enumerate(essentia_dists):
            # reconstruct pair (i, j)
            kk = k
            lo, hi = 0, n - 2
            while lo <= hi:
                mid = (lo + hi) // 2
                prefix = mid * (2 * n - mid - 1) // 2
                if prefix <= kk:
                    lo = mid + 1
                else:
                    hi = mid - 1
            i = lo - 1
            if i < 0 or i >= n - 1:
                continue
            prefix_i = i * (2 * n - i - 1) // 2
            if kk < prefix_i or kk >= prefix_i + (n - i - 1):
                continue
            j = i + 1 + (kk - prefix_i)
            if i == track:
                other = j
            elif j == track:
                other = i
            else:
                continue
            ess_nbrs.append((d, other))
        ess_nbrs.sort(key=lambda x: x[0])
        for _, other in ess_nbrs[:3]:
            ess_top3.add(other)

        # collect all neighbors for this track under clap
        clap_nbrs: list[tuple[float, int]] = []
        for k, d in enumerate(clap_dists):
            kk = k
            lo, hi = 0, n - 2
            while lo <= hi:
                mid = (lo + hi) // 2
                prefix = mid * (2 * n - mid - 1) // 2
                if prefix <= kk:
                    lo = mid + 1
                else:
                    hi = mid - 1
            i = lo - 1
            if i < 0 or i >= n - 1:
                continue
            prefix_i = i * (2 * n - i - 1) // 2
            if kk < prefix_i or kk >= prefix_i + (n - i - 1):
                continue
            j = i + 1 + (kk - prefix_i)
            if i == track:
                other = j
            elif j == track:
                other = i
            else:
                continue
            clap_nbrs.append((d, other))
        clap_nbrs.sort(key=lambda x: x[0])
        for _, other in clap_nbrs[:3]:
            clap_top3.add(other)

        overlap = len(ess_top3 & clap_top3)
        top3_overlap_sum += overlap / 3.0

    exact_match_rate = exact_matches / n
    top3_overlap_mean = top3_overlap_sum / n

    return {
        "exact_match_rate": exact_match_rate,
        "top3_overlap_mean": top3_overlap_mean,
    }


def clap_walk(
    seed_idx: int,
    clap_vecs: list[list[float]],
    limit: int = 9,
    schedule: list[str] | None = None,
) -> list[dict]:
    """Greedy walk across CLAP pairwise-distance bands.

    Mirrors generate_playlist.py:119-161 behaviour.

    Args:
        seed_idx: starting track index
        clap_vecs: list of 512-dim (or other) CLAP embedding vectors
        limit: number of tracks to select (default 9)
        schedule: band order per step; defaults to ["Near", "Mid", "Far", "Mid", "Near"]

    Returns:
        list of dicts, each with keys:
            position (int), track_idx (int), band (str), distance (float), reason (str|None)
        distance is rounded to 4 decimal places.
    """
    if schedule is None:
        schedule = ["Near", "Mid", "Far", "Mid", "Near"]

    n = len(clap_vecs)
    if n <= 1:
        return []

    # ---- Precompute all-pair cosine distances ----
    cos_dist: list[list[float]] = [[0.0] * n for _ in range(n)]
    all_pairwise: list[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance(clap_vecs[i], clap_vecs[j])
            cos_dist[i][j] = d
            cos_dist[j][i] = d
            all_pairwise.append(d)

    # ---- Compute quantile cutoffs ----
    sorted_all = sorted(all_pairwise)
    n_all = len(sorted_all)
    q33_idx = int(0.33 * n_all)
    q66_idx = int(0.66 * n_all)
    # Clamp indices to valid range
    q33_idx = max(0, min(q33_idx, n_all - 1))
    q66_idx = max(0, min(q66_idx, n_all - 1))
    q33 = sorted_all[q33_idx]
    q66 = sorted_all[q66_idx]

    # ---- Greedy walk ----
    visited = [False] * n
    visited[seed_idx] = True

    current = seed_idx
    result: list[dict] = []

    for step in range(limit):
        band_name = schedule[step % len(schedule)].capitalize()  # "Near", "Mid", "Far"

        # gather unvisited tracks with distances from current
        unvisited = [i for i in range(n) if not visited[i]]
        if not unvisited:
            break

        # compute distances from current to each unvisited
        dists: list[tuple[float, int]] = []
        for i in unvisited:
            d = cosine_distance(clap_vecs[current], clap_vecs[i])
            dists.append((d, i))
        dists.sort(key=lambda x: x[0])  # nearest first

        band = band_name
        selected: int | None = None
        reason: str | None = None
        actual_dist: float | None = None

        if band == "Near":
            # Near: d <= q33
            candidates = [(d, i) for d, i in dists if d <= q33 + 1e-12]
            if candidates:
                candidates.sort(key=lambda x: x[0])
                selected = candidates[0][1]
                actual_dist = candidates[0][0]
            else:
                # Fallback to global nearest
                selected = dists[0][1]
                actual_dist = dists[0][0]
                reason = f"Fallback: scheduled Near empty, global-nearest (actual {actual_dist:.4f})"

        elif band == "Mid":
            # Mid: q33 < d <= q66
            candidates = [(d, i) for d, i in dists if q33 < d <= q66]
            if candidates:
                candidates.sort(key=lambda x: x[0])
                selected = candidates[0][1]
                actual_dist = candidates[0][0]
            else:
                selected = dists[0][1]
                actual_dist = dists[0][0]
                reason = f"Fallback: scheduled Mid empty, global-nearest (actual {actual_dist:.4f})"

        elif band == "Far":
            # Far: farthest unvisited (no hold axis)
            # Pick the unvisited track with maximum distance from current
            # "Far above" means d > q66, but if none qualify, fallback to nearest
            far_candidates = [(d, i) for d, i in dists if d > q66]
            if far_candidates:
                far_candidates.sort(key=lambda x: -x[0])  # farthest first
                selected = far_candidates[0][1]
                actual_dist = far_candidates[0][0]
            else:
                # Fallback to global nearest
                selected = dists[0][1]
                actual_dist = dists[0][0]
                reason = f"Fallback: scheduled Far empty, global-nearest (actual {actual_dist:.4f})"

        if selected is None:
            break  # should not happen if unvisited is non-empty

        visited[selected] = True
        distance_rounded = round(actual_dist, 4)

        result.append(
            {
                "position": step + 1,
                "track_idx": selected,
                "band": band,
                "distance": distance_rounded,
                "reason": reason,
            }
        )

        current = selected

    return result