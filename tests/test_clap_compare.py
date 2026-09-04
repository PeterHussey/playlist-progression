"""Tests for stdlib-only CLAP vs Essentia comparison module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cosine_distance_known_values():
    from src.recommender.clap_compare import cosine_distance
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert abs(cosine_distance([1.0, 0.0], [0.0, 1.0]) - 1.0) < 1e-9
    assert abs(cosine_distance([1.0, 1.0], [2.0, 2.0]) - 0.0) < 1e-9
    assert cosine_distance([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 1.0


def test_spearman_perfect_and_inverse():
    from src.recommender.clap_compare import rank_spearman
    assert abs(rank_spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) - 1.0) < 1e-9
    assert abs(rank_spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) + 1.0) < 1e-9
    assert rank_spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


def test_pairwise_report_counts_and_agreement():
    from src.recommender.clap_compare import pairwise_report, nn_agreement
    ess = [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]
    clap = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    rep = pairwise_report(ess, clap, means=[0.0, 0.0], stddevs=[1.0, 1.0])
    assert rep["n_pairs"] == 3
    assert -1.0 <= rep["spearman"] <= 1.0
    assert len(rep["essentia_dists"]) == 3 and len(rep["clap_dists"]) == 3
    agr = nn_agreement(rep["essentia_dists"], rep["clap_dists"], n=3)
    assert 0.0 <= agr["exact_match_rate"] <= 1.0
    assert 0.0 <= agr["top3_overlap_mean"] <= 1.0


def test_clap_walk_deterministic_fallback_labels():
    from src.recommender.clap_compare import clap_walk
    vecs = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [-1.0, 0.0]]
    first = clap_walk(seed_idx=0, clap_vecs=vecs, limit=3)
    second = clap_walk(seed_idx=0, clap_vecs=vecs, limit=3)
    assert first == second
    assert [e["position"] for e in first] == [1, 2, 3]
    assert len({e["track_idx"] for e in first}) == 3
    assert all(e["track_idx"] != 0 for e in first)
    # C1 fix: every "Fallback:" reason must end with a band label (Near/Mid/Far),
    # never a bare distance number
    for entry in first:
        if entry["reason"] and entry["reason"].startswith("Fallback:"):
            import re
            assert re.search(r'\(actual (Near|Mid|Far)\)$', entry["reason"]), \
                f"Fallback reason does not end with band label: {entry['reason']}"
    for entry in second:
        if entry["reason"] and entry["reason"].startswith("Fallback:"):
            import re
            assert re.search(r'\(actual (Near|Mid|Far)\)$', entry["reason"]), \
                f"Fallback reason does not end with band label: {entry['reason']}"
