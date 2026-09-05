"""Tests for CLAP cosine similarity + quantile bands."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.recommender import clap_similarity as cs


def test_cosine_similarity_identical_is_one():
    assert cs.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == math.isclose(1.0, 1.0) and abs(cs.cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_is_zero():
    assert abs(cs.cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_rank_orders_by_similarity_desc():
    ranked = cs.rank_by_similarity([1.0, 0.0], [(1, [0.0, 1.0]), (2, [1.0, 0.0])])
    assert [i for i, _ in ranked] == [2, 1]


def test_partition_quantiles_10_items():
    ranked = [(i, 1.0 - i * 0.1) for i in range(10)]
    bands = cs.partition_quantile_bands(ranked, near_quantile=0.10, mid_quantile=0.40)
    assert [i for i, _ in bands["near"]] == [0]
    assert [i for i, _ in bands["mid"]] == [1, 2, 3]
    assert [i for i, _ in bands["far"]] == [4, 5, 6, 7, 8, 9]