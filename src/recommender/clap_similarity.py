"""CLAP cosine similarity + quantile bands (pure functions, no DB)."""
from __future__ import annotations
import math


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return [0.0 for _ in vec]
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        raise ValueError("cosine_similarity requires non-empty vectors")
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    na, nb = l2_normalize(a), l2_normalize(b)
    return sum(x * y for x, y in zip(na, nb))


def rank_by_similarity(seed: list[float], candidates: list[tuple[int, list[float]]]) -> list[tuple[int, float]]:
    scored = [(cid, cosine_similarity(seed, vec)) for cid, vec in candidates]
    scored.sort(key=lambda p: p[1], reverse=True)
    return scored


def partition_quantile_bands(ranked: list[tuple[int, float]], near_quantile: float = 0.10, mid_quantile: float = 0.40) -> dict[str, list[tuple[int, float]]]:
    n = len(ranked)
    if n == 0:
        return {"near": [], "mid": [], "far": []}
    n_near = math.ceil(n * near_quantile)
    n_mid = math.ceil(n * mid_quantile) - n_near
    near = ranked[:n_near]
    mid = ranked[n_near:n_near + n_mid]
    far = ranked[n_near + n_mid:]
    return {"near": near, "mid": mid, "far": far}
