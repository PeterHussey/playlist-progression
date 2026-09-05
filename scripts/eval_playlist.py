#!/usr/bin/env python3
"""Old-vs-new playlist eval: anchor adherence, band separation, rank correlation.

Usage:
    python scripts/eval_playlist.py --db database/playlist.db --seed-title "substring" --limit 9
Compares the legacy Essentia sampler against PlaylistSampler on the same seed.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def spearman(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n != len(y) or n < 2:
        raise ValueError("spearman needs 2+ paired values")
    rx = _ranks(x)
    ry = _ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def band_means(entries: list[dict]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for e in entries:
        acc.setdefault(str(e.get("band", "?")), []).append(float(e.get("distance", 0.0)))
    return {b: sum(v) / len(v) for b, v in acc.items() if v}


def anchor_adherence(entries_meta: list[dict]) -> dict[str, float]:
    n = len(entries_meta)
    if not n:
        return {"mood": 1.0, "key": 1.0, "tempo": 1.0}
    out = {}
    for k in ("mood", "key", "tempo"):
        flag = f"{k}_ok"
        out[k] = sum(1 for e in entries_meta if e.get(flag)) / n
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Compare legacy vs CLAP samplers on one seed.")
    p.add_argument("--db", type=Path, default=Path("database/playlist.db"))
    p.add_argument("--seed-title", type=str, default=None)
    p.add_argument("--limit", type=int, default=9)
    args = p.parse_args(argv)
    print(f"eval: db={args.db} seed={args.seed_title!r} limit={args.limit}")
    print("eval: build the 30-50 track eval library first (local only); full DB comparison lands here.")


if __name__ == "__main__":
    main()