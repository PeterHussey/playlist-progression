#!/usr/bin/env python3
"""Old-vs-new playlist eval: anchor adherence, band separation, rank correlation.

Usage:
    python scripts/eval_playlist.py --db database/playlist.db --seed-title "substring" --limit 9
Compares the legacy Essentia sampler against PlaylistSampler on the same seed.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender.track import Track
from src.recommender.feature_converter import AXIS_NAMES, convert
from src.recommender.branch_sampler import BranchSampler
from src.recommender.constraint_filter import (
    _parsed_sidecar, key_slot, key_steps, key_verdict, tempo_ok,
)
from src.recommender.clap_similarity import cosine_similarity


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


def _anchor_meta(seed: Track, entries: list[dict], tracks_by_id: dict[int, Track]) -> list[dict]:
    """Compute per-consecutive-pair mood_ok / key_ok / tempo_ok metadata."""
    meta: list[dict] = []
    prev = seed
    for e in entries:
        nxt = tracks_by_id.get(e["track_id"])
        if nxt is None:
            meta.append({"mood_ok": True, "key_ok": True, "tempo_ok": True})
            prev = nxt or prev
            continue
        prev_raw = _parsed_sidecar(prev.feature_json)
        nxt_raw = _parsed_sidecar(nxt.feature_json)
        # key check (default max_steps=2)
        ps = key_slot((prev_raw.get("key") or {}).get("key"),
                       (prev_raw.get("key") or {}).get("mode") or (prev_raw.get("key") or {}).get("scale"))
        ns = key_slot((nxt_raw.get("key") or {}).get("key"),
                       (nxt_raw.get("key") or {}).get("mode") or (nxt_raw.get("key") or {}).get("scale"))
        steps = key_steps(ps, ns)
        k_ok = key_verdict(steps, 2) != "block"
        # tempo check (default 10%)
        pb = float((prev_raw.get("tempo") or {}).get("bpm", 0.0) or 0.0)
        nb = float((nxt_raw.get("tempo") or {}).get("bpm", 0.0) or 0.0)
        t_ok = True
        if pb > 0 and nb > 0:
            t_ok = tempo_ok(pb, nb, 0.10)
        # mood check (empty box → always ok)
        m_ok = True
        meta.append({"mood_ok": m_ok, "key_ok": k_ok, "tempo_ok": t_ok})
        prev = nxt
    return meta


def _run_legacy(seed: Track, tracks: list[tuple[Track, dict]], limit: int) -> list[dict]:
    """Run legacy BranchSampler path and return entry dicts."""
    feature_vectors = [convert(t.feature_json) for t, _ in tracks]
    n_axes = len(AXIS_NAMES)
    means, stddevs = [], []
    for i in range(n_axes):
        values = [fv[i] for fv in feature_vectors]
        mean = sum(values) / len(values)
        means.append(mean)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stddevs.append(math.sqrt(variance) if variance > 0 else 1.0)
    axis_index = {name: i for i, name in enumerate(AXIS_NAMES)}
    sampler = BranchSampler(axis_weights=[1.0] * n_axes, axis_index=axis_index,
                            axis_stddevs=stddevs, axis_means=means)
    schedule = ["Near", "Mid", "Far", "Mid", "Near"]
    visited = {seed.id}
    current = seed
    entries: list[dict] = []
    for step in range(limit):
        band = schedule[step % len(schedule)]
        candidates = [t for t, _ in tracks if t.id not in visited]
        selected = None
        fell_back = False
        if band == "Near":
            near_list = sampler.select_near(current, candidates)
            selected = near_list[0] if near_list else None
        elif band == "Mid":
            mid_list = sampler.select_mid(current, candidates)
            selected = mid_list[0] if mid_list else None
        elif band == "Far":
            far_list = sampler.select_directed_jump(current, candidates, "tempo.bpm")
            selected = far_list[0] if far_list else None
        if selected is None:
            fell_back = True
            best = None
            best_dist = float("inf")
            for cand in candidates:
                d = sampler.compute_distance(current, cand)
                if d < best_dist:
                    best_dist = d
                    best = cand
            selected = best
        if selected is None:
            break
        d = sampler.compute_distance(current, selected)
        entries.append({"position": step + 1, "track_id": selected.id,
                        "title": selected.get_title(), "artist": selected.get_artist(),
                        "band": band, "distance": round(d, 4),
                        "reason": f"{'Fallback: ' if fell_back else ''}{band}"})
        visited.add(selected.id)
        current = selected
    return entries


def _run_clap(seed: Track, tracks: list[tuple[Track, dict]], limit: int) -> list[dict]:
    """Run PlaylistSampler (clap) path and return entry dicts."""
    from src.recommender.playlist_sampler import PlaylistSampler
    config = {
        "near_quantile": 0.10,
        "mid_quantile": 0.40,
        "mood_box": {},
        "key_max_steps": 2,
        "tempo_max_step_pct": 0.10,
        "tempo_drift_per_step": 0.0,
        "band_schedule": ["Near", "Mid", "Far", "Mid", "Near"],
        "allow_missing_clap": False,
        "allow_missing_mood": False,
    }
    sampler = PlaylistSampler(config)
    sampler.load_library([t for t, _ in tracks])
    return sampler.generate(seed, limit)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Compare legacy vs CLAP samplers on one seed.")
    p.add_argument("--db", type=Path, default=Path("database/playlist.db"))
    p.add_argument("--seed-title", type=str, default=None)
    p.add_argument("--limit", type=int, default=9)
    args = p.parse_args(argv)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM tracks ORDER BY id").fetchall()
    tracks: list[tuple[Track, dict]] = []
    for row in rows:
        clap_emb = None if row["clap_embedding"] is None else json.loads(row["clap_embedding"])
        t = Track(id=row["id"], file_path=Path(row["file_path"]),
                  title=row["title"], artist=row["artist"],
                  duration_sec=row["duration_sec"] or 0.0,
                  feature_json=row["feature_json"],
                  clap_embedding=clap_emb)
        vec = convert(row["feature_json"])
        raw = json.loads(row["feature_json"] or "{}")
        t.set_features(vec)
        tracks.append((t, raw))
    conn.close()

    if not tracks:
        print(f"Error: no tracks in {args.db}", file=sys.stderr)
        sys.exit(1)

    # Resolve seed
    seed = None
    if args.seed_title is not None:
        matching = [t for t, _ in tracks if args.seed_title.lower() in (t.title or "").lower()]
        if not matching:
            print(f"No tracks found with '{args.seed_title}' in title", file=sys.stderr)
            sys.exit(1)
        if len(matching) > 1:
            print(f"Multiple matches for '{args.seed_title}': {', '.join(str(t.id) for t in matching)}",
                  file=sys.stderr)
            sys.exit(1)
        seed = matching[0]
    else:
        seed = max(tracks, key=lambda item: item[0].id)[0]

    tracks_by_id = {t.id: t for t, _ in tracks}

    # ---- Legacy path ----
    legacy_entries = _run_legacy(seed, tracks, args.limit)
    legacy_meta = _anchor_meta(seed, legacy_entries, tracks_by_id)
    legacy_adh = anchor_adherence(legacy_meta)
    legacy_bm = band_means(legacy_entries)
    print(f"\n=== Legacy (Essentia) path ===")
    print(f"Playlist length: {len(legacy_entries)}")
    print(f"Anchor adherence: {legacy_adh}")
    print(f"Band means: {legacy_bm}")
    key_violations = sum(1 for m in legacy_meta if not m["key_ok"])
    if key_violations:
        print(f"Key violation steps: {key_violations}")

    # ---- CLAP path ----
    clap_entries = _run_clap(seed, tracks, args.limit)
    clap_meta = _anchor_meta(seed, clap_entries, tracks_by_id)
    clap_adh = anchor_adherence(clap_meta)
    clap_bm = band_means(clap_entries)
    print(f"\n=== CLAP path ===")
    print(f"Playlist length: {len(clap_entries)}")
    print(f"Anchor adherence: {clap_adh}")
    print(f"Band means: {clap_bm}")
    key_violations = sum(1 for m in clap_meta if not m["key_ok"])
    if key_violations:
        print(f"Key violation steps: {key_violations}")

    # ---- Spearman between Essentia distance and CLAP cosine distance ----
    # Build paired distances for consecutive entries that exist in both playlists
    all_ids = [e["track_id"] for e in legacy_entries]
    clap_dist_map = {e["track_id"]: e["distance"] for e in clap_entries}
    essentia_dists = []
    clap_dists = []
    prev_id = seed.id
    for e in legacy_entries:
        tid = e["track_id"]
        if tid in clap_dist_map:
            # Essentia distance: recompute from features
            prev_t = tracks_by_id.get(prev_id)
            cur_t = tracks_by_id.get(tid)
            if prev_t is not None and cur_t is not None and prev_t.features and cur_t.features:
                n = len(prev_t.features)
                ez = sum((a - b) ** 2 for a, b in zip(prev_t.features, cur_t.features))
                ed = (ez / n) ** 0.5
            else:
                ed = e["distance"]
            essentia_dists.append(ed)
            clap_dists.append(clap_dist_map[tid])
        prev_id = tid

    print(f"\n=== Cross-metric correlation ===")
    if len(essentia_dists) >= 2:
        rho = spearman(essentia_dists, clap_dists)
        print(f"Spearman (Essentia dist vs CLAP dist): {rho:.4f}  (n={len(essentia_dists)})")
    else:
        print("Spearman: n/a (need 2+ paired entries)")


if __name__ == "__main__":
    main()
