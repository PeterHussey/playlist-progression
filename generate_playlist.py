#!/usr/bin/env python3
"""Generate branch playlist from database and write summary."""
import sqlite3, json
from pathlib import Path
import sys, math
sys.path.insert(0, '.')
from src.recommender.track import Track
from src.recommender.branch_sampler import BranchSampler
from src.recommender.playlist_writer import write_playlist, make_entry
from src.recommender.feature_converter import AXIS_NAMES, convert
DB = Path("database/playlist.db")

def main():
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM tracks ORDER BY id").fetchall()
    tracks = []
    feature_vectors = []
    for row in rows:
        t = Track(id=row["id"], file_path=Path(row["file_path"]),
                  title=row["title"], artist=row["artist"],
                  duration_sec=row["duration_sec"] or 0.0,
                  feature_json=row["feature_json"])
        vec = convert(row["feature_json"])
        raw = json.loads(row["feature_json"] or "{}")
        t.set_features(vec)
        tracks.append((t, raw))
        feature_vectors.append(vec)
    conn.close()
    n_axes = len(AXIS_NAMES)
    means = []
    stddevs = []
    for i in range(n_axes):
        values = [fv[i] for fv in feature_vectors]
        mean = sum(values) / len(values)
        means.append(mean)
        variance = sum((v - mean)**2 for v in values) / len(values)
        stddevs.append(math.sqrt(variance) if variance > 0 else 1.0)
        print(f"Axis {AXIS_NAMES[i]}: mean={mean:.2f}, stddev={stddevs[-1]:.2f}")
    axis_index = {name: i for i, name in enumerate(AXIS_NAMES)}
    sampler = BranchSampler(axis_weights=[1.0]*n_axes, axis_index=axis_index,
                            axis_stddevs=stddevs, axis_means=means)
    seed = next((t for t, _ in tracks if t.id == 17), None)
    if seed is None:
        seed = max(tracks, key=lambda item: item[0].id)[0]
    visited = {seed.id}
    current = seed
    playlist_entries = []
    schedule = ["Near", "Mid", "Far", "Mid", "Near"]
    for step in range(9):
        band = schedule[step % len(schedule)]
        candidates = [t for t, _ in tracks if t.id not in visited]
        selected = None
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
            best = None; best_dist = float("inf")
            for cand in candidates:
                d = sampler.compute_distance(current, cand)
                if d < best_dist: best_dist = d; best = cand
            selected = best
        if selected is None: break
        d = sampler.compute_distance(current, selected)
        playlist_entries.append(make_entry(
            position=step+1, track_id=selected.id,
            title=selected.get_title(), artist=selected.get_artist(),
            band=band, distance=round(d, 4),
            reason=f"{band} band transition from seed"))
        visited.add(selected.id)
        current = selected
        print(f"  {band}: id={selected.id} '{selected.get_title() or 'Unknown'}' distance={d:.3f}")
    write_playlist(Path("branch_playlist.json"), seed, playlist_entries)
    selected_ids = [seed.id] + [e["track_id"] for e in playlist_entries]
    selected_data = [item for item in tracks if item[0].id in selected_ids]
    selected_data.sort(key=lambda x: selected_ids.index(x[0].id))
    lines = []
    lines.append("=" * 60)
    lines.append("PLAYLIST SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Seed: id={seed.id} '{seed.get_title() or 'Unknown'}' by {seed.get_artist() or 'Unknown'}")
    lines.append("Schedule: Near -> Mid -> Far -> Mid -> Near (10 tracks)")
    lines.append("Hold axis: tempo.bpm (BPM)")
    lines.append("")
    for idx, (t, raw) in enumerate(selected_data):
        band_label = playlist_entries[idx-1]["band"] if idx > 0 else "SEED"
        lines.append(f"{idx}. id={t.id} | {band_label}")
        lines.append(f"   '{t.get_title() or 'Unknown'}' - {t.get_artist() or 'Unknown'}")
        loud = raw.get("loudness", {})
        tempo = raw.get("tempo", {})
        spec = raw.get("spectral", {})
        rhythm = raw.get("rhythm", {})
        mood = raw.get("mood", {})
        lines.append(f"   Duration: {raw.get('duration_sec', 0):.1f}s | Loudness: {loud.get('integrated', 0):.1f} LUFS (range {loud.get('range', 0):.1f})")
        lines.append(f"   Tempo: {tempo.get('bpm', 0):.0f} BPM (conf {tempo.get('confidence', 0):.2f}) | Key: {raw.get('key', {}).get('scale', 'N/A')}")
        lines.append(f"   Spectral: centroid={spec.get('centroid', 0):.0f} Hz, rolloff={spec.get('rolloff', 0):.0f} Hz, flatness={spec.get('flatness', 0):.3f}")
        lines.append(f"   Rhythm: danceability={rhythm.get('danceability', 0):.2f}, onset_rate={rhythm.get('onset_rate', 0):.1f}/sec")
        lines.append(f"   Mood: happy={mood.get('happy', 0):.2f}, sad={mood.get('sad', 0):.2f}, aggressive={mood.get('aggressive', 0):.2f}, "
                     f"relaxed={mood.get('relaxed', 0):.2f}, electronic={mood.get('electronic', 0):.2f}, party={mood.get('party', 0):.2f}, acoustic={mood.get('acoustic', 0):.2f}")
        lines.append("")
    Path("playlist_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nPlaylist JSON: branch_playlist.json")
    print(f"Text summary: playlist_summary.txt")
if __name__ == "__main__":
    main()
