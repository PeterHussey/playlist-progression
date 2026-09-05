"""Tests for eval metric helpers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts import eval_playlist as ev


def test_spearman_perfect_and_inverse():
    assert abs(ev.spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-9
    assert abs(ev.spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) + 1.0) < 1e-9


def test_band_means_separates():
    entries = [{"band": "Near", "distance": 0.1}, {"band": "Near", "distance": 0.3},
               {"band": "Far", "distance": 0.8}]
    means = ev.band_means(entries)
    assert means["Near"] < means["Far"]


def test_anchor_adherence_counts_violations():
    meta = [{"mood_ok": True, "key_ok": True, "tempo_ok": True},
            {"mood_ok": True, "key_ok": False, "tempo_ok": True}]
    out = ev.anchor_adherence(meta)
    assert out == {"mood": 1.0, "key": 0.5, "tempo": 1.0}