#!/usr/bin/env python3
"""TDD test: spectral descriptors must be frame-wise, not full-track FFT.

Regression guard for backlog item 6: es.Spectrum()(audio) on the full track
is the same class of bug as the key flaw (commit ba7c8d3). Correct path is
FrameGenerator -> Windowing -> Spectrum -> Centroid/RollOff/Flatness + mean.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_compute_spectral_uses_framewise_pipeline():
    """compute_spectral_descriptors must aggregate frame-wise means."""
    from unittest.mock import patch, MagicMock
    import numpy as np

    import scripts.extract_essentia as ex

    assert hasattr(ex, "compute_spectral_descriptors"), (
        "missing compute_spectral_descriptors() — frame-wise helper not implemented"
    )

    audio = np.random.randn(44100).astype("float32")

    # Fake frame pipeline: 3 frames, known descriptor values
    fake_frames = [np.ones(2048, dtype="float32") * i for i in range(3)]

    with patch("essentia.standard.FrameGenerator", create=True) as mock_fg, \
         patch("essentia.standard.Windowing", create=True) as mock_win, \
         patch("essentia.standard.Spectrum", create=True) as mock_spec, \
         patch("essentia.standard.Centroid", create=True) as mock_cent, \
         patch("essentia.standard.RollOff", create=True) as mock_roll, \
         patch("essentia.standard.Flatness", create=True) as mock_flat:

        mock_fg.return_value = fake_frames
        # Window/Spectrum pass through
        mock_win.return_value = MagicMock(side_effect=lambda f: f)
        mock_spec.return_value = MagicMock(side_effect=lambda f: f)
        # Descriptor values per frame
        mock_cent.return_value = MagicMock(side_effect=[1000.0, 2000.0, 3000.0])
        mock_roll.return_value = MagicMock(side_effect=[2000.0, 4000.0, 6000.0])
        mock_flat.return_value = MagicMock(side_effect=[0.1, 0.2, 0.3])

        centroid, rolloff, flatness = ex.compute_spectral_descriptors(audio)

        # FrameGenerator must be used (not a single Spectrum(audio) call)
        mock_fg.assert_called_once()
        assert mock_cent.return_value.call_count == 3
        assert mock_roll.return_value.call_count == 3
        assert mock_flat.return_value.call_count == 3

        # Means across frames
        assert centroid == 2000.0
        assert rolloff == 4000.0
        assert abs(flatness - 0.2) < 1e-6


def test_no_fulltrack_spectrum_call():
    """Source must not call Spectrum() on the full-track audio buffer."""
    src = Path(__file__).parent.parent / "scripts" / "extract_essentia.py"
    text = src.read_text()
    # Buggy patterns: spec = spectrum(audio) / Spectrum()(audio)
    assert "spectrum(audio)" not in text, (
        "found full-track spectrum(audio) — must use FrameGenerator loop"
    )
    assert "Spectrum()(audio)" not in text, (
        "found full-track Spectrum()(audio) — must use FrameGenerator loop"
    )
