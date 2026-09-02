#!/usr/bin/env python3
"""Tests for mood extraction functionality."""
import pytest
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.network
def test_mood_models_download():
    """Test that mood models can be downloaded."""
    from scripts.extract_essentia import download_mood_models

    models_dir = Path("test_models")
    try:
        moods = ["happy"]  # Test with just one mood
        download_mood_models(moods, models_dir)

        # Check that model file was created
        model_path = models_dir / "mood_happy-musicnn-msd-1.pb"
        assert model_path.exists()
    finally:
        # Cleanup
        if models_dir.exists():
            import shutil
            shutil.rmtree(models_dir)


def test_mood_extraction_with_mock():
    """Test mood extraction with mocked TensorFlow."""
    from unittest.mock import patch, MagicMock
    import numpy as np

    # Mock TensorFlow components
    with patch('essentia.standard.TensorflowPredictMusiCNN') as mock_predict:
        mock_predict.return_value = MagicMock(return_value=np.array([0.8]))

        from scripts.extract_essentia import extract_mood

        # Create dummy audio
        audio = np.random.randn(16000)  # 1 second at 16kHz

        # Test extraction
        scores = extract_mood(audio, models_dir=Path("test_models"))

        assert isinstance(scores, dict)
        assert "happy" in scores