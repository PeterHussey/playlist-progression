"""Tests for configurable extraction timeout and related interfaces."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_resolve_timeout_precedence(monkeypatch):
    """resolve_timeout: explicit > env > default."""
    from src.recommender.feature_extractor import resolve_timeout

    monkeypatch.delenv("EXTRACT_TIMEOUT_SEC", raising=False)
    assert resolve_timeout(None, "EXTRACT_TIMEOUT_SEC", 180) == 180

    monkeypatch.setenv("EXTRACT_TIMEOUT_SEC", "240")
    assert resolve_timeout(None, "EXTRACT_TIMEOUT_SEC", 180) == 240

    assert resolve_timeout(60, "EXTRACT_TIMEOUT_SEC", 180) == 60


def test_run_script_uses_default_180(monkeypatch):
    """DEFAULT_TIMEOUT_SEC == 180 and TIMEOUT_SECONDS kept for backward compat."""
    import src.recommender.feature_extractor as fe

    assert fe.DEFAULT_TIMEOUT_SEC == 180
    # TIMEOUT_SECONDS kept as alias; may or may not equal 180
    assert hasattr(fe, "TIMEOUT_SECONDS")


def test_resolve_timeout_bad_env_falls_back(monkeypatch):
    """Non-numeric env value falls back to default."""
    from src.recommender.feature_extractor import resolve_timeout

    monkeypatch.setenv("EXTRACT_TIMEOUT_SEC", "not-a-number")
    assert resolve_timeout(None, "EXTRACT_TIMEOUT_SEC", 180) == 180


def test_resolve_timeout_empty_env_falls_back(monkeypatch):
    """Empty env string falls back to default."""
    from src.recommender.feature_extractor import resolve_timeout

    monkeypatch.setenv("EXTRACT_TIMEOUT_SEC", "")
    assert resolve_timeout(None, "EXTRACT_TIMEOUT_SEC", 180) == 180


def test_run_script_signature_accepts_timeout():
    """run_script signature includes timeout kwarg."""
    import inspect
    from src.recommender.feature_extractor import run_script

    sig = inspect.signature(run_script)
    assert "timeout" in sig.parameters, "run_script must accept a 'timeout' parameter"


def test_extract_essentia_signature_accepts_timeout():
    """extract_essentia signature includes timeout kwarg."""
    import inspect
    from src.recommender.feature_extractor import extract_essentia

    sig = inspect.signature(extract_essentia)
    assert "timeout" in sig.parameters, "extract_essentia must accept a 'timeout' parameter"


def test_ensure_mood_models_exists():
    """ensure_mood_models function exists."""
    from src.recommender.feature_extractor import ensure_mood_models

    assert callable(ensure_mood_models)
