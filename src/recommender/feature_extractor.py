"""Feature extractor — runs Python extraction scripts as subprocess wrappers.

Calls the existing scripts extract_essentia.py and extract_clap.py via
subprocess.run() with a configurable timeout and exit-code checking.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


# Legacy constant kept for backward compatibility
TIMEOUT_SECONDS = 30

# Configurable timeout defaults (seconds)
DEFAULT_TIMEOUT_SEC = 180
DSP_TIMEOUT_SEC = 60
MOOD_TIMEOUT_SEC = 180


def resolve_timeout(explicit: int | None, env_name: str, default: int) -> int:
    """Resolve the effective timeout: explicit arg > env var > default.

    Args:
        explicit: caller-supplied timeout (highest priority)
        env_name: environment variable name to check
        default: fallback value

    Returns:
        Effective timeout in seconds
    """
    if explicit is not None:
        return int(explicit)
    raw = os.environ.get(env_name)
    if raw is not None and str(raw).strip() != "":
        try:
            return int(str(raw).strip())
        except ValueError:
            pass
    return default


def run_script(script: Path, audio_path: Path, output_path: Path, timeout: int | None = None) -> None:
    """Run a Python extraction script via subprocess.run().

    Args:
        script: filename of the script (e.g. "extract_essentia.py")
        audio_path: absolute path to the input audio file
        output_path: absolute path for the JSON sidecar output
        timeout: per-file timeout in seconds (default 180, env EXTRACT_TIMEOUT_SEC)

    Raises:
        RuntimeError: if the process times out or returns non-zero exit code
    """
    eff = resolve_timeout(timeout, "EXTRACT_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC)
    cmd = [sys.executable, str(script), str(audio_path), str(output_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=eff,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Script {script} timed out after {eff}s"
        ) from e

    if result.returncode != 0:
        raise RuntimeError(
            f"Script {script} exited with code {result.returncode}"
            + (f"\nstderr: {result.stderr}" if result.stderr else "")
        )

    # Script succeeded — it wrote the JSON sidecar already
    if not output_path.exists():
        raise RuntimeError(f"Script {script} did not produce output at {output_path}")


def extract_essentia(audio_path: Path, output_path: Path, timeout: int | None = None) -> None:
    """Run extract_essentia.py on an audio file.

    Args:
        audio_path: path to the input audio file
        output_path: path where the JSON sidecar will be written
        timeout: per-file timeout in seconds (default 180, env EXTRACT_TIMEOUT_SEC)
    """
    script = Path(__file__).parent.parent.parent / "scripts" / "extract_essentia.py"
    run_script(script, audio_path, output_path, timeout=timeout)


def extract_clap(audio_path: Path, output_path: Path) -> None:
    """Run extract_clap.py on an audio file.

    Args:
        audio_path: path to the input audio file
        output_path: path where the JSON sidecar will be written
    """
    script = Path(__file__).parent.parent.parent / "scripts" / "extract_clap.py"
    run_script(script, audio_path, output_path)


def ensure_mood_models(models_dir: Path | None = None, timeout: int | None = None) -> None:
    """Prefetch mood classification models via extract_essentia.py --prefetch.

    Args:
        models_dir: directory to store downloaded models (default "models")
        timeout: timeout in seconds (default 180, env EXTRACT_TIMEOUT_SEC)

    Raises:
        RuntimeError: if the prefetch subprocess fails or times out
    """
    script = Path(__file__).parent.parent.parent / "scripts" / "extract_essentia.py"
    cmd = [sys.executable, str(script), "--prefetch"]
    if models_dir is not None:
        cmd += ["--models-dir", str(models_dir)]
    eff = resolve_timeout(timeout, "EXTRACT_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=eff)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Prefetch timed out after {eff}s") from e
    if result.returncode != 0:
        raise RuntimeError(f"Prefetch failed: {result.stderr or result.returncode}")