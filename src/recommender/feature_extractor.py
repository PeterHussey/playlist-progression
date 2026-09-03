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


def extract_essentia(audio_path: Path, output_path: Path, timeout: int | None = None, no_mood: bool = False, mood_only: bool = False) -> None:
    """Run extract_essentia.py on an audio file.

    Args:
        audio_path: path to the input audio file
        output_path: path where the JSON sidecar will be written
        timeout: per-file timeout in seconds (default 180, env EXTRACT_TIMEOUT_SEC)
        no_mood: if True, skip mood extraction (DSP only)
        mood_only: if True, run only mood extraction on existing sidecar
    """
    script = Path(__file__).parent.parent.parent / "scripts" / "extract_essentia.py"
    cmd = [sys.executable, str(script), str(audio_path), str(output_path)]
    if no_mood:
        cmd.append("--no-mood")
    if mood_only:
        cmd.append("--mood-only")
    eff = resolve_timeout(timeout, "EXTRACT_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=eff)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Script {script} timed out after {eff}s") from e
    if result.returncode != 0:
        raise RuntimeError(
            f"Script {script} exited with code {result.returncode}"
            + (f"\nstderr: {result.stderr}" if result.stderr else "")
        )
    if not output_path.exists():
        raise RuntimeError(f"Script {script} did not produce output at {output_path}")


def extract_clap(audio_path: Path, output_path: Path) -> None:
    """Run extract_clap.py on an audio file.

    Args:
        audio_path: path to the input audio file
        output_path: path where the JSON sidecar will be written
    """
    script = Path(__file__).parent.parent.parent / "scripts" / "extract_clap.py"
    run_script(script, audio_path, output_path)


def timeout_for(phase: str, explicit: int | None = None) -> int:
    """Resolve the effective timeout for a specific extraction phase.

    Args:
        phase: extraction phase ("dsp" or "mood")
        explicit: caller-supplied timeout (highest priority)

    Returns:
        Effective timeout in seconds

    Raises:
        ValueError: if phase is not "dsp" or "mood"
    """
    if phase == "dsp":
        return resolve_timeout(explicit, "EXTRACT_DSP_TIMEOUT_SEC", DSP_TIMEOUT_SEC)
    if phase == "mood":
        return resolve_timeout(explicit, "EXTRACT_MOOD_TIMEOUT_SEC", MOOD_TIMEOUT_SEC)
    raise ValueError(f"Unknown phase: {phase}")


def run_batch(manifest_path: Path, summary_path: Path, timeout: int | None = None, no_mood: bool = False, models_dir: Path | None = None) -> dict:
    """Run extract_essentia.py --batch on a manifest of audio files.

    The batch worker loads TF graphs once and reuses them across tracks,
    amortising the ~5 s init cost.

    Args:
        manifest_path: path to manifest JSON (list of {audio_path, output_path})
        summary_path: path where summary JSON will be written
        timeout: overall subprocess timeout in seconds (None uses default)
        no_mood: if True, pass --no-mood to skip mood extraction
        models_dir: directory for mood classification models (passed to script)

    Returns:
        Summary dict: {"ok": [str], "failed": [{"output": str, "error": str}]}

    Raises:
        RuntimeError: if the batch subprocess itself fails or times out
    """
    script = Path(__file__).parent.parent.parent / "scripts" / "extract_essentia.py"
    eff = resolve_timeout(timeout, "EXTRACT_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC)
    cmd = [sys.executable, str(script), "--batch", str(manifest_path)]
    if no_mood:
        cmd.append("--no-mood")
    if models_dir is not None:
        cmd += ["--models-dir", str(models_dir)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=eff)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Batch worker timed out after {eff}s") from e

    if result.returncode != 0:
        raise RuntimeError(
            f"Batch worker failed (rc={result.returncode})"
            + (f"\nstderr: {result.stderr}" if result.stderr else "")
        )

    summary_file = Path(str(summary_path))
    if not summary_file.exists():
        raise RuntimeError(f"Batch worker did not produce summary at {summary_file}")

    return json.loads(summary_file.read_text())


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
