"""Feature extractor — runs Python extraction scripts as subprocess wrappers.

Calls the existing scripts extract_essentia.py and extract_clap.py via
subprocess.run() with a 30-second timeout and exit-code checking.
"""

import json
import subprocess
import sys
from pathlib import Path


# Hard timeout per subprocess in seconds (matches Java FeatureExtractor default)
TIMEOUT_SECONDS = 30


def run_script(script: Path, audio_path: Path, output_path: Path) -> None:
    """Run a Python extraction script via subprocess.run().

    Args:
        script: filename of the script (e.g. "extract_essentia.py")
        audio_path: absolute path to the input audio file
        output_path: absolute path for the JSON sidecar output

    Raises:
        RuntimeError: if the process times out or returns non-zero exit code
    """
    cmd = [sys.executable, str(script), str(audio_path), str(output_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Script {script} timed out after {TIMEOUT_SECONDS}s"
        ) from e

    if result.returncode != 0:
        raise RuntimeError(
            f"Script {script} exited with code {result.returncode}"
            + (f"\nstderr: {result.stderr}" if result.stderr else "")
        )

    # Script succeeded — it wrote the JSON sidecar already
    if not output_path.exists():
        raise RuntimeError(f"Script {script} did not produce output at {output_path}")


def extract_essentia(audio_path: Path, output_path: Path) -> None:
    """Run extract_essentia.py on an audio file.

    Args:
        audio_path: path to the input audio file
        output_path: path where the JSON sidecar will be written
    """
    script = Path(__file__).parent.parent.parent / "scripts" / "extract_essentia.py"
    run_script(script, audio_path, output_path)


def extract_clap(audio_path: Path, output_path: Path) -> None:
    """Run extract_clap.py on an audio file.

    Args:
        audio_path: path to the input audio file
        output_path: path where the JSON sidecar will be written
    """
    script = Path(__file__).parent.parent.parent / "scripts" / "extract_clap.py"
    run_script(script, audio_path, output_path)