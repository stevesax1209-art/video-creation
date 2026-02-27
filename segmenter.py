"""
Segmenter: Splits an audio file into fixed-length segments using ffprobe to
obtain the true duration, then builds a deterministic list of segment dicts.

Each segment covers exactly SEGMENT_DURATION seconds except the final segment
which may be shorter (never padded beyond the actual audio length).
A ``segments.json`` file is written to ``job_dir`` when supplied so that runs
can be reproduced and inspected for debugging.
"""

import json
import subprocess
from pathlib import Path

# Target duration per segment in seconds
SEGMENT_DURATION = 2.5


def segment_audio(mp3_path: str, job_dir: str | None = None) -> list[dict]:
    """
    Split an audio file into ~2.5-second segments.

    Args:
        mp3_path: Path to the MP3 (or any ffprobe-readable audio file).
        job_dir:  Optional directory; when given, ``segments.json`` is saved
                  there for debugging.

    Returns:
        Ordered list of segment dicts::

            [{"i": 0, "start": 0.0, "end": 2.5, "duration": 2.5}, ...]

        The last segment may have a ``duration`` shorter than
        ``SEGMENT_DURATION`` if the audio length is not an exact multiple.
    """
    duration = _probe_duration(mp3_path)
    segments = _build_segments(duration)

    if job_dir is not None:
        _save_json(segments, job_dir)

    return segments


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _probe_duration(mp3_path: str) -> float:
    """Return audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            mp3_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for '{mp3_path}': {result.stderr.strip()}")
    if not result.stdout.strip():
        raise RuntimeError(f"ffprobe returned no duration for '{mp3_path}'")
    return float(result.stdout.strip())


def _build_segments(duration: float) -> list[dict]:
    """Build the deterministic segment list for a given total duration."""
    total = round(duration, 3)  # canonical end boundary
    segments: list[dict] = []
    i = 0
    start = 0.0
    while start < total:
        end = min(round(start + SEGMENT_DURATION, 3), total)
        segments.append({
            "i":        i,
            "start":    start,
            "end":      end,
            "duration": round(end - start, 3),
        })
        start = end
        i += 1
    return segments


def _save_json(segments: list[dict], job_dir: str) -> None:
    """Persist segments to ``<job_dir>/segments.json``."""
    path = Path(job_dir) / "segments.json"
    with open(path, "w") as fh:
        json.dump(segments, fh, indent=2)
