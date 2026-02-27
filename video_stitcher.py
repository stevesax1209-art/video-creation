"""
Video Stitcher: Concatenates Sora-generated MP4 clips and overlays the
uploaded MP3 audio track using ffmpeg subprocess calls.
"""

import os
import subprocess
import tempfile
from pathlib import Path


def stitch_clips(
    clip_paths: list[str],
    audio_path: str,
    output_path: str,
) -> str:
    """
    Concatenate MP4 clips and overlay the MP3 audio track.

    Uses ffmpeg concat demuxer for lossless stream copy where possible,
    then re-encodes the merged silent video once before adding audio.

    Args:
        clip_paths:   Ordered list of existing MP4 file paths.
        audio_path:   Path to the MP3 audio file.
        output_path:  Destination path for the final MP4.

    Returns:
        output_path on success.

    Raises:
        ValueError:   If clip_paths is empty.
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    if not clip_paths:
        raise ValueError("No clips to stitch.")

    concat_file = ""
    silent_path = output_path + ".silent.mp4"

    try:
        # Write ffmpeg concat list
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as fh:
            concat_file = fh.name
            for clip in clip_paths:
                abs_clip = os.path.abspath(clip)
                # Escape single quotes in paths
                escaped = abs_clip.replace("'", "'\\''")
                fh.write(f"file '{escaped}'\n")

        # Step 1 – concatenate clips into a silent video
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "20", "-pix_fmt", "yuv420p",
            "-an",
            silent_path,
        ])

        # Step 2 – overlay audio, trim to the shorter stream
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-i", silent_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ])

    finally:
        if concat_file:
            Path(concat_file).unlink(missing_ok=True)
        Path(silent_path).unlink(missing_ok=True)

    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_ffmpeg(cmd: list[str]) -> None:
    """Run an ffmpeg command and raise RuntimeError on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Decode safely to avoid cut mid-sequence issues in stderr
        stderr = result.stderr.encode("utf-8", errors="replace").decode("utf-8")[-600:]
        raise RuntimeError(f"ffmpeg error: {stderr}")
