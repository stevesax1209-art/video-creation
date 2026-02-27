"""
Video Generator: Combines images (optional) and an MP3 audio track into an
MP4 music video using FFmpeg.

The previous MoviePy Ken Burns / ImageMagick caption-overlay path has been
removed.  FFmpeg handles all encoding so the app works without ImageMagick.
When no images are supplied a plain dark-gradient background is used so that
"vision prompt only" jobs still produce a valid MP4.
"""

import os
import subprocess
import tempfile
from pathlib import Path

# Output resolution
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_FPS = 24


def generate_video(
    audio_path: str,
    image_paths: list[str],
    captions: list[str],  # kept for API compatibility, no longer rendered
    output_path: str,
) -> str:
    """
    Generate an MP4 music video from images (optional) and audio via FFmpeg.

    Args:
        audio_path:  Path to the input MP3 file.
        image_paths: List of image file paths (may be empty for vision-only jobs).
        captions:    Ignored — caption overlays have been removed.
        output_path: Destination path for the output MP4.

    Returns:
        The output_path on success.
    """
    if image_paths:
        return _generate_from_images(audio_path, image_paths, output_path)
    return _generate_from_gradient(audio_path, output_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_from_images(
    audio_path: str,
    image_paths: list[str],
    output_path: str,
) -> str:
    """Build a slideshow from still images and overlay audio with FFmpeg."""
    audio_duration = _probe_duration(audio_path)
    per_image = audio_duration / len(image_paths)

    concat_file = None
    silent_path = output_path + ".silent.mp4"

    try:
        with tempfile.TemporaryDirectory() as tmp:
            slide_paths: list[str] = []
            for i, img_path in enumerate(image_paths):
                slide = os.path.join(tmp, f"slide_{i:04d}.mp4")
                _run([
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", img_path,
                    "-t", str(per_image),
                    "-vf", (
                        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
                        ":force_original_aspect_ratio=decrease,"
                        f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:"
                        "(ow-iw)/2:(oh-ih)/2,setsar=1"
                    ),
                    "-c:v", "libx264", "-preset", "fast",
                    "-crf", "20", "-pix_fmt", "yuv420p",
                    "-r", str(OUTPUT_FPS), "-an",
                    slide,
                ])
                slide_paths.append(slide)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as fh:
                concat_file = fh.name
                for sp in slide_paths:
                    fh.write(f"file '{sp}'\n")

            _run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c", "copy", "-an",
                silent_path,
            ])

        _run([
            "ffmpeg", "-y",
            "-i", silent_path,
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path,
        ])

    finally:
        if concat_file is not None:
            Path(concat_file).unlink(missing_ok=True)
        Path(silent_path).unlink(missing_ok=True)

    return output_path


def _generate_from_gradient(audio_path: str, output_path: str) -> str:
    """Create a plain dark-background clip and overlay audio (vision-only jobs)."""
    audio_duration = _probe_duration(audio_path)
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", (
            f"color=c=0x1a1a2e:size={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}"
            f":rate={OUTPUT_FPS}:duration={audio_duration}"
        ),
        "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path,
    ])
    return output_path


def _probe_duration(path: str) -> float:
    """Return duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe is required to read audio duration but was not found. "
            "Please install ffmpeg (which includes ffprobe) and ensure it is in your system PATH."
        )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for '{path}': {result.stderr}")
    if not result.stdout.strip():
        raise RuntimeError(f"ffprobe returned no duration for '{path}'")
    return float(result.stdout.strip())


def _run(cmd: list[str]) -> None:
    """Run a subprocess command, raising RuntimeError on non-zero exit."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is required to generate video but was not found. "
            "Please install ffmpeg and ensure it is in your system PATH."
        )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr[-600:]}")
