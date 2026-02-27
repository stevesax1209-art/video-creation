"""
Clip Provider: A thin interface layer for short-video-clip generation.

The public API is a single function::

    generate_clip(prompt, duration, resolution) -> path_to_mp4

The current implementation is a **stub** that generates a simple dark-gradient
placeholder clip using FFmpeg so the full pipeline can run end-to-end without
any AI video API credentials.

To swap in a real generator (e.g. Sora, Runway, Kling) replace
``_generate_stub_clip`` with a provider-specific implementation while keeping
the ``generate_clip`` signature stable.
"""

import subprocess
import tempfile
from pathlib import Path

DEFAULT_RESOLUTION = "1280x720"
DEFAULT_FPS = 24


def generate_clip(
    prompt: str,
    duration: float,
    resolution: str = DEFAULT_RESOLUTION,
    output_path: str | None = None,
) -> str:
    """
    Generate a short video clip for the given prompt and duration.

    Args:
        prompt:      Text prompt describing the desired clip content.
                     (Used by real providers; the stub ignores it.)
        duration:    Clip length in seconds.
        resolution:  Output resolution as ``"WxH"`` (e.g. ``"1280x720"``).
        output_path: Destination path for the MP4.  When ``None`` a temporary
                     file is created automatically.

    Returns:
        Absolute path to the generated MP4 file.
    """
    return _generate_stub_clip(prompt, duration, resolution, output_path)


# ---------------------------------------------------------------------------
# Stub implementation
# ---------------------------------------------------------------------------

def _generate_stub_clip(
    prompt: str,
    duration: float,
    resolution: str,
    output_path: str | None,
) -> str:
    """
    Generate a plain dark-gradient placeholder clip using FFmpeg.

    The clip is a solid color (``#1a1a2e``) at the requested resolution and
    duration with no audio track — ready to be muxed with audio downstream.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        output_path = tmp.name

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    width, height = resolution.split("x", 1)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", (
            f"color=c=0x1a1a2e:size={width}x{height}"
            f":rate={DEFAULT_FPS}:duration={duration}"
        ),
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-pix_fmt", "yuv420p",
        "-an",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"clip_provider: ffmpeg error generating stub clip: "
            f"{result.stderr[-400:]}"
        )

    return output_path
