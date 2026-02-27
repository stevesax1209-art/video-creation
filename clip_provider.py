"""
Clip Provider: A thin interface layer for short-video-clip generation.

The public API is a single function::

    generate_clip(prompt, duration, resolution) -> path_to_mp4

The current implementation is a **stub** that generates a placeholder clip
using FFmpeg so the full pipeline can run end-to-end without AI video API
credentials.  When ``ref_images`` are supplied the stub creates a slide-show
from those images so uploaded character photos actually appear in the clip and
thumbnail.  When no images are available it falls back to a solid dark-gradient
colour.

To swap in a real generator (e.g. Sora, Runway, Kling) replace
``_generate_stub_clip`` with a provider-specific implementation while keeping
the ``generate_clip`` signature stable.
"""

import subprocess
import tempfile
from pathlib import Path

DEFAULT_RESOLUTION = "1280x720"
DEFAULT_FPS = 24
# Maximum number of reference images used in the slide-show stub.
# Kept small to limit ffmpeg filter-graph complexity and peak memory usage.
_MAX_SLIDESHOW_IMAGES = 5


def generate_clip(
    prompt: str,
    duration: float,
    resolution: str = DEFAULT_RESOLUTION,
    output_path: str | None = None,
    ref_images: list[str] | None = None,
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
        ref_images:  Optional list of reference image paths to use as the
                     slide-show source.  The stub cycles through these images
                     so the uploaded character photos appear in the clip.

    Returns:
        Absolute path to the generated MP4 file.
    """
    return _generate_stub_clip(prompt, duration, resolution, output_path, ref_images)


# ---------------------------------------------------------------------------
# Stub implementation
# ---------------------------------------------------------------------------

def _generate_stub_clip(
    prompt: str,
    duration: float,
    resolution: str,
    output_path: str | None,
    ref_images: list[str] | None = None,
) -> str:
    """
    Generate a placeholder clip using FFmpeg.

    If *ref_images* are provided (and exist on disk) the stub builds a
    slide-show from those images so the uploaded character photos appear in
    both the clip and the review thumbnail.  Otherwise it falls back to a
    solid dark-gradient colour (``#1a1a2e``).
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        output_path = tmp.name

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    width, height = resolution.split("x", 1)

    # Use uploaded reference images when available
    valid_images = [p for p in (ref_images or []) if Path(p).exists()]
    if valid_images:
        return _generate_slideshow_clip(valid_images, float(duration), width, height, output_path)

    # Dark gradient fallback (no images available)
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


def _generate_slideshow_clip(
    images: list[str],
    duration: float,
    width: str,
    height: str,
    output_path: str,
) -> str:
    """
    Cycle through *images* to produce a slide-show stub clip.

    Each image is shown for ``duration / n`` seconds (minimum 1 s each).
    Images are scaled and cropped to fill the target resolution.
    """
    # Cap at _MAX_SLIDESHOW_IMAGES to keep the ffmpeg filter graph manageable
    images = images[:_MAX_SLIDESHOW_IMAGES]
    n = len(images)
    secs_per = max(1.0, duration / n)

    args = ["ffmpeg", "-y"]
    for img in images:
        args += ["-loop", "1", "-t", str(secs_per), "-i", img]

    concat_inputs = "".join(f"[{i}:v]" for i in range(n))
    scale_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )
    filter_complex = (
        f"{concat_inputs}concat=n={n}:v=1:a=0[raw];"
        f"[raw]{scale_filter}[out]"
    )

    args += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-pix_fmt", "yuv420p",
        "-an",
        "-t", str(duration),
        output_path,
    ]

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"clip_provider: ffmpeg slideshow error: {result.stderr[-400:]}"
        )

    return output_path
