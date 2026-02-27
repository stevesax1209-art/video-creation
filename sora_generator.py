"""
Sora Video Generator: Creates cinematic videos using OpenAI's Sora text-to-video API.

Flow:
  1. Submit a video generation job via client.video.generations.create().
  2. Poll client.video.generations.retrieve() until the job succeeds or fails.
  3. Download the resulting MP4 to local storage.
"""

import os
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "sora-2"
DEFAULT_RESOLUTION = "1280x720"
DEFAULT_DURATION = 10

SORA_POLL_INTERVAL = 10   # seconds between status checks
SORA_POLL_TIMEOUT = 600   # 10-minute hard timeout
DOWNLOAD_TIMEOUT = 120    # seconds to wait for video download


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_sora_video(
    prompt: str,
    output_path: str,
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    duration: int = DEFAULT_DURATION,
    progress_callback=None,
) -> str:
    """
    Generate a video with Sora, wait for it to finish, and save the MP4 locally.

    Args:
        prompt:            Text description of the video to generate.
        output_path:       Local filesystem path where the MP4 will be saved.
        model:             Sora model variant ("sora-2" or "sora-2-pro").
        resolution:        Video dimensions ("1280x720" or "720x1280").
        duration:          Clip length in seconds (1–20).
        progress_callback: Optional callable(progress_pct: int, message: str).

    Returns:
        output_path on success.

    Raises:
        RuntimeError:  If Sora reports a generation failure.
        TimeoutError:  If the job does not complete within SORA_POLL_TIMEOUT seconds.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    _cb(progress_callback, 20, "Submitting video request to Sora…")

    # Step 1 – submit generation job
    generation = client.video.generations.create(
        model=model,
        prompt=prompt,
        size=resolution,
        duration=duration,
        n=1,
    )
    generation_id = generation.id

    _cb(progress_callback, 30, "Job queued. Sora is generating your video…")

    # Step 2 – poll until terminal state
    elapsed = 0
    while elapsed < SORA_POLL_TIMEOUT:
        time.sleep(SORA_POLL_INTERVAL)
        elapsed += SORA_POLL_INTERVAL

        result = client.video.generations.retrieve(generation_id)
        status = result.status

        if status == "succeeded":
            video_url = result.data[0].url
            _cb(progress_callback, 85, "Downloading your video…")
            _download_video(video_url, output_path)
            _cb(progress_callback, 100, "Video ready!")
            return output_path

        if status == "failed":
            error_detail = getattr(result, "error", "unknown error")
            raise RuntimeError(f"Sora video generation failed: {error_detail}")

        # queued / processing – report incremental progress
        pct = min(30 + int((elapsed / SORA_POLL_TIMEOUT) * 55), 84)
        _cb(progress_callback, pct, f"Generating your cinematic video… ({elapsed}s elapsed)")

    raise TimeoutError(f"Sora video generation timed out after {SORA_POLL_TIMEOUT} seconds.")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _cb(callback, pct: int, message: str):
    """Safely invoke the optional progress callback."""
    if callback is not None:
        try:
            callback(pct, message)
        except Exception:
            pass


def _download_video(url: str, output_path: str) -> None:
    """Stream-download a video from *url* and write it to *output_path*."""
    resp = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    with open(output_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                fh.write(chunk)
