"""
Sora Video Generator: Creates cinematic video clips using OpenAI's Sora
text-to-video API.

Single-clip API:  create_sora_video()
Batch API:        generate_scene_clips()  – used by the music video worker.

Flow per clip:
  1. Submit a generation job via client.video.generations.create().
  2. Poll client.video.generations.retrieve() until succeeded / failed.
  3. Download the resulting MP4 to local storage.
  4. On failure, retry once with a wider-shot fallback prompt.
"""

import os
import time
from pathlib import Path

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "sora-2"
DEFAULT_RESOLUTION = "1280x720"
DEFAULT_DURATION = 10

SORA_POLL_INTERVAL = 10   # seconds between status checks
SORA_POLL_TIMEOUT = 600   # 10-minute hard timeout per clip
DOWNLOAD_TIMEOUT = 120    # seconds to wait for video download


# ---------------------------------------------------------------------------
# Single-clip public API
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
    Generate a single video clip with Sora, poll until complete, and save
    the MP4 locally.

    Args:
        prompt:            Text description of the video to generate.
        output_path:       Local filesystem path where the MP4 will be saved.
        model:             Sora model variant ("sora-2" or "sora-2-pro").
        resolution:        Video dimensions (e.g. "1280x720", "1920x1080").
        duration:          Clip length in seconds (1–20).
        progress_callback: Optional callable(progress_pct: int, message: str).

    Returns:
        output_path on success.

    Raises:
        RuntimeError:  If Sora reports a generation failure.
        TimeoutError:  If the job does not complete within SORA_POLL_TIMEOUT.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    _cb(progress_callback, 20, "Submitting video request to Sora…")

    generation_id = _submit_job(client, prompt, model, resolution, duration)

    _cb(progress_callback, 30, "Job queued. Sora is generating your video…")

    video_url = _poll_until_done(
        client, generation_id,
        on_progress=lambda elapsed: _cb(
            progress_callback,
            min(30 + int((elapsed / SORA_POLL_TIMEOUT) * 55), 84),
            f"Generating your cinematic video… ({elapsed}s elapsed)",
        ),
    )

    _cb(progress_callback, 85, "Downloading your video…")
    _download_video(video_url, output_path)
    _cb(progress_callback, 100, "Video ready!")
    return output_path


# ---------------------------------------------------------------------------
# Batch clip generation (used by the music video worker)
# ---------------------------------------------------------------------------

def generate_scene_clips(
    scenes: list[dict],
    clips_dir: str,
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    progress_callback=None,
) -> list[str]:
    """
    Generate one Sora clip per scene dict sequentially.

    Each *scene* dict must contain:
      - prompt:    str  – Sora text prompt
      - duration:  int  – clip length in seconds

    On a generation failure the clip is retried once with a wider-shot
    fallback prompt.  If the retry also fails the scene is skipped and
    generation continues so the rest of the video is not lost.

    Args:
        scenes:            List of scene dicts (from scene_planner.plan_scenes).
        clips_dir:         Directory where individual MP4 clips are saved.
        model:             Sora model variant.
        resolution:        Video dimensions.
        progress_callback: Optional callable(progress_pct: int, message: str).

    Returns:
        Ordered list of saved clip file paths (may be shorter than *scenes*
        if some clips could not be generated).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    clips_dir_path = Path(clips_dir)
    clips_dir_path.mkdir(parents=True, exist_ok=True)

    total = len(scenes)
    saved_clips: list[str] = []

    for idx, scene in enumerate(scenes):
        # Progress: 20 % → 88 % spread across all clips
        scene_pct = 20 + int(((idx) / total) * 68)
        _cb(
            progress_callback,
            scene_pct,
            f"Generating scene {idx + 1} of {total} ({scene.get('arc', '')} arc)…",
        )

        clip_path = str(clips_dir_path / f"clip_{idx:04d}.mp4")
        prompt = scene["prompt"]
        duration = int(scene.get("duration", 3))

        # First attempt
        try:
            gen_id = _submit_job(client, prompt, model, resolution, duration)
            url = _poll_until_done(client, gen_id)
            _download_video(url, clip_path)
            saved_clips.append(clip_path)
            continue
        except Exception:
            pass

        # Retry with a wider, simpler shot (identity fallback)
        fallback_prompt = _make_wider_shot_prompt(prompt)
        try:
            gen_id = _submit_job(client, fallback_prompt, model, resolution, duration)
            url = _poll_until_done(client, gen_id)
            _download_video(url, clip_path)
            saved_clips.append(clip_path)
        except Exception:
            # Skip this clip entirely rather than aborting the whole job
            pass

    return saved_clips


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _submit_job(
    client: OpenAI,
    prompt: str,
    model: str,
    resolution: str,
    duration: int,
) -> str:
    """Submit a Sora generation job and return its ID."""
    generation = client.video.generations.create(
        model=model,
        prompt=prompt,
        size=resolution,
        duration=duration,
        n=1,
    )
    return generation.id


def _poll_until_done(
    client: OpenAI,
    generation_id: str,
    on_progress=None,
) -> str:
    """
    Poll until the generation succeeds or fails.

    Returns the video URL on success.
    Raises RuntimeError on failure, TimeoutError on timeout.
    """
    elapsed = 0
    while elapsed < SORA_POLL_TIMEOUT:
        time.sleep(SORA_POLL_INTERVAL)
        elapsed += SORA_POLL_INTERVAL

        result = client.video.generations.retrieve(generation_id)
        status = result.status

        if status == "succeeded":
            return result.data[0].url

        if status == "failed":
            error_detail = getattr(result, "error", "unknown error")
            raise RuntimeError(f"Sora generation failed: {error_detail}")

        if on_progress:
            try:
                on_progress(elapsed)
            except Exception:
                pass

    raise TimeoutError(
        f"Sora generation timed out after {SORA_POLL_TIMEOUT} seconds."
    )


def _make_wider_shot_prompt(original_prompt: str) -> str:
    """
    Produce a simpler wide-shot fallback prompt that reduces identity risk.
    Prepends a wide-shot instruction and strips character close-up cues.
    """
    return (
        "Wide establishing shot, figures small in frame. "
        + original_prompt[:300]
        + " Wide angle, no close faces."
    )


def _cb(callback, pct: int, message: str) -> None:
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

