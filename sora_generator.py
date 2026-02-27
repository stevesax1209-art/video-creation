"""
Sora Video Generator: Generates short cinematic clips (~5 s each) using
OpenAI's Sora text-to-video API.

Batch flow (generate_scene_clips):
  1. For each scene: submit → poll → download → identity QC.
  2. On QC fail: retry same prompt.
  3. On second QC fail: retry with a wider-shot fallback prompt.
  4. If third attempt also fails: skip clip (rather than abort the whole job).

Single-clip helper (create_sora_video) is kept for direct use.
"""

import os
import time
from pathlib import Path

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "sora-2"
DEFAULT_RESOLUTION = "1280x720"
DEFAULT_DURATION = 5

SORA_POLL_INTERVAL = 10    # seconds between polls
SORA_POLL_TIMEOUT  = 600   # max wait per clip (10 min)
DOWNLOAD_TIMEOUT   = 120   # seconds for HTTP download


# ---------------------------------------------------------------------------
# Batch clip generation
# ---------------------------------------------------------------------------

def generate_scene_clips(
    scenes: list[dict],
    clips_dir: str,
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    progress_callback=None,
    char_descriptions: dict[str, str] | None = None,
) -> list[str]:
    """
    Generate one Sora clip per scene dict with identity QC and retry.

    Each *scene* dict must have:
      - prompt:     str  – Sora text prompt
      - duration:   int  – clip length in seconds
      - characters: list – character names (used for QC)

    Returns ordered list of saved MP4 paths (may be shorter than *scenes*
    if some clips could not be generated after retries).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    clips_dir_path = Path(clips_dir)
    clips_dir_path.mkdir(parents=True, exist_ok=True)
    char_descriptions = char_descriptions or {}

    total = len(scenes)
    saved_clips: list[str] = []

    for idx, scene in enumerate(scenes):
        # Progress: 20 % → 88 % spread across all clips
        pct = 20 + int((idx / total) * 68)
        _cb(progress_callback, pct,
            f"Generating scene {idx + 1}/{total} ({scene.get('arc', '')} · {scene.get('intensity', '')} energy)…")

        clip_path = str(clips_dir_path / f"clip_{idx:04d}.mp4")
        prompt   = scene["prompt"]
        duration = int(scene.get("duration", DEFAULT_DURATION))

        # Determine which character to QC (first locked char in this scene)
        qc_char = scene.get("characters", [None])[0]
        qc_desc = char_descriptions.get(qc_char, "") if qc_char else ""

        if _generate_with_qc(client, prompt, duration, model, resolution,
                              clip_path, qc_char, qc_desc):
            saved_clips.append(clip_path)

    return saved_clips


def _generate_with_qc(
    client: OpenAI,
    prompt: str,
    duration: int,
    model: str,
    resolution: str,
    clip_path: str,
    qc_char: str | None,
    qc_desc: str,
) -> bool:
    """
    Try to generate a clip and pass identity QC.
    Attempt 1: original prompt.
    Attempt 2 (QC fail): original prompt again.
    Attempt 3 (2nd QC fail): wider-shot fallback.
    Returns True if a clip was saved, False if all attempts failed.
    """
    for attempt in range(3):
        current_prompt = _make_wider_shot_prompt(prompt) if attempt == 2 else prompt
        try:
            gen_id = _submit_job(client, current_prompt, model, resolution, duration)
            url    = _poll_until_done(client, gen_id)
            _download_video(url, clip_path)
        except Exception:
            continue

        # Identity QC
        if qc_char and qc_desc:
            from identity_qc import check_clip_identity
            qc = check_clip_identity(clip_path, qc_char, qc_desc)
            if qc["passed"]:
                return True
            # QC failed – delete clip and retry
            Path(clip_path).unlink(missing_ok=True)
        else:
            return True  # No QC required

    # All attempts exhausted – skip this clip
    return False


# ---------------------------------------------------------------------------
# Single-clip public API (kept for direct use)
# ---------------------------------------------------------------------------

def create_sora_video(
    prompt: str,
    output_path: str,
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    duration: int = DEFAULT_DURATION,
    progress_callback=None,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    _cb(progress_callback, 20, "Submitting video request to Sora…")

    gen_id = _submit_job(client, prompt, model, resolution, duration)
    _cb(progress_callback, 30, "Job queued. Sora is generating your video…")

    url = _poll_until_done(
        client, gen_id,
        on_progress=lambda elapsed: _cb(
            progress_callback,
            min(30 + int((elapsed / SORA_POLL_TIMEOUT) * 55), 84),
            f"Generating… ({elapsed}s elapsed)",
        ),
    )

    _cb(progress_callback, 85, "Downloading your video…")
    _download_video(url, output_path)
    _cb(progress_callback, 100, "Video ready!")
    return output_path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _submit_job(client: OpenAI, prompt: str, model: str, resolution: str, duration: int) -> str:
    generation = client.video.generations.create(
        model=model,
        prompt=prompt,
        size=resolution,
        duration=duration,
        n=1,
    )
    return generation.id


def _poll_until_done(client: OpenAI, generation_id: str, on_progress=None) -> str:
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

    raise TimeoutError(f"Sora generation timed out after {SORA_POLL_TIMEOUT}s.")


def _make_wider_shot_prompt(original_prompt: str) -> str:
    """
    Wider-shot fallback: reduces identity pressure and slows camera motion.
    Truncates at a word boundary to avoid cutting mid-word.
    """
    # Truncate to ~300 chars at a word boundary
    truncated = original_prompt[:300]
    if len(original_prompt) > 300:
        last_space = truncated.rfind(" ")
        if last_space > 200:
            truncated = truncated[:last_space]
    return (
        "Wide establishing shot, figures small in frame, slow smooth motion. "
        + truncated
        + " Wide angle lens, no close-up faces, slow and steady camera."
    )


def _cb(callback, pct: int, message: str) -> None:
    if callback is not None:
        try:
            callback(pct, message)
        except Exception:
            pass


def _download_video(url: str, output_path: str) -> None:
    resp = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    with open(output_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                fh.write(chunk)
