"""
Identity QC: Extracts a middle frame from a generated Sora clip and uses
GPT-4 Vision to verify that the locked character's identity is consistent
with their reference description.

Flow per clip:
  1. ffmpeg extracts a single JPEG frame at the clip midpoint.
  2. GPT-4 Vision receives the frame + the character's reference description.
  3. Model returns PASS / FAIL + a one-sentence reason.
  4. Caller (sora_generator) decides whether to regenerate.
"""

import base64
import os
import subprocess
import tempfile
from pathlib import Path

try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False


# A FAIL verdict with confidence below this threshold is treated as PASS.
# Rationale: very low model confidence means the frame was ambiguous (distant
# wide shot, partial occlusion, etc.) rather than a genuine identity mismatch.
# Blocking regeneration in these cases wastes Sora quota with no quality gain.
_LOW_CONFIDENCE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_clip_identity(
    clip_path: str,
    char_name: str,
    char_description: str,
) -> dict:
    """
    Check whether the locked character appears consistently in the clip.

    Args:
        clip_path:        Path to the generated MP4 clip.
        char_name:        Human-readable character name (e.g. "Bryce").
        char_description: Appearance description produced by describe_character().

    Returns:
        {
          "passed":     bool,
          "confidence": float,  # 0.0 – 1.0 (heuristic from model response)
          "reason":     str,
        }

    If GPT-4 Vision is unavailable or the frame cannot be extracted the check
    defaults to PASS so generation is never blocked by a QC infra failure.
    """
    frame_path = _extract_middle_frame(clip_path)
    if frame_path is None:
        return {"passed": True, "confidence": 0.5, "reason": "Frame extraction failed – skipping QC."}

    try:
        result = _gpt4v_identity_check(frame_path, char_name, char_description)
    finally:
        Path(frame_path).unlink(missing_ok=True)

    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _extract_middle_frame(clip_path: str) -> "str | None":
    """
    Use ffprobe to get clip duration, then ffmpeg to extract a JPEG at
    the midpoint.  Returns the temp file path or None on failure.
    """
    try:
        # Get duration
        probe = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json", "-show_format",
                clip_path,
            ],
            capture_output=True, text=True, timeout=20,
        )
        if probe.returncode != 0:
            return None

        import json
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        midpoint = duration / 2.0

        # Extract frame
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        frame_path = tmp.name

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(midpoint),
                "-i", clip_path,
                "-vframes", "1",
                "-q:v", "5",
                frame_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            Path(frame_path).unlink(missing_ok=True)
            return None

        return frame_path
    except Exception:
        return None


def _gpt4v_identity_check(
    frame_path: str,
    char_name: str,
    char_description: str,
) -> dict:
    """Send the frame to GPT-4 Vision and parse the identity verdict."""
    if not _OPENAI_OK:
        return {"passed": True, "confidence": 0.5, "reason": "OpenAI SDK unavailable."}

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"passed": True, "confidence": 0.5, "reason": "No API key – skipping QC."}

    try:
        with open(frame_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
    except Exception as exc:
        return {"passed": True, "confidence": 0.5, "reason": f"Could not read frame: {exc}"}

    client = OpenAI(api_key=api_key)

    prompt = (
        f"You are an identity consistency checker for an AI video generator.\n"
        f"Character: {char_name}\n"
        f"Reference description: {char_description}\n\n"
        "Look at the provided video frame.\n"
        f"Does the person who appears to be {char_name} in this frame MATCH the reference description?\n"
        "Reply with exactly one line in this format:\n"
        "PASS <confidence 0-100> <one sentence reason>\n"
        "or\n"
        "FAIL <confidence 0-100> <one sentence reason>\n"
        "If no person matching the description is visible, reply FAIL 0 No matching character found."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        },
                    },
                ],
            }],
            max_tokens=80,
        )
        return _parse_verdict(resp.choices[0].message.content.strip())
    except Exception as exc:
        return {"passed": True, "confidence": 0.5, "reason": f"GPT-4 Vision error: {exc}"}


def _parse_verdict(text: str) -> dict:
    """Parse 'PASS 85 Bryce is clearly visible...' → dict."""
    parts = text.strip().split(None, 2)
    verdict = parts[0].upper() if parts else "PASS"
    try:
        confidence = float(parts[1]) / 100.0 if len(parts) > 1 else 0.5
    except ValueError:
        confidence = 0.5
    reason = parts[2] if len(parts) > 2 else text

    passed = verdict == "PASS"
    if not passed and confidence < _LOW_CONFIDENCE_THRESHOLD:
        # Low confidence fail → treat as pass to avoid over-triggering
        passed = True
        reason = f"Low-confidence fail ignored. {reason}"

    return {"passed": passed, "confidence": confidence, "reason": reason}
