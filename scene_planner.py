"""
Scene Planner: Takes energy segments from the audio analyzer, describes cast
characters via GPT-4 Vision, then generates one cinematic Sora prompt per
segment that:
  - Follows a structured emotional arc
  - Matches the segment's energy level (calm vs. dynamic visuals)
  - Obeys all Phase 1 engine rules
"""

import base64
import math
import os
import re
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

# ---------------------------------------------------------------------------
# Engine constants
# ---------------------------------------------------------------------------

BRYCE_SCENE_FRACTION = 0.70

STYLE_DESCRIPTIONS: dict[str, str] = {
    "documentary": (
        "documentary handheld cinematography, intimate and authentic, "
        "natural available light, slight organic camera movement"
    ),
    "cinematic": (
        "photorealistic cinematic, wide dynamic range, dramatic motivated lighting, "
        "steady professional camera, shallow depth of field"
    ),
}

CAST_MAP: dict[str, list[str]] = {
    "bryce":         ["Bryce"],
    "bryce_brian":   ["Bryce", "Brian"],
    "bryce_carmen":  ["Bryce", "Carmen"],
}

_ARC = [
    (0.00, 0.10, "opening",  "a quiet, intimate human moment – stillness and reflection"),
    (0.10, 0.70, "middle",   "genuine human interaction, warmth, mutual support and presence"),
    (0.70, 0.90, "lift",     "connection and hope, a sense of rising energy and shared purpose"),
    (0.90, 1.00, "ending",   "relief, dignity and peace – a quiet resolution"),
]

_CAMERA_MOVES: dict[str, list[str]] = {
    "low":    ["slow push-in", "gentle dolly forward", "static wide", "slow tilt up"],
    "medium": ["smooth tracking shot", "subtle handheld drift", "slow pan left",
               "gentle crane down", "soft rack focus"],
    "high":   ["dynamic tracking", "motivated handheld", "quick push-in",
               "energetic arc shot", "fluid dolly through"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def describe_character(name: str, image_paths: list[str]) -> str:
    if not image_paths or not _openai_available:
        return name
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return name

    client = OpenAI(api_key=api_key)
    images_content: list[dict] = []
    for path in image_paths[:2]:
        try:
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            ext = Path(path).suffix.lower().lstrip(".")
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            images_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"},
            })
        except Exception:
            continue

    if not images_content:
        return name

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Describe {name}'s consistent physical appearance in under 20 words "
                            "for AI video prompts. Focus on face, hair, skin tone, clothing style."
                        ),
                    },
                    *images_content,
                ],
            }],
            max_tokens=60,
        )
        return f"{name} ({resp.choices[0].message.content.strip()})"
    except Exception:
        return name


def plan_scenes(
    cast_key: str,
    style_key: str,
    character_descriptions: dict[str, str],
    audio_segments: list[dict],
    lyrics: Optional[str] = None,
) -> list[dict]:
    """Return one scene dict per audio segment."""
    characters = CAST_MAP.get(cast_key, ["Bryce"])
    style_desc = STYLE_DESCRIPTIONS.get(style_key, STYLE_DESCRIPTIONS["cinematic"])
    num_scenes = len(audio_segments)
    bryce_count = math.ceil(num_scenes * BRYCE_SCENE_FRACTION)

    if _openai_available and os.environ.get("OPENAI_API_KEY") and num_scenes > 0:
        scene_texts = _plan_with_openai(num_scenes, characters, lyrics, style_desc, character_descriptions)
    else:
        scene_texts = _fallback_texts(num_scenes)

    secondary = [c for c in characters if c != "Bryce"]
    scenes: list[dict] = []

    for i, seg in enumerate(audio_segments):
        arc_label = _arc_label_for(i, num_scenes)
        intensity = seg.get("intensity", "medium")
        camera = _pick_camera_move(intensity, i)

        if i < bryce_count:
            chars = ["Bryce"]
            char_hint = character_descriptions.get("Bryce", "Bryce")
        else:
            chars = secondary or ["Bryce"]
            char_hint = ", ".join(character_descriptions.get(c, c) for c in chars)

        base_text = scene_texts[i] if i < len(scene_texts) else scene_texts[-1]

        full_prompt = (
            f"{base_text} "
            f"Featuring {char_hint}. "
            f"{style_desc}. "
            f"{camera}. "
            "Elderly seniors as dignified background figures. "
            "No text overlays, no logos, no lip sync. "
            "Authentic and dignified."
        )

        scenes.append({
            "prompt":     full_prompt,
            "duration":   int(seg.get("clip_secs", 5)),
            "characters": chars,
            "arc":        arc_label,
            "intensity":  intensity,
        })

    return scenes


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _plan_with_openai(num_scenes, characters, lyrics, style_desc, char_descs):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    bryce_count = math.ceil(num_scenes * BRYCE_SCENE_FRACTION)
    secondary = [c for c in characters if c != "Bryce"]

    char_block = "\n".join(f"- {char_descs.get(c, c)}" for c in characters)
    arc_block = "\n".join(
        f"  Scenes {int(s * num_scenes) + 1}-{int(e * num_scenes)}: {label} - {desc}"
        for s, e, label, desc in _ARC
    )

    system = (
        "You are a cinematic AI music video director writing short scene descriptions "
        f"for a Sora text-to-video generator.\n\n"
        f"CHARACTERS:\n{char_block}\n\n"
        f"EMOTIONAL ARC:\n{arc_block}\n\n"
        "RULES:\n"
        f"1. Bryce appears in scenes 1-{bryce_count} (of {num_scenes}).\n"
        f"2. Secondary ({', '.join(secondary) if secondary else 'none'}) in remaining scenes.\n"
        "3. Never more than 3 visible faces per frame.\n"
        "4. Elderly seniors as dignified background figures in some scenes.\n"
        "5. No lip sync, no text overlays, no logos.\n\n"
        f"Write EXACTLY {num_scenes} descriptions (max 2 sentences and max 30 words each).\n"
        "Return ONLY a numbered list."
    )

    user_parts = [f"Generate {num_scenes} scene descriptions."]
    if lyrics:
        user_parts.append(f"Lyrics:\n{lyrics[:3000]}")

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "\n\n".join(user_parts)},
            ],
            max_tokens=4096,
            temperature=0.8,
        )
        return _parse_numbered_list(resp.choices[0].message.content.strip(), num_scenes)
    except Exception:
        return _fallback_texts(num_scenes)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

_FALLBACK_BY_ARC = {
    "opening": [
        "Bryce sits quietly in a softly lit room, hands folded, morning light.",
        "Close-up of Bryce's face, contemplative, gentle ambient stillness.",
    ],
    "middle": [
        "Bryce walks slowly through a community center, nodding to passing elders.",
        "Bryce and an elderly person share a warm, silent moment at a table.",
        "Wide shot of a garden courtyard, seniors in background, Bryce in foreground.",
        "Bryce gently places a hand on an elder's shoulder in a caring gesture.",
        "Bryce leans forward attentively, genuine care in his expression.",
    ],
    "lift": [
        "Bryce and an elder laugh softly together, a moment of shared joy.",
        "Wide shot, Bryce and seniors walking outside in warm sunlight.",
    ],
    "ending": [
        "Bryce seated in quiet dignity, late afternoon golden light, peaceful.",
        "Close-up of hands, young and old, resting together in quiet solidarity.",
    ],
}


def _fallback_texts(num_scenes: int) -> list[str]:
    texts: list[str] = []
    idx: dict[str, int] = {k: 0 for k in _FALLBACK_BY_ARC}
    for i in range(num_scenes):
        arc = _arc_label_for(i, num_scenes)
        pool = _FALLBACK_BY_ARC[arc]
        texts.append(pool[idx[arc] % len(pool)])
        idx[arc] += 1
    return texts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arc_label_for(scene_index: int, num_scenes: int) -> str:
    frac = scene_index / max(num_scenes - 1, 1)
    for start, end, label, _ in _ARC:
        if start <= frac < end:
            return label
    return "ending"


def _pick_camera_move(intensity: str, seed: int) -> str:
    moves = _CAMERA_MOVES.get(intensity, _CAMERA_MOVES["medium"])
    return moves[seed % len(moves)]


def _parse_numbered_list(raw: str, num_scenes: int) -> list[str]:
    lines = raw.splitlines()
    prompts: list[str] = []
    for line in lines:
        cleaned = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
        cleaned = re.sub(r"^\s*[-*]\s*", "", cleaned).strip()
        if cleaned:
            prompts.append(cleaned)
    fallback = "Bryce in a quiet moment, cinematic warm light."
    while len(prompts) < num_scenes:
        prompts.append(prompts[-1] if prompts else fallback)
    return prompts[:num_scenes]
