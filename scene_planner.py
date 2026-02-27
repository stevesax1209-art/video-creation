"""
Scene Planner: Analyzes the uploaded MP3, describes cast characters via
GPT-4 Vision, then generates 60–90 cinematic scene prompts that follow
a structured emotional arc and obey all V1 engine rules.
"""

import base64
import json
import math
import os
import re
import subprocess
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

SCENE_DURATION_SECS = 3          # target seconds per Sora clip
MIN_SCENES = 30
MAX_SCENES = 90
BRYCE_SCENE_FRACTION = 0.70      # Bryce must appear in ≥ 70 % of scenes
MAX_FACES_PER_FRAME = 3

STYLE_DESCRIPTIONS: dict[str, str] = {
    "documentary": (
        "documentary handheld cinematography, intimate and authentic, "
        "natural available light, slight organic camera movement, raw emotion"
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

# Emotional arc: (start_fraction, end_fraction, arc_label, arc_description)
_ARC = [
    (0.00, 0.10, "opening",    "a quiet, intimate human moment – stillness and reflection"),
    (0.10, 0.70, "middle",     "genuine human interaction, warmth, mutual support and presence"),
    (0.70, 0.90, "lift",       "connection and hope, a sense of rising energy and shared purpose"),
    (0.90, 1.00, "ending",     "relief, dignity and peace – a quiet resolution"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_audio_duration(mp3_path: str) -> float:
    """Return MP3 duration in seconds using ffprobe (falls back to 210 s)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                mp3_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
    except Exception:
        pass
    return 210.0  # ~3.5 minutes fallback


def describe_character(name: str, image_paths: list[str]) -> str:
    """
    Use GPT-4 Vision to produce a short, consistent appearance description for
    *name* from up to 2 reference images.  Falls back to just the name.
    """
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
            # Guess MIME type from extension
            ext = Path(path).suffix.lower().lstrip(".")
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            images_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                    "detail": "low",
                },
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
                            "for use in AI video prompts. Focus on face, hair color/style, "
                            "skin tone, and typical clothing style. Be specific and concise."
                        ),
                    },
                    *images_content,
                ],
            }],
            max_tokens=60,
        )
        description = resp.choices[0].message.content.strip()
        return f"{name} ({description})"
    except Exception:
        return name


def plan_scenes(
    mp3_path: str,
    cast_key: str,
    lyrics: Optional[str],
    style_key: str,
    character_descriptions: dict[str, str],
) -> list[dict]:
    """
    Return an ordered list of scene dicts, each containing:
      - prompt:     str   – full Sora generation prompt
      - duration:   int   – clip length in seconds
      - characters: list  – which characters appear in this scene
      - arc:        str   – emotional arc label

    Args:
        mp3_path:                Path to the uploaded MP3 file.
        cast_key:                One of "bryce", "bryce_brian", "bryce_carmen".
        lyrics:                  Optional song lyrics for sentiment context.
        style_key:               "documentary" or "cinematic".
        character_descriptions:  {char_name: description_string} from describe_character().
    """
    duration = get_audio_duration(mp3_path)
    num_scenes = max(MIN_SCENES, min(MAX_SCENES, int(duration / SCENE_DURATION_SECS)))

    characters = CAST_MAP.get(cast_key, ["Bryce"])
    style_desc = STYLE_DESCRIPTIONS.get(style_key, STYLE_DESCRIPTIONS["cinematic"])

    if _openai_available and os.environ.get("OPENAI_API_KEY"):
        scenes = _plan_with_openai(
            num_scenes, characters, lyrics, style_desc, character_descriptions
        )
    else:
        scenes = _plan_fallback(num_scenes, characters, style_desc, character_descriptions)

    return scenes


# ---------------------------------------------------------------------------
# OpenAI-based scene planning
# ---------------------------------------------------------------------------

def _plan_with_openai(
    num_scenes: int,
    characters: list[str],
    lyrics: Optional[str],
    style_desc: str,
    char_descs: dict[str, str],
) -> list[dict]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    bryce_count = math.ceil(num_scenes * BRYCE_SCENE_FRACTION)
    secondary = [c for c in characters if c != "Bryce"]
    secondary_count = num_scenes - bryce_count

    # Build character description block for the prompt
    char_block = "\n".join(
        f"- {char_descs.get(c, c)}" for c in characters
    )

    arc_block = "\n".join(
        f"  Scenes {int(s * num_scenes) + 1}–{int(e * num_scenes)}: {label} – {desc}"
        for s, e, label, desc in _ARC
    )

    system = (
        "You are a cinematic AI music video director creating scene descriptions "
        f"for a Sora text-to-video generator.\n\n"
        f"CHARACTERS:\n{char_block}\n\n"
        f"EMOTIONAL ARC (follow strictly):\n{arc_block}\n\n"
        "ENGINE RULES (never break):\n"
        f"1. Bryce must appear in scenes 1–{bryce_count} (at least {bryce_count} of {num_scenes} scenes).\n"
        f"2. Secondary characters ({', '.join(secondary) if secondary else 'none'}) may appear "
        f"in up to {secondary_count} scenes.\n"
        "3. Never more than 3 visible faces in any frame.\n"
        "4. Elderly seniors appear as dignified background/support characters in several scenes.\n"
        "5. No lip sync, no text overlays, no logos, no titles.\n"
        f"6. Visual style: {style_desc}.\n\n"
        f"Write EXACTLY {num_scenes} scene descriptions.\n"
        "Each description: 1–2 cinematic sentences, max 30 words.\n"
        "Return ONLY a numbered list, one scene per line, no extra commentary."
    )

    user_parts = [f"Generate {num_scenes} scene descriptions."]
    if lyrics:
        user_parts.append(f"Song lyrics (use for emotional pacing):\n{lyrics[:3000]}")

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
        raw = resp.choices[0].message.content.strip()
        prompts = _parse_numbered_list(raw, num_scenes)
        return _build_scene_list(
            prompts, characters, style_desc, char_descs, num_scenes, bryce_count
        )
    except Exception:
        return _plan_fallback(num_scenes, characters, style_desc, char_descs)


def _build_scene_list(
    prompts: list[str],
    characters: list[str],
    style_desc: str,
    char_descs: dict[str, str],
    num_scenes: int,
    bryce_count: int,
) -> list[dict]:
    secondary = [c for c in characters if c != "Bryce"]
    scenes: list[dict] = []
    for i, prompt in enumerate(prompts):
        if i < bryce_count:
            chars = ["Bryce"]
            char_hint = char_descs.get("Bryce", "Bryce")
        else:
            chars = secondary or ["Bryce"]
            char_hint = ", ".join(char_descs.get(c, c) for c in chars)

        arc_label = _arc_label_for(i, num_scenes)
        full_prompt = (
            f"{prompt} "
            f"Featuring {char_hint}. "
            f"{style_desc}. "
            "Elderly seniors as dignified background figures. "
            "No text overlays, no logos, no lip sync. "
            "Authentic and dignified."
        )
        scenes.append({
            "prompt": full_prompt,
            "duration": SCENE_DURATION_SECS,
            "characters": chars,
            "arc": arc_label,
        })
    return scenes


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_FALLBACK_ARCS = {
    "opening": [
        "Bryce sits quietly in a softly lit room, hands folded, morning light through a window.",
        "A close-up of Bryce's face, contemplative, gentle ambient noise.",
        "Bryce stands at a window looking out, soft golden morning light.",
    ],
    "middle": [
        "Bryce walks slowly through a community center hallway, nodding to passing elders.",
        "Bryce and an elderly woman share a warm, silent moment seated at a table.",
        "Wide shot of a garden courtyard, seniors in background, Bryce in foreground.",
        "Bryce gently assists an elder with a gentle touch on their shoulder.",
        "Elderly seniors gathered in soft afternoon light, Bryce among them.",
        "Bryce listening attentively, leaning forward with genuine care.",
    ],
    "lift": [
        "Bryce and an elder laugh softly together, a moment of shared joy.",
        "Wide shot, Bryce and a group of seniors walking outside in warm sunlight.",
        "Bryce smiles warmly, seniors in background, a sense of rising hope.",
    ],
    "ending": [
        "Bryce seated in quiet dignity, late afternoon golden light, peaceful.",
        "A wide, still shot – Bryce and elderly figures in a calm garden, fading light.",
        "Close-up of hands, young and old, resting together in quiet solidarity.",
    ],
}


def _plan_fallback(
    num_scenes: int,
    characters: list[str],
    style_desc: str,
    char_descs: dict[str, str],
) -> list[dict]:
    bryce_count = math.ceil(num_scenes * BRYCE_SCENE_FRACTION)
    bryce_desc = char_descs.get("Bryce", "Bryce")
    secondary = [c for c in characters if c != "Bryce"]

    scenes: list[dict] = []
    arc_pool: dict[str, int] = {k: 0 for k in _FALLBACK_ARCS}

    for i in range(num_scenes):
        arc_label = _arc_label_for(i, num_scenes)
        arc_templates = _FALLBACK_ARCS[arc_label]
        template = arc_templates[arc_pool[arc_label] % len(arc_templates)]
        arc_pool[arc_label] += 1

        if i < bryce_count:
            chars = ["Bryce"]
            char_hint = bryce_desc
        else:
            chars = secondary or ["Bryce"]
            char_hint = ", ".join(char_descs.get(c, c) for c in chars)

        full_prompt = (
            f"{template} "
            f"Featuring {char_hint}. "
            f"{style_desc}. "
            "Elderly seniors as dignified background figures. "
            "No text overlays, no logos, no lip sync."
        )
        scenes.append({
            "prompt": full_prompt,
            "duration": SCENE_DURATION_SECS,
            "characters": chars,
            "arc": arc_label,
        })
    return scenes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arc_label_for(scene_index: int, num_scenes: int) -> str:
    frac = scene_index / max(num_scenes - 1, 1)
    for start, end, label, _ in _ARC:
        if start <= frac < end:
            return label
    return "ending"


def _parse_numbered_list(raw: str, num_scenes: int) -> list[str]:
    lines = raw.splitlines()
    prompts: list[str] = []
    for line in lines:
        cleaned = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
        cleaned = re.sub(r"^\s*[-*]\s*", "", cleaned).strip()
        if cleaned:
            prompts.append(cleaned)
    fallback = "Bryce in a quiet moment, cinematic warm light, elderly figures in background."
    while len(prompts) < num_scenes:
        prompts.append(prompts[-1] if prompts else fallback)
    return prompts[:num_scenes]
