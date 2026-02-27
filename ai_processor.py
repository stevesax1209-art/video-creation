"""
AI Processor: Uses OpenAI to generate descriptive scene prompts
from the user's vision description and image context.
Falls back to rule-based prompts if no API key is configured.
"""

import os
import re
import textwrap

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


def _get_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or not _openai_available:
        return None
    return OpenAI(api_key=api_key)


def generate_scene_prompts(description: str, num_scenes: int) -> list[str]:
    """
    Given a user's vision description and the number of scenes (images),
    return a list of short scene captions/prompts (one per scene).
    """
    if num_scenes <= 0:
        return []

    client = _get_client()
    if client:
        return _generate_with_openai(client, description, num_scenes)
    return _generate_fallback(description, num_scenes)


def _generate_with_openai(client, description: str, num_scenes: int) -> list[str]:
    """Call OpenAI ChatCompletion to produce scene captions."""
    system_prompt = (
        "You are a creative music video director. "
        "Given the director's vision description, produce exactly {n} short, "
        "evocative scene captions (max 12 words each) that will appear as "
        "on-screen text overlays in a music video. "
        "Return ONLY the captions as a numbered list, one per line."
    ).format(n=num_scenes)

    user_prompt = (
        f"Vision description:\n{description}\n\n"
        f"Generate {num_scenes} scene captions for the music video."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.8,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_numbered_list(raw, num_scenes)
    except Exception:
        return _generate_fallback(description, num_scenes)


def _parse_numbered_list(raw: str, num_scenes: int) -> list[str]:
    """Parse a numbered list response into a plain list of strings."""
    lines = raw.splitlines()
    captions = []
    for line in lines:
        # Strip leading numbering like "1." "1)" "- " etc.
        cleaned = re.sub(r"^\s*[\d]+[.)]\s*", "", line).strip()
        cleaned = re.sub(r"^\s*[-*]\s*", "", cleaned).strip()
        if cleaned:
            captions.append(cleaned)
    # Pad or trim to exactly num_scenes
    while len(captions) < num_scenes:
        captions.append(captions[-1] if captions else "")
    return captions[:num_scenes]


def _generate_fallback(description: str, num_scenes: int) -> list[str]:
    """
    Simple rule-based fallback: split the description into roughly equal
    chunks and use the first sentence of each chunk as a caption.
    """
    # Split description into sentences
    sentences = re.split(r"(?<=[.!?])\s+", description.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [f"Scene {i + 1}" for i in range(num_scenes)]

    # Distribute sentences across scenes
    captions = []
    step = max(1, len(sentences) // num_scenes)
    for i in range(num_scenes):
        idx = min(i * step, len(sentences) - 1)
        caption = sentences[idx]
        # Trim to ~12 words for overlay legibility
        words = caption.split()
        if len(words) > 12:
            caption = " ".join(words[:12]) + "…"
        captions.append(caption)

    return captions
