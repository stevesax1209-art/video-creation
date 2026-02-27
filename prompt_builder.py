"""
Prompt Builder: Generates consistent, varied prompts per 2.5-second segment
with hard identity locks and cinematic structure.

Inputs:
  - vision_text:   string describing overall narrative context
  - segments:      list of dicts from segmenter (i, start, end, duration)
  - style_preset:  "photorealistic" | "documentary" | "cinematic" | "digital_art"
  - characters:    dict of enabled characters:
                     {"bryce": True | "<description>", "brian": bool | str, ...}
                   Bryce is required; Brian and Carmen are optional.
  - song_metadata: optional dict (filename, bpm, mood_tags)
  - job_dir:       optional path; when given, prompts.json is written there

Output:
  List of prompt dicts, length == len(segments), each with:
    i, start, end, prompt, negative_prompt, character_set
"""

import json
import math
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Identity locks (MVP defaults – override by passing a description string)
# ---------------------------------------------------------------------------

DEFAULT_IDENTITY_LOCKS: dict[str, str] = {
    "bryce": (
        'middle-aged man, short salt-and-pepper hair, trimmed goatee, '
        'warm approachable expression, black jacket'
    ),
    "brian": (
        'man, builder appearance, strong presence, casual work clothing'
    ),
    "carmen": (
        'woman, warm personality, professional casual attire'
    ),
}

# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------

STYLE_PRESETS: dict[str, str] = {
    "documentary": (
        "documentary handheld cinematography, 35mm film grain, shallow depth of field, "
        "intimate and authentic, natural available light, slight organic camera movement"
    ),
    "photorealistic": (
        "photorealistic rendering, natural lighting, lifelike textures, "
        "true-to-life color grading, high dynamic range"
    ),
    "cinematic": (
        "photorealistic cinematic, wide dynamic range, dramatic motivated lighting, "
        "steady professional camera, shallow depth of field"
    ),
    "digital_art": (
        "digital art style, stylized painterly rendering, vivid colors, "
        "crisp edges, illustrative aesthetic"
    ),
}

# ---------------------------------------------------------------------------
# Negative prompt – hard rules that appear in EVERY generated prompt
# ---------------------------------------------------------------------------

NEGATIVE_PROMPT_BASE = (
    "lip syncing, singing mouth performance, open mouth singing, "
    "on-screen text, captions, subtitles, logos, watermarks, title cards, "
    "morphing faces, face deformation, flickering, identity change, "
    "more than 3 faces in frame, blurry faces, text overlay"
)

# ---------------------------------------------------------------------------
# Scene variety: locations and actions (rotated deterministically by index)
# ---------------------------------------------------------------------------

_LOCATIONS = [
    "a warmly lit community center common room",
    "an outdoor courtyard with afternoon sunlight",
    "a quiet hallway with natural window light",
    "a comfortable reception area",
    "a sunlit garden terrace",
    "an intimate therapy room with soft ambient light",
    "a modern open-plan office space",
    "a cozy kitchen and dining area",
    "a landscaped outdoor walking path",
    "a small conference or meeting room",
]

_ACTIONS = [
    "sharing a quiet reflective moment",
    "in a warm coaching conversation",
    "reviewing documents together at a table",
    "walking side by side and talking",
    "greeting and welcoming an elder",
    "helping attentively with a task",
    "sitting across from each other in dialogue",
    "offering a gentle guiding hand on a shoulder",
    "standing together looking at something off-frame",
    "sharing a soft, genuine smile",
]

# ---------------------------------------------------------------------------
# Story arc phases
# (fraction_start, fraction_end, label, scene_description)
# ---------------------------------------------------------------------------

_ARC_PHASES = [
    (0.00, 0.20, "isolated_quiet",
     "intimate solitude, quiet contemplation, stillness"),
    (0.20, 0.60, "interactions_coaching",
     "genuine human connection, coaching warmth, mutual support"),
    (0.60, 1.00, "community_hopeful",
     "community togetherness, hopeful warmth, shared purpose"),
]

# ---------------------------------------------------------------------------
# Camera moves (rotated deterministically by index)
# ---------------------------------------------------------------------------

_CAMERA_MOVES = [
    "slow push-in, subtle handheld drift",
    "static wide shot, gentle rack focus",
    "smooth tracking shot left to right",
    "slow dolly forward, shallow depth of field",
    "medium close-up, gentle tilt up",
    "wide establishing shot, slow pan",
    "over-the-shoulder framing, slight handheld",
    "low-angle medium shot, slow zoom out",
    "tight two-shot, gentle dolly back",
    "aerial-style crane down to eye level",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_prompts(
    vision_text: str,
    segments: list[dict],
    style_preset: str,
    characters: dict,
    song_metadata: Optional[dict] = None,
    job_dir: Optional[str] = None,
) -> list[dict]:
    """
    Generate one prompt dict per segment.

    Args:
        vision_text:    Overall narrative context for the video.
        segments:       List of segment dicts from segmenter
                        (each must have keys: i, start, end).
        style_preset:   One of "photorealistic", "documentary",
                        "cinematic", "digital_art".
        characters:     Dict of enabled characters.  Keys are lowercase
                        names ("bryce", "brian", "carmen").  Values are
                        ``True`` to use the default identity lock or a
                        custom description string.
                        Bryce is required; "bryce" must be present and truthy.
        song_metadata:  Optional dict with keys: filename, bpm, mood_tags.
        job_dir:        Optional directory.  When given, ``prompts.json``
                        is written there so the run can be reproduced.

    Returns:
        List of prompt dicts of length ``len(segments)``, each containing::

            {
              "i":               int,
              "start":           float,
              "end":             float,
              "prompt":          str,
              "negative_prompt": str,
              "character_set":   list[str],
              "arc":             str,  # "isolated_quiet" | "interactions_coaching" | "community_hopeful"
            }

    Raises:
        ValueError: if "bryce" is not enabled in characters.
    """
    if not characters.get("bryce"):
        raise ValueError("Bryce is required and must be enabled in characters.")

    n = len(segments)
    if n == 0:
        return []

    style_desc = STYLE_PRESETS.get(style_preset, STYLE_PRESETS["cinematic"])

    # Build identity lock strings for each enabled character
    identity_locks = _build_identity_locks(characters)

    # Determine secondary characters and assign segment slots
    secondary_chars = [k for k in characters if k != "bryce" and characters.get(k)]
    bryce_slots, secondary_slots = _distribute_characters(n, secondary_chars)

    # Optional mood hint from song metadata
    mood_hint = _mood_hint(song_metadata)

    prompts: list[dict] = []
    prev_scene_brief = ""

    for seg in segments:
        i = seg["i"]
        start = seg["start"]
        end = seg["end"]
        frac = i / max(n - 1, 1)

        # Character set for this segment
        char_set = _get_character_set(i, bryce_slots, secondary_slots, secondary_chars)

        # ── Line 1: Identity lock ─────────────────────────────────────────
        lock_parts = [identity_locks[c] for c in char_set if c in identity_locks]
        lock_line = "; ".join(lock_parts) + "."

        # ── Line 2: Scene beat ────────────────────────────────────────────
        _arc_label, arc_desc = _arc_for_frac(frac)
        location = _LOCATIONS[i % len(_LOCATIONS)]
        action = _ACTIONS[i % len(_ACTIONS)]
        scene_line = (
            f"{arc_desc.capitalize()}. "
            f"{mood_hint}"
            f"{vision_text.strip('. ')+'. ' if vision_text else ''}"
            f"{action.capitalize()} in {location}."
        )

        # ── Line 3: Camera ────────────────────────────────────────────────
        camera_line = _CAMERA_MOVES[i % len(_CAMERA_MOVES)] + "."

        # ── Line 4: Lighting (cool → warm over timeline) ──────────────────
        lighting_line = _lighting_for_progress(frac) + "."

        # ── Line 5: Continuity ────────────────────────────────────────────
        if prev_scene_brief:
            continuity_line = f"Continuous scene from: {prev_scene_brief}."
        else:
            continuity_line = "Opening scene, establishing the space."

        # ── Assemble ──────────────────────────────────────────────────────
        prompt = (
            f"[IDENTITY] {lock_line} "
            f"[SCENE] {scene_line} "
            f"[CAMERA] {camera_line} "
            f"[LIGHTING] {lighting_line} "
            f"[CONTINUITY] {continuity_line} "
            f"[STYLE] {style_desc}. "
            "Elderly seniors as dignified background figures. "
            "No more than 3 faces in frame. "
            "Smooth realistic motion, stable faces, no morphing."
        )

        prev_scene_brief = f"{action} in {location}"

        prompts.append({
            "i": i,
            "start": start,
            "end": end,
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT_BASE,
            "character_set": char_set,
            "arc": _arc_label,
        })

    if job_dir is not None:
        _save_json(prompts, job_dir)

    return prompts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_identity_locks(characters: dict) -> dict[str, str]:
    """Return identity lock strings keyed by lowercase character name."""
    locks: dict[str, str] = {}
    for char_key, value in characters.items():
        if not value:
            continue
        name = char_key.capitalize()
        if isinstance(value, str) and len(value) > 5:
            desc = value
        else:
            desc = DEFAULT_IDENTITY_LOCKS.get(char_key, char_key)
        locks[char_key] = f'{name}: "{desc}"'
    return locks


def _distribute_characters(
    n: int,
    secondary_chars: list[str],
) -> tuple[list[int], dict[str, list[int]]]:
    """
    Assign segment slots to Bryce and secondary characters.

    Rules:
    - Bryce appears in >= 70% of segments.
    - Each secondary character appears in 10–25% of segments (non-contiguous).

    Returns:
        bryce_slots:     sorted list of segment indices that include Bryce.
        secondary_slots: dict {char_name: sorted list of segment indices}.
    """
    secondary_slots: dict[str, list[int]] = {}
    all_secondary: set[int] = set()

    for idx, char in enumerate(secondary_chars):
        # Target ~17% of segments (midpoint of 10–25%)
        target = _clamp_count(n, lo=0.10, mid=0.17, hi=0.25)
        slots = _spread_slots(
            n=n,
            target=target,
            skip_first=True,    # Never the very first segment (arc opener)
            offset_factor=idx + 1,
            existing=all_secondary,
        )
        secondary_slots[char] = slots
        all_secondary.update(slots)

    # Bryce gets all segments not assigned exclusively to secondaries
    bryce_slots = [i for i in range(n) if i not in all_secondary]

    # Ensure Bryce meets the 70% floor
    bryce_needed = math.ceil(n * 0.70)
    if len(bryce_slots) < bryce_needed:
        extra_needed = bryce_needed - len(bryce_slots)
        # Add some secondary-only slots so Bryce co-appears
        for s_idx in sorted(all_secondary):
            if extra_needed <= 0:
                break
            bryce_slots.append(s_idx)
            extra_needed -= 1
        bryce_slots.sort()

    return bryce_slots, secondary_slots


def _clamp_count(n: int, lo: float, mid: float, hi: float) -> int:
    """Return a count clamped to [ceil(n*lo), floor(n*hi)], targeting round(n*mid)."""
    return max(math.ceil(n * lo), min(round(n * mid), math.floor(n * hi)))


def _spread_slots(
    n: int,
    target: int,
    skip_first: bool,
    offset_factor: int,
    existing: set,
) -> list[int]:
    """
    Return up to *target* evenly-spread segment indices from [start, n),
    avoiding indices in *existing*.
    """
    start = 1 if skip_first else 0
    available = [i for i in range(start, n) if i not in existing]

    if not available or target <= 0:
        return []

    target = min(target, len(available))
    step = max(1, len(available) // target)
    offset = (offset_factor * step // 2) % step

    slots: list[int] = []
    for j in range(target):
        pick = offset + j * step
        if pick < len(available):
            slots.append(available[pick])

    return sorted(slots)


def _get_character_set(
    i: int,
    bryce_slots: list[int],
    secondary_slots: dict[str, list[int]],
    secondary_chars: list[str],
) -> list[str]:
    """Return the list of character keys appearing in segment *i*."""
    chars: list[str] = []
    if i in bryce_slots:
        chars.append("bryce")
    for char in secondary_chars:
        if i in secondary_slots.get(char, []):
            chars.append(char)
    return chars if chars else ["bryce"]


def _arc_for_frac(frac: float) -> tuple[str, str]:
    """Return (arc_label, arc_description) for timeline fraction *frac*."""
    for s, e, label, desc in _ARC_PHASES:
        if s <= frac < e:
            return label, desc
    # Last phase catches frac == 1.0
    return _ARC_PHASES[-1][2], _ARC_PHASES[-1][3]


def _lighting_for_progress(frac: float) -> str:
    """Cool dawn → neutral day → warm golden hour over the timeline."""
    if frac < 0.25:
        return "cool blue-grey morning light, soft diffused shadows"
    if frac < 0.50:
        return "neutral midday light, even soft-box quality illumination"
    if frac < 0.75:
        return "warm afternoon light, gentle golden undertones"
    return "warm golden-hour light, rich amber tones, deep soft shadows"


def _mood_hint(song_metadata: Optional[dict]) -> str:
    """Build a short mood prefix from optional song metadata."""
    if not song_metadata:
        return ""
    tags = song_metadata.get("mood_tags") or []
    if not tags:
        return ""
    return f"Mood: {', '.join(str(t) for t in tags)}. "


def _save_json(prompts: list[dict], job_dir: str) -> None:
    """Write prompts to ``<job_dir>/prompts.json``."""
    path = Path(job_dir) / "prompts.json"
    with open(path, "w") as fh:
        json.dump(prompts, fh, indent=2)
