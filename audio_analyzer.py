"""
Audio Analyzer: Splits an MP3 into ~2.5-second segments and measures the
RMS energy of each segment so that the scene planner can match visual
intensity to the music's dynamics.

Uses pydub (which wraps ffmpeg) for decoding.  Falls back gracefully to
evenly-distributed dummy segments if pydub is unavailable.
"""

import math
from typing import Optional

try:
    from pydub import AudioSegment as _AudioSegment
    _PYDUB_OK = True
except Exception:
    _PYDUB_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEGMENT_MS = 2500          # target segment length in milliseconds
CLIP_DURATION_SECS = 5     # Sora clip duration per segment (≥ Sora minimum)
FALLBACK_DURATION_SECS = 210.0  # ~3.5 minutes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_audio(mp3_path: str) -> dict:
    """
    Analyze an MP3 and return a dict with:
      - duration_secs:  float  – total audio length
      - segments:       list   – one dict per ~2.5-second window:
            {
              index:      int,
              start_ms:   int,
              end_ms:     int,
              duration_ms:int,
              rms:        float,   # raw RMS amplitude
              intensity:  str,     # "low" | "medium" | "high"
              clip_secs:  int,     # Sora clip duration to generate
            }

    All segment durations are normalised so that consecutive clips cover
    the whole audio track without gaps.
    """
    if not _PYDUB_OK:
        return _dummy_analysis(FALLBACK_DURATION_SECS)

    try:
        audio = _AudioSegment.from_mp3(mp3_path)
    except Exception:
        return _dummy_analysis(FALLBACK_DURATION_SECS)

    total_ms = len(audio)
    total_secs = total_ms / 1000.0

    # Split into windows
    windows: list[_AudioSegment] = []
    boundaries: list[tuple[int, int]] = []
    pos = 0
    while pos < total_ms:
        end = min(pos + SEGMENT_MS, total_ms)
        windows.append(audio[pos:end])
        boundaries.append((pos, end))
        pos = end

    # Compute RMS for each window
    rms_values = [w.rms for w in windows]

    # Derive intensity thresholds from the distribution.
    # Guard against very short audio files that produce fewer than 3 segments.
    sorted_rms = sorted(rms_values)
    n = len(sorted_rms)
    if n < 3:
        low_thresh  = sorted_rms[0]
        high_thresh = sorted_rms[-1]
    else:
        low_thresh  = sorted_rms[n // 3]
        high_thresh = sorted_rms[(2 * n) // 3]

    segments: list[dict] = []
    for i, (rms, (start_ms, end_ms)) in enumerate(zip(rms_values, boundaries)):
        duration_ms = end_ms - start_ms
        intensity = _classify(rms, low_thresh, high_thresh)
        clip_secs = _clip_secs_for_intensity(intensity)
        segments.append({
            "index":       i,
            "start_ms":    start_ms,
            "end_ms":      end_ms,
            "duration_ms": duration_ms,
            "rms":         float(rms),
            "intensity":   intensity,
            "clip_secs":   clip_secs,
        })

    return {
        "duration_secs": total_secs,
        "segments":       segments,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify(rms: float, low_thresh: float, high_thresh: float) -> str:
    if rms < low_thresh:
        return "low"
    if rms >= high_thresh:
        return "high"
    return "medium"


def _clip_secs_for_intensity(intensity: str) -> int:
    """
    Map energy intensity to clip duration.
    Higher energy → shorter, punchier cuts (5 s).
    Lower energy  → slightly longer, slower clips (7 s).
    """
    return {
        "low":    7,
        "medium": 5,
        "high":   5,
    }.get(intensity, 5)


def _dummy_analysis(total_secs: float) -> dict:
    """Return evenly spaced dummy segments when pydub is unavailable."""
    total_ms = int(total_secs * 1000)
    segments: list[dict] = []
    pos = 0
    i = 0
    while pos < total_ms:
        end = min(pos + SEGMENT_MS, total_ms)
        segments.append({
            "index":       i,
            "start_ms":    pos,
            "end_ms":      end,
            "duration_ms": end - pos,
            "rms":         0.0,
            "intensity":   "medium",
            "clip_secs":   CLIP_DURATION_SECS,
        })
        pos = end
        i += 1
    return {"duration_secs": total_secs, "segments": segments}
