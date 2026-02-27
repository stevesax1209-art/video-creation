"""
Video Generator: Combines uploaded images and an MP3 audio track
into an MP4 music video with Ken Burns (zoom/pan) effects and
AI-generated text overlays.
"""

import os
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

# Output resolution
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_FPS = 24
OUTPUT_BITRATE = "4000k"

# Ken Burns zoom range (1.0 = original, 1.15 = 15% zoom)
ZOOM_START = 1.0
ZOOM_END = 1.15

# Text overlay settings
FONT_SIZE = 40
CAPTION_DISPLAY_DURATION_FRACTION = 0.6  # Show caption for first 60% of scene
CAPTION_DISPLAY_FRACTION = CAPTION_DISPLAY_DURATION_FRACTION


def _resize_and_crop(pil_img: Image.Image, width: int, height: int) -> np.ndarray:
    """Resize image to fill target dimensions (cover), then center-crop."""
    img_ratio = pil_img.width / pil_img.height
    target_ratio = width / height

    if img_ratio > target_ratio:
        # Image is wider than target: fit by height
        new_h = height
        new_w = int(pil_img.width * height / pil_img.height)
    else:
        # Image is taller than target: fit by width
        new_w = width
        new_h = int(pil_img.height * width / pil_img.width)

    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    pil_img = pil_img.crop((left, top, left + width, top + height))

    return np.array(pil_img.convert("RGB"))


def _ken_burns_frame(base_frame: np.ndarray, t: float, duration: float) -> np.ndarray:
    """Apply a Ken Burns zoom-in effect: zoom from ZOOM_START to ZOOM_END."""
    h, w = base_frame.shape[:2]
    progress = t / duration if duration > 0 else 0
    zoom = ZOOM_START + (ZOOM_END - ZOOM_START) * progress

    # Compute cropped region
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)
    x_start = (w - crop_w) // 2
    y_start = (h - crop_h) // 2

    cropped = base_frame[y_start : y_start + crop_h, x_start : x_start + crop_w]
    resized = np.array(
        Image.fromarray(cropped).resize((w, h), Image.BILINEAR)
    )
    return resized


def _make_scene_clip(
    image_path: str,
    duration: float,
    caption: str,
) -> CompositeVideoClip:
    """Create a single scene clip with Ken Burns effect and optional caption."""
    pil_img = Image.open(image_path).convert("RGB")
    base_frame = _resize_and_crop(pil_img, OUTPUT_WIDTH, OUTPUT_HEIGHT)

    video_clip = ImageClip(base_frame, duration=duration).fl(
        lambda gf, t: _ken_burns_frame(base_frame, t, duration),
    )
    video_clip = video_clip.set_fps(OUTPUT_FPS)

    layers = [video_clip]

    # Add caption overlay if provided
    if caption and caption.strip():
        caption_duration = duration * CAPTION_DISPLAY_FRACTION
        try:
            txt_clip = (
                TextClip(
                    caption,
                    fontsize=FONT_SIZE,
                    color="white",
                    stroke_color="black",
                    stroke_width=2,
                    method="caption",
                    size=(OUTPUT_WIDTH - 80, None),
                    align="center",
                )
                .set_position(("center", OUTPUT_HEIGHT - 120))
                .set_start(0)
                .set_duration(caption_duration)
                .crossfadeout(0.5)
            )
            layers.append(txt_clip)
        except Exception:
            # TextClip may fail if ImageMagick is unavailable – skip caption gracefully
            pass

    composite = CompositeVideoClip(layers, size=(OUTPUT_WIDTH, OUTPUT_HEIGHT))
    composite = composite.set_duration(duration)
    return composite


def generate_video(
    audio_path: str,
    image_paths: list[str],
    captions: list[str],
    output_path: str,
) -> str:
    """
    Generate an MP4 music video.

    Args:
        audio_path:  Path to the input MP3 file.
        image_paths: List of image file paths (1–6 images).
        captions:    AI-generated captions, one per image.
        output_path: Destination path for the output MP4.

    Returns:
        The output_path on success.
    """
    if not image_paths:
        raise ValueError("At least one image is required.")

    audio_clip = AudioFileClip(audio_path)
    total_duration = audio_clip.duration

    # Distribute duration equally across all scenes
    scene_duration = total_duration / len(image_paths)

    scene_clips = []
    for i, img_path in enumerate(image_paths):
        caption = captions[i] if i < len(captions) else ""
        clip = _make_scene_clip(img_path, scene_duration, caption)
        scene_clips.append(clip)

    # Concatenate all scenes
    final_video = concatenate_videoclips(scene_clips, method="compose")
    final_video = final_video.set_audio(audio_clip)

    final_video.write_videofile(
        output_path,
        fps=OUTPUT_FPS,
        codec="libx264",
        audio_codec="aac",
        bitrate=OUTPUT_BITRATE,
        temp_audiofile=output_path + ".temp_audio.m4a",
        remove_temp=True,
        logger=None,
    )

    audio_clip.close()
    final_video.close()

    return output_path
