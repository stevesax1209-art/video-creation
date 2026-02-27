"""
AI Music Video Generator – Flask Application (Phase 1 / Freebeat-style)
========================================================================
Freebeat-style pipeline:
  1. Upload MP3
  2. Analyze audio duration + energy per ~2.5-second segment
  3. GPT-4 Vision describes each locked character from reference photos
  4. Scene planner generates one prompt per segment (energy-aware, arc-driven)
  5. Sora generates a short clip (~5 s) per segment with identity QC + retry
  6. ffmpeg stitches clips + overlays original MP3
  7. Export final MP4
"""

import os
import shutil
import uuid
import threading
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    abort,
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER  = BASE_DIR / "uploads"
GENERATED_FOLDER = BASE_DIR / "generated"

ALLOWED_AUDIO = {"mp3"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "webp"}

MAX_AUDIO_SIZE_MB = 60
MAX_IMAGE_SIZE_MB = 10
MAX_REF_IMAGES_PER_CHAR = 10
MIN_REF_IMAGES_PER_CHAR = 3

CAST_OPTIONS = {
    "bryce":        "Bryce only",
    "bryce_brian":  "Bryce + Brian",
    "bryce_carmen": "Bryce + Carmen",
}

STYLE_OPTIONS = {
    "documentary": "Documentary Handheld",
    "cinematic":   "Photorealistic Cinematic",
}

SORA_MODELS   = ["sora-2", "sora-2-pro"]
DEFAULT_MODEL = "sora-2"

# Max characters (2) × max ref images each → ceiling for upload size calculation
_MAX_LOCKED_CHARS = 2

UPLOAD_FOLDER.mkdir(exist_ok=True)
GENERATED_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = (
    MAX_AUDIO_SIZE_MB + _MAX_LOCKED_CHARS * MAX_REF_IMAGES_PER_CHAR * MAX_IMAGE_SIZE_MB
) * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(24).hex())

# In-memory job store {job_id: {status, progress, message, output}}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename: str, allowed: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def _job_update(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _cleanup_job_dir(job_dir: Path):
    try:
        shutil.rmtree(job_dir, ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _music_video_worker(
    job_id: str,
    job_dir: Path,
    mp3_path: Path,
    cast_key: str,
    ref_image_paths: dict[str, list[Path]],
    lyrics: str,
    style_key: str,
    model: str,
):
    output_path = GENERATED_FOLDER / f"{job_id}.mp4"
    clips_dir   = job_dir / "clips"

    try:
        # ── Step 1: Analyze audio energy ─────────────────────────────────
        _job_update(job_id, status="processing", progress=3,
                    message="Analyzing audio energy and splitting into segments…")

        from audio_analyzer import analyze_audio
        audio_data = analyze_audio(str(mp3_path))
        segments   = audio_data["segments"]
        total_secs = audio_data["duration_secs"]

        _job_update(job_id, progress=8,
                    message=f"Audio analyzed – {len(segments)} segments over {total_secs:.0f}s")

        # ── Step 2: Describe characters ───────────────────────────────────
        _job_update(job_id, progress=10,
                    message="Describing characters from reference images…")

        from scene_planner import describe_character
        char_descs: dict[str, str] = {}
        for char_name, img_paths in ref_image_paths.items():
            char_descs[char_name] = describe_character(
                char_name, [str(p) for p in img_paths]
            )

        # ── Step 3: Plan scenes ───────────────────────────────────────────
        _job_update(job_id, progress=14,
                    message=f"Planning {len(segments)} cinematic scenes…")

        from scene_planner import plan_scenes
        scenes = plan_scenes(
            cast_key=cast_key,
            style_key=style_key,
            character_descriptions=char_descs,
            audio_segments=segments,
            lyrics=lyrics or None,
        )

        _job_update(job_id, progress=18,
                    message=f"Scene plan ready – generating {len(scenes)} clips…")

        # ── Step 4: Generate clips with QC ───────────────────────────────
        from sora_generator import generate_scene_clips

        def _clip_progress(pct: int, msg: str):
            _job_update(job_id, progress=pct, message=msg)

        saved_clips = generate_scene_clips(
            scenes=scenes,
            clips_dir=str(clips_dir),
            model=model,
            progress_callback=_clip_progress,
            char_descriptions=char_descs,
        )

        if not saved_clips:
            raise RuntimeError(
                "No clips were generated successfully. "
                "Check your OPENAI_API_KEY and Sora quota."
            )

        # ── Step 5: Stitch + audio ────────────────────────────────────────
        _job_update(job_id, progress=90,
                    message=f"Stitching {len(saved_clips)} clips and adding audio…")

        from video_stitcher import stitch_clips
        stitch_clips(
            clip_paths=saved_clips,
            audio_path=str(mp3_path),
            output_path=str(output_path),
        )

        _job_update(
            job_id,
            status="complete",
            progress=100,
            message=f"Your music video is ready! ({len(saved_clips)} scenes)",
            output=output_path.name,
        )

    except Exception as exc:
        _job_update(job_id, status="error", progress=0, message=f"Error: {exc}")

    finally:
        _cleanup_job_dir(job_dir)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        cast_options=CAST_OPTIONS,
        style_options=STYLE_OPTIONS,
        sora_models=SORA_MODELS,
        default_model=DEFAULT_MODEL,
        min_ref_images=MIN_REF_IMAGES_PER_CHAR,
    )


@app.route("/generate", methods=["POST"])
def generate():
    """Validate inputs and launch the background music-video worker."""

    # ── MP3 ───────────────────────────────────────────────────────────────
    audio_file = request.files.get("audio")
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "Please upload an MP3 file."}), 400
    if not _allowed_file(audio_file.filename, ALLOWED_AUDIO):
        return jsonify({"error": "Audio must be an MP3 file."}), 400

    # ── Cast ──────────────────────────────────────────────────────────────
    cast_key = request.form.get("cast", "").strip()
    if cast_key not in CAST_OPTIONS:
        return jsonify({"error": "Invalid cast selection."}), 400

    # ── Reference images ──────────────────────────────────────────────────
    from scene_planner import CAST_MAP
    characters = CAST_MAP[cast_key]

    ref_image_files: dict[str, list] = {}
    for char in characters:
        field = f"ref_{char.lower()}"
        files = [f for f in request.files.getlist(field) if f and f.filename]
        if any(not _allowed_file(f.filename, ALLOWED_IMAGE) for f in files):
            return jsonify({"error": f"Unsupported image format for {char}."}), 400
        if len(files) < MIN_REF_IMAGES_PER_CHAR:
            return jsonify({
                "error": (
                    f"Please upload at least {MIN_REF_IMAGES_PER_CHAR} "
                    f"reference images for {char}."
                )
            }), 400
        ref_image_files[char] = files[:MAX_REF_IMAGES_PER_CHAR]

    # ── Style + model ─────────────────────────────────────────────────────
    style_key = request.form.get("style", "cinematic").strip()
    if style_key not in STYLE_OPTIONS:
        return jsonify({"error": "Invalid style selection."}), 400

    model = request.form.get("model", DEFAULT_MODEL).strip()
    if model not in SORA_MODELS:
        return jsonify({"error": "Invalid model."}), 400

    # ── Lyrics (optional) ─────────────────────────────────────────────────
    lyrics = request.form.get("lyrics", "").strip()

    # ── Save uploads ──────────────────────────────────────────────────────
    job_id  = uuid.uuid4().hex
    job_dir = UPLOAD_FOLDER / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    mp3_path = job_dir / secure_filename(audio_file.filename)
    audio_file.save(str(mp3_path))

    saved_ref: dict[str, list[Path]] = {}
    for char, files in ref_image_files.items():
        char_dir = job_dir / f"ref_{char.lower()}"
        char_dir.mkdir(exist_ok=True)
        saved: list[Path] = []
        for i, img in enumerate(files):
            ext = img.filename.rsplit(".", 1)[-1].lower()
            p = char_dir / f"{i:02d}.{ext}"
            img.save(str(p))
            saved.append(p)
        saved_ref[char] = saved

    # ── Register + start worker ───────────────────────────────────────────
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "progress": 0,
            "message": "Queued…", "output": None,
        }

    threading.Thread(
        target=_music_video_worker,
        args=(job_id, job_dir, mp3_path, cast_key, saved_ref, lyrics, style_key, model),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id}), 202


@app.route("/status/<job_id>")
def status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id: str):
    # Validate job_id – hex string only, prevents path traversal
    if not all(c in "0123456789abcdef" for c in job_id) or len(job_id) != 32:
        abort(400)
    output_path = GENERATED_FOLDER / f"{job_id}.mp4"
    if not output_path.exists():
        abort(404)
    return send_file(
        str(output_path),
        as_attachment=True,
        download_name="music_video.mp4",
        mimetype="video/mp4",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
