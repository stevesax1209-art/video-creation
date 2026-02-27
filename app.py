"""
Music Video Generator – Flask Application
==========================================
Allows users to upload an MP3 audio track, up to 6 images, and a
2500-word max description of their vision.  AI-generated scene
captions are applied as on-screen overlays and the result is an
MP4 music video available for download.
"""

import os
import uuid
import threading
import json
import time
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
UPLOAD_FOLDER = BASE_DIR / "uploads"
GENERATED_FOLDER = BASE_DIR / "generated"

ALLOWED_AUDIO = {"mp3"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "gif", "webp"}

MAX_IMAGES = 6
MAX_DESCRIPTION_WORDS = 2500
MAX_AUDIO_SIZE_MB = 50
MAX_IMAGE_SIZE_MB = 10

UPLOAD_FOLDER.mkdir(exist_ok=True)
GENERATED_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = (MAX_AUDIO_SIZE_MB + MAX_IMAGES * MAX_IMAGE_SIZE_MB) * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(24).hex())

# In-memory job store  {job_id: {"status": ..., "progress": ..., "message": ..., "output": ...}}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename: str, allowed: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def _word_count(text: str) -> int:
    return len(text.split())


def _job_update(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _cleanup_uploads(paths: list[Path]):
    """Remove temporary upload files."""
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Background video generation worker
# ---------------------------------------------------------------------------

def _generate_worker(
    job_id: str,
    audio_path: Path,
    image_paths: list[Path],
    description: str,
    output_path: Path,
):
    """Run in a background thread; updates job status as it progresses."""
    try:
        _job_update(job_id, status="processing", progress=10, message="Analyzing your vision with AI…")

        from ai_processor import generate_scene_prompts
        captions = generate_scene_prompts(description, len(image_paths))

        _job_update(job_id, progress=30, message="Building video scenes…")

        from video_generator import generate_video
        generate_video(
            audio_path=str(audio_path),
            image_paths=[str(p) for p in image_paths],
            captions=captions,
            output_path=str(output_path),
        )

        _job_update(
            job_id,
            status="complete",
            progress=100,
            message="Your music video is ready!",
            output=output_path.name,
        )
    except Exception as exc:
        _job_update(job_id, status="error", progress=0, message=f"Error: {exc}")
    finally:
        _cleanup_uploads([audio_path] + image_paths)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", max_images=MAX_IMAGES, max_words=MAX_DESCRIPTION_WORDS)


@app.route("/generate", methods=["POST"])
def generate():
    """Accept uploads and kick off background video generation."""

    # --- Validate audio ---
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400
    audio_file = request.files["audio"]
    if not audio_file.filename or not _allowed_file(audio_file.filename, ALLOWED_AUDIO):
        return jsonify({"error": "Please upload a valid MP3 file."}), 400

    # --- Validate images ---
    image_files = request.files.getlist("images")
    image_files = [f for f in image_files if f and f.filename]
    if not image_files:
        return jsonify({"error": "Please upload at least one image."}), 400
    if len(image_files) > MAX_IMAGES:
        return jsonify({"error": f"Maximum {MAX_IMAGES} images allowed."}), 400
    for img in image_files:
        if not _allowed_file(img.filename, ALLOWED_IMAGE):
            return jsonify({"error": f"Unsupported image format: {img.filename}"}), 400

    # --- Validate description ---
    description = request.form.get("description", "").strip()
    if not description:
        return jsonify({"error": "Please provide a vision description."}), 400
    if _word_count(description) > MAX_DESCRIPTION_WORDS:
        return jsonify({"error": f"Description exceeds {MAX_DESCRIPTION_WORDS} words."}), 400

    # --- Save files ---
    job_id = uuid.uuid4().hex
    job_dir = UPLOAD_FOLDER / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    audio_path = job_dir / secure_filename(audio_file.filename)
    audio_file.save(str(audio_path))

    saved_images: list[Path] = []
    for i, img_file in enumerate(image_files):
        ext = img_file.filename.rsplit(".", 1)[-1].lower()
        img_path = job_dir / f"image_{i:02d}.{ext}"
        img_file.save(str(img_path))
        saved_images.append(img_path)

    output_path = GENERATED_FOLDER / f"{job_id}.mp4"

    # --- Register job and start background thread ---
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "progress": 0, "message": "Queued…", "output": None}

    thread = threading.Thread(
        target=_generate_worker,
        args=(job_id, audio_path, saved_images, description, output_path),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id}), 202


@app.route("/status/<job_id>")
def status(job_id: str):
    """Poll for job status."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id: str):
    """Download the generated MP4."""
    # Validate job_id is a hex string to prevent path traversal
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
