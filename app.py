"""
Sora Cinematic Video Generator – Flask Application
===================================================
Accepts a text prompt together with model, resolution, and duration
preferences, submits a generation job to OpenAI Sora, polls for
completion, stores the resulting MP4 locally, and serves a download link.
"""

import os
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
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
GENERATED_FOLDER = BASE_DIR / "generated"

SORA_MODELS = ["sora-2", "sora-2-pro"]
SORA_RESOLUTIONS = ["1280x720", "720x1280"]
DEFAULT_MODEL = "sora-2"
DEFAULT_RESOLUTION = "1280x720"
SORA_DURATION_MIN = 1
SORA_DURATION_MAX = 20
SORA_DURATION_DEFAULT = 10

MAX_PROMPT_WORDS = 2500

GENERATED_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB – text only
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(24).hex())

# In-memory job store  {job_id: {"status": ..., "progress": ..., "message": ..., "output": ...}}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _job_update(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


# ---------------------------------------------------------------------------
# Background Sora worker
# ---------------------------------------------------------------------------

def _sora_worker(
    job_id: str,
    prompt: str,
    model: str,
    resolution: str,
    duration: int,
    output_path: Path,
):
    """Run in a background thread; submits to Sora and polls until complete."""
    try:
        _job_update(job_id, status="processing", progress=10, message="Connecting to Sora…")

        from sora_generator import create_sora_video

        def _progress(pct: int, msg: str):
            _job_update(job_id, progress=pct, message=msg)

        create_sora_video(
            prompt=prompt,
            output_path=str(output_path),
            model=model,
            resolution=resolution,
            duration=duration,
            progress_callback=_progress,
        )

        _job_update(
            job_id,
            status="complete",
            progress=100,
            message="Your cinematic video is ready!",
            output=output_path.name,
        )
    except Exception as exc:
        _job_update(job_id, status="error", progress=0, message=f"Error: {exc}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        sora_models=SORA_MODELS,
        sora_resolutions=SORA_RESOLUTIONS,
        default_model=DEFAULT_MODEL,
        duration_min=SORA_DURATION_MIN,
        duration_max=SORA_DURATION_MAX,
        duration_default=SORA_DURATION_DEFAULT,
        max_words=MAX_PROMPT_WORDS,
    )


@app.route("/generate", methods=["POST"])
def generate():
    """Validate the prompt and settings, then kick off a Sora generation job."""

    # --- Prompt ---
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Please provide a video prompt."}), 400
    if _word_count(prompt) > MAX_PROMPT_WORDS:
        return jsonify({"error": f"Prompt exceeds {MAX_PROMPT_WORDS} words."}), 400

    # --- Model ---
    model = request.form.get("model", "sora-2").strip()
    if model not in SORA_MODELS:
        return jsonify({"error": f"Invalid model. Choose from: {', '.join(SORA_MODELS)}."}), 400

    # --- Resolution ---
    resolution = request.form.get("resolution", "1280x720").strip()
    if resolution not in SORA_RESOLUTIONS:
        return jsonify({"error": f"Invalid resolution. Choose from: {', '.join(SORA_RESOLUTIONS)}."}), 400

    # --- Duration ---
    try:
        duration = int(request.form.get("duration", SORA_DURATION_DEFAULT))
    except (TypeError, ValueError):
        return jsonify({"error": "Duration must be an integer."}), 400
    if not (SORA_DURATION_MIN <= duration <= SORA_DURATION_MAX):
        return jsonify({"error": f"Duration must be between {SORA_DURATION_MIN} and {SORA_DURATION_MAX} seconds."}), 400

    # --- Register job and start background thread ---
    job_id = uuid.uuid4().hex
    output_path = GENERATED_FOLDER / f"{job_id}.mp4"

    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "progress": 0, "message": "Queued…", "output": None}

    thread = threading.Thread(
        target=_sora_worker,
        args=(job_id, prompt, model, resolution, duration, output_path),
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
        download_name="sora_video.mp4",
        mimetype="video/mp4",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
