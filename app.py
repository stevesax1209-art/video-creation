"""
AI Music Video Generator – Flask Application (Phase 1 / Freebeat-style)
========================================================================
Freebeat-style pipeline:
  1. Upload MP3
  2. Analyze audio duration + energy per ~2.5-second segment
  3. GPT-4 Vision describes each locked character from reference photos
  4. prompt_builder generates one structured prompt per segment with identity
     locks, story arc, scene variety, and hard negative-prompt rules
  5. Sora generates a short clip (~5 s) per segment with identity QC + retry
  6. ffmpeg stitches clips + overlays original MP3
  7. Export final MP4
  8. /review/<job_id> lets users flag bad clips for targeted regeneration
"""

import json
import os
import shutil
import subprocess
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
UPLOAD_FOLDER    = BASE_DIR / "uploads"
GENERATED_FOLDER = BASE_DIR / "generated"
REVIEW_FOLDER    = GENERATED_FOLDER  # review data lives under GENERATED_FOLDER/<job_id>/

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


def _extract_thumbnail(clip_path: str, thumb_path: str) -> bool:
    """Extract a single JPEG thumbnail from *clip_path* at 0.8 s offset."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", "0.8",
                "-i", clip_path,
                "-frames:v", "1",
                "-q:v", "5",
                thumb_path,
            ],
            capture_output=True, text=True, timeout=20,
        )
        return result.returncode == 0
    except Exception:
        return False


def _persist_review_data(
    job_id: str,
    clip_paths: list[str],
    scenes: list[dict],
    prompt_items: list[dict],
    char_descs: dict,
    mp3_path: Path,
    model: str,
    ref_image_paths: dict[str, list[str]] | None = None,
) -> Path:
    """
    Copy clips, audio, thumbnails, and reference images to a persistent review
    directory so they survive the upload-folder cleanup and can be accessed via
    the /review route.

    Returns the review directory path.
    """
    review_dir = GENERATED_FOLDER / job_id
    clips_dest  = review_dir / "clips"
    thumbs_dest = review_dir / "thumbs"
    refs_dest   = review_dir / "refs"
    clips_dest.mkdir(parents=True, exist_ok=True)
    thumbs_dest.mkdir(parents=True, exist_ok=True)
    refs_dest.mkdir(parents=True, exist_ok=True)

    # Copy the original MP3 so re-stitch can use it later
    mp3_dest = review_dir / mp3_path.name
    try:
        shutil.copy2(str(mp3_path), str(mp3_dest))
    except Exception:
        pass  # Re-stitch will gracefully skip audio if missing

    # Copy reference images so targeted regen can use them as a slide-show stub.
    # Directory names are lowercased for filesystem compatibility; the manifest
    # stores absolute paths so casing is not used for lookup.
    saved_ref_paths: dict[str, list[str]] = {}
    for char, paths in (ref_image_paths or {}).items():
        char_dir = refs_dest / char.lower()
        char_dir.mkdir(exist_ok=True)
        char_saved: list[str] = []
        for p in paths:
            try:
                dest_img = char_dir / Path(p).name
                shutil.copy2(p, str(dest_img))
                char_saved.append(str(dest_img))
            except Exception:
                pass
        if char_saved:
            saved_ref_paths[char] = char_saved

    clip_names: list[str] = []
    thumb_names: list[str] = []

    for i, src in enumerate(clip_paths):
        name = f"clip_{i:04d}.mp4"
        dest = str(clips_dest / name)
        try:
            shutil.copy2(src, dest)
            clip_names.append(name)
        except Exception:
            clip_names.append("")

        # Extract thumbnail
        thumb_name = f"thumb_{i:04d}.jpg"
        thumb_path = str(thumbs_dest / thumb_name)
        ok = _extract_thumbnail(src, thumb_path)
        thumb_names.append(thumb_name if ok else "")

    # Save review manifest
    manifest = {
        "job_id":       job_id,
        "model":        model,
        "char_descs":   char_descs,
        "mp3_name":     mp3_path.name,
        "clips":        clip_names,
        "thumbs":       thumb_names,
        "scenes":       scenes,
        "prompt_items": prompt_items,
        "ref_image_paths": saved_ref_paths,
    }
    with open(review_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    return review_dir


def _load_review_manifest(job_id: str) -> dict | None:
    """Load the review manifest for *job_id*, or return None if not found."""
    path = GENERATED_FOLDER / job_id / "manifest.json"
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


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

        # ── Step 3: Build per-segment prompts ────────────────────────────
        _job_update(job_id, progress=14,
                    message=f"Building {len(segments)} structured prompts with identity locks…")

        from scene_planner import CAST_MAP
        from prompt_builder import build_prompts
        characters_list = CAST_MAP.get(cast_key, ["Bryce"])
        # Use GPT descriptions if available, else default identity lock
        characters = {
            c.lower(): (char_descs.get(c) or True)
            for c in characters_list
        }
        # Normalize segments from audio_analyzer format to prompt_builder format
        norm_segs = [
            {
                "i":     seg.get("index", i),
                "start": seg.get("start_ms", i * 2500) / 1000.0,
                "end":   seg.get("end_ms", (i + 1) * 2500) / 1000.0,
            }
            for i, seg in enumerate(segments)
        ]
        prompt_items = build_prompts(
            vision_text=(
                lyrics or
                "A compassionate caregiver supporting elderly residents with dignity."
            ),
            segments=norm_segs,
            style_preset=style_key,
            characters=characters,
            job_dir=str(job_dir),
        )
        # Convert to scene dicts expected by generate_scene_clips()
        scenes = [
            {
                "prompt":     item["prompt"],
                "duration":   int(seg.get("clip_secs", 5)),
                "characters": [c.capitalize() for c in item["character_set"]],
                "arc":        item.get("arc", "middle"),
                "intensity":  seg.get("intensity", "medium"),
            }
            for seg, item in zip(segments, prompt_items)
        ]

        _job_update(job_id, progress=18,
                    message=f"Prompts ready – generating {len(scenes)} clips…")

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
            ref_image_paths={
                char: [str(p) for p in paths]
                for char, paths in ref_image_paths.items()
            },
        )

        if not saved_clips:
            raise RuntimeError(
                "No clips were generated successfully. "
                "Check your OPENAI_API_KEY and Sora quota."
            )

        # ── Step 4b: Persist clips + thumbnails for review ────────────────
        _job_update(job_id, progress=88,
                    message="Extracting review thumbnails…")
        review_dir = _persist_review_data(
            job_id=job_id,
            clip_paths=saved_clips,
            scenes=scenes,
            prompt_items=prompt_items,
            char_descs=char_descs,
            mp3_path=mp3_path,
            model=model,
            ref_image_paths={
                char: [str(p) for p in paths]
                for char, paths in ref_image_paths.items()
            },
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
            review_url=f"/review/{job_id}",
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


@app.route("/review/<job_id>")
def review(job_id: str):
    """Manual QC review page: shows one thumbnail per clip for flagging."""
    if not all(c in "0123456789abcdef" for c in job_id) or len(job_id) != 32:
        abort(400)
    manifest = _load_review_manifest(job_id)
    if manifest is None:
        abort(404)
    return render_template("review.html", manifest=manifest, job_id=job_id)


@app.route("/thumb/<job_id>/<int:idx>")
def thumb(job_id: str, idx: int):
    """Serve a clip thumbnail by index."""
    if not all(c in "0123456789abcdef" for c in job_id) or len(job_id) != 32:
        abort(400)
    manifest = _load_review_manifest(job_id)
    if manifest is None:
        abort(404)
    thumbs = manifest.get("thumbs", [])
    if idx < 0 or idx >= len(thumbs) or not thumbs[idx]:
        abort(404)
    thumb_path = GENERATED_FOLDER / job_id / "thumbs" / thumbs[idx]
    if not thumb_path.exists():
        abort(404)
    return send_file(str(thumb_path), mimetype="image/jpeg")


@app.route("/regen/<job_id>", methods=["POST"])
def regen(job_id: str):
    """
    Trigger targeted regeneration of specific clips.

    Expects JSON body: ``{"clip_indices": [2, 5, 11]}``

    Starts a background thread and returns 202 immediately.
    Poll ``/status/<job_id>`` for status updates.
    """
    if not all(c in "0123456789abcdef" for c in job_id) or len(job_id) != 32:
        abort(400)

    data = request.get_json(silent=True) or {}
    clip_indices = data.get("clip_indices", [])
    if not isinstance(clip_indices, list) or not clip_indices:
        return jsonify({"error": "clip_indices must be a non-empty list."}), 400

    manifest = _load_review_manifest(job_id)
    if manifest is None:
        return jsonify({"error": "Review data not found for this job."}), 404

    with _jobs_lock:
        job = _jobs.get(job_id)
        if job and job.get("status") == "regenerating":
            return jsonify({"error": "Regeneration already in progress."}), 409
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(
            status="regenerating", progress=0,
            message=f"Regenerating {len(clip_indices)} clip(s)…",
        )

    threading.Thread(
        target=_regen_worker,
        args=(job_id, clip_indices, manifest),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id, "regenerating": len(clip_indices)}), 202


# ---------------------------------------------------------------------------
# Targeted regeneration worker
# ---------------------------------------------------------------------------

def _regen_worker(job_id: str, clip_indices: list[int], manifest: dict):
    """Regenerate specific clips and re-stitch the final video."""
    review_dir  = GENERATED_FOLDER / job_id
    clips_dir   = review_dir / "clips"
    thumbs_dir  = review_dir / "thumbs"
    output_path = GENERATED_FOLDER / f"{job_id}.mp4"
    mp3_name    = manifest.get("mp3_name", "")

    try:
        scenes    = manifest["scenes"]
        clip_names = manifest["clips"]
        char_descs = manifest.get("char_descs", {})
        model      = manifest.get("model", "sora-2")

        total = len(clip_indices)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        client = None
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

        for step, idx in enumerate(clip_indices):
            pct = int((step / total) * 70)
            _job_update(job_id, progress=pct,
                        message=f"Regenerating clip {idx + 1} ({step + 1}/{total})…")

            if idx < 0 or idx >= len(scenes):
                continue

            clip_path = str(clips_dir / f"clip_{idx:04d}.mp4")
            scene     = scenes[idx]

            from sora_generator import _generate_with_qc
            if client:
                qc_char  = scene.get("characters", [None])[0]
                qc_desc  = char_descs.get(qc_char, "") if qc_char else ""
                ok = _generate_with_qc(
                    client=client,
                    prompt=scene["prompt"],
                    duration=int(scene.get("duration", 5)),
                    model=model,
                    resolution="1280x720",
                    clip_path=clip_path,
                    qc_char=qc_char,
                    qc_desc=qc_desc,
                )
            else:
                # Collect any persisted reference images for the slide-show stub
                regen_ref_images: list[str] = []
                for paths in manifest.get("ref_image_paths", {}).values():
                    regen_ref_images.extend(paths)
                from clip_provider import generate_clip as _stub
                _stub(
                    prompt=scene["prompt"],
                    duration=int(scene.get("duration", 5)),
                    output_path=clip_path,
                    ref_images=regen_ref_images if regen_ref_images else None,
                )
                ok = True

            if ok:
                # Refresh thumbnail
                thumb_path = str(thumbs_dir / f"thumb_{idx:04d}.jpg")
                _extract_thumbnail(clip_path, thumb_path)

        # Re-stitch with updated clips
        _job_update(job_id, progress=75, message="Re-stitching video with updated clips…")
        all_clips = [
            str(clips_dir / name)
            for name in clip_names
            if name and (clips_dir / name).exists()
        ]
        if not all_clips:
            raise RuntimeError("No clips available to stitch after regeneration.")

        # Locate the original MP3 from the upload folder (already moved to
        # generated/<job_id>/ during persist) or skip audio if gone.
        mp3_candidates = list((review_dir).glob("*.mp3"))
        if not mp3_candidates and mp3_name:
            mp3_candidates = [review_dir / mp3_name]

        from video_stitcher import stitch_clips
        mp3_file = mp3_candidates[0] if mp3_candidates and Path(mp3_candidates[0]).exists() else None
        if mp3_file:
            stitch_clips(
                clip_paths=all_clips,
                audio_path=str(mp3_file),
                output_path=str(output_path),
            )
        else:
            # No audio available – concatenate clips silently
            from video_stitcher import _run_ffmpeg
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
                concat_file = fh.name
                for clip in all_clips:
                    escaped = os.path.abspath(clip).replace("'", "'\\''")
                    fh.write(f"file '{escaped}'\n")
            try:
                _run_ffmpeg([
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_file,
                    "-c:v", "libx264", "-preset", "fast",
                    "-crf", "20", "-pix_fmt", "yuv420p",
                    "-an",
                    str(output_path),
                ])
            finally:
                Path(concat_file).unlink(missing_ok=True)

        _job_update(
            job_id,
            status="complete",
            progress=100,
            message=f"Regeneration complete – {len(clip_indices)} clip(s) updated.",
            output=output_path.name,
            review_url=f"/review/{job_id}",
        )

    except Exception as exc:
        _job_update(job_id, status="error", progress=0,
                    message=f"Regeneration error: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
