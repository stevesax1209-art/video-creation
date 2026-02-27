# 🎬 AI Music Video Generator

A web application that turns your original music, photos, and creative vision into an MP4 music video using AI-generated scene captions and Ken Burns-style animations.

![AI Music Video Generator UI](https://github.com/user-attachments/assets/7cb11d3f-f8ca-42a6-b373-6ab2765952a5)

## Features

- **Upload an MP3** — original or non-copyrighted music (up to 50 MB)
- **Upload up to 6 photos** — of yourself, friends, or any subjects (JPG, PNG, GIF, WebP)
- **Describe your vision** — up to 2,500 words; the app feeds this to AI to craft per-scene captions
- **AI-powered scene generation** — uses OpenAI GPT-3.5 (optional) to write evocative text overlays; falls back to sentence-extraction when no API key is provided
- **Ken Burns effect** — smooth zoom-in animation on every image for a cinematic feel
- **Background processing** — live progress bar while the video is being generated
- **MP4 download** — 1280×720 H.264 video with AAC audio

## Quick Start

### Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on `PATH`
- ImageMagick (optional — needed for text overlays; install via `apt install imagemagick` or `brew install imagemagick`)

### Installation

```bash
git clone https://github.com/stevesax1209-art/video-creation.git
cd video-creation
pip install -r requirements.txt
```

### Configuration (optional)

Create a `.env` file to enable OpenAI scene caption generation:

```
OPENAI_API_KEY=sk-...
SECRET_KEY=some-random-secret
PORT=5000
```

If `OPENAI_API_KEY` is omitted the app still works — it extracts captions directly from your description text.

### Run

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## How It Works

1. **Upload** your MP3, up to 6 photos, and your vision description.
2. **AI analysis** — the description is sent to OpenAI (or processed locally) to generate one scene caption per photo.
3. **Video assembly** — each photo becomes a scene whose duration equals `total_audio_length / number_of_photos`. A Ken Burns zoom is applied. Captions are overlaid as text.
4. **Export** — all scenes are concatenated and mixed with the original audio to produce a 1280×720 MP4.
5. **Download** — a download link appears once generation is complete.

## Project Structure

```
video-creation/
├── app.py               # Flask web server (routes, job queue)
├── ai_processor.py      # OpenAI / fallback caption generation
├── video_generator.py   # moviepy-based video assembly
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Single-page UI
├── static/
│   ├── css/style.css    # Dark-theme styling
│   └── js/main.js       # Upload, preview, progress polling
├── uploads/             # Temporary upload storage (auto-created)
└── generated/           # Output MP4 files (auto-created)
```

## Notes

- Only upload music you own or have explicit permission to use.
- Generated videos are stored in `generated/` and are not automatically deleted; manage disk space as needed.
- For production deployment use a WSGI server (e.g. `gunicorn app:app`) behind a reverse proxy.