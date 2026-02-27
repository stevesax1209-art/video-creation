/**
 * main.js – client-side logic for the Music Video Generator
 *
 * Handles:
 *  - Drag-and-drop + click-to-select for audio and images
 *  - Image preview grid with remove buttons
 *  - Word-count live update with limit enforcement
 *  - Form submission via Fetch API
 *  - Progress polling and result/error display
 */

"use strict";

// ---------------------------------------------------------------------------
// Constants (must match server-side values)
// ---------------------------------------------------------------------------
const MAX_IMAGES = 6;
const MAX_DESCRIPTION_WORDS = 2500;
const POLL_INTERVAL_MS = 2000;

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const form           = document.getElementById("upload-form");
const audioInput     = document.getElementById("audio-input");
const audioDrop      = document.getElementById("audio-drop");
const audioLabel     = document.getElementById("audio-label");
const imagesInput    = document.getElementById("images-input");
const imagesDrop     = document.getElementById("images-drop");
const imagesLabel    = document.getElementById("images-label");
const previewGrid    = document.getElementById("preview-grid");
const descTextarea   = document.getElementById("description");
const wordCountEl    = document.getElementById("word-count");
const wordCounter    = document.querySelector(".word-counter");
const submitBtn      = document.getElementById("submit-btn");

const mainContent    = document.getElementById("main-content");
const progressPanel  = document.getElementById("progress-panel");
const progressBar    = document.getElementById("progress-bar");
const progressMsg    = document.getElementById("progress-message");

const resultPanel    = document.getElementById("result-panel");
const downloadLink   = document.getElementById("download-link");
const startOverBtn   = document.getElementById("start-over-btn");

const errorPanel     = document.getElementById("error-panel");
const errorMessage   = document.getElementById("error-message");
const retryBtn       = document.getElementById("retry-btn");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
/** @type {File[]} */
let selectedImages = [];
let pollTimer = null;

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------
function countWords(text) {
  return text.trim() === "" ? 0 : text.trim().split(/\s+/).length;
}

function showPanel(panel) {
  [progressPanel, resultPanel, errorPanel].forEach(p => p.classList.add("hidden"));
  form.classList.add("hidden");
  document.querySelector(".submit-row")?.classList.add("hidden");
  panel.classList.remove("hidden");
}

function resetUI() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  form.classList.remove("hidden");
  progressPanel.classList.add("hidden");
  resultPanel.classList.add("hidden");
  errorPanel.classList.add("hidden");
  submitBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// Audio drop-zone
// ---------------------------------------------------------------------------
setupDropZone(audioDrop, audioInput, {
  onFiles(files) {
    const mp3 = Array.from(files).find(f => f.name.toLowerCase().endsWith(".mp3"));
    if (!mp3) { alert("Please select an MP3 file."); return; }
    // Transfer to real input
    const dt = new DataTransfer();
    dt.items.add(mp3);
    audioInput.files = dt.files;
    audioDrop.classList.add("has-file");
    audioLabel.textContent = `✅ ${mp3.name}`;
  },
});

audioInput.addEventListener("change", () => {
  if (audioInput.files.length) {
    audioDrop.classList.add("has-file");
    audioLabel.textContent = `✅ ${audioInput.files[0].name}`;
  }
});

// ---------------------------------------------------------------------------
// Images drop-zone
// ---------------------------------------------------------------------------
setupDropZone(imagesDrop, imagesInput, {
  onFiles(files) {
    const imgs = Array.from(files).filter(f =>
      /\.(jpe?g|png|gif|webp)$/i.test(f.name)
    );
    if (!imgs.length) { alert("Please select JPG, PNG, GIF or WebP images."); return; }
    addImages(imgs);
  },
});

imagesInput.addEventListener("change", () => {
  if (imagesInput.files.length) {
    addImages(Array.from(imagesInput.files));
    // Reset input so same file can be re-added after removal
    imagesInput.value = "";
  }
});

function addImages(files) {
  for (const f of files) {
    if (selectedImages.length >= MAX_IMAGES) break;
    selectedImages.push(f);
  }
  renderPreviews();
}

function renderPreviews() {
  previewGrid.innerHTML = "";
  selectedImages.forEach((file, idx) => {
    const thumb = document.createElement("div");
    thumb.className = "preview-thumb";

    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.alt = file.name;

    const removeBtn = document.createElement("button");
    removeBtn.className = "remove-btn";
    removeBtn.type = "button";
    removeBtn.textContent = "×";
    removeBtn.setAttribute("aria-label", `Remove ${file.name}`);
    removeBtn.addEventListener("click", () => {
      selectedImages.splice(idx, 1);
      renderPreviews();
    });

    thumb.appendChild(img);
    thumb.appendChild(removeBtn);
    previewGrid.appendChild(thumb);
  });

  const count = selectedImages.length;
  if (count > 0) {
    imagesDrop.classList.add("has-file");
    imagesLabel.textContent = `✅ ${count} photo${count !== 1 ? "s" : ""} selected (max ${MAX_IMAGES})`;
  } else {
    imagesDrop.classList.remove("has-file");
    imagesLabel.textContent = `Click or drag & drop up to ${MAX_IMAGES} photos here`;
  }
}

// ---------------------------------------------------------------------------
// Word counter
// ---------------------------------------------------------------------------
descTextarea.addEventListener("input", updateWordCount);

function updateWordCount() {
  const wc = countWords(descTextarea.value);
  wordCountEl.textContent = wc;
  if (wc > MAX_DESCRIPTION_WORDS) {
    wordCounter.classList.add("over-limit");
  } else {
    wordCounter.classList.remove("over-limit");
  }
}

// ---------------------------------------------------------------------------
// Form submit
// ---------------------------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Client-side validation
  if (!audioInput.files.length) {
    alert("Please upload an MP3 file.");
    return;
  }
  if (selectedImages.length === 0) {
    alert("Please upload at least one photo.");
    return;
  }
  if (!descTextarea.value.trim()) {
    alert("Please describe your vision.");
    return;
  }
  if (countWords(descTextarea.value) > MAX_DESCRIPTION_WORDS) {
    alert(`Description must be ${MAX_DESCRIPTION_WORDS} words or fewer.`);
    return;
  }

  submitBtn.disabled = true;

  const fd = new FormData();
  fd.append("audio", audioInput.files[0]);
  selectedImages.forEach(img => fd.append("images", img));
  fd.append("description", descTextarea.value);

  try {
    const res = await fetch("/generate", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || "Upload failed. Please try again.");
      submitBtn.disabled = false;
      return;
    }
    startPolling(data.job_id);
  } catch (err) {
    showError("Network error: " + err.message);
    submitBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Progress polling
// ---------------------------------------------------------------------------
function startPolling(jobId) {
  // Show progress panel immediately
  showProgressPanel(0, "Starting…");
  form.classList.add("hidden");
  progressPanel.classList.remove("hidden");

  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/status/${jobId}`);
      if (!res.ok) { clearInterval(pollTimer); pollTimer = null; return; }
      const job = await res.json();

      updateProgress(job.progress, job.message);

      if (job.status === "complete") {
        clearInterval(pollTimer);
        pollTimer = null;
        showResult(jobId);
      } else if (job.status === "error") {
        clearInterval(pollTimer);
        pollTimer = null;
        showError(job.message);
      }
    } catch (_) {
      // Transient network error – keep polling
    }
  }, POLL_INTERVAL_MS);
}

function showProgressPanel(pct, msg) {
  progressPanel.classList.remove("hidden");
  updateProgress(pct, msg);
}

function updateProgress(pct, msg) {
  progressBar.style.width = `${pct}%`;
  progressMsg.textContent = msg || "";
}

// ---------------------------------------------------------------------------
// Result & error
// ---------------------------------------------------------------------------
function showResult(jobId) {
  downloadLink.href = `/download/${jobId}`;
  progressPanel.classList.add("hidden");
  resultPanel.classList.remove("hidden");
}

function showError(msg) {
  errorMessage.textContent = msg;
  progressPanel.classList.add("hidden");
  errorPanel.classList.remove("hidden");
}

startOverBtn.addEventListener("click", () => {
  location.reload();
});

retryBtn.addEventListener("click", () => {
  resetUI();
});

// ---------------------------------------------------------------------------
// Generic drop-zone setup
// ---------------------------------------------------------------------------
function setupDropZone(zone, input, { onFiles }) {
  zone.addEventListener("click", (e) => {
    if (e.target.closest(".remove-btn")) return;
    input.click();
  });

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });

  zone.addEventListener("dragleave", () => {
    zone.classList.remove("drag-over");
  });

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) {
      onFiles(e.dataTransfer.files);
    }
  });
}
