/**
 * main.js – AI Music Video Generator (Phase 1 / Freebeat-style)
 *
 * Handles:
 *  - MP3 drag-and-drop / click-to-select
 *  - Cast radio cards driving dynamic reference-image sections
 *  - Per-character drag-and-drop image upload with thumbnail previews
 *  - Style radio cards
 *  - Form submit via Fetch API (multipart)
 *  - Progress polling every 3 s with live message
 *  - Result / error display
 */

"use strict";

const MIN_REF_IMAGES  = 3;
const POLL_INTERVAL   = 3000;   // ms

// ── DOM refs ──────────────────────────────────────────────────────────────
const form       = document.getElementById("upload-form");
const submitBtn  = document.getElementById("submit-btn");
const audioInput = document.getElementById("audio-input");
const audioDrop  = document.getElementById("audio-drop");
const audioLabel = document.getElementById("audio-label");

const progressPanel = document.getElementById("progress-panel");
const progressBar   = document.getElementById("progress-bar");
const progressMsg   = document.getElementById("progress-message");
const resultPanel   = document.getElementById("result-panel");
const downloadLink  = document.getElementById("download-link");
const startOverBtn  = document.getElementById("start-over-btn");
const errorPanel    = document.getElementById("error-panel");
const errorMessage  = document.getElementById("error-message");
const retryBtn      = document.getElementById("retry-btn");

// ── Per-character state ───────────────────────────────────────────────────
const CHARS = ["bryce", "brian", "carmen"];
/** @type {Record<string, File[]>} */
const refImages = { bryce: [], brian: [], carmen: [] };

// ── Audio drop-zone ───────────────────────────────────────────────────────
setupDropZone(audioDrop, audioInput, {
  onFiles(files) {
    const mp3 = Array.from(files).find(f => /\.mp3$/i.test(f.name));
    if (!mp3) { alert("Please select an MP3 file."); return; }
    const dt = new DataTransfer();
    dt.items.add(mp3);
    audioInput.files = dt.files;
    setAudioUI(mp3.name);
  },
});
audioInput.addEventListener("change", () => {
  if (audioInput.files.length) setAudioUI(audioInput.files[0].name);
});
function setAudioUI(name) {
  audioDrop.classList.add("has-file");
  audioLabel.textContent = "✅ " + name;
}

// ── Cast radio cards ──────────────────────────────────────────────────────
document.querySelectorAll('input[name="cast"]').forEach(r => r.addEventListener("change", updateCast));
function updateCast() {
  const val = document.querySelector('input[name="cast"]:checked')?.value ?? "bryce";
  document.getElementById("ref-brian").classList.toggle("hidden",  !val.includes("brian"));
  document.getElementById("ref-carmen").classList.toggle("hidden", !val.includes("carmen"));
  syncRadioCards("cast-cards");
}

// ── Style radio cards ─────────────────────────────────────────────────────
document.querySelectorAll('input[name="style"]').forEach(r => r.addEventListener("change", () => syncRadioCards("style-cards")));

function syncRadioCards(containerId) {
  document.querySelectorAll(`#${containerId} .radio-card`).forEach(card => {
    card.classList.toggle("selected", card.querySelector("input").checked);
  });
}

// Initialise radio card visuals and cast section visibility on page load
updateCast();
syncRadioCards("style-cards");


CHARS.forEach(char => {
  const dropEl  = document.getElementById(`${char}-drop`);
  const inputEl = document.getElementById(`${char}-input`);
  if (!dropEl || !inputEl) return;

  setupDropZone(dropEl, inputEl, {
    onFiles(files) { addRefImages(char, Array.from(files).filter(isImage)); },
  });
  inputEl.addEventListener("change", () => {
    if (inputEl.files.length) {
      addRefImages(char, Array.from(inputEl.files).filter(isImage));
      inputEl.value = "";
    }
  });
});

function isImage(f) { return /\.(jpe?g|png|webp)$/i.test(f.name); }

function addRefImages(char, files) {
  refImages[char].push(...files);
  renderRefPreviews(char);
}

function renderRefPreviews(char) {
  const grid    = document.getElementById(`${char}-preview`);
  const dropEl  = document.getElementById(`${char}-drop`);
  const labelEl = document.getElementById(`${char}-label`);
  const countEl = document.getElementById(`${char}-count`);
  if (!grid) return;

  grid.innerHTML = "";
  refImages[char].forEach((file, idx) => {
    const thumb = document.createElement("div");
    thumb.className = "preview-thumb";

    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.alt = file.name;

    const rmBtn = document.createElement("button");
    rmBtn.className = "remove-btn";
    rmBtn.type = "button";
    rmBtn.textContent = "×";
    rmBtn.setAttribute("aria-label", "Remove " + file.name);
    rmBtn.addEventListener("click", () => {
      refImages[char].splice(idx, 1);
      renderRefPreviews(char);
    });

    thumb.appendChild(img);
    thumb.appendChild(rmBtn);
    grid.appendChild(thumb);
  });

  const n = refImages[char].length;
  const ok = n >= MIN_REF_IMAGES;
  dropEl.classList.toggle("has-file", ok);
  labelEl.textContent = n > 0 ? `✅ ${n} photo${n !== 1 ? "s" : ""} selected` : "Click or drag photos here";
  if (countEl) {
    countEl.textContent = `${n} / ${MIN_REF_IMAGES}+ photos`;
    countEl.style.color = ok ? "#3ecf8e" : "var(--accent)";
  }
}

// ── Form submit ───────────────────────────────────────────────────────────
form.addEventListener("submit", async e => {
  e.preventDefault();

  if (!audioInput.files.length) { alert("Please upload an MP3 file."); return; }

  const castVal = document.querySelector('input[name="cast"]:checked')?.value ?? "bryce";
  const requiredChars = ["bryce"];
  if (castVal.includes("brian"))  requiredChars.push("brian");
  if (castVal.includes("carmen")) requiredChars.push("carmen");

  for (const char of requiredChars) {
    if (refImages[char].length < MIN_REF_IMAGES) {
      const name = char.charAt(0).toUpperCase() + char.slice(1);
      alert(`Please upload at least ${MIN_REF_IMAGES} reference photos for ${name}.`);
      return;
    }
  }

  submitBtn.disabled = true;

  const fd = new FormData();
  fd.append("audio", audioInput.files[0]);
  fd.append("cast",  castVal);
  fd.append("style", document.querySelector('input[name="style"]:checked')?.value ?? "cinematic");
  fd.append("model", document.getElementById("model-select").value);
  const lyrics = (document.getElementById("lyrics")?.value ?? "").trim();
  if (lyrics) fd.append("lyrics", lyrics);
  for (const char of requiredChars) refImages[char].forEach(img => fd.append(`ref_${char}`, img));

  try {
    const res  = await fetch("/generate", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) { showError(data.error || "Request failed."); submitBtn.disabled = false; return; }
    startPolling(data.job_id);
  } catch (err) {
    showError("Network error: " + err.message);
    submitBtn.disabled = false;
  }
});

// ── Polling ───────────────────────────────────────────────────────────────
let pollTimer = null;

function startPolling(jobId) {
  form.classList.add("hidden");
  document.querySelector(".submit-row")?.classList.add("hidden");
  progressPanel.classList.remove("hidden");
  setProgress(0, "Starting…");

  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/status/${jobId}`);
      if (!res.ok) return;
      const job = await res.json();
      setProgress(job.progress, job.message);
      if (job.status === "complete") { clearPoll(); showResult(jobId, job.review_url); }
      else if (job.status === "error") { clearPoll(); showError(job.message); }
    } catch (_) {}
  }, POLL_INTERVAL);
}

function clearPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }
function setProgress(pct, msg) {
  progressBar.style.width = `${pct}%`;
  progressMsg.textContent = msg ?? "";
}

// ── Result / error ────────────────────────────────────────────────────────
const reviewLink = document.getElementById("review-link");

function showResult(jobId, reviewUrl) {
  downloadLink.href = `/download/${jobId}`;
  if (reviewLink) {
    if (reviewUrl) {
      reviewLink.href = reviewUrl;
      reviewLink.style.display = "";
    } else {
      reviewLink.style.display = "none";
    }
  }
  progressPanel.classList.add("hidden");
  resultPanel.classList.remove("hidden");
}
function showError(msg) {
  errorMessage.textContent = msg;
  progressPanel.classList.add("hidden");
  errorPanel.classList.remove("hidden");
}

startOverBtn.addEventListener("click", () => location.reload());
retryBtn.addEventListener("click", () => {
  clearPoll();
  form.classList.remove("hidden");
  [progressPanel, resultPanel, errorPanel].forEach(p => p.classList.add("hidden"));
  submitBtn.disabled = false;
});

// ── Drop-zone helper ──────────────────────────────────────────────────────
function setupDropZone(zone, input, { onFiles }) {
  zone.addEventListener("click",    e  => { if (!e.target.closest(".remove-btn")) input.click(); });
  zone.addEventListener("dragover", e  => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave",()  => zone.classList.remove("drag-over"));
  zone.addEventListener("drop",     e  => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files);
  });
}
