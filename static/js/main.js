/**
 * main.js – client-side logic for the Sora Cinematic Video Generator
 *
 * Handles:
 *  - Word-count live update with limit enforcement
 *  - Form submission via Fetch API
 *  - Progress polling and result/error display
 */

"use strict";

// ---------------------------------------------------------------------------
// Constants (must match server-side values)
// ---------------------------------------------------------------------------
const MAX_PROMPT_WORDS = 2500;
const POLL_INTERVAL_MS = 3000;

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const form           = document.getElementById("upload-form");
const promptTextarea = document.getElementById("prompt");
const wordCountEl    = document.getElementById("word-count");
const wordCounter    = document.querySelector(".word-counter");
const submitBtn      = document.getElementById("submit-btn");

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
let pollTimer = null;

// ---------------------------------------------------------------------------
// Word counter
// ---------------------------------------------------------------------------
promptTextarea.addEventListener("input", updateWordCount);

function countWords(text) {
  return text.trim() === "" ? 0 : text.trim().split(/\s+/).length;
}

function updateWordCount() {
  const wc = countWords(promptTextarea.value);
  wordCountEl.textContent = wc;
  if (wc > MAX_DESCRIPTION_WORDS) {
    wordCounter.classList.add("over-limit");
  } else {
    wordCounter.classList.remove("over-limit");
  }
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function showProgressPanel(pct, msg) {
  form.classList.add("hidden");
  document.querySelector(".submit-row")?.classList.add("hidden");
  progressPanel.classList.remove("hidden");
  updateProgress(pct, msg);
}

function updateProgress(pct, msg) {
  progressBar.style.width = `${pct}%`;
  progressMsg.textContent = msg || "";
}

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
// Form submit
// ---------------------------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const prompt = promptTextarea.value.trim();
  if (!prompt) {
    alert("Please describe your video.");
    return;
  }
  if (countWords(prompt) > MAX_PROMPT_WORDS) {
    alert(`Prompt must be ${MAX_PROMPT_WORDS} words or fewer.`);
    return;
  }

  submitBtn.disabled = true;

  const fd = new FormData(form);

  try {
    const res = await fetch("/generate", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || "Request failed. Please try again.");
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
  showProgressPanel(0, "Starting…");

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

// ---------------------------------------------------------------------------
// Button handlers
// ---------------------------------------------------------------------------
startOverBtn.addEventListener("click", () => {
  location.reload();
});

retryBtn.addEventListener("click", () => {
  resetUI();
});
