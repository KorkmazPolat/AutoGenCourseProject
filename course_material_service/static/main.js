document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("course-form");
  const overlay = document.getElementById("loading-overlay");
  const bar = document.getElementById("progress-bar");
  const pct = document.getElementById("progress-text");
  const hint = document.getElementById("progress-hint");

  const dropZone = document.getElementById("document-dropzone");
  const fileInput = document.getElementById("document-input");
  const browseButton = document.getElementById("document-browse");
  const statusMessage = document.getElementById("document-status");

  const hintMessages = [
    "Drafting plan…",
    "Outlining slides…",
    "Writing narration…",
    "Rendering visuals…",
    "Finalizing output…",
  ];

  let formProgressTimer = null;
  let uploadInFlight = false;

  const STATUS_CLASSES = ["text-gray-500", "text-indigo-600", "text-green-600", "text-red-600"];

  function setStatus(message, tone = "muted") {
    if (!statusMessage) return;
    statusMessage.textContent = message;
    statusMessage.classList.remove(...STATUS_CLASSES);
    const mapping = {
      muted: "text-gray-500",
      info: "text-indigo-600",
      pending: "text-indigo-600",
      success: "text-green-600",
      error: "text-red-600",
    };
    statusMessage.classList.add(mapping[tone] || mapping.muted);
  }

  function setDropZoneState(state) {
    if (!dropZone) return;
    dropZone.classList.remove("dragover", "success", "error", "uploading");
    if (state) {
      dropZone.classList.add(state);
    }
  }

  function clearFileSelector() {
    if (fileInput) {
      fileInput.value = "";
    }
  }

  function validateFile(file) {
    if (!file) {
      return "No document selected.";
    }
    const name = file.name.toLowerCase();
    const type = (file.type || "").toLowerCase();
    if (!name.endsWith(".pdf") && !type.includes("pdf")) {
      return "Please upload a PDF file.";
    }
    if (file.size === 0) {
      return "The selected file is empty.";
    }
    const maxBytes = 25 * 1024 * 1024; // 25 MB
    if (file.size > maxBytes) {
      return "PDF must be 25 MB or smaller.";
    }
    return "";
  }

  async function uploadDocument(file) {
    if (!dropZone) return;
    setDropZoneState(null);
    const error = validateFile(file);
    if (error) {
      setDropZoneState("error");
      setStatus(error, "error");
      clearFileSelector();
      return;
    }

    uploadInFlight = true;
    setDropZoneState("uploading");
    setStatus(`Uploading ${file.name}…`, "pending");

    const formData = new FormData();
    formData.append("file", file, file.name);

    try {
      const response = await fetch("/documents/upload", {
        method: "POST",
        body: formData,
        credentials: "same-origin",
      });

      if (!response.ok) {
        let detail = "Upload failed. Please try again.";
        try {
          const payload = await response.json();
          detail = payload.detail || detail;
        } catch {
          detail = await response.text() || detail;
        }
        throw new Error(detail);
      }

      const result = await response.json();
      setDropZoneState("success");
      const pages = typeof result.pages === "number" ? result.pages : "?";
      const chunks = typeof result.chunks === "number" ? result.chunks : "?";
      setStatus(`Indexed ${pages} pages (${chunks} chunks) from ${result.filename}.`, "success");
    } catch (err) {
      setDropZoneState("error");
      const message = err instanceof Error ? err.message : "Upload failed. Please try again.";
      setStatus(message, "error");
    } finally {
      uploadInFlight = false;
      clearFileSelector();
    }
  }

  function handleFileSelection(fileList) {
    if (!fileList || fileList.length === 0) {
      return;
    }
    const file = fileList[0];
    uploadDocument(file);
  }

  // Drop zone interactions
  dropZone?.addEventListener("click", () => {
    if (uploadInFlight) return;
    fileInput?.click();
  });

  browseButton?.addEventListener("click", (event) => {
    event.preventDefault();
    if (uploadInFlight) return;
    fileInput?.click();
  });

  dropZone?.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (uploadInFlight) return;
    dropZone.classList.add("dragover");
  });

  dropZone?.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone?.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
    if (uploadInFlight) return;
    handleFileSelection(event.dataTransfer?.files);
  });

  fileInput?.addEventListener("change", (event) => {
    if (uploadInFlight) {
      clearFileSelector();
      return;
    }
    handleFileSelection(event.target?.files);
  });

  function startFormProgress() {
    if (!overlay || formProgressTimer) return;
    overlay.classList.remove("hidden");
    let progress = 0;
    let hintIndex = 0;

    formProgressTimer = setInterval(() => {
      progress = Math.min(progress + Math.max(1, Math.round((100 - progress) / 18)), 96);
      if (bar) bar.style.width = `${progress}%`;
      if (pct) pct.textContent = `${progress}%`;
      if (hint && hintIndex < hintMessages.length) {
        hint.textContent = hintMessages[hintIndex];
        if (progress > (hintIndex + 1) * (100 / hintMessages.length)) {
          hintIndex += 1;
        }
      }
    }, 250);
  }

  function stopFormProgress() {
    if (formProgressTimer) {
      clearInterval(formProgressTimer);
      formProgressTimer = null;
    }
    if (bar) bar.style.width = "100%";
    if (pct) pct.textContent = "100%";
  }

  form?.addEventListener("submit", (event) => {
    if (uploadInFlight) {
      event.preventDefault();
      setStatus("Please wait for the PDF upload to finish before submitting.", "error");
      return;
    }
    startFormProgress();
  });

  window.addEventListener("beforeunload", () => {
    stopFormProgress();
  });
});
