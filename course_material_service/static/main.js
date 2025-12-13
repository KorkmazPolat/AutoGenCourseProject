// static/main.js

/**
 * Profesyonel form doğrulama (validation) modülü
 * Bir formu (formId) alır ve tüm 'required' alanları için
 * gerçek zamanlı doğrulama ve submit butonu kontrolü ekler.
 */
function setupFormValidation(formId) {
    const form = document.getElementById(formId);
    // Formu bulamazsa hiçbir şey yapma
    if (!form) return; 

    const submitButton = form.querySelector('button[type="submit"]');
    // 'required' olan tüm input ve textarea'ları bul
    const requiredInputs = form.querySelectorAll("[required]");

    /**
     * Tek bir input alanını doğrular ve görsel geri bildirimi (UI) günceller.
     * @param {HTMLInputElement | HTMLTextAreaElement} input - Doğrulanacak input alanı.
     * @returns {boolean} - Alanın geçerli (valid) olup olmadığı.
     */
    function validateInput(input) {
        // HTML5 Validity API'sini kullan (required, type="email", minlength vb. kontrol eder)
        const isValid = input.validity.valid;

        if (isValid) {
            input.classList.remove("is-invalid");
            input.classList.add("is-valid");
            // İlişkili hata mesajını gizle
            const errorElement = document.getElementById(`${input.id}-error`);
            if (errorElement) {
                errorElement.style.display = "none";
            }
        } else {
            input.classList.remove("is-valid");
            input.classList.add("is-invalid");
            // İlişkili hata mesajını göster
            const errorElement = document.getElementById(`${input.id}-error`);
            if (errorElement) {
                // HTML5'in varsayılan mesajını (veya bizim özel mesajımızı) kullan
                if (input.validationMessage && !errorElement.textContent.includes(input.validationMessage)) {
                     // Sadece minlength veya type hataları için özel mesajı göster
                     if(input.validity.tooShort) {
                        errorElement.textContent = `Password must be at least ${input.minLength} characters.`;
                     } else if (input.validity.typeMismatch) {
                        errorElement.textContent = "Please enter a valid email address.";
                     } else if (input.validity.valueMissing) {
                        errorElement.textContent = "This field is required.";
                     }
                }
                errorElement.style.display = "block";
            }
        }
        return isValid;
    }

    /**
     * Formdaki TÜM zorunlu alanları kontrol eder ve
     * submit butonunu etkinleştirir veya devre dışı bırakır.
     */
    function checkFormValidity() {
        if (!submitButton) return;

        let isFormFullyValid = true;
        // Tüm zorunlu alanları tek tek kontrol et
        requiredInputs.forEach(input => {
            // Eğer BİR tanesi bile geçersizse, formu geçersiz say
            if (!input.validity.valid) {
                isFormFullyValid = false;
            }
        });

        // Butonun durumunu güncelle
        submitButton.disabled = !isFormFullyValid;
    }

    // --- Event Listeners Ekle ---

    // 1. Her bir input için: 'input' (yazarken) veya 'blur' (odaktan çıkınca)
    //    alanını doğrula ve ardından tüm formun durumunu kontrol et.
    requiredInputs.forEach(input => {
        // 'input' olayı daha hızlı geri bildirim sağlar
        input.addEventListener("input", () => {
            validateInput(input);
        });
        // Alanı terk ettiğinde (blur) de kontrol et
        input.addEventListener("blur", () => {
            validateInput(input);
        });

        // Herhangi bir input değiştiğinde, tüm formun geçerliliğini tekrar kontrol et
        input.addEventListener("input", checkFormValidity);
    });

    // 2. Sayfa yüklendiğinde, formu ve butonu varsayılan duruma ayarla
    checkFormValidity();
}


/**
 * PDF Yükleme (RAG) modülü
 */
function setupPdfUpload() {
    const dropZone = document.getElementById("document-dropzone");
    // Eğer bu element sayfada yoksa (örn. login sayfasındayız), hiçbir şey yapma.
    if (!dropZone) return;

    const fileInput = document.getElementById("document-input");
    const browseButton = document.getElementById("document-browse");
    const statusMessage = document.getElementById("document-status");

    let uploadInFlight = false;
    const STATUS_CLASSES = ["text-gray-500", "text-indigo-600", "text-green-600", "text-red-600"];

    function setStatus(message, tone = "muted") {
        if (!statusMessage) return;
        statusMessage.textContent = message;
        statusMessage.classList.remove(...STATUS_CLASSES);
        const mapping = {
          muted: "text-gray-500", info: "text-indigo-600", pending: "text-indigo-600",
          success: "text-green-600", error: "text-red-600"
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
        if (fileInput) fileInput.value = "";
    }

    function validateFile(file) {
        if (!file) return "No document selected.";
        const name = file.name.toLowerCase();
        const type = (file.type || "").toLowerCase();
        if (!name.endsWith(".pdf") && !type.includes("pdf")) return "Please upload a PDF file.";
        if (file.size === 0) return "The selected file is empty.";
        const maxBytes = 25 * 1024 * 1024; // 25 MB
        if (file.size > maxBytes) return "PDF must be 25 MB or smaller.";
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
                method: "POST", body: formData, credentials: "same-origin",
            });
            if (!response.ok) {
                let detail = "Upload failed. Please try again.";
                try {
                    const payload = await response.json();
                    detail = payload.detail || detail;
                } catch { detail = await response.text() || detail; }
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
        if (!fileList || fileList.length === 0) return;
        uploadDocument(fileList[0]);
    }

    // --- Event Listeners Ekle ---
    dropZone.addEventListener("click", () => {
        if (uploadInFlight) return;
        fileInput?.click();
    });
    browseButton.addEventListener("click", (event) => {
        event.preventDefault();
        if (uploadInFlight) return;
        fileInput?.click();
    });
    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        if (uploadInFlight) return;
        dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropZone.classList.remove("dragover");
        if (uploadInFlight) return;
        handleFileSelection(event.dataTransfer?.files);
    });
    fileInput.addEventListener("change", (event) => {
        if (uploadInFlight) {
            clearFileSelector();
            return;
        }
        handleFileSelection(event.target?.files);
    });
}


/**
 * Dashboard (index.html) yükleme ekranı modülü - AJAX + job polling on the same page.
 */
function setupLoadingOverlay() {
    const form = document.getElementById("course-form");
    if (!form) return;

    const overlay = document.getElementById("loading-overlay");
    const bar = document.getElementById("progress-bar");
    const pct = document.getElementById("progress-text");
    const hint = document.getElementById("progress-hint");
    const statusMessage = document.getElementById("document-status");

    let submitting = false;
    let uploadInFlight = statusMessage && statusMessage.textContent.includes("Uploading");

    const showOverlay = (message = "Generating…") => {
        if (!overlay) return;
        overlay.classList.remove("hidden");
        overlay.style.display = "flex";
        overlay.style.zIndex = "9999";
        overlay.setAttribute("aria-hidden", "false");
        if (bar) bar.style.width = "5%";
        if (pct) pct.textContent = "0%";
        if (hint) hint.textContent = message;
    };

    const updateOverlay = (progress, messageText) => {
        if (bar && typeof progress === "number") bar.style.width = `${progress}%`;
        if (pct && typeof progress === "number") pct.textContent = `${progress}%`;
        if (hint && messageText) hint.textContent = messageText;
    };

    const hideOverlay = () => {
        if (!overlay) return;
        overlay.classList.add("hidden");
        overlay.style.display = "none";
        overlay.setAttribute("aria-hidden", "true");
    };

    const pollStatus = async (jobId) => {
        try {
            const res = await fetch(`/agentic-jobs/${jobId}`, { headers: { "Accept": "application/json" } });
            if (!res.ok) throw new Error("Status fetch failed");
            const data = await res.json();
            updateOverlay(data.progress ?? 0, data.message || "Generating…");

            if (data.status === "completed") {
                window.location.href = `/agentic-jobs/${jobId}/view`;
                return;
            }
            if (data.status === "failed") {
                alert(data.error || "Generation failed. Please try again.");
                hideOverlay();
                submitting = false;
                return;
            }
        } catch (err) {
            console.warn("Polling error", err);
        }
        setTimeout(() => pollStatus(jobId), 700);
    };

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (submitting) return;
        if (uploadInFlight) {
            alert("Please wait for the PDF upload to finish before submitting.");
            return;
        }

        submitting = true;
        const startedAt = performance.now();
        showOverlay("Agents are researching, planning, and writing content…");

        const formData = new FormData(form);
        let jobId = null;

        try {
            const res = await fetch(form.action || "/agentic-jobs/start", {
                method: "POST",
                body: formData,
                headers: {
                    "Accept": "application/json",
                    "X-Requested-With": "XMLHttpRequest",
                },
            });
            if (!res.ok) throw new Error("Failed to start generation");
            const payload = await res.json();
            jobId = payload.job_id;
        } catch (err) {
            console.error("Start job failed", err);
            alert("Could not start generation. Please try again.");
            submitting = false;
            hideOverlay();
            return;
        }

        const elapsed = performance.now() - startedAt;
        const minVisible = 300;
        if (elapsed < minVisible) {
            await new Promise((resolve) => setTimeout(resolve, minVisible - elapsed));
        }

        if (jobId) {
            pollStatus(jobId);
        } else {
            hideOverlay();
            submitting = false;
        }
    });
}


// --- Uygulamayı Başlat ---
// Sayfa yüklendiğinde, ilgili modülleri çalıştır
document.addEventListener("DOMContentLoaded", () => {
    // 1. Form Doğrulama modülünü çalıştır
    // 'login-form' ID'li formu (login.html) ve 
    // 'course-form' ID'li formu (index.html) arayacak
    setupFormValidation("login-form");
    setupFormValidation("course-form"); // Henüz 'index.html'i güncellemedik, ama bu hazır olacak
    
    // 2. PDF Yükleme modülünü çalıştır (sadece /dashboard'da çalışır)
    setupPdfUpload();

    // 3. Yükleme Ekranı modülünü çalıştır (sadece /dashboard'da çalışır)
    setupLoadingOverlay();
});
