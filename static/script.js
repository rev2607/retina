/* ═══════════════════════════════════════════════════════════════════════════════
   RetinaRisk — Frontend Logic
   Handles: file upload, API calls, loading animation, results rendering, modal
   ═══════════════════════════════════════════════════════════════════════════════ */

// ─── State ──────────────────────────────────────────────────────────────────────
let selectedFiles = [];
let analysisResults = [];

// ─── DOM References ─────────────────────────────────────────────────────────────
const pageUpload   = document.getElementById("page-upload");
const pageLoading  = document.getElementById("page-loading");
const pageResults  = document.getElementById("page-results");

const dropzone     = document.getElementById("dropzone");
const fileInput    = document.getElementById("file-input");
const previewStrip = document.getElementById("preview-strip");
const fileCount    = document.getElementById("file-count");
const btnAnalyze   = document.getElementById("btn-analyze");

const loadingStatus = document.getElementById("loading-status");
const progressBar   = document.getElementById("progress-bar");
const progressLabel = document.getElementById("progress-label");

const btnBack       = document.getElementById("btn-back");
const metricsRow    = document.getElementById("metrics-row");
const errorsSection = document.getElementById("errors-section");
const highRiskSec   = document.getElementById("high-risk-section");
const lowRiskSec    = document.getElementById("low-risk-section");
const clinicalGrid  = document.getElementById("clinical-grid");

const modalOverlay  = document.getElementById("modal-overlay");
const modalClose    = document.getElementById("modal-close");
const modalTitle    = document.getElementById("modal-title");
const modalOriginal = document.getElementById("modal-original");
const modalGradcam  = document.getElementById("modal-gradcam");
const modalDownload = document.getElementById("modal-download");


// ─── Page Navigation ────────────────────────────────────────────────────────────
function showPage(id) {
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    document.getElementById(id).classList.add("active");
    window.scrollTo({ top: 0, behavior: "smooth" });
}


// ─── Dropzone Events ────────────────────────────────────────────────────────────
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    addFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", () => {
    addFiles(fileInput.files);
    fileInput.value = "";          // reset so re-selecting same file works
});


// ─── File Management ────────────────────────────────────────────────────────────
function addFiles(fileList) {
    for (const file of fileList) {
        if (!file.type.match(/^image\/(jpeg|png|jpg)$/i)) continue;
        // Avoid exact duplicates by name+size
        if (selectedFiles.some(f => f.name === file.name && f.size === file.size)) continue;
        selectedFiles.push(file);
    }
    renderPreviews();
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderPreviews();
}

function renderPreviews() {
    previewStrip.innerHTML = "";
    selectedFiles.forEach((file, i) => {
        const thumb = document.createElement("div");
        thumb.className = "preview-thumb";

        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.alt = file.name;

        const btn = document.createElement("button");
        btn.className = "remove-btn";
        btn.textContent = "✕";
        btn.onclick = e => { e.stopPropagation(); removeFile(i); };

        thumb.appendChild(img);
        thumb.appendChild(btn);
        previewStrip.appendChild(thumb);
    });

    fileCount.textContent = selectedFiles.length
        ? `${selectedFiles.length} image(s) selected`
        : "";

    btnAnalyze.disabled = selectedFiles.length === 0;
}


// ─── Analyze Button ─────────────────────────────────────────────────────────────
btnAnalyze.addEventListener("click", () => {
    if (selectedFiles.length === 0) return;
    startAnalysis();
});


// ─── Analysis Flow ──────────────────────────────────────────────────────────────
async function startAnalysis() {
    // Switch to loading page
    showPage("page-loading");

    // Animated loading steps
    const steps = [
        "🔬 Preprocessing retinal images …",
        "🧠 Running deep-learning inference …",
        "🔥 Generating Grad-CAM heatmaps …",
        "📊 Classifying cardiovascular risk …",
        "✅ Finalizing results …",
    ];

    let stepIdx = 0;
    progressBar.style.width = "0%";
    loadingStatus.textContent = steps[0];

    // Animate fake progress (minimum 3s visible)
    const fakeInterval = setInterval(() => {
        stepIdx = Math.min(stepIdx + 1, steps.length - 1);
        loadingStatus.textContent = steps[stepIdx];
    }, 700);

    // Smooth progress bar animation
    let fakeProgress = 0;
    const progressInterval = setInterval(() => {
        fakeProgress = Math.min(fakeProgress + 2, 70);
        progressBar.style.width = fakeProgress + "%";
        progressLabel.textContent = `Processing …`;
    }, 100);

    // Remember when we started for minimum display time
    const startTime = Date.now();

    // Build form data
    const formData = new FormData();
    selectedFiles.forEach(f => formData.append("images", f));
    formData.append("bp", document.getElementById("bp").value);
    formData.append("cholesterol", document.getElementById("cholesterol").value);
    formData.append("bmi", document.getElementById("bmi").value);
    formData.append("diabetes", document.getElementById("diabetes").value);

    try {
        const res = await fetch("/api/analyze", { method: "POST", body: formData });
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();
        analysisResults = data.results;
    } catch (err) {
        clearInterval(fakeInterval);
        clearInterval(progressInterval);
        alert("Analysis failed: " + err.message);
        showPage("page-upload");
        return;
    }

    // Ensure at least 3.5 seconds of loading screen
    const elapsed = Date.now() - startTime;
    const remaining = Math.max(0, 3500 - elapsed);

    clearInterval(fakeInterval);
    clearInterval(progressInterval);

    // Finish progress smoothly
    loadingStatus.textContent = steps[steps.length - 1];
    progressBar.style.width = "90%";

    await sleep(remaining);

    progressBar.style.width = "100%";
    progressLabel.textContent = "✅ Analysis complete!";
    await sleep(500);

    renderResults();
    showPage("page-results");
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}


// ─── Render Results ─────────────────────────────────────────────────────────────
function renderResults() {
    const highRisk = [];
    const lowRisk  = [];
    const errors   = [];

    analysisResults.forEach(r => {
        if (r.error)                     errors.push(r);
        else if (r.risk.percent >= 60)   highRisk.push(r);
        else                             lowRisk.push(r);
    });

    const totalValid = highRisk.length + lowRisk.length;

    // Metrics
    metricsRow.innerHTML = `
        ${metricCard(analysisResults.length, "Total Images", "#fff")}
        ${metricCard(totalValid, "Valid Scans", "var(--accent)")}
        ${metricCard(highRisk.length, "🔴 High Risk", "var(--red)")}
        ${metricCard(lowRisk.length, "🟢 Low Risk", "var(--green)")}
    `;

    // Errors
    errorsSection.innerHTML = "";
    if (errors.length) {
        errorsSection.innerHTML = `<h4 style="color:var(--red);margin-bottom:0.6rem;">⚠️ Rejected / Failed Images (${errors.length})</h4>`;
        errors.forEach(r => {
            const imgTag = r.original
                ? `<img src="data:image/png;base64,${r.original}" alt="${r.name}">`
                : "";
            errorsSection.innerHTML += `
                <div class="error-card">
                    ${imgTag}
                    <div>
                        <div class="error-name">${esc(r.name)}</div>
                        <div class="error-text">${esc(r.error)}</div>
                    </div>
                </div>`;
        });
    }

    // High Risk
    highRiskSec.innerHTML = "";
    if (highRisk.length) {
        highRiskSec.innerHTML = `<div class="risk-group-header high">🔴 High Risk Images</div>`;
        highRisk.forEach((r, i) => highRiskSec.innerHTML += resultCardHTML(r, `h${i}`));
    }

    // Low Risk
    lowRiskSec.innerHTML = "";
    if (lowRisk.length) {
        lowRiskSec.innerHTML = `<div class="risk-group-header low">🟢 Low Risk Images</div>`;
        lowRisk.forEach((r, i) => lowRiskSec.innerHTML += resultCardHTML(r, `l${i}`));
    }

    // Attach heatmap button handlers
    document.querySelectorAll(".btn-heatmap").forEach(btn => {
        btn.addEventListener("click", () => {
            const idx = parseInt(btn.dataset.idx);
            const r = analysisResults[idx];
            openHeatmapModal(r);
        });
    });

    // Clinical summary
    const bp = document.getElementById("bp").value;
    const chol = document.getElementById("cholesterol").value;
    const bmi = document.getElementById("bmi").value;
    const diab = document.getElementById("diabetes").value;

    clinicalGrid.innerHTML = `
        ${clinicalItem(bp + " mmHg", "Blood Pressure")}
        ${clinicalItem(chol + " mg/dL", "Cholesterol")}
        ${clinicalItem(parseFloat(bmi).toFixed(1), "BMI")}
        ${clinicalItem(diab === "1" ? "Yes" : "No", "Diabetes")}
    `;
}


// ─── HTML Builders ──────────────────────────────────────────────────────────────
function metricCard(value, label, color) {
    return `<div class="metric-card">
        <div class="metric-value" style="color:${color}">${value}</div>
        <div class="metric-label">${label}</div>
    </div>`;
}

function clinicalItem(val, label) {
    return `<div class="clinical-item">
        <div class="cval">${val}</div>
        <div class="clbl">${label}</div>
    </div>`;
}

function resultCardHTML(r, id) {
    const risk = r.risk;
    const color = risk.color;
    const pct = risk.percent.toFixed(1);
    const globalIdx = analysisResults.indexOf(r);

    const heatmapBtn = r.gradcam
        ? `<button class="btn-heatmap" data-idx="${globalIdx}">🔥 View Grad-CAM Heatmap</button>`
        : `<button class="btn-heatmap" disabled>Heatmap unavailable</button>`;

    return `
    <div class="result-card" id="card-${id}">
        <div class="result-top">
            <div class="result-info">
                <div class="file-name">📄 ${esc(r.name)}</div>
                <span class="risk-badge" style="background:${color}22;color:${color};border:1px solid ${color}44;">
                    ${risk.emoji} ${risk.label}
                </span>
            </div>
            <div class="score-box">
                <div class="score-value" style="color:${color}">${pct}%</div>
                <div class="score-label">Risk Score</div>
            </div>
        </div>

        <div class="risk-bar-wrapper">
            <div class="risk-bar" style="width:${pct}%;background:${color};"></div>
        </div>

        <div class="interpretation">💡 ${esc(risk.interpretation)}</div>

        <div class="result-image">
            <img src="data:image/png;base64,${r.original}" alt="Retinal scan of ${esc(r.name)}">
        </div>

        ${heatmapBtn}
    </div>`;
}

function esc(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
}


// ─── Heatmap Modal ──────────────────────────────────────────────────────────────
function openHeatmapModal(r) {
    modalTitle.textContent = `🔥 Grad-CAM — ${r.name}`;
    modalOriginal.src = `data:image/png;base64,${r.original}`;
    modalGradcam.src  = `data:image/png;base64,${r.gradcam}`;
    modalDownload.href = `data:image/png;base64,${r.gradcam}`;
    modalDownload.download = `gradcam_${r.name.replace(/\.[^.]+$/, "")}.png`;
    modalOverlay.classList.add("open");
}

function closeModal() {
    modalOverlay.classList.remove("open");
}

modalClose.addEventListener("click", closeModal);
modalOverlay.addEventListener("click", e => {
    if (e.target === modalOverlay) closeModal();
});
document.addEventListener("keydown", e => {
    if (e.key === "Escape") closeModal();
});


// ─── Back Button ────────────────────────────────────────────────────────────────
btnBack.addEventListener("click", () => {
    showPage("page-upload");
});
