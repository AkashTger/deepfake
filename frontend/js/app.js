/**
 * DeepGuard AI — Frontend Application
 * 2-Page SPA with video support, Chart.js, and SentinelAI aesthetics
 */

const API = "";
let cachedDashboardData = null;
let selectedFile = null;
let currentPage = "detect";
let chartsBuilt = false;

const $ = (sel) => document.querySelector(sel);
const uploadZone = $("#upload-zone");
const fileInput = $("#file-input");
const previewContainer = $("#preview-container");
const previewImage = $("#preview-image");
const previewVideo = $("#preview-video");
const clearBtn = $("#clear-btn");
const analyzeBtn = $("#analyze-btn");
const resultsCard = $("#results-card");
const explainGrid = $("#explain-grid");
const statusBadge = $("#status-badge");
const statusText = $(".status-text");

// ═══════════════════════════════════════════════════════
// PAGE NAVIGATION
// ═══════════════════════════════════════════════════════

function navigateTo(page) {
    if (page === currentPage) return;
    currentPage = page;

    document.querySelectorAll(".sidebar-link").forEach((link) => {
        link.classList.toggle("active", link.dataset.page === page);
    });

    document.querySelectorAll(".page").forEach((p) => {
        if (p.id === `page-${page}`) {
            p.style.display = "block";
            void p.offsetWidth;
            p.classList.add("active");
        } else {
            p.classList.remove("active");
            setTimeout(() => {
                if (!p.classList.contains("active")) p.style.display = "none";
            }, 400);
        }
    });

    window.scrollTo({ top: 0, behavior: "smooth" });

    if (page === "research" && cachedDashboardData && !chartsBuilt) {
        setTimeout(() => buildCharts(cachedDashboardData), 200);
    }
}

// ═══════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    setupUpload();
    loadDashboard();
    setupScrollAnimations();
});

async function checkHealth() {
    try {
        const res = await fetch(`${API}/api/health`);
        const data = await res.json();
        if (data.model_loaded) {
            statusBadge.className = "status-badge status-online";
            statusText.textContent = `Ready · ${data.device.toUpperCase()}`;
        } else {
            statusBadge.className = "status-badge status-offline";
            statusText.textContent = "No Model";
        }
    } catch {
        statusBadge.className = "status-badge status-offline";
        statusText.textContent = "Offline";
    }
}

function setupScrollAnimations() {
    const obs = new IntersectionObserver(entries => {
        entries.forEach(e => { if (e.isIntersecting) e.target.classList.add("visible"); });
    }, { threshold: 0.1 });
    document.querySelectorAll(".fade-up").forEach(el => obs.observe(el));
}

// ═══════════════════════════════════════════════════════
// FILE UPLOAD (Image + Video)
// ═══════════════════════════════════════════════════════

function setupUpload() {
    uploadZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });
    uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
    uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });
    clearBtn.addEventListener("click", clearUpload);
    analyzeBtn.addEventListener("click", analyzeFile);
}

function handleFile(file) {
    const isImage = file.type.startsWith("image/");
    const isVideo = file.type.startsWith("video/");

    if (!isImage && !isVideo) {
        alert("Please upload an image or video file.");
        return;
    }
    if (file.size > 50 * 1024 * 1024) {
        alert("File too large. Maximum 50MB for videos, 10MB for images.");
        return;
    }

    selectedFile = file;
    uploadZone.style.display = "none";
    previewContainer.style.display = "block";

    if (isImage) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.display = "block";
            previewVideo.style.display = "none";
        };
        reader.readAsDataURL(file);
    } else {
        previewVideo.src = URL.createObjectURL(file);
        previewVideo.style.display = "block";
        previewImage.style.display = "none";
    }

    analyzeBtn.disabled = false;
    resultsCard.style.display = "none";
    explainGrid.style.display = "none";
    $("#video-results").style.display = "none";
}

function clearUpload() {
    selectedFile = null;
    fileInput.value = "";
    previewImage.src = "";
    previewImage.style.display = "none";
    if (previewVideo.src) URL.revokeObjectURL(previewVideo.src);
    previewVideo.src = "";
    previewVideo.style.display = "none";
    uploadZone.style.display = "block";
    previewContainer.style.display = "none";
    analyzeBtn.disabled = true;
    resultsCard.style.display = "none";
    explainGrid.style.display = "none";
    $("#video-results").style.display = "none";
}

// ═══════════════════════════════════════════════════════
// ANALYZE (Image or Video)
// ═══════════════════════════════════════════════════════

async function analyzeFile() {
    if (!selectedFile) return;

    const btnText = $(".btn-text");
    const btnLoader = $(".btn-loader");
    analyzeBtn.disabled = true;

    const isVideo = selectedFile.type.startsWith("video/");

    if (isVideo) {
        btnText.textContent = "Extracting frames...";
        btnLoader.style.display = "block";
        await analyzeVideo();
    } else {
        btnText.textContent = "Analyzing...";
        btnLoader.style.display = "block";
        await analyzeImage();
    }

    btnText.textContent = "Analyze";
    btnLoader.style.display = "none";
    analyzeBtn.disabled = false;
}

async function analyzeImage() {
    try {
        const formData = new FormData();
        formData.append("file", selectedFile);
        const res = await fetch(`${API}/api/analyze`, { method: "POST", body: formData });
        if (!res.ok) { const err = await res.json(); throw new Error(err.detail || "Failed"); }
        const data = await res.json();
        renderResults(data);
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

async function analyzeVideo() {
    try {
        // Try server-side video analysis first
        const formData = new FormData();
        formData.append("file", selectedFile);
        const res = await fetch(`${API}/api/analyze-video`, { method: "POST", body: formData });

        if (res.ok) {
            const data = await res.json();
            renderVideoResults(data);
            return;
        }

        // Fallback: client-side frame extraction
        const frames = await extractFrames(selectedFile, 6);
        const results = [];

        for (let i = 0; i < frames.length; i++) {
            $(".btn-text").textContent = `Analyzing frame ${i + 1}/${frames.length}...`;
            const blob = await canvasToBlob(frames[i].canvas);
            const fd = new FormData();
            fd.append("file", blob, `frame_${i}.jpg`);
            try {
                const r = await fetch(`${API}/api/analyze`, { method: "POST", body: fd });
                if (r.ok) {
                    const d = await r.json();
                    d.timestamp = frames[i].time;
                    results.push(d);
                }
            } catch (e) {
                console.warn(`Frame ${i} failed:`, e);
            }
        }

        if (results.length > 0) {
            renderVideoResults({ frames: results, total_frames: results.length });
        } else {
            alert("Could not analyze any frames from the video.");
        }
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

function extractFrames(file, count) {
    return new Promise((resolve) => {
        const video = document.createElement("video");
        video.preload = "metadata";
        video.muted = true;
        video.src = URL.createObjectURL(file);

        video.onloadedmetadata = () => {
            const duration = video.duration;
            const interval = duration / (count + 1);
            const frames = [];
            let idx = 0;

            video.onseeked = () => {
                const canvas = document.createElement("canvas");
                canvas.width = Math.min(video.videoWidth, 512);
                canvas.height = Math.min(video.videoHeight, 512);
                const ctx = canvas.getContext("2d");
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                frames.push({ canvas, time: video.currentTime.toFixed(1) });
                idx++;
                if (idx < count) {
                    video.currentTime = interval * (idx + 1);
                } else {
                    URL.revokeObjectURL(video.src);
                    resolve(frames);
                }
            };
            video.currentTime = interval;
        };
    });
}

function canvasToBlob(canvas) {
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
}

// ═══════════════════════════════════════════════════════
// RENDER RESULTS
// ═══════════════════════════════════════════════════════

function renderResults(data) {
    const isFake = data.prediction === "FAKE";
    const banner = $("#verdict-banner");
    banner.className = `verdict-banner ${isFake ? "verdict-fake" : "verdict-real"}`;
    $("#verdict-icon").textContent = isFake ? "⚠️" : "✅";
    $("#verdict-label").textContent = data.prediction;
    $("#verdict-confidence").textContent = `${data.confidence}% confidence`;

    const fixBanner = $("#fix-banner");
    if (data.fix_applied) {
        fixBanner.style.display = "flex";
        $("#fix-disagreement").textContent = data.disagreement;
    } else {
        fixBanner.style.display = "none";
    }

    setTimeout(() => {
        animateBar("cnn-bar", "cnn-value", data.cnn_prob);
        animateBar("vit-bar", "vit-value", data.vit_prob);
        animateBar("ensemble-bar", "ensemble-value", data.fake_probability);
    }, 100);

    $("#ensemble-strategy").textContent = data.ensemble_strategy;
    setBase64Image("face-crop-img", data.face_crop);
    setBase64Image("gradcam-img", data.gradcam);
    setBase64Image("ela-img", data.ela);
    setBase64Image("fft-img", data.fft);

    resultsCard.style.display = "flex";
    explainGrid.style.display = "grid";
    $("#video-results").style.display = "none";
    resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderVideoResults(data) {
    const frames = data.frames || [];
    if (frames.length === 0) return;

    // Summary
    const fakeCount = frames.filter(f => f.prediction === "FAKE").length;
    const realCount = frames.length - fakeCount;
    const avgConf = (frames.reduce((s, f) => s + f.fake_probability, 0) / frames.length).toFixed(1);
    const verdict = fakeCount > realCount ? "FAKE" : "REAL";
    const verdictColor = verdict === "FAKE" ? "var(--red)" : "var(--green)";

    const summary = $("#video-summary");
    summary.innerHTML = `
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">
            <span style="font-size:2rem">${verdict === "FAKE" ? "⚠️" : "✅"}</span>
            <div>
                <div style="font-family:var(--font-mono);font-size:1.4rem;font-weight:800;color:${verdictColor};letter-spacing:1px">${verdict}</div>
                <div style="font-size:.82rem;color:var(--muted)">Video Analysis — ${frames.length} frames sampled</div>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border);border-radius:8px;overflow:hidden">
            <div style="background:var(--card);padding:.75rem;text-align:center">
                <div style="font-family:var(--font-mono);font-size:1.2rem;font-weight:700;color:var(--red)">${fakeCount}</div>
                <div style="font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em">Fake Frames</div>
            </div>
            <div style="background:var(--card);padding:.75rem;text-align:center">
                <div style="font-family:var(--font-mono);font-size:1.2rem;font-weight:700;color:var(--green)">${realCount}</div>
                <div style="font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em">Real Frames</div>
            </div>
            <div style="background:var(--card);padding:.75rem;text-align:center">
                <div style="font-family:var(--font-mono);font-size:1.2rem;font-weight:700;color:var(--blue)">${avgConf}%</div>
                <div style="font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em">Avg P(Fake)</div>
            </div>
        </div>
    `;

    // Frame cards
    const container = $("#video-frames");
    container.innerHTML = "";
    frames.forEach((f, i) => {
        const isFake = f.prediction === "FAKE";
        const card = document.createElement("div");
        card.className = "frame-card";
        card.innerHTML = `
            ${f.face_crop ? `<img src="data:image/png;base64,${f.face_crop}" alt="Frame ${i + 1}"/>` : ""}
            <div class="frame-label">Frame ${i + 1}${f.timestamp ? ` · ${f.timestamp}s` : ""}</div>
            <div class="frame-result" style="color:${isFake ? "var(--red)" : "var(--green)"}">${f.prediction} ${f.fake_probability}%</div>
        `;
        container.appendChild(card);
    });

    // Show first frame's explainability in the main panel
    const first = frames[0];
    if (first) {
        setBase64Image("face-crop-img", first.face_crop);
        setBase64Image("gradcam-img", first.gradcam);
        setBase64Image("ela-img", first.ela);
        setBase64Image("fft-img", first.fft);
        explainGrid.style.display = "grid";
    }

    resultsCard.style.display = "none";
    $("#video-results").style.display = "block";
    summary.scrollIntoView({ behavior: "smooth", block: "start" });
}

function animateBar(barId, valueId, percent) {
    const bar = document.getElementById(barId);
    const value = document.getElementById(valueId);
    bar.style.width = `${Math.min(percent, 100)}%`;
    value.textContent = `${percent}%`;
    if (percent > 70) value.style.color = "var(--red)";
    else if (percent > 40) value.style.color = "var(--amber)";
    else value.style.color = "var(--green)";
}

function setBase64Image(id, base64) {
    const img = document.getElementById(id);
    if (base64) img.src = `data:image/png;base64,${base64}`;
}

// ═══════════════════════════════════════════════════════
// DASHBOARD (Chart.js)
// ═══════════════════════════════════════════════════════

async function loadDashboard() {
    try {
        const res = await fetch(`${API}/api/dashboard`);
        const data = await res.json();
        cachedDashboardData = data;

        // Update metrics
        const m = data.metrics || {};
        if (m.auc) $("#m-auc").textContent = m.auc > 1 ? m.auc.toFixed(3) : m.auc.toFixed(3);
        if (m.sn34) $("#m-sn34").textContent = m.sn34 > 1 ? m.sn34.toFixed(3) : m.sn34.toFixed(3);
        if (m.f1) $("#m-f1").textContent = m.f1 > 1 ? m.f1.toFixed(3) : m.f1.toFixed(3);
        if (data.device) $("#m-device").textContent = data.device.toUpperCase();
    } catch {
        // Dashboard is optional
    }
}

function buildCharts(data) {
    if (chartsBuilt) return;
    if (!data.training_history || data.training_history.length === 0) return;

    chartsBuilt = true;
    const history = data.training_history;
    const GRID = "rgba(74,26,42,0.5)";
    const TICK = "#a07080";

    const baseOpts = {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { color: GRID }, ticks: { color: TICK, font: { size: 10 } } },
            y: { grid: { color: GRID }, ticks: { color: TICK, font: { size: 10 } } }
        }
    };

    const epochs = history.map(h => h.epoch);

    // Loss chart
    const lossCtx = document.getElementById("lossChart");
    if (lossCtx) {
        new Chart(lossCtx, {
            type: "line",
            data: {
                labels: epochs,
                datasets: [
                    { label: "Train", data: history.map(h => h.train_loss), borderColor: "#e63946", borderWidth: 2, pointRadius: 0, tension: 0.4 },
                    { label: "Val", data: history.map(h => h.val_loss), borderColor: "#ff4d5e", borderWidth: 2, pointRadius: 0, tension: 0.4, borderDash: [5, 3] }
                ]
            },
            options: {
                ...baseOpts,
                plugins: { legend: { display: true, labels: { color: TICK, font: { size: 10 }, boxWidth: 10 } } },
                scales: {
                    x: { ...baseOpts.scales.x, title: { display: true, text: "Epoch", color: TICK, font: { size: 10 } } },
                    y: { ...baseOpts.scales.y, title: { display: true, text: "Loss", color: TICK, font: { size: 10 } } }
                }
            }
        });
    }

    // AUC chart
    const aucCtx = document.getElementById("aucChart");
    if (aucCtx) {
        new Chart(aucCtx, {
            type: "line",
            data: {
                labels: epochs,
                datasets: [{ data: history.map(h => h.val_auc), borderColor: "#00ff9d", borderWidth: 2, fill: true, backgroundColor: "rgba(0,255,157,.05)", pointRadius: 0, tension: 0.4 }]
            },
            options: {
                ...baseOpts,
                scales: {
                    x: { ...baseOpts.scales.x, title: { display: true, text: "Epoch", color: TICK, font: { size: 10 } } },
                    y: { ...baseOpts.scales.y, title: { display: true, text: "AUC", color: TICK, font: { size: 10 } }, min: 0.4, max: 1.02 }
                }
            }
        });
    }

    // SN34 chart
    const sn34Ctx = document.getElementById("sn34Chart");
    if (sn34Ctx) {
        const bestEp = history.reduce((a, b) => b.val_sn34 > a.val_sn34 ? b : a, history[0]);
        new Chart(sn34Ctx, {
            type: "line",
            data: {
                labels: epochs,
                datasets: [
                    { data: history.map(h => h.val_sn34), borderColor: "#d2a8ff", borderWidth: 2, fill: true, backgroundColor: "rgba(210,168,255,.05)", pointRadius: 0, tension: 0.4 },
                    { data: history.map(h => h.epoch === bestEp.epoch ? h.val_sn34 : null), pointRadius: 8, pointBackgroundColor: "#ffb400", borderColor: "transparent", type: "scatter" }
                ]
            },
            options: {
                ...baseOpts,
                scales: {
                    x: { ...baseOpts.scales.x, title: { display: true, text: "Epoch", color: TICK, font: { size: 10 } } },
                    y: { ...baseOpts.scales.y, title: { display: true, text: "SN34", color: TICK, font: { size: 10 } }, min: 0.5, max: 1.0 }
                }
            }
        });
    }
}

// Redraw on resize
let resizeTimer = null;
window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        // Chart.js handles its own resize
    }, 250);
});
