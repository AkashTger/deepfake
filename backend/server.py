"""
FastAPI server — DeepGuard AI Deepfake Detection
Implements the GAN detection fix (disagreement-aware ensemble).

Security notes:
- CORS is open (allow_origins=["*"]) — suitable for local/demo use only.
- For production: restrict origins, add authentication, use HTTPS.
- Upload size is limited to 10 MB server-side.
"""

import os
import sys
import io
import json
import base64
import logging
import urllib.request
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageEnhance
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Setup paths ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model_files"
FRONTEND_DIR = ROOT / "frontend"
BACKEND_DIR = ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepguard")

# ── Lifespan ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application):
    init_face_detector()
    load_model()
    yield

# ── App ──────────────────────────────────────────────────────
app = FastAPI(title="DeepGuard AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
face_net = None
face_cascade = None
IMG_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

preprocess_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ═══════════════════════════════════════════════════════════════
# FACE DETECTION
# ═══════════════════════════════════════════════════════════════

def init_face_detector():
    """Initialize OpenCV DNN face detector, fallback to Haar cascade."""
    global face_net, face_cascade

    proto_path = str(BACKEND_DIR / "deploy.prototxt")
    caffe_path = str(BACKEND_DIR / "res10_300x300_ssd_iter_140000.caffemodel")

    if os.path.exists(proto_path) and os.path.exists(caffe_path):
        face_net = cv2.dnn.readNetFromCaffe(proto_path, caffe_path)
        logger.info("Face detector: OpenCV DNN SSD ✅")
        return

    # Try to download — prototxt and caffemodel are at different repos
    PROTO_URLS = [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
    ]
    CAFFE_URLS = [
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    ]

    def _download(url_list, dest):
        for url in url_list:
            try:
                logger.info(f"Downloading {os.path.basename(dest)} from {url[:80]}...")
                urllib.request.urlretrieve(url, dest)
                return True
            except Exception as e:
                logger.warning(f"  Failed: {e}")
                if os.path.exists(dest):
                    os.remove(dest)
        return False

    if _download(PROTO_URLS, proto_path) and _download(CAFFE_URLS, caffe_path):
        face_net = cv2.dnn.readNetFromCaffe(proto_path, caffe_path)
        logger.info("Face detector: OpenCV DNN SSD ✅")
        return

    # Fallback to Haar cascade — still functional, just less accurate
    logger.warning("All DNN download mirrors failed. Using Haar Cascade fallback.")
    logger.warning("For better accuracy, manually download the DNN files:")
    logger.warning("  deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel")
    logger.warning(f"  Place them in: {BACKEND_DIR}")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    logger.info("Face detector: Haar Cascade (fallback) ⚠️")


def detect_face(image_rgb: np.ndarray, margin: float = 0.3) -> np.ndarray:
    """Detect and crop the largest face, returns (IMG_SIZE, IMG_SIZE, 3)."""
    h, w = image_rgb.shape[:2]

    if face_net is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image_rgb, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        face_net.setInput(blob)
        detections = face_net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype(int))

        if boxes:
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
            x1, y1, x2, y2 = boxes[np.argmax(areas)]
            fw_, fh_ = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - fw_ * margin))
            y1 = max(0, int(y1 - fh_ * margin))
            x2 = min(w, int(x2 + fw_ * margin))
            y2 = min(h, int(y2 + fh_ * margin))
            crop = image_rgb[y1:y2, x1:x2]
            if crop.size > 0:
                return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    elif face_cascade is not None:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces) > 0:
            areas = [fw * fh for (_, _, fw, fh) in faces]
            x, y, fw, fh = faces[np.argmax(areas)]
            x1 = max(0, int(x - fw * margin))
            y1 = max(0, int(y - fh * margin))
            x2 = min(w, int(x + fw + fw * margin))
            y2 = min(h, int(y + fh + fh * margin))
            crop = image_rgb[y1:y2, x1:x2]
            if crop.size > 0:
                return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    # Fallback: center crop
    s = min(h, w)
    y1, x1 = (h - s) // 2, (w - s) // 2
    crop = image_rgb[y1 : y1 + s, x1 : x1 + s]
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))


# ═══════════════════════════════════════════════════════════════
# GAN FIX — SMART ENSEMBLE
# ═══════════════════════════════════════════════════════════════

def smart_ensemble(cnn_prob: float, vit_prob: float) -> dict:
    """
    Disagreement-aware ensemble — fixes GAN detection.

    Problem: ConvNeXt gives 5-10% on GAN fakes (no local artifacts),
             Swin gives 80-88% (detects global inconsistencies).
             Simple average = (0.07 + 0.84)/2 = 0.455 → FALSE NEGATIVE.

    Fix: When branches strongly disagree, trust the more suspicious one.
    """
    original_avg = (cnn_prob + vit_prob) / 2.0
    disagreement = abs(cnn_prob - vit_prob)

    if disagreement > 0.4:
        # Strong disagreement → one branch found something → trust it
        fixed_prob = max(cnn_prob, vit_prob)
        strategy = "max (disagreement fix)"
        fix_applied = True
    else:
        # Agreement → normal average
        fixed_prob = original_avg
        strategy = "average"
        fix_applied = False

    return {
        "final_prob": fixed_prob,
        "original_avg": original_avg,
        "strategy": strategy,
        "fix_applied": fix_applied,
        "disagreement": round(disagreement, 4),
    }


# ═══════════════════════════════════════════════════════════════
# EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════

def generate_gradcam(face_rgb: np.ndarray, input_tensor: torch.Tensor) -> np.ndarray:
    """Generate Grad-CAM heatmap from ConvNeXt branch."""
    if model is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    target_layer = model.cnn_branch.backbone.stages[3].blocks[-1]

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["val"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["val"] = grad_out[0].detach()

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.eval()
        t = input_tensor.clone().requires_grad_(True).to(device)
        ensemble_logit, _, _ = model(t)

        model.zero_grad()
        ensemble_logit.backward()

        act = activations["val"]
        grad = gradients["val"]
        weights = grad.mean(dim=[2, 3], keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}")
        cam = np.zeros((IMG_SIZE, IMG_SIZE))
    finally:
        h_fwd.remove()
        h_bwd.remove()

    # Overlay on face image
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(face_resized, 0.5, heatmap, 0.5, 0)
    return overlay


def generate_ela(face_rgb: np.ndarray, quality: int = 90) -> np.ndarray:
    """Error Level Analysis — reveals compression inconsistencies."""
    pil = Image.fromarray(face_rgb)
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)

    diff = ImageChops.difference(pil, recompressed)
    extrema = diff.getextrema()
    max_val = max(e[1] for e in extrema)
    if max_val == 0:
        max_val = 1
    scale = 255.0 / max_val
    enhanced = ImageEnhance.Brightness(diff).enhance(scale)
    return np.array(enhanced)


def generate_fft(face_rgb: np.ndarray) -> np.ndarray:
    """FFT spectrum — GAN images show unnatural grid patterns."""
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    mag = 20 * np.log(np.abs(f_shift) + 1e-8)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_uint8 = np.uint8(mag)
    colored = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def np_to_base64(img: np.ndarray) -> str:
    """Convert numpy image to base64 PNG string."""
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_model():
    """Load the trained ensemble model."""
    global model

    weights_path = MODEL_DIR / "ensemble_best.safetensors"
    config_path = MODEL_DIR / "model_config.yaml"

    if not weights_path.exists():
        logger.warning(f"Model weights not found at {weights_path}")
        logger.warning("Place ensemble_best.safetensors in model_files/")
        return False

    try:
        from model import EnsembleDeepfakeDetector

        if config_path.exists():
            m = EnsembleDeepfakeDetector(config_path=str(config_path))
        else:
            m = EnsembleDeepfakeDetector()

        m.load_safetensors(str(weights_path))
        m = m.to(device)
        m.eval()
        model = m
        logger.info(f"Model loaded on {device} ✅")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════

# Startup is handled by the lifespan context manager above


@app.get("/")
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
        "face_detector": "DNN" if face_net else ("Haar" if face_cascade else "none"),
    }


MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an uploaded image for deepfake detection."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Place ensemble_best.safetensors in model_files/",
        )

    # Read image with size limit enforced server-side
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",
        )

    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Detect face
    face = detect_face(image_np)
    face_pil = Image.fromarray(face)

    # Preprocess
    input_tensor = preprocess_transform(face_pil).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        ensemble_logit, cnn_logit, vit_logit = model(input_tensor)

    cnn_prob = torch.sigmoid(cnn_logit.float()).item()
    vit_prob = torch.sigmoid(vit_logit.float()).item()
    original_prob = torch.sigmoid(ensemble_logit.float()).item()

    # Apply GAN fix
    ensemble_result = smart_ensemble(cnn_prob, vit_prob)
    final_prob = ensemble_result["final_prob"]

    prediction = "FAKE" if final_prob >= 0.5 else "REAL"
    confidence = max(final_prob, 1 - final_prob)

    # Explainability
    gradcam_img = generate_gradcam(face, input_tensor)
    ela_img = generate_ela(face)
    fft_img = generate_fft(face)

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 1),
        "fake_probability": round(final_prob * 100, 1),
        "original_ensemble_prob": round(original_prob * 100, 1),
        "cnn_prob": round(cnn_prob * 100, 1),
        "vit_prob": round(vit_prob * 100, 1),
        "ensemble_strategy": ensemble_result["strategy"],
        "fix_applied": ensemble_result["fix_applied"],
        "disagreement": round(ensemble_result["disagreement"] * 100, 1),
        "face_crop": np_to_base64(face),
        "gradcam": np_to_base64(gradcam_img),
        "ela": np_to_base64(ela_img),
        "fft": np_to_base64(fft_img),
    }


MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50 MB for videos


@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze a video by extracting frames and running detection on each."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Place ensemble_best.safetensors in model_files/",
        )

    contents = await file.read()
    if len(contents) > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Video too large. Maximum size is {MAX_VIDEO_SIZE // (1024 * 1024)} MB.",
        )

    # Save to temp file for OpenCV
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(contents)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not read video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration = total_frames / fps if fps > 0 else 0

        # Sample up to 8 frames evenly
        num_samples = min(8, max(1, total_frames))
        sample_indices = [int(i * total_frames / num_samples) for i in range(num_samples)]

        frame_results = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp = round(idx / fps, 1) if fps > 0 else 0

            # Detect face
            face = detect_face(frame_rgb)
            face_pil = Image.fromarray(face)

            # Preprocess
            input_tensor = preprocess_transform(face_pil).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                ensemble_logit, cnn_logit, vit_logit = model(input_tensor)

            cnn_prob = torch.sigmoid(cnn_logit.float()).item()
            vit_prob = torch.sigmoid(vit_logit.float()).item()

            ensemble_result = smart_ensemble(cnn_prob, vit_prob)
            final_prob = ensemble_result["final_prob"]
            prediction = "FAKE" if final_prob >= 0.5 else "REAL"
            confidence = max(final_prob, 1 - final_prob)

            # Explainability (only for first frame to save time)
            if len(frame_results) == 0:
                gradcam_img = generate_gradcam(face, input_tensor)
                ela_img = generate_ela(face)
                fft_img = generate_fft(face)
                gradcam_b64 = np_to_base64(gradcam_img)
                ela_b64 = np_to_base64(ela_img)
                fft_b64 = np_to_base64(fft_img)
            else:
                gradcam_b64 = None
                ela_b64 = None
                fft_b64 = None

            frame_results.append({
                "frame_index": idx,
                "timestamp": timestamp,
                "prediction": prediction,
                "confidence": round(confidence * 100, 1),
                "fake_probability": round(final_prob * 100, 1),
                "cnn_prob": round(cnn_prob * 100, 1),
                "vit_prob": round(vit_prob * 100, 1),
                "ensemble_strategy": ensemble_result["strategy"],
                "fix_applied": ensemble_result["fix_applied"],
                "face_crop": np_to_base64(face),
                "gradcam": gradcam_b64,
                "ela": ela_b64,
                "fft": fft_b64,
            })

        cap.release()

        # Summary
        fake_count = sum(1 for f in frame_results if f["prediction"] == "FAKE")
        avg_prob = sum(f["fake_probability"] for f in frame_results) / max(len(frame_results), 1)

        return {
            "total_frames": len(frame_results),
            "duration": round(duration, 1),
            "fps": round(fps, 1),
            "fake_frame_count": fake_count,
            "real_frame_count": len(frame_results) - fake_count,
            "avg_fake_probability": round(avg_prob, 1),
            "verdict": "FAKE" if fake_count > len(frame_results) / 2 else "REAL",
            "frames": frame_results,
        }
    finally:
        os.unlink(tmp.name)


@app.get("/api/dashboard")
async def dashboard():
    """Return training metrics and model summary."""
    bundle_path = MODEL_DIR / "web_app_bundle.json"
    summary_path = MODEL_DIR / "model_summary.json"
    history_path = MODEL_DIR / "training_history.json"

    data = {}

    if bundle_path.exists():
        with open(bundle_path) as f:
            data = json.load(f)
    else:
        if summary_path.exists():
            with open(summary_path) as f:
                data["metrics"] = json.load(f).get("in_distribution", {})
        if history_path.exists():
            with open(history_path) as f:
                data["training_history"] = json.load(f)

    data["model_loaded"] = model is not None
    data["device"] = str(device)
    return data


# Serve static frontend files
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    # Create assets dir if needed
    (FRONTEND_DIR / "assets").mkdir(parents=True, exist_ok=True)

    port = int(os.environ.get("PORT", 7860))

    print("\n" + "=" * 55)
    print("  [DeepGuard AI] Deepfake Detection Server")
    print("=" * 55)
    print(f"  Device    : {device}")
    print(f"  Model dir : {MODEL_DIR}")
    print(f"  Frontend  : {FRONTEND_DIR}")
    print(f"  URL       : http://localhost:{port}")
    print("=" * 55 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
