# 🛡️ DeepGuard AI — Deepfake Detection

A state-of-the-art deepfake face detection web application using a dual-backbone deep learning ensemble with explainable AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/DiddiAkash/deepfake-application)

---

## 🌐 Live Demo

The application is hosted and running live on Hugging Face Spaces!
👉 **[Try the Deepfake Application Here](https://huggingface.co/spaces/DiddiAkash/deepfake-application)**


## ✨ Features

- **Dual-Backbone Ensemble** — ConvNeXt-V2 (local pixel artifacts) + Swin-V2 (global inconsistencies)
- **GAN Detection Fix** — Disagreement-aware smart ensemble that trusts the suspicious branch when models disagree
- **Triple Explainability** — Grad-CAM attention maps, Error Level Analysis (ELA), FFT frequency spectrum
- **Real-Time Inference** — &lt;2s per image on GPU, ~5s on CPU
- **Premium Web UI** — Dark glassmorphism theme with drag-and-drop upload
- **Training Dashboard** — Live metrics, training curves, model performance visualization

## 📊 Performance

| Metric | In-Distribution (FF++ + GAN) | Cross-Dataset (CelebDF) |
|--------|------------------------------|------------------------|
| AUC    | 93.9%                        | 77.4%                  |
| F1     | 91.3%                        | 71.6%                  |
| SN34   | 88.7%                        | 69.2%                  |

## 🏗️ Architecture

```
ConvNeXt-V2 Base ──┐
  (local artifacts) ├──▶ Temperature-Scaled Soft Voting ──▶ Smart Ensemble ──▶ REAL/FAKE
Swin-V2 Base ──────┘       (disagreement-aware)
  (global patterns)
```

**The GAN Fix:** When ConvNeXt and Swin disagree by >40%, the ensemble trusts the more suspicious branch instead of averaging — fixing false negatives on GAN-generated faces.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA GPU recommended (CPU works but slower)

### Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd main

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Download model weights (~700 MB)
#    Place ensemble_best.safetensors in model_files/
#    See model_files/README.md for download instructions

# 4. Start the server
python backend/server.py
# Or on Windows, just double-click: start.bat

# 5. Open http://localhost:8000
```

## 📁 Project Structure

```
main/
├── start.bat                    # One-click launcher (Windows)
├── backend/
│   ├── server.py                # FastAPI server (API + inference + explainability)
│   ├── model.py                 # Model architecture (inference version)
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── index.html               # Single-page application
│   ├── css/style.css            # Premium dark theme
│   └── js/app.js                # Upload, API calls, chart rendering
├── model_files/
│   ├── ensemble_best.safetensors  # Trained weights (~700 MB)
│   ├── model_config.yaml          # Model & training config
│   ├── model_summary.json         # Performance metrics
│   ├── training_history.json      # 20-epoch training log
│   ├── web_app_bundle.json        # Dashboard data bundle
│   └── model.py                   # Training-time model definition
└── docs_and_experiments/          # Training notebooks, logs, and experiment docs
```

## 🔧 API Endpoints

| Method | Path             | Description                                |
|--------|------------------|--------------------------------------------|
| GET    | `/`              | Serves the web UI                          |
| GET    | `/api/health`    | Model status, device, face detector type   |
| POST   | `/api/analyze`   | Upload image → prediction + explainability |
| GET    | `/api/dashboard` | Training metrics and history               |

## 🧪 Training Notebooks

| Notebook | Purpose |
|----------|---------|
| `sdpmajor-1` | Data preparation & manifest creation |
| `final-1` (NB1) | Data pipeline & configuration |
| `final-2` (NB2) | Model training (20 epochs on Kaggle GPU) |
| `final 3` (NB3) | Evaluation, explainability samples, web bundle |

## 🔒 Security Notes

- CORS is open (`allow_origins=["*"]`) — intended for **local/demo use only**
- Upload size limited to **10 MB** server-side
- For production: add authentication, restrict CORS origins, use HTTPS

## 📝 License

SDP Major Project — Academic Use

---

*Built with PyTorch, FastAPI, timm, and OpenCV*
