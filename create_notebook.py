import json

cells = []

def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":src.strip().split("\n")})

def code(src):
    cells.append({"cell_type":"code","metadata":{},"source":src.strip().split("\n"),"outputs":[],"execution_count":None})

md("""# Notebook 3 — Explainability & Web App Bundle (v4)
**Prerequisites:** Add these datasets as input before running:
1. `assualttger/sdp-preprocessing` — preprocessed .npy face crops
2. `assualttger/spd-processing2` — manifest.json, model_config.yaml from NB1
3. `assualttger/final-2-v4-output` — ensemble_best.safetensors, model_summary.json, training_history.json

**Important:** This version uses the v4 model with Dropout(0.3) layers.
**Outputs:** web_app_bundle.json, Grad-CAM / ELA / FFT images, ROC/PR curves""")

code("""# == CELL 0: Copy files + Write v4 model.py (with Dropout) ==
import os, shutil, sys

NB1_FILES = ["manifest.json", "model_config.yaml"]
NB2_FILES = ["ensemble_best.safetensors", "model_summary.json", "training_history.json"]

def find_and_copy(filename, required=True):
    for root, _, files in os.walk("/kaggle/input"):
        if filename in files:
            src = os.path.join(root, filename)
            dst = f"/kaggle/working/{filename}"
            shutil.copy(src, dst)
            print(f"  OK {filename:<40} <- {src}")
            return True
    if required:
        print(f"  MISSING {filename:<40} — add the dataset as input")
    return False

print("--- From NB1 (assualttger/spd-processing2) ---")
for f in NB1_FILES:
    find_and_copy(f)

print("\\n--- From NB2 (assualttger/final-2-v4-output) ---")
for f in NB2_FILES:
    find_and_copy(f)

# CRITICAL: model.py MUST match the architecture used in NB2 v4
# v4 added Dropout(0.3) before FC layers — without this, weights won't load
model_py = \"\"\"
import torch, torch.nn as nn, timm, yaml
from safetensors.torch import save_file, load_file

class ConvNextBranch(nn.Module):
    def __init__(self, model_name="convnextv2_base", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, drop_path_rate=0.2)
        self.backbone.set_grad_checkpointing(enable=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.backbone.num_features, 1)
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(self.dropout(features)), features

class SwinBranch(nn.Module):
    def __init__(self, model_name="swinv2_base_window12to16_192to256", pretrained=True):
        super().__init__()
        try:
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, drop_path_rate=0.2)
        except Exception:
            self.backbone = timm.create_model(
                "swinv2_base_window12_256", pretrained=pretrained, num_classes=0, drop_path_rate=0.2)
        self.backbone.set_grad_checkpointing(enable=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.backbone.num_features, 1)
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(self.dropout(features)), features

class EnsembleDeepfakeDetector(nn.Module):
    def __init__(self, config_path="/kaggle/working/model_config.yaml"):
        super().__init__()
        cfg = yaml.safe_load(open(config_path))["model"]
        self.cnn_branch = ConvNextBranch(cfg["ensemble"]["cnn_branch"], cfg["pretrained"])
        self.vit_branch = SwinBranch(cfg["ensemble"]["vit_branch"], cfg["pretrained"])
        self.temperature_cnn = nn.Parameter(torch.ones(1))
        self.temperature_vit = nn.Parameter(torch.ones(1))
    def forward(self, x):
        cnn_logits, _ = self.cnn_branch(x)
        vit_logits, _ = self.vit_branch(x)
        ensemble = (cnn_logits / self.temperature_cnn + vit_logits / self.temperature_vit) / 2.0
        return ensemble, cnn_logits / self.temperature_cnn, vit_logits / self.temperature_vit
    def save_safetensors(self, path):
        save_file(dict(self.state_dict()), path)
        print(f"Saved -> {path}")
    def load_safetensors(self, path):
        self.load_state_dict(load_file(path))
        print(f"Loaded <- {path}")
\"\"\"
with open("/kaggle/working/model.py", "w") as f:
    f.write(model_py.strip())
print("\\n  model.py written (v4 with Dropout 0.3)")

sys.path.insert(0, "/kaggle/working")

critical = ["manifest.json", "model_config.yaml", "model.py", "ensemble_best.safetensors"]
print("\\n--- File check ---")
all_ok = True
for f in critical:
    exists = os.path.exists(f"/kaggle/working/{f}")
    print(f"  {'OK' if exists else 'MISSING'} /kaggle/working/{f}")
    all_ok = all_ok and exists
print("\\nAll files ready!" if all_ok else "\\nMissing files - check datasets above.")""")

code("""# == CELL 1: Imports ==
import torch, numpy as np, yaml, json, os, sys, math
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageChops, ImageEnhance
import cv2
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score, roc_curve, matthews_corrcoef,
    brier_score_loss, precision_recall_curve, average_precision_score
)
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "/kaggle/working")
os.makedirs("/kaggle/working/gradcam_samples", exist_ok=True)
os.makedirs("/kaggle/working/ela_samples", exist_ok=True)
os.makedirs("/kaggle/working/fft_samples", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")""")

code("""# == CELL 2: Load Model & Manifest ==
from model import EnsembleDeepfakeDetector

config   = yaml.safe_load(open("/kaggle/working/model_config.yaml"))
manifest = json.load(open("/kaggle/working/manifest.json"))

val_data   = manifest["val"]
celeb_data = manifest["celeb_test"]
IMG_SIZE   = 256

import glob
found_npy = glob.glob("/kaggle/input/**/rgb/*.npy", recursive=True)
if found_npy:
    ACTUAL_PREP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(found_npy[0])))
    def fix_path(p):
        marker = "/preprocessed/"
        return (ACTUAL_PREP_ROOT + p[p.index(marker):]) if marker in p else p
    def remap(samples):
        for s in samples:
            for k in ["rgb_path","fft_path","hist_path"]:
                if k in s: s[k] = fix_path(s[k])
        return samples
    sample0 = val_data[0].get("rgb_path","")
    if not os.path.exists(sample0):
        val_data   = remap(val_data)
        celeb_data = remap(celeb_data)
        print(f"Paths remapped -> {val_data[0]['rgb_path']}")

model = EnsembleDeepfakeDetector("/kaggle/working/model_config.yaml").to(device)
model.load_safetensors("/kaggle/working/ensemble_best.safetensors")
model.eval()
print("\\nModel loaded (v4 with Dropout)")""")

code("""# == CELL 3: Inference on Val Samples ==
import random; random.seed(42)

normalize = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def predict_npy(sample):
    rgb = np.load(sample["rgb_path"])
    t   = normalize(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, cnn_l, vit_l = model(t)
        prob     = torch.sigmoid(logits).item()
        cnn_prob = torch.sigmoid(cnn_l).item()
        vit_prob = torch.sigmoid(vit_l).item()
    return {
        "ensemble_prob": round(prob, 4),
        "cnn_prob":      round(cnn_prob, 4),
        "vit_prob":      round(vit_prob, 4),
        "prediction":    "FAKE" if prob >= 0.5 else "REAL",
        "confidence":    round(max(prob, 1-prob), 4),
        "true_label":    sample["label"],
    }

real_s = [s for s in val_data if s["label"]==0 and os.path.exists(s.get("rgb_path",""))]
fake_s = [s for s in val_data if s["label"]==1 and os.path.exists(s.get("rgb_path",""))]
sample_set = random.sample(real_s, min(250, len(real_s))) + random.sample(fake_s, min(250, len(fake_s)))

results = [predict_npy(s) for s in sample_set]
json.dump(results, open("/kaggle/working/inference_results.json","w"), indent=2)

y_true = np.array([r["true_label"]    for r in results])
y_prob = np.array([r["ensemble_prob"] for r in results])
print(f"Inference on {len(results)} samples")
print(f"AUC : {roc_auc_score(y_true, y_prob):.4f}")""")

code("""# == CELL 4: ROC & Precision-Recall Curves ==
def sn34_score(y_true, y_prob):
    y_pred     = (y_prob >= 0.5).astype(int)
    norm_mcc   = (matthews_corrcoef(y_true, y_pred) + 1) / 2
    norm_brier = 1 - brier_score_loss(y_true, y_prob)
    return float(math.pow(max(norm_mcc,0)**1.2 * max(norm_brier,0)**1.8, 1/3.0))

fpr, tpr, _  = roc_curve(y_true, y_prob)
prec, rec, _ = precision_recall_curve(y_true, y_prob)
sn34 = sn34_score(y_true, y_prob)
auc  = roc_auc_score(y_true, y_prob)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22"); ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#30363d")

axes[0].plot(fpr, tpr, color="#58a6ff", lw=2, label=f"AUC={auc:.4f}")
axes[0].plot([0,1],[0,1],"--",color="#8b949e")
axes[0].fill_between(fpr, tpr, alpha=0.1, color="#58a6ff")
axes[0].set_title("ROC Curve", color="white", fontsize=13)
axes[0].set_xlabel("FPR", color="#8b949e"); axes[0].set_ylabel("TPR", color="#8b949e")
axes[0].legend(facecolor="#21262d", labelcolor="white")

axes[1].plot(rec, prec, color="#3fb950", lw=2,
             label=f"AP={average_precision_score(y_true,y_prob):.4f}")
axes[1].fill_between(rec, prec, alpha=0.1, color="#3fb950")
axes[1].set_title("Precision-Recall", color="white", fontsize=13)
axes[1].set_xlabel("Recall", color="#8b949e"); axes[1].set_ylabel("Precision", color="#8b949e")
axes[1].legend(facecolor="#21262d", labelcolor="white")

fig.suptitle(f"Model Performance  (SN34: {sn34:.4f})", color="white", fontsize=14)
plt.tight_layout()
plt.savefig("/kaggle/working/roc_pr_curves.png", dpi=150, facecolor="#0d1117")
plt.show()""")

code("""# == CELL 5: Confidence Distribution ==
real_probs = [r["ensemble_prob"] for r in results if r["true_label"] == 0]
fake_probs = [r["ensemble_prob"] for r in results if r["true_label"] == 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22"); ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#30363d")

axes[0].hist(real_probs, bins=30, alpha=0.75, color="#2ea043", label="Real")
axes[0].hist(fake_probs, bins=30, alpha=0.75, color="#f85149", label="Fake")
axes[0].axvline(0.5, color="#e3b341", ls="--", lw=1.5, label="threshold")
axes[0].set_title("Score Distribution", color="white", fontsize=13)
axes[0].set_xlabel("P(Fake)", color="#8b949e")
axes[0].legend(facecolor="#21262d", labelcolor="white")

sns.kdeplot([max(p,1-p) for p in real_probs], ax=axes[1], color="#2ea043",
            lw=2, fill=True, alpha=0.3, label="Real")
sns.kdeplot([max(p,1-p) for p in fake_probs], ax=axes[1], color="#f85149",
            lw=2, fill=True, alpha=0.3, label="Fake")
axes[1].set_title("Confidence Distribution", color="white", fontsize=13)
axes[1].set_xlabel("Confidence", color="#8b949e")
axes[1].legend(facecolor="#21262d", labelcolor="white")

fig.suptitle("Prediction Confidence Analysis", color="white", fontsize=14)
plt.tight_layout()
plt.savefig("/kaggle/working/confidence_distribution.png", dpi=150, facecolor="#0d1117")
plt.show()""")

code("""# == CELL 6: Grad-CAM Heatmaps ==
!pip install -q torchcam
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

model.cnn_branch.backbone.set_grad_checkpointing(enable=False)
target_layer = model.cnn_branch.backbone.stages[3].blocks[2]
cam_extractor = GradCAM(model.cnn_branch.backbone, target_layer=target_layer)

cam_samples = (
    random.sample([s for s in val_data if s["label"]==0 and os.path.exists(s.get("rgb_path",""))], 5) +
    random.sample([s for s in val_data if s["label"]==1 and os.path.exists(s.get("rgb_path",""))], 5)
)

gradcam_manifest = []
model.eval()

for i, s in enumerate(cam_samples):
    rgb = np.load(s["rgb_path"])
    pil = Image.fromarray(rgb).convert("RGB")
    t   = normalize(pil).unsqueeze(0).to(device)

    out = model.cnn_branch.backbone(t)
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    heatmap = to_pil_image(activation_map[0].squeeze(0), mode="F")
    result  = overlay_mask(pil.resize((IMG_SIZE, IMG_SIZE)), heatmap, alpha=0.5)

    out_path = f"/kaggle/working/gradcam_samples/gradcam_{i:02d}_label{s['label']}.png"
    result.save(out_path)
    gradcam_manifest.append({"path": out_path, "label": s["label"]})

model.cnn_branch.backbone.set_grad_checkpointing(enable=True)
json.dump(gradcam_manifest, open("/kaggle/working/gradcam_manifest.json","w"), indent=2)
print(f"{len(gradcam_manifest)} Grad-CAM images saved")""")

code("""# == CELL 7: ELA & FFT Analysis ==
def ela(rgb_arr, quality=90):
    orig = Image.fromarray(rgb_arr)
    orig.save("/tmp/_ela.jpg", "JPEG", quality=quality)
    diff  = ImageChops.difference(orig, Image.open("/tmp/_ela.jpg"))
    scale = 255.0 / max(max(e[1] for e in diff.getextrema()), 1)
    return ImageEnhance.Brightness(diff).enhance(scale)

def fft(rgb_arr):
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag  = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)
    mag  = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(np.uint8(mag))

ela_manifest, fft_manifest = [], []
for i, s in enumerate(cam_samples):
    rgb = np.load(s["rgb_path"])
    ela_path = f"/kaggle/working/ela_samples/ela_{i:02d}.png"
    ela(rgb).save(ela_path)
    ela_manifest.append(ela_path)
    fft_path = f"/kaggle/working/fft_samples/fft_{i:02d}.png"
    fft(rgb).save(fft_path)
    fft_manifest.append(fft_path)

print(f"ELA: {len(ela_manifest)} | FFT: {len(fft_manifest)} images saved")""")

code("""# == CELL 8: Explainability Panel ==
N = 3
fig, axes = plt.subplots(N, 3, figsize=(15, 5*N))
fig.patch.set_facecolor("#0d1117")
col_titles = ["Grad-CAM", "ELA (Error Level)", "FFT Spectrum"]

for row in range(N):
    gc    = gradcam_manifest[row]
    imgs  = [Image.open(gc["path"]),
             Image.open(ela_manifest[row]),
             Image.open(fft_manifest[row])]
    cmaps = [None, "hot", "plasma"]
    label = "FAKE" if gc["label"] == 1 else "REAL"
    color = "#f85149" if label == "FAKE" else "#2ea043"

    for col, (img, cmap, title) in enumerate(zip(imgs, cmaps, col_titles)):
        ax = axes[row][col]
        ax.set_facecolor("#161b22")
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if row == 0:
            ax.set_title(title, color="white", fontsize=12)
        if col == 0:
            ax.set_ylabel(label, color=color, fontsize=11,
                          fontweight="bold", rotation=0, labelpad=45)

fig.suptitle("Explainability Panel - ConvNeXt-V2 Feature Analysis",
             color="white", fontsize=14)
plt.tight_layout()
plt.savefig("/kaggle/working/explainability_panel.png", dpi=150, facecolor="#0d1117")
plt.show()""")

code("""# == CELL 9: Build web_app_bundle.json ==
training_history = json.load(open("/kaggle/working/training_history.json"))
model_summary    = json.load(open("/kaggle/working/model_summary.json"))

bundle = {
    "meta": {
        "title":         "SOTA Explainable Deepfake Detection (v4 Anti-Overfit)",
        "architecture": "ConvNeXt-V2-Base + Swin-V2-Base Ensemble",
        "regularization": "Dropout(0.3) + DropPath(0.2) + LabelSmoothing(0.05)",
        "datasets":     ["FaceForensics++ C23", "140K Real & Fake", "Celeb-DF v2 (test)"],
        "explainability": ["Grad-CAM", "ELA", "FFT"]
    },
    "metrics":      model_summary.get("in_distribution", {}),
    "cross_dataset": model_summary.get("cross_dataset_celeb", {}),
    "training_history": training_history,
    "roc_curve": {
        "fpr": [round(float(v), 4) for v in fpr],
        "tpr": [round(float(v), 4) for v in tpr],
    },
    "pr_curve": {
        "precision": [round(float(v), 4) for v in prec],
        "recall":    [round(float(v), 4) for v in rec],
    },
    "prediction_distribution": {
        "real": [round(p, 3) for p in real_probs],
        "fake": [round(p, 3) for p in fake_probs],
    },
    "inference_sample": results[:50],
    "plots": [
        "training_curves.png", "confusion_matrix.png",
        "roc_pr_curves.png", "confidence_distribution.png",
        "explainability_panel.png"
    ]
}

json.dump(bundle, open("/kaggle/working/web_app_bundle.json", "w"), indent=2)

print("=" * 55)
print("PIPELINE COMPLETE")
print("=" * 55)
for fname in sorted(os.listdir("/kaggle/working/")):
    p = f"/kaggle/working/{fname}"
    if os.path.isfile(p):
        sz = os.path.getsize(p)
        unit = "MB" if sz > 1e6 else "KB"
        print(f"  {fname:<45} {sz/(1e6 if sz>1e6 else 1024):.1f} {unit}")""")

# Build notebook JSON
nb = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.12"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [
                {"sourceType": "datasetVersion", "sourceId": 15414201, "datasetId": 9860925, "databundleVersionId": 16330882},
                {"sourceType": "datasetVersion", "sourceId": 15597636, "datasetId": 9980563, "databundleVersionId": 16530538}
            ],
            "dockerImageVersionId": 31329,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells
}

out = r"c:\Users\akash\Desktop\main\final-3-v4.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Created: {out}")
print(f"Cells: {len(cells)}")
