# ===============================================================
# ANTI-OVERFIT FIXES FOR final-2 (1).ipynb
# ===============================================================
# Copy each section into the corresponding CELL in your notebook.
# Changes are marked with # ← CHANGED or # ← NEW
#
# Your original training had severe overfitting:
#   Epoch 3:  train=0.19, val=0.30 (1.6× gap)
#   Epoch 20: train=0.03, val=0.69 (24× gap — severe)
#
# These fixes target the root causes.
# ===============================================================


# ══════════════════════════════════════════════════════════════
# CELL 0: model.py — ADD DROPOUT (replace the model_py string)
# ══════════════════════════════════════════════════════════════

model_py = '''
import torch, torch.nn as nn, timm, yaml
from safetensors.torch import save_file, load_file


class ConvNextBranch(nn.Module):
    """ConvNeXt-V2: detects local pixel-level manipulation artifacts."""
    def __init__(self, model_name="convnextv2_base", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, drop_path_rate=0.2)
        self.backbone.set_grad_checkpointing(enable=True)
        self.dropout = nn.Dropout(0.3)                     # ← NEW: prevents memorization
        self.fc = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(self.dropout(features)), features   # ← CHANGED: dropout before FC


class SwinBranch(nn.Module):
    """Swin-V2: detects global lighting/shadow inconsistencies."""
    def __init__(self, model_name="swinv2_base_window12to16_192to256", pretrained=True):
        super().__init__()
        try:
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, drop_path_rate=0.2)
        except Exception:
            self.backbone = timm.create_model(
                "swinv2_base_window12_256", pretrained=pretrained, num_classes=0, drop_path_rate=0.2)
        self.backbone.set_grad_checkpointing(enable=True)
        self.dropout = nn.Dropout(0.3)                     # ← NEW: prevents memorization
        self.fc = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(self.dropout(features)), features   # ← CHANGED: dropout before FC


class EnsembleDeepfakeDetector(nn.Module):
    """Soft-voting ensemble with Dropout + Drop Path anti-overfit."""
    def __init__(self, config_path="/kaggle/working/model_config.yaml"):
        super().__init__()
        cfg = yaml.safe_load(open(config_path))["model"]
        self.cnn_branch = ConvNextBranch(cfg["ensemble"]["cnn_branch"], cfg["pretrained"])
        self.vit_branch = SwinBranch(cfg["ensemble"]["vit_branch"],     cfg["pretrained"])
        self.temperature_cnn = nn.Parameter(torch.ones(1))
        self.temperature_vit = nn.Parameter(torch.ones(1))

    def forward(self, x):
        cnn_logits, _ = self.cnn_branch(x)
        vit_logits, _ = self.vit_branch(x)
        ensemble = (cnn_logits / self.temperature_cnn + vit_logits / self.temperature_vit) / 2.0
        return ensemble, cnn_logits / self.temperature_cnn, vit_logits / self.temperature_vit

    def save_safetensors(self, path):
        save_file(dict(self.state_dict()), path)
        print(f"Saved → {path}")

    def load_safetensors(self, path):
        self.load_state_dict(load_file(path))
        print(f"Loaded ← {path}")
'''


# ══════════════════════════════════════════════════════════════
# CELL 2: Hyperparameters — ADD EARLY STOPPING + LABEL SMOOTHING
# ══════════════════════════════════════════════════════════════

BATCH_SIZE       = 16
ACCUM_STEPS      = 4
EPOCHS           = 15           # ← CHANGED: 20 → 15 (early stopping will likely trigger earlier)
LR               = 1e-5
WEIGHT_DECAY     = 1e-2
IMG_SIZE         = 256
PATIENCE         = 5            # ← NEW: stop if no improvement for 5 epochs
LABEL_SMOOTHING  = 0.05         # ← NEW: soft labels prevent overconfidence
WARMUP_EPOCHS    = 2            # ← NEW: freeze backbones during warmup


# ══════════════════════════════════════════════════════════════
# CELL 4: STRONGER DATA AUGMENTATION
# ══════════════════════════════════════════════════════════════

# Replace the entire train_transform with this:

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),

    # ← CHANGED: Much stronger color augmentation
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.3,   # was 0.1
            contrast=0.3,     # was 0.1
            saturation=0.2,   # was missing
            hue=0.05          # was missing
        )
    ], p=0.5),                # was 0.3

    # ← CHANGED: Stronger blur
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))  # was kernel=3, no sigma range
    ], p=0.3),                # was 0.1

    transforms.RandomRotation(15),  # ← CHANGED: was 8°

    # ← NEW: Slight spatial distortions
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),

    # ← NEW: Force color-independent feature learning
    transforms.RandomGrayscale(p=0.05),

    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),

    # ← NEW: Random patch erasure (simulates occlusion)
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
])


# ══════════════════════════════════════════════════════════════
# CELL 7: TRAINING LOOP — ALL OVERFIT FIXES INTEGRATED
# ══════════════════════════════════════════════════════════════

# Replace the entire CELL 7 with this:

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ← CHANGED: Proper warmup + cosine schedule (was just CosineAnnealing)
warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-7)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, [warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
)

scaler = GradScaler()

history, best_sn34_val, best_epoch = [], 0.0, 0
no_improve = 0                                             # ← NEW: early stopping counter
BEST = "/kaggle/working/ensemble_best.safetensors"


# ← NEW: Layer freezing helper
def set_backbone_frozen(mdl, frozen=True):
    """Freeze/unfreeze backbone layers (keep FC + temperature trainable)."""
    m = mdl.module if hasattr(mdl, "module") else mdl
    for param in m.cnn_branch.backbone.parameters():
        param.requires_grad = not frozen
    for param in m.vit_branch.backbone.parameters():
        param.requires_grad = not frozen
    status = "FROZEN" if frozen else "UNFROZEN"
    trainable = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  Backbones {status} — trainable params: {trainable:,}")


print(f"Training {EPOCHS} epochs | LR={LR} | WD={WEIGHT_DECAY} | LabelSmooth={LABEL_SMOOTHING}")
print(f"Batch={BATCH_SIZE}×{ACCUM_STEPS} (eff {BATCH_SIZE*ACCUM_STEPS}) | AMP=ON | GradCkpt=ON")
print(f"Early stopping: patience={PATIENCE} | Dropout=0.3 | DropPath=0.2")
print("=" * 72)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ← NEW: Freeze backbones during warmup, unfreeze after
    if epoch == 1:
        set_backbone_frozen(model, frozen=True)
    elif epoch == WARMUP_EPOCHS + 1:
        set_backbone_frozen(model, frozen=False)

    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Ep {epoch:02d}/{EPOCHS}", leave=False)):
        x, y = x.to(device), y.to(device)

        # ← NEW: Label smoothing — prevents overconfident predictions
        y_smooth = y * (1 - LABEL_SMOOTHING) + (1 - y) * LABEL_SMOOTHING

        with autocast():
            m = model.module if hasattr(model, "module") else model
            logits, _, _ = m(x)
            loss = criterion(logits, y_smooth) / ACCUM_STEPS  # ← CHANGED: use y_smooth

        scaler.scale(loss).backward()
        train_loss += loss.item() * ACCUM_STEPS

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    scheduler.step()
    avg_train = train_loss / len(train_loader)
    val_loss, val_auc, val_sn34, _, _ = evaluate(model, val_loader)
    current_lr = optimizer.param_groups[0]["lr"]

    # ← NEW: Track train-val gap for monitoring
    gap = val_loss - avg_train

    flag = ""
    if val_sn34 > best_sn34_val:
        best_sn34_val, best_epoch = val_sn34, epoch
        (model.module if hasattr(model, "module") else model).save_safetensors(BEST)
        no_improve = 0                                     # ← NEW: reset counter
        flag = " 🏆"
    else:
        no_improve += 1                                    # ← NEW: increment counter

    rec = {"epoch": epoch, "train_loss": round(avg_train, 5), "val_loss": round(val_loss, 5),
           "val_auc": round(val_auc, 5), "val_sn34": round(val_sn34, 5),
           "lr": round(current_lr, 8), "elapsed": round(time.time() - t0, 1),
           "gap": round(gap, 4)}                           # ← NEW: track gap
    history.append(rec)
    print(f"Ep {epoch:02d} | loss {avg_train:.4f}→{val_loss:.4f} (gap {gap:.3f}) | "
          f"AUC {val_auc:.4f} | SN34 {val_sn34:.4f} | LR {current_lr:.2e}{flag}")

    # ← NEW: Early stopping check
    if no_improve >= PATIENCE:
        print(f"\n⏹️  Early stopping at epoch {epoch} — no improvement for {PATIENCE} epochs")
        break

json.dump(history, open("/kaggle/working/training_history.json", "w"), indent=2)
print(f"\nBest SN34 = {best_sn34_val:.4f}  (epoch {best_epoch})")
if no_improve < PATIENCE:
    print(f"Training completed all {EPOCHS} epochs without early stopping")
