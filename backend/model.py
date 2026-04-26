"""
Model definitions — MUST match the architecture used during training (NB2).
Loads weights from ensemble_best.safetensors.
"""

import torch
import torch.nn as nn
import timm
import yaml
from safetensors.torch import load_file


class ConvNextBranch(nn.Module):
    """ConvNeXt-V2: detects local pixel-level manipulation artifacts."""

    def __init__(self, model_name="convnextv2_base", pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(self.dropout(features)), features


class SwinBranch(nn.Module):
    """Swin-V2: detects global lighting/shadow inconsistencies."""

    def __init__(self, model_name="swinv2_base_window12to16_192to256", pretrained=False):
        super().__init__()
        try:
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0
            )
        except Exception:
            self.backbone = timm.create_model(
                "swinv2_base_window12_256", pretrained=pretrained, num_classes=0
            )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(self.dropout(features)), features


class EnsembleDeepfakeDetector(nn.Module):
    """
    Soft-voting ensemble of ConvNeXt-V2 and Swin-V2.
    Architecture must match the trained weights exactly.
    """

    def __init__(self, config: dict = None, config_path: str = None):
        super().__init__()

        if config is None and config_path is not None:
            with open(config_path) as f:
                config = yaml.safe_load(f)

        if config is None:
            config = {
                "model": {
                    "ensemble": {
                        "cnn_branch": "convnextv2_base",
                        "vit_branch": "swinv2_base_window12to16_192to256",
                    },
                    "pretrained": False,
                }
            }

        cfg = config["model"]
        self.cnn_branch = ConvNextBranch(
            cfg["ensemble"]["cnn_branch"], cfg.get("pretrained", False)
        )
        self.vit_branch = SwinBranch(
            cfg["ensemble"]["vit_branch"], cfg.get("pretrained", False)
        )
        self.temperature_cnn = nn.Parameter(torch.ones(1))
        self.temperature_vit = nn.Parameter(torch.ones(1))

    def forward(self, x):
        cnn_logits, cnn_feat = self.cnn_branch(x)
        vit_logits, vit_feat = self.vit_branch(x)
        scaled_cnn = cnn_logits / self.temperature_cnn
        scaled_vit = vit_logits / self.temperature_vit
        ensemble = (scaled_cnn + scaled_vit) / 2.0
        return ensemble, scaled_cnn, scaled_vit

    def load_safetensors(self, path):
        state = load_file(path)
        self.load_state_dict(state)
        print(f"Loaded weights from {path}")
