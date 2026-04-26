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