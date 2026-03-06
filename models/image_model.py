import torch
import torch.nn as nn
from torchvision import models

class DeepShieldImageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)  # ❗ RAW LOGITS (no sigmoid)
