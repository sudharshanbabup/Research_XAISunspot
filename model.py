"""
model.py
========
ResNet50-based ISN regression model + weighted-MSE loss.

Architecture (Section IV of paper):
  ResNet50 (ImageNet pretrained)
  → Global Average Pooling
  → FC-512 → ReLU → Dropout(0.3)
  → FC-1  (linear scalar output = predicted ISN)

Two-phase training:
  Phase 1 (epochs 1–20)   : Stages 1–3 frozen, only head trained
  Phase 2 (epochs 21–200) : Stages 4–5 unfrozen for fine-tuning

Weighted MSE (Eq. 1 in paper):
  L_w = (1/N) Σ w(y_i)(ŷ_i - y_i)²
  w(y_i) = 1 + λ · 1[y_i ≥ threshold]      λ=2.0, threshold=150
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ── Model ────────────────────────────────────────────────────────────────────

class ResNet50ISN(nn.Module):
    """
    ResNet50 backbone adapted for scalar ISN regression.

    Parameters
    ----------
    pretrained : bool   -- initialise from ImageNet-1K weights
    dropout    : float  -- dropout probability before the output FC
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()

        # Load backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Replace 1000-class head with regression head
        in_features = backbone.fc.in_features          # 2048
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),                         # scalar output
        )
        self.backbone = backbone

        # Track phase for logging
        self._phase = 0

    # ── Phase-1 : freeze stages 1–3 ─────────────────────────────────────────
    def freeze_stages_1_3(self) -> None:
        """
        Freeze conv1, bn1, layer1, layer2, layer3.
        Only the regression head (and later layer4) will be trained.
        Phase 1: epochs 1–20.
        """
        frozen_prefixes = ("conv1", "bn1", "layer1", "layer2", "layer3")
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(p) for p in frozen_prefixes):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        self._phase = 1
        self._log_trainable("Phase 1 (head only)")

    # ── Phase-2 : unfreeze stages 4–5 ───────────────────────────────────────
    def unfreeze_stages_4_5(self) -> None:
        """
        Unfreeze layer4 and the FC regression head while keeping
        stages 1–3 explicitly frozen.
        Phase 2: epochs 21–200.
        """
        frozen_prefixes = ("conv1", "bn1", "layer1", "layer2", "layer3")
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(p) for p in frozen_prefixes):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        self._phase = 2
        self._log_trainable("Phase 2 (layer4 + head)")

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W) normalised image tensor

        Returns
        -------
        (B,) predicted ISN values
        """
        return self.backbone(x).squeeze(1)

    # ── Utility ──────────────────────────────────────────────────────────────
    def _log_trainable(self, label: str) -> None:
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"  [{label}] trainable params: {n_train:,} / {n_total:,} "
              f"({100 * n_train / n_total:.1f}%)")

    @property
    def phase(self) -> int:
        return self._phase


# ── Loss ─────────────────────────────────────────────────────────────────────

class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss (Eq. 1 in paper) to address high-ISN class imbalance.

        L_w = (1/N) Σ_i  w(y_i) · (ŷ_i − y_i)²

    where  w(y_i) = 1 + λ · 1[y_i ≥ threshold]

    With λ=2.0 and threshold=150, high-activity samples contribute
    3× more to the loss, equivalent to synthetic 3× oversampling.

    Parameters
    ----------
    threshold : float  -- ISN level above which extra weight is applied
    lam       : float  -- extra weight multiplier  (w = 1 + lam  for ISN ≥ threshold)
    """

    def __init__(self, threshold: float = 150.0, lam: float = 2.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.lam       = lam

    def forward(
        self,
        pred:   torch.Tensor,   # (B,) predicted ISN
        target: torch.Tensor,   # (B,) ground-truth ISN
    ) -> torch.Tensor:
        weights = 1.0 + self.lam * (target >= self.threshold).float()
        return (weights * (pred - target).pow(2)).mean()

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, lam={self.lam}"


# ── Convenience factory ───────────────────────────────────────────────────────

def build_model(
    pretrained: bool = True,
    dropout:    float = 0.3,
    device:     str   = "cpu",
) -> ResNet50ISN:
    """Construct and move model to device."""
    model = ResNet50ISN(pretrained=pretrained, dropout=dropout)
    return model.to(device)


def build_loss(threshold: float = 150.0, lam: float = 2.0) -> WeightedMSELoss:
    """Construct the weighted-MSE criterion."""
    return WeightedMSELoss(threshold=threshold, lam=lam)
