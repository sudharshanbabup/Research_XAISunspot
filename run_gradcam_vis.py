"""
run_gradcam_vis.py
==================
Generate Grad-CAM saliency overlays and compute quantitative XAI metrics.

Outputs
-------
  results/gradcam/low_isn_overlay.png
  results/gradcam/moderate_isn_overlay.png
  results/gradcam/high_isn_overlay.png
  results/gradcam/xai_metrics.json

Usage
-----
  python run_gradcam_vis.py \\
      --ckpt  checkpoints/best_resnet50_isn.pt \\
      --csv   data/image_isn_pairs.csv \\
      --image_dir data/images/ \\
      --output_dir results/gradcam/
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from dataset     import make_loaders, HMIDataset, IMAGENET_MEAN, IMAGENET_STD
from model       import build_model
from gradcam     import GradCAM
from xai_metrics import run_xai_evaluation


# ── Denormalise tensor to uint8 numpy ────────────────────────────────────────

def denorm(tensor):
    """
    Reverse ImageNet normalisation and return H×W uint8 numpy grayscale.
    """
    t = tensor.squeeze(0)                        # (3, H, W)
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD ).view(3, 1, 1)
    t = (t * s + m).clamp(0, 1)
    img = (t[0].numpy() * 255).astype(np.uint8)  # use channel 0 (grayscale)
    return img


# ── Save a side-by-side overlay panel ────────────────────────────────────────

def save_overlay_panel(
    orig_img:  np.ndarray,     # H×W uint8 grayscale
    saliency:  np.ndarray,     # H×W float in [0,1]
    isn_gt:    float,
    isn_pred:  float,
    out_path:  str,
    title:     str = "",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # Left: original HMI image
    axes[0].imshow(orig_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"HMI image\nISN = {isn_gt:.0f}", fontsize=10)
    axes[0].axis("off")

    # Right: Grad-CAM overlay
    heatmap = cm.jet(saliency)[..., :3]
    base    = np.stack([orig_img / 255.0] * 3, axis=-1)
    overlay = np.clip(0.55 * base + 0.45 * heatmap, 0, 1)
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Grad-CAM saliency\npred = {isn_pred:.1f} ISN", fontsize=10)
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    ckpt_path:   str = "checkpoints/best_resnet50_isn.pt",
    csv_path:    str = "data/image_isn_pairs.csv",
    image_dir:   str = "data/images/",
    output_dir:  str = "results/gradcam/",
    n_vis:       int = 3,       # one per ISN regime
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model(pretrained=False, device=str(device))
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Build test dataset (no shuffle) ──────────────────────────────────────
    loaders = make_loaders(csv_path, image_dir, batch_size=1)
    test_ds = HMIDataset(csv_path, image_dir, split="test", augment=False)
    gcam    = GradCAM(model)

    # ── Find representative images per ISN regime ─────────────────────────────
    #   Low     : ISN  < 20
    #   Moderate: 50 ≤ ISN < 100
    #   High    : ISN ≥ 150

    regime_targets = {
        "low_isn"      : lambda v: v < 20,
        "moderate_isn" : lambda v: 50 <= v < 100,
        "high_isn"     : lambda v: v >= 150,
    }
    regime_done = {k: False for k in regime_targets}

    print("\nGenerating Grad-CAM overlays …")
    for idx in range(len(test_ds)):
        img_tensor, isn_gt = test_ds[idx]
        isn_gt = float(isn_gt)

        for regime, check in regime_targets.items():
            if regime_done[regime] or not check(isn_gt):
                continue

            img_in = img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                isn_pred = float(model(img_in).item())

            saliency = gcam.generate(img_in)       # (H, W)
            orig_img = denorm(img_tensor)           # H×W uint8

            label = regime.replace("_", " ").title()
            save_overlay_panel(
                orig_img, saliency,
                isn_gt, isn_pred,
                out_path=os.path.join(output_dir, f"{regime}_overlay.png"),
                title=f"{label}  (ISN gt={isn_gt:.0f},  pred={isn_pred:.1f})",
            )
            regime_done[regime] = True

        if all(regime_done.values()):
            break

    # ── Quantitative XAI metrics ──────────────────────────────────────────────
    print("\nComputing quantitative XAI metrics (may take a few minutes) …")
    xai_results = run_xai_evaluation(
        model, loaders["test"], gcam, device=str(device))

    metrics_path = os.path.join(output_dir, "xai_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(xai_results, f, indent=2)
    print(f"  Saved: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualisation + XAI metric computation")
    parser.add_argument("--ckpt",       default="checkpoints/best_resnet50_isn.pt")
    parser.add_argument("--csv",        default="data/image_isn_pairs.csv")
    parser.add_argument("--image_dir",  default="data/images/")
    parser.add_argument("--output_dir", default="results/gradcam/")
    args = parser.parse_args()

    main(
        ckpt_path  = args.ckpt,
        csv_path   = args.csv,
        image_dir  = args.image_dir,
        output_dir = args.output_dir,
    )
