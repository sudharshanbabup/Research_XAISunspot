"""
xai_metrics.py
==============
Quantitative XAI faithfulness metrics (Table VIII in paper):

  1. Deletion AUC (Samek et al. 2021)
     Pixels masked in descending saliency order; R² re-evaluated at
     each step. AUC under the R²-vs-fraction curve.
     Lower AUC → faster R² degradation → more faithful saliency map.
     ResNet50+Aug: 0.31  |  Random-mask baseline: 0.47

  2. Simulated IoU
     Grad-CAM map thresholded at 70% of peak value → binary mask.
     Compared with synthetic sunspot masks from HMI intensity thresholding.
     IoU = |intersection| / |union|.
     ResNet50+Aug: 0.61

Usage:
  from xai_metrics import deletion_auc, simulated_iou
"""

import numpy as np
import torch
from sklearn.metrics import r2_score
from tqdm import tqdm


# ── Deletion AUC ─────────────────────────────────────────────────────────────

def deletion_auc(
    model,
    loader,
    gradcam,
    n_steps:     int   = 10,
    device:      str   = "cpu",
    verbose:     bool  = True,
) -> float:
    """
    Compute deletion AUC (Samek et al. 2021, [8] in paper).

    Algorithm
    ---------
    For fraction f ∈ {0.1, 0.2, …, 1.0}:
      1. Generate Grad-CAM saliency map for each test image.
      2. Mask the top-f fraction of pixels (highest saliency first).
      3. Re-evaluate R² on the masked images.
    AUC = area under the R²(f) curve (trapezoid rule).

    A genuinely informative saliency map causes rapid R² degradation
    (low AUC); a random mask degrades R² more slowly (high AUC ≈ 0.47).

    Parameters
    ----------
    model    : ResNet50ISN (eval mode)
    loader   : DataLoader  (test split, any batch size)
    gradcam  : GradCAM instance
    n_steps  : number of deletion fractions in [0.1, 1.0]
    device   : 'cpu' or 'cuda'

    Returns
    -------
    auc : float
    """
    fractions    = np.linspace(0.1, 1.0, n_steps)
    r2_per_step  = []

    model.eval()
    dev = torch.device(device)

    for frac in tqdm(fractions, desc="Deletion AUC", disable=not verbose):
        all_preds, all_tgts = [], []

        for imgs, labels in loader:
            imgs   = imgs.to(dev)

            # Generate saliency maps for each image in the mini-batch
            with torch.enable_grad():
                batch_saliency = []
                for j in range(len(imgs)):
                    cam = gradcam.generate(imgs[j:j+1])   # (H, W) in [0,1]
                    batch_saliency.append(cam)

            # Build pixel masks: top-frac by saliency
            masks = []
            for cam in batch_saliency:
                flat  = cam.flatten()
                k     = max(1, int(frac * len(flat)))
                # Use -k-th value as threshold (partition trick)
                thresh = np.partition(flat, -k)[-k]
                mask   = (cam >= thresh).astype(np.float32)    # (H, W)
                masks.append(mask)

            # Apply masks and run forward pass
            mask_tensor = torch.from_numpy(
                np.stack(masks, axis=0)          # (B, H, W)
            ).unsqueeze(1).to(dev)               # (B, 1, H, W)

            imgs_masked = imgs * (1.0 - mask_tensor)

            with torch.no_grad():
                preds = model(imgs_masked)

            all_preds.append(preds.cpu().numpy())
            all_tgts.append(labels.numpy())

        r2 = r2_score(
            np.concatenate(all_tgts),
            np.concatenate(all_preds),
        )
        r2_per_step.append(max(float(r2), 0.0))

    # Trapezoid AUC
    auc = float(np.trapz(r2_per_step, fractions))

    if verbose:
        print(f"\n  Deletion AUC = {auc:.3f}  "
              f"(random-mask baseline ≈ 0.47; ours: 0.31)")
        print(f"  R² at each deletion step: "
              + "  ".join(f"{r:.3f}" for r in r2_per_step))

    return auc


# ── Simulated IoU ─────────────────────────────────────────────────────────────

def simulated_iou(
    model,
    loader,
    gradcam,
    saliency_threshold:  float = 0.70,
    intensity_threshold: float = 0.85,
    device:              str   = "cpu",
    verbose:             bool  = True,
) -> float:
    """
    Compute simulated IoU between Grad-CAM masks and synthetic sunspot masks.

    Grad-CAM binary mask
    --------------------
      Saliency map thresholded at `saliency_threshold` × peak value.

    Synthetic sunspot mask
    ----------------------
      HMI intensity thresholding: pixels whose (un-normalised) intensity
      falls below `intensity_threshold` are labelled as sunspot candidates.
      In practice we approximate this from the normalised tensor values.

    IoU = |Grad-CAM mask ∩ sunspot mask| / |Grad-CAM mask ∪ sunspot mask|

    Parameters
    ----------
    saliency_threshold  : float  -- fraction of peak Grad-CAM used for binarisation
    intensity_threshold : float  -- un-normalised intensity fraction for sunspot mask

    Returns
    -------
    mean_iou : float
    """
    model.eval()
    dev      = torch.device(device)
    iou_list = []

    for imgs, _ in tqdm(loader, desc="Simulated IoU", disable=not verbose):
        imgs = imgs.to(dev)

        for j in range(len(imgs)):
            img_j = imgs[j:j+1]                         # (1, 3, H, W)

            # ── Grad-CAM binary mask ──────────────────────────────────────────
            with torch.enable_grad():
                cam = gradcam.generate(img_j)            # (H, W) in [0,1]
            sal_mask = (cam >= saliency_threshold * cam.max()).astype(np.uint8)

            # ── Synthetic sunspot mask from HMI intensity ─────────────────────
            # Un-normalise channel 0 (all channels identical for grayscale HMI)
            # ImageNet: mean=0.485, std=0.229  →  intensity ≈ pixel*std + mean
            mean0, std0 = 0.485, 0.229
            intensity   = (img_j[0, 0].cpu().numpy() * std0 + mean0)
            # Sunspots are dark (low intensity); threshold at fraction of median
            disk_median = float(np.median(intensity[intensity > 0.1]))
            ssn_mask    = (intensity < intensity_threshold * disk_median
                           ).astype(np.uint8)

            if ssn_mask.sum() == 0:                      # skip blank images
                continue

            intersection = int((sal_mask & ssn_mask).sum())
            union        = int((sal_mask | ssn_mask).sum())
            if union > 0:
                iou_list.append(intersection / union)

    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0

    if verbose:
        print(f"\n  Simulated IoU = {mean_iou:.3f}  "
              f"(n_samples={len(iou_list)}, target ≈ 0.61)")

    return mean_iou


# ── Combined evaluation ───────────────────────────────────────────────────────

def run_xai_evaluation(
    model,
    loader,
    gradcam,
    device: str = "cpu",
) -> dict:
    """
    Run both XAI metrics and print a summary table.

    Returns
    -------
    dict with keys 'deletion_auc' and 'simulated_iou'
    """
    print("\n" + "═" * 50)
    print("  Quantitative Grad-CAM Faithfulness Evaluation")
    print("═" * 50)

    auc = deletion_auc(model, loader, gradcam, device=device)
    iou = simulated_iou(model, loader, gradcam, device=device)

    print("\n  ┌─────────────────────┬────────┬──────────────┐")
    print("  │ Metric              │  Ours  │  Target      │")
    print("  ├─────────────────────┼────────┼──────────────┤")
    print(f"  │ Deletion AUC  (↓)  │ {auc:.3f}  │  0.31        │")
    print(f"  │ Simulated IoU (↑)  │ {iou:.3f}  │  0.61        │")
    print("  └─────────────────────┴────────┴──────────────┘")

    return dict(deletion_auc=auc, simulated_iou=iou)
