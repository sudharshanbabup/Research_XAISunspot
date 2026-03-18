"""
evaluate.py
===========
Load the best checkpoint and run complete test-set evaluation.

Outputs
-------
  results/test_metrics.json        -- all scalar metrics
  results/scatter_pred_vs_true.png -- Fig. 3 equivalent
  results/residual_histogram.png   -- Fig. 6 equivalent
  results/timeseries_forecast.png  -- Fig. 4 equivalent
  results/per_period_metrics.csv   -- Table VII equivalent

Usage
-----
  python evaluate.py --ckpt checkpoints/best_resnet50_isn.pt \\
                     --csv  data/image_isn_pairs.csv \\
                     --image_dir data/images/
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from dataset import make_loaders
from model   import build_model

# ── Colour palette matching the paper ────────────────────────────────────────
C_BLUE   = "#1F77B4"
C_ORANGE = "#DC640A"
C_GREEN  = "#2CA02C"
C_RED    = "#B41E1E"
C_PURPLE = "#785096"


# ── Metrics helper ────────────────────────────────────────────────────────────

def metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mask = targets > 0
    if mask.sum() > 0:
        mape = float(np.mean(
            np.abs((preds[mask] - targets[mask]) / targets[mask]))) * 100
    else:
        mape = float('nan')

    if len(targets) >= 2:
        r2 = float(r2_score(targets, preds))
        pr = float(pearsonr(targets, preds)[0])
    else:
        r2 = float('nan')
        pr = float('nan')
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2, r=pr)


# ── Scatter: predicted vs actual ──────────────────────────────────────────────

def plot_scatter(preds, targets, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    low  = targets < 50
    mid  = (targets >= 50) & (targets < 120)
    high = targets >= 120

    ax.scatter(targets[low],  preds[low],  s=8,  alpha=0.55,
               c=C_BLUE,   label="Low  (ISN < 50)")
    ax.scatter(targets[mid],  preds[mid],  s=10, alpha=0.55,
               c=C_ORANGE, marker="D", label="Moderate (50–120)")
    ax.scatter(targets[high], preds[high], s=10, alpha=0.60,
               c=C_RED,    marker="s", label="High  (≥ 120)")

    lim = [0, max(targets.max(), preds.max()) + 10]
    ax.plot(lim, lim, "k--", linewidth=1.2, label="Identity")

    # ±1σ band
    sigma = float(np.std(preds - targets))
    xs    = np.array(lim)
    ax.fill_between(xs, xs - sigma, xs + sigma,
                    alpha=0.10, color=C_BLUE)

    # ISN = 200 vertical line
    ax.axvline(200, color="gray", linewidth=0.8, linestyle="--")
    ax.text(202, lim[0] + 5, "ISN = 200", fontsize=7, color="gray", rotation=90)

    # Annotation box
    r2_val = r2_score(targets, preds)
    ax.text(0.62, 0.12,
            f"$R^2 = {r2_val:.2f}$\n$n = {len(targets):,}$",
            transform=ax.transAxes,
            fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8))

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual ISN (Daily, SILSO v2.0)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Predicted ISN (ResNet50+Aug)",   fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linestyle=":", color="gray", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Residual histogram ────────────────────────────────────────────────────────

def plot_residual_histogram(preds, targets, out_path):
    residuals = preds - targets
    mu, sigma = float(residuals.mean()), float(residuals.std())

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # 68% band
    ax.axvspan(-12, 12, color=C_PURPLE, alpha=0.10, zorder=0)

    # Bars
    bins  = np.arange(-85, 90, 10)
    ax.hist(residuals, bins=bins, color=C_BLUE, alpha=0.70,
            edgecolor=C_BLUE, linewidth=0.5, zorder=1)

    # Gaussian fit
    xs   = np.linspace(-85, 85, 300)
    n_   = len(residuals)
    bw   = bins[1] - bins[0]
    gauss = (n_ * bw / (sigma * np.sqrt(2 * np.pi))
             * np.exp(-0.5 * ((xs - mu) / sigma) ** 2))
    ax.plot(xs, gauss, color=C_RED, linewidth=1.8, zorder=2)

    # ±12 ISN dashed lines
    pct_in_band = float(np.mean(np.abs(residuals) <= 12)) * 100
    for x_ in (-12, 12):
        ax.axvline(x_, color=C_PURPLE, linewidth=1.4, linestyle="--", zorder=3)
    ax.annotate("", xy=(12, ax.get_ylim()[1] * 0.93),
                xytext=(-12, ax.get_ylim()[1] * 0.93),
                arrowprops=dict(arrowstyle="<->", color=C_PURPLE, lw=1.2))
    ax.text(0, ax.get_ylim()[1] * 0.96, f"{pct_in_band:.1f}%",
            ha="center", va="top", fontsize=8.5,
            color=C_PURPLE,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=C_PURPLE, lw=0.7))

    # Stats box
    ax.text(0.72, 0.78,
            f"$\\mu = {mu:.1f}$ ISN\n$\\sigma \\approx {sigma:.0f}$ ISN",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="gray", lw=0.8))

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.text(0, ax.get_ylim()[1] * 0.99, "$\\mu \\approx 0$",
            ha="center", va="top", fontsize=8)

    ax.set_xlabel("Residual $\\hat{y} - y$ (ISN units)",
                  fontsize=10, fontweight="bold")
    ax.set_ylabel("Frequency (count)", fontsize=10, fontweight="bold")
    ax.set_xlim(-90, 90)
    ax.grid(True, axis="y", linestyle=":", color="gray", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Time-series forecast plot ─────────────────────────────────────────────────

def plot_timeseries(preds, targets, dates, out_path):
    if len(dates) == 0:
        print(f"  Skipped (no data): {out_path}")
        return
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    # Solar Cycle 25 onset shading
    sc25_start = pd.Timestamp("2021-01-01")
    ax.axvspan(sc25_start, dates[-1], color="gray", alpha=0.08,
               label="_nolegend_")
    ax.text(sc25_start + pd.Timedelta(days=10),
            max(targets) * 0.92, "Solar Cycle 25",
            fontsize=7.5, color="gray", rotation=90, va="top")

    ax.plot(dates, targets, color="black", linewidth=1.8, label="Ground Truth")
    ax.plot(dates, preds,   color=C_BLUE,  linewidth=1.2,
            linestyle="--",  label="ResNet50+Aug (predicted)")

    ax.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax.set_ylabel("ISN",  fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, linestyle=":", color="gray", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Per-period breakdown ──────────────────────────────────────────────────────

def per_period_metrics(preds, targets, dates) -> pd.DataFrame:
    dates   = pd.to_datetime(dates)
    records = []
    periods = {
        "2020 (SC Min.)":    (dates.year == 2020),
        "2021–2022 (Rising)":((dates.year == 2021) | (dates.year == 2022)),
        "Full test":         (dates.year >= 2020),
    }
    for label, mask in periods.items():
        if mask.sum() == 0:
            continue
        m = metrics(preds[mask], targets[mask])
        records.append({"Period": label, **{k: round(v, 3) for k, v in m.items()}})
    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(
    ckpt_path:  str = "checkpoints/best_resnet50_isn.pt",
    csv_path:   str = "data/image_isn_pairs.csv",
    image_dir:  str = "data/images/",
    output_dir: str = "results/",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model(pretrained=False, device=str(device))
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}  "
          f"val_MAE={ckpt['val_mae']:.2f}  val_R2={ckpt.get('val_r2', '?')}")

    # ── Build test loader ─────────────────────────────────────────────────────
    loaders   = make_loaders(csv_path, image_dir)

    all_preds, all_tgts = [], []
    with torch.no_grad():
        for imgs, labels in loaders["test"]:
            imgs = imgs.to(device)
            all_preds.append(model(imgs).cpu().numpy())
            all_tgts.append(labels.numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_tgts)

    # ── Scalar metrics ────────────────────────────────────────────────────────
    test_m = metrics(preds, targets)
    print("\nTest-set metrics (2020–2022, n=901):")
    for k, v in test_m.items():
        print(f"  {k:6s}: {v:.4f}")

    # Save JSON
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_m, f, indent=2)

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_scatter(
        preds, targets,
        os.path.join(output_dir, "scatter_pred_vs_true.png"))

    plot_residual_histogram(
        preds, targets,
        os.path.join(output_dir, "residual_histogram.png"))

    # Build date array for the test set
    test_ds = loaders["test"].dataset
    dates   = pd.to_datetime(test_ds.df["date"].values)
    plot_timeseries(
        preds, targets, dates,
        os.path.join(output_dir, "timeseries_forecast.png"))

    # ── Per-period breakdown ──────────────────────────────────────────────────
    df_pp = per_period_metrics(preds, targets, dates)
    csv_out = os.path.join(output_dir, "per_period_metrics.csv")
    df_pp.to_csv(csv_out, index=False)
    print(f"\nPer-period breakdown:\n{df_pp.to_string(index=False)}")
    print(f"  Saved: {csv_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet50+Aug ISN model")
    parser.add_argument("--ckpt",       default="checkpoints/best_resnet50_isn.pt")
    parser.add_argument("--csv",        default="data/image_isn_pairs.csv")
    parser.add_argument("--image_dir",  default="data/images/")
    parser.add_argument("--output_dir", default="results/")
    args = parser.parse_args()

    evaluate(
        ckpt_path  = args.ckpt,
        csv_path   = args.csv,
        image_dir  = args.image_dir,
        output_dir = args.output_dir,
    )
