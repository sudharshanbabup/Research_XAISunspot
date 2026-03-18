"""
train.py
========
Two-phase training loop for ResNet50 ISN regression.

Phase 1 (epochs 1–20)  : backbone stages 1–3 frozen; only head trained.
Phase 2 (epochs 21–200): stages 4–5 unfrozen for task-specific fine-tuning.

Optimiser : Adam  lr=1e-4, weight_decay=1e-5
Scheduler : ReduceLROnPlateau  factor=0.5, patience=10
Early stop: patience=25

Usage:
  python train.py --csv data/image_isn_pairs.csv --image_dir data/images/
"""

import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from dataset import make_loaders
from model   import build_model, build_loss


# ── Hyper-parameters ─────────────────────────────────────────────────────────
HPARAMS = dict(
    lr             = 1e-4,
    weight_decay   = 1e-5,
    batch_size     = 32,
    max_epochs     = 200,
    phase1_epochs  = 20,      # freeze stages 1–3 for this many epochs
    patience_lr    = 10,      # ReduceLROnPlateau patience
    patience_es    = 25,      # early-stopping patience
    seed           = 42,
    dropout        = 0.3,
    wloss_threshold= 150.0,
    wloss_lam      = 2.0,
)


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(
    preds:   np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """Return MAE, RMSE, MAPE, R², Pearson-r."""
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mask = targets > 0
    mape = float(np.mean(np.abs((preds[mask] - targets[mask])
                                / targets[mask]))) * 100 if mask.any() else 0.0
    r2   = float(r2_score(targets, preds))
    pr   = float(pearsonr(targets, preds)[0])
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2, r=pr)


# ── Single epoch ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device) \
        -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_tgts = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        all_preds.append(preds.detach().cpu().numpy())
        all_tgts.append(labels.cpu().numpy())

    p = np.concatenate(all_preds)
    t = np.concatenate(all_tgts)
    return total_loss / len(loader.dataset), float(np.mean(np.abs(p - t)))


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device) \
        -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_tgts = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)
        total_loss += loss.item() * len(labels)
        all_preds.append(preds.cpu().numpy())
        all_tgts.append(labels.cpu().numpy())

    p = np.concatenate(all_preds)
    t = np.concatenate(all_tgts)
    return total_loss / len(loader.dataset), compute_metrics(p, t), p, t


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    csv_path:   str = "data/image_isn_pairs.csv",
    image_dir:  str = "data/images/",
    ckpt_dir:   str = "checkpoints/",
    num_workers: int = 4,
) -> Tuple[object, dict, dict]:

    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed(HPARAMS["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = make_loaders(
        csv_path, image_dir,
        batch_size=HPARAMS["batch_size"],
        num_workers=num_workers,
    )

    # ── Model & loss ──────────────────────────────────────────────────────────
    model = build_model(pretrained=True, dropout=HPARAMS["dropout"],
                        device=str(device))
    model.freeze_stages_1_3()

    criterion = build_loss(
        threshold=HPARAMS["wloss_threshold"],
        lam=HPARAMS["wloss_lam"],
    ).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HPARAMS["lr"],
        weight_decay=HPARAMS["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=0.5, patience=HPARAMS["patience_lr"],
    )

    best_val_mae   = float("inf")
    no_improve     = 0
    history        = dict(train_loss=[], train_mae=[],
                          val_loss=[],   val_mae=[],   lr=[]) 

    print(f"
{'─'*65}")
    print(f" Epoch │ Train Loss │ Train MAE │  Val MAE  │  Val R²  │  time")
    print(f"{'─'*65}")

    for epoch in range(1, HPARAMS["max_epochs"] + 1):

        # ── Phase transition ──────────────────────────────────────────────────
        if epoch == HPARAMS["phase1_epochs"] + 1:
            print(f"
  ▶ Epoch {epoch}: switching to Phase 2 – unfreezing stages 4–5")
            model.unfreeze_stages_4_5()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=HPARAMS["lr"],
                weight_decay=HPARAMS["weight_decay"],
            )
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min",
                factor=0.5, patience=HPARAMS["patience_lr"],
            )
            no_improve   = 0
            best_val_mae = float("inf")

        t0 = time.time()
        tr_loss, tr_mae = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device)
        vl_loss, vl_met, _, _ = evaluate_epoch(
            model, loaders["val"], criterion, device)

        val_mae = vl_met["MAE"]
        scheduler.step(vl_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        # ── Log ───────────────────────────────────────────────────────────────
        for k, v in zip(
            ["train_loss","train_mae","val_loss","val_mae","lr"],
            [tr_loss, tr_mae, vl_loss, val_mae, lr_now],
        ): 
            history[k].append(v)

        print(f"  {epoch:4d}  │  {tr_loss:8.2f}  │  {tr_mae:7.2f}  │"
              f"  {val_mae:7.2f}  │  {vl_met['R2']:6.3f}  │"
              f"  {time.time()-t0:.1f}s")

        # ── Checkpoint ───────────────────────────────────────────────────────
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve   = 0
            ckpt_path    = os.path.join(ckpt_dir, "best_resnet50_isn.pt")
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "val_mae":        val_mae,
                "val_r2":         vl_met["R2"],
                "hparams":        HPARAMS,
            }, ckpt_path)
            print(f"         ✓ checkpoint saved  val_MAE={best_val_mae:.2f}")
        else:
            no_improve += 1
            if no_improve >= HPARAMS["patience_es"]:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {HPARAMS['patience_es']} epochs)")
                break

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(" Loading best checkpoint for test-set evaluation …")
    ckpt = torch.load(os.path.join(ckpt_dir, "best_resnet50_isn.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    _, test_metrics, test_preds, test_targets = evaluate_epoch(
        model, loaders["test"], criterion, device)

    print("\n TEST METRICS (2020–2022, n=901)")
    for k, v in test_metrics.items():
        print(f"  {k:6s}: {v:.4f}")

    np.save(os.path.join(ckpt_dir, "test_preds.npy"),   test_preds)
    np.save(os.path.join(ckpt_dir, "test_targets.npy"), test_targets)
    print(f"\n  Predictions saved to {ckpt_dir}")

    return model, history, test_metrics


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet50+Aug for ISN regression")
    parser.add_argument("--csv",        default="data/image_isn_pairs.csv")
    parser.add_argument("--image_dir",  default="data/images/")
    parser.add_argument("--ckpt_dir",   default="checkpoints/")
    parser.add_argument("--num_workers",type=int, default=4)
    args = parser.parse_args()

    train(
        csv_path   = args.csv,
        image_dir  = args.image_dir,
        ckpt_dir   = args.ckpt_dir,
        num_workers= args.num_workers,
    )
