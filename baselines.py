"""
baselines.py
============
ARIMA and LSTM time-series baselines for ISN prediction.

Both operate solely on the daily ISN record (no image input),
providing a fair comparison against the image-based CNN approach.

Usage
-----
  python baselines.py --silso_csv data/SN_d_tot_V2.0.csv
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr


# ══════════════════════════════════════════════════════════════════════════════
#  ARIMA(5,1,2) Baseline
# ══════════════════════════════════════════════════════════════════════════════

def run_arima(
    isn_series:  pd.Series,
    test_size:   int  = 901,
    order:       tuple = (5, 1, 2),
    verbose:     bool  = True,
) -> tuple:
    """
    Rolling one-step-ahead ARIMA forecast on daily ISN.

    The model is re-fitted at every test step (true online forecasting).
    Matches exactly the split used by the CNN: train ≤ 2018,
    val = 2019, test = 2020-2022.

    Requires: pip install statsmodels

    Parameters
    ----------
    isn_series : pd.Series  -- full daily ISN series (sorted by date)
    test_size  : int        -- number of test steps
    order      : tuple      -- (p, d, q) for ARIMA

    Returns
    -------
    preds   : np.ndarray  shape (test_size,)
    targets : np.ndarray  shape (test_size,)
    metrics : dict
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise ImportError("pip install statsmodels")

    train_vals = isn_series.iloc[:-test_size].values.astype(float)
    test_vals  = isn_series.iloc[-test_size:].values.astype(float)
    history    = list(train_vals)
    preds      = []

    if verbose:
        from tqdm import tqdm
        iterator = tqdm(range(test_size), desc=f"ARIMA{order}")
    else:
        iterator = range(test_size)

    for t in iterator:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model  = ARIMA(history, order=order)
            result = model.fit()
        preds.append(float(result.forecast(steps=1)[0]))
        history.append(test_vals[t])

    preds   = np.array(preds)
    targets = test_vals

    m = _metrics(preds, targets)
    if verbose:
        print(f"\nARIMA{order} test metrics:")
        _print_metrics(m)
    return preds, targets, m


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM Baseline
# ══════════════════════════════════════════════════════════════════════════════

class ISNWindowDataset(Dataset):
    """Sliding-window dataset for LSTM.  X: window of ISN; y: next ISN."""

    def __init__(self, series: np.ndarray, window: int = 30) -> None:
        self.X = []
        self.y = []
        for i in range(len(series) - window):
            self.X.append(series[i : i + window])
            self.y.append(series[i + window])
        self.X = torch.from_numpy(np.array(self.X, dtype=np.float32)).unsqueeze(-1)
        self.y = torch.from_numpy(np.array(self.y, dtype=np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ISNLSTMModel(nn.Module):
    """Two-layer LSTM with dropout for ISN one-step-ahead prediction."""

    def __init__(
        self,
        input_size:  int   = 1,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, seq_len, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)   # (B,)


def run_lstm(
    isn_series:  pd.Series,
    test_size:   int   = 901,
    window:      int   = 30,
    hidden_size: int   = 64,
    num_layers:  int   = 2,
    lr:          float = 1e-3,
    epochs:      int   = 50,
    batch_size:  int   = 64,
    seed:        int   = 42,
    verbose:     bool  = True,
) -> tuple:
    """
    Train and evaluate the LSTM ISN baseline.

    Parameters
    ----------
    isn_series : pd.Series  -- full daily ISN series
    test_size  : int        -- number of test-set samples
    window     : int        -- look-back window length

    Returns
    -------
    preds, targets, metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vals   = isn_series.values.astype(np.float32)
    # Normalise by training-set max for stable training
    train_max = float(vals[:-test_size].max())
    if train_max == 0:
        train_max = 1.0
    vals_norm = vals / train_max

    # Split
    n_train  = len(vals) - test_size - window
    train_ds = ISNWindowDataset(vals_norm[:n_train + window], window)
    test_ds  = ISNWindowDataset(vals_norm[n_train:],          window)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model     = ISNLSTMModel(hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # ── Training ──────────────────────────────────────────────────────────────
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(y)
        scheduler.step()
        if verbose and ep % 10 == 0:
            print(f"  LSTM epoch {ep:3d}/{epochs}  "
                  f"train_loss={ep_loss / len(train_ds):.4f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for X, y in test_loader:
            p = model(X.to(device)).cpu().numpy() * train_max
            all_preds.append(p)
            all_tgts.append(y.numpy() * train_max)

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_tgts)

    m = _metrics(preds, targets)
    if verbose:
        print(f"\nLSTM test metrics:")
        _print_metrics(m)
    return preds, targets, m


# ══════════════════════════════════════════════════════════════════════════════
#  Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def _metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(targets, preds))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mask = targets > 0
    if mask.any():
        mape = float(np.mean(
            np.abs((preds[mask] - targets[mask]) / targets[mask]))) * 100
    else:
        mape = 0.0
    r2   = float(r2_score(targets, preds))
    try:
        pr = float(pearsonr(targets, preds)[0])
        if np.isnan(pr):
            pr = 0.0
    except Exception:
        pr = 0.0
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2, r=pr)


def _print_metrics(m: dict) -> None:
    for k, v in m.items():
        print(f"  {k:6s}: {v:.4f}")


def load_silso(csv_path: str) -> pd.Series:
    """
    Load SILSO SN_d_tot_V2.0.csv and return a daily ISN pd.Series indexed by date.

    SILSO CSV format (semicolon-separated, no header):
      year;month;day;frac_year;ISN;error;n_obs;definitive

    Download: https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.csv
    """
    df = pd.read_csv(
        csv_path, sep=";", header=None,
        names=["year", "month", "day", "frac", "ISN", "err", "n_obs", "definitive"])
    df = df[df["ISN"] >= 0].copy()
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date").sort_index()
    return df["ISN"].astype(float)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ARIMA and LSTM ISN baselines")
    parser.add_argument("--silso_csv",
                        default="data/SN_d_tot_V2.0.csv",
                        help="Path to SILSO SN_d_tot_V2.0.csv")
    parser.add_argument("--test_size", type=int, default=901)
    parser.add_argument("--skip_arima", action="store_true",
                        help="Skip ARIMA (slow rolling re-fit)")
    args = parser.parse_args()

    isn = load_silso(args.silso_csv)
    print(f"Loaded ISN series: {len(isn)} days  "
          f"({isn.index[0].date()} – {isn.index[-1].date()})")

    results = {}

    if not args.skip_arima:
        print("\n── ARIMA(5,1,2) ─────────────────────────────────────")
        _, _, m_arima = run_arima(isn, test_size=args.test_size)
        results["ARIMA"] = m_arima

    print("\n── LSTM (2-layer, hidden=64, window=30) ────────────")
    _, _, m_lstm = run_lstm(isn, test_size=args.test_size)
    results["LSTM"] = m_lstm

    print("\n── Summary ─────────────────────────────────────────")
    print(f"{'Model':<10}  {'MAE':>6}  {'RMSE':>6}  {'MAPE%':>6}  {'R2':>6}")
    for name, m in results.items():
        print(f"{name:<10}  {m['MAE']:6.1f}  {m['RMSE']:6.1f}  "
              f"{m['MAPE']:6.1f}  {m['R2']:6.3f}")
