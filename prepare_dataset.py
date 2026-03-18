"""
prepare_dataset.py
==================
Download SDO/HMI continuum images from JSOC and pair them with
SILSO ISN v2.0 daily values to build the training dataset.

Data sources
------------
  SDO/HMI images : http://jsoc.stanford.edu
    Series  : hmi.Ic_45s   (6173 Å continuum intensity)
    Cadence : 1 image/day at 12:00 UTC
    Period  : 2010-01-01 to 2022-12-31
    Register free account at: http://jsoc.stanford.edu/ajax/register_email.html

  SILSO ISN v2.0 : https://www.sidc.be/silso/datafiles
    Direct CSV    : https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.csv

Prerequisites
-------------
  pip install drms astropy numpy pillow tqdm pandas

Output
------
  data/images/          -- 224×224 PNG files  (~1 MB each, ~4 GB total)
  data/SN_d_tot_V2.0.csv
  data/image_isn_pairs.csv   -- master pairing CSV used by dataset.py

Usage
-----
  # Step 1: download SILSO ISN CSV manually from URL above, place in data/
  # Step 2:
  python prepare_dataset.py --email your@email.com

  # If JSOC is unavailable, use the simulated dataset generator:
  python prepare_dataset.py --simulate --n_samples 4231
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SILSO_URL  = "https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.csv"
JSOC_BASE  = "http://jsoc.stanford.edu"
IMAGE_DIR  = Path("data/images")
SILSO_CSV  = Path("data/SN_d_tot_V2.0.csv")
PAIRS_CSV  = Path("data/image_isn_pairs.csv")
IMG_SIZE   = 224
START_DATE = "2010-01-01"
END_DATE   = "2022-12-31"


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: Download SILSO ISN
# ══════════════════════════════════════════════════════════════════════════════

def download_silso(out_path: Path = SILSO_CSV) -> pd.DataFrame:
    """Download SILSO SN_d_tot_V2.0.csv if not already present."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"  SILSO CSV already exists: {out_path}")
    else:
        print(f"  Downloading SILSO ISN from {SILSO_URL} …")
        try:
            import urllib.request
            urllib.request.urlretrieve(SILSO_URL, out_path)
            print(f"  Saved to {out_path}")
        except Exception as e:
            raise RuntimeError(
                f"Could not download SILSO CSV: {e}\n"
                f"Please download manually from {SILSO_URL} "
                f"and place it at {out_path}"
            )

    df = pd.read_csv(
        out_path, sep=";", header=None,
        names=["year", "month", "day", "frac", "ISN", "err", "n_obs", "def"])
    df = df[df["ISN"] >= 0].copy()
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date").sort_index()
    print(f"  Loaded {len(df):,} ISN records "
          f"({df.index[0].date()} – {df.index[-1].date()})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: Download HMI images from JSOC
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_hmi(data: np.ndarray, header: dict,
                    size: int = IMG_SIZE) -> Image.Image:
    """
    Disk-centre crop, median-intensity normalisation → 8-bit PNG.

    Steps:
      1. Extract disk centre and radius from FITS header
      2. Crop a square around the disk with a small margin
      3. Normalise by the disk-integrated median  (paper Section III.B)
      4. Clip to [0, 1.5] and scale to uint8
      5. Resize to size × size (bilinear)
    """
    cx   = int(header.get("CRPIX1", data.shape[1] // 2))
    cy   = int(header.get("CRPIX2", data.shape[0] // 2))
    r    = int(header.get("R_SUN",  min(data.shape) * 0.45))
    half = r + 20

    x1, x2 = max(0, cx - half), min(data.shape[1], cx + half)
    y1, y2 = max(0, cy - half), min(data.shape[0], cy + half)
    crop = data[y1:y2, x1:x2].astype(np.float32)

    med  = float(np.nanmedian(crop[crop > 0])) if (crop > 0).any() else 1.0
    crop = crop / med
    crop = np.clip(crop, 0.0, 1.5)
    crop = (crop / 1.5 * 255.0).astype(np.uint8)

    img = Image.fromarray(crop).resize((size, size), Image.Resampling.BILINEAR)
    # Convert grayscale to RGB (channel-replicated) for ImageNet normalisation
    return img.convert("RGB")


def download_hmi_images(
    email:       str,
    image_dir:   Path = IMAGE_DIR,
    start_date:  str  = START_DATE,
    end_date:    str  = END_DATE,
) -> list:
    """
    Query JSOC for hmi.Ic_45s and download one image per day at 12:00 UTC.

    Requires: pip install drms astropy
    """
    try:
        import drms
    except ImportError:
        raise ImportError("pip install drms astropy")

    image_dir.mkdir(parents=True, exist_ok=True)
    client = drms.Client(email=email)

    query_str = (f"hmi.Ic_45s[{start_date.replace('-','.')}"
                 f"-{end_date.replace('-','.')}/@1d]"
                 f"[?T_OBS$='_120000_TAI'?]")
    print(f"  JSOC query: {query_str}")

    keys, segments = client.query(
        query_str,
        key=["T_OBS", "QUALITY"],
        seg=["continuum"],
    )
    print(f"  Found {len(keys):,} records")

    downloaded = []
    for i, (_, row) in enumerate(tqdm(keys.iterrows(),
                                      total=len(keys),
                                      desc="Downloading HMI")):
        if row["QUALITY"] != 0:          # skip bad quality exposures
            continue
        t_obs    = str(row["T_OBS"]).replace(":", "").replace(".", "").replace(" ", "_")
        out_path = image_dir / f"{t_obs}.png"

        if out_path.exists():
            downloaded.append(str(out_path))
            continue

        try:
            seg_url = JSOC_BASE + segments.iloc[i]["continuum"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from astropy.io import fits as pyfits
                with pyfits.open(seg_url) as hdul:
                    data   = hdul[1].data.astype(np.float32)
                    header = dict(hdul[1].header)
            img = _preprocess_hmi(data, header)
            img.save(out_path)
            downloaded.append(str(out_path))
        except Exception as exc:
            print(f"  WARN: {t_obs}: {exc}")

    print(f"  Downloaded {len(downloaded):,} images to {image_dir}")
    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: Build image–ISN pairing CSV
# ══════════════════════════════════════════════════════════════════════════════

def build_pairs_csv(
    image_dir: Path = IMAGE_DIR,
    silso_df:  pd.DataFrame = None,
    out_path:  Path = PAIRS_CSV,
    silso_csv: Path = SILSO_CSV,
) -> pd.DataFrame:
    """
    Scan image_dir for PNG files, parse the date from each filename,
    look up the corresponding SILSO ISN value, and write the pairing CSV.

    Filename format expected: YYYYMMDD_HHMMSS_TAI.png
    (as produced by download_hmi_images above)
    """
    if silso_df is None:
        silso_df = download_silso(silso_csv)

    isn_index = silso_df["ISN"].copy()
    isn_index.index = isn_index.index.normalize()

    records = []
    for fname in sorted(image_dir.glob("*.png")):
        try:
            date_str = fname.stem[:8]                  # YYYYMMDD
            date     = pd.Timestamp(
                f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            ).normalize()
            if date in isn_index.index:
                isn_val = float(isn_index[date])
                if isn_val >= 0:
                    records.append({
                        "image": fname.name,
                        "date":  date.strftime("%Y-%m-%d"),
                        "ISN":   isn_val,
                    })
        except Exception as exc:
            warnings.warn(f"  Skipping {fname.name}: {exc}")

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Paired {len(df):,} image–ISN records → {out_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Simulated dataset generator (no JSOC access required)
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_solar_disk(isn: float, size: int = IMG_SIZE,
                           rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a plausible synthetic HMI-style solar disk image (uint8).

    The image is a gray disk with dark circular sunspot blobs
    whose count and size scale with ISN.
    Intended for code-path testing only; not scientifically valid.
    """
    if rng is None:
        rng = np.random.default_rng()

    img  = np.full((size, size), 200, dtype=np.float32)
    cx, cy = size // 2, size // 2
    r_disk  = int(size * 0.44)

    # Draw disk
    yy, xx = np.ogrid[:size, :size]
    dist    = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    outside = dist > r_disk
    img[outside] = 0

    # Limb darkening
    limb    = 1.0 - 0.6 * (dist / r_disk) ** 2
    img     = np.where(outside, 0, img * np.clip(limb, 0, 1))

    # Sunspot blobs: count ~ ISN / 12 (rough ISN definition)
    n_spots = max(0, int(isn / 12) + rng.integers(0, 2))
    for _ in range(n_spots):
        lat_deg = rng.uniform(-30, 30)
        lon_deg = rng.uniform(-70, 70)
        sx  = cx + int(r_disk * np.sin(np.radians(lon_deg)))
        sy  = cy - int(r_disk * np.sin(np.radians(lat_deg)))
        sr  = rng.integers(3, 10)
        yy2, xx2 = np.ogrid[:size, :size]
        spot_mask = (xx2 - sx) ** 2 + (yy2 - sy) ** 2 < sr ** 2
        img[spot_mask & ~outside] *= rng.uniform(0.3, 0.6)

    # Add noise
    img += rng.normal(0, 2, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_simulated_dataset(
    silso_csv:  Path = SILSO_CSV,
    image_dir:  Path = IMAGE_DIR,
    pairs_csv:  Path = PAIRS_CSV,
    n_samples:  int  = 4231,
    seed:       int  = 42,
) -> pd.DataFrame:
    """
    Create a simulated dataset of synthetic HMI-style images paired
    with real SILSO ISN values.

    Used for unit testing and CI pipelines where JSOC access is unavailable.
    The images are synthetic; results will NOT reproduce the paper metrics.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Load or create dummy ISN series
    if silso_csv.exists():
        silso_df = download_silso(silso_csv)
        isn_vals = silso_df.reset_index()
    else:
        print("  SILSO CSV not found – generating synthetic ISN series.")
        dates    = pd.date_range(START_DATE, END_DATE, freq="D")[:n_samples]
        t        = np.linspace(0, 2 * np.pi * 12 / 11, len(dates))
        isn_vals_arr = np.clip(
            64 + 55 * np.sin(t) + rng.normal(0, 10, len(dates)), 0, 310)
        isn_vals = pd.DataFrame({"date": dates, "ISN": isn_vals_arr})

    isn_vals = isn_vals.head(n_samples).copy()
    if len(isn_vals) < n_samples:
        warnings.warn(
            f"Requested {n_samples} samples but only {len(isn_vals)} "
            "ISN records are available. Proceeding with available data."
        )
    records  = []

    print(f"  Generating {len(isn_vals):,} synthetic HMI images …")
    for _, row in tqdm(isn_vals.iterrows(), total=len(isn_vals)):
        date_str = pd.Timestamp(row["date"]).strftime("%Y%m%d")
        fname    = f"{date_str}_120000_TAI.png"
        out_path = image_dir / fname

        if not out_path.exists():
            arr = _synthetic_solar_disk(float(row["ISN"]), size=IMG_SIZE, rng=rng)
            img = Image.fromarray(arr).convert("RGB")
            img.save(out_path)

        records.append({
            "image": fname,
            "date":  pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "ISN":   float(row["ISN"]),
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df.to_csv(pairs_csv, index=False)
    print(f"  Dataset ready: {len(df):,} pairs → {pairs_csv}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare SDO/HMI + SILSO ISN dataset")
    parser.add_argument("--email",     default="",
                        help="JSOC registered email (required for real download)")
    parser.add_argument("--image_dir", default="data/images/")
    parser.add_argument("--silso_csv", default="data/SN_d_tot_V2.0.csv")
    parser.add_argument("--pairs_csv", default="data/image_isn_pairs.csv")
    parser.add_argument("--simulate",  action="store_true",
                        help="Generate synthetic dataset (no JSOC needed)")
    parser.add_argument("--n_samples", type=int, default=4231,
                        help="Number of samples for --simulate mode")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    silso_csv = Path(args.silso_csv)
    pairs_csv = Path(args.pairs_csv)

    if args.simulate:
        print("=== Simulated Dataset Mode ===")
        print("WARNING: Synthetic images only. Not suitable for paper results.")
        generate_simulated_dataset(
            silso_csv  = silso_csv,
            image_dir  = image_dir,
            pairs_csv  = pairs_csv,
            n_samples  = args.n_samples,
        )
    else:
        if not args.email:
            parser.error(
                "--email required for JSOC download.\n"
                "Register free at: http://jsoc.stanford.edu/ajax/register_email.html\n"
                "Or use --simulate for synthetic data.")

        print("=== Real Dataset Download ===")
        print(f"JSOC email : {args.email}")

        # 1. SILSO ISN
        print("\n[1/3] SILSO ISN")
        silso_df = download_silso(silso_csv)

        # 2. HMI images
        print("\n[2/3] HMI continuum images from JSOC")
        download_hmi_images(email=args.email, image_dir=image_dir)

        # 3. Build pairing CSV
        print("\n[3/3] Building image–ISN pairs CSV")
        build_pairs_csv(image_dir=image_dir,
                        silso_df=silso_df,
                        out_path=pairs_csv)

    print("\nDone! Run training with:")
    print(f"  python train.py --csv {pairs_csv} --image_dir {image_dir}/")
