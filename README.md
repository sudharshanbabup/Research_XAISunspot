# ☀️ Sunspot ISN Regression — SDO/HMI + Grad-CAM

> ** Sunspot Number Estimation from SDO/HMI Images Using Deep Learning and Grad-CAM**

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Backbone-ResNet50-0078D4?style=flat-square" />
  <img src="https://img.shields.io/badge/XAI-Grad--CAM-4CAF50?style=flat-square" />
  <img src="https://img.shields.io/badge/Conference-IEEE-00629B?style=flat-square&logo=ieee" />
  <img src="https://img.shields.io/badge/Domain-Space%20Weather-FF6F00?style=flat-square" />
</p>

**Authors:** Pandava Sudharshan Babu · Kashish · Deepthi M · Shyma Chandrasekharan · Omprakash Gottam

**Affiliations:** CMR University, Bengaluru &nbsp;|&nbsp; KLEF, Hyderabad



---

## Overview

This repository provides the complete implementation for regressing the **International Sunspot Number (ISN)** directly from full-disk **SDO/HMI continuum images** using a ResNet50 backbone fine-tuned with staged layer unfreezing, combined with quantitatively validated **Grad-CAM explainability**.

Key contributions:
- End-to-end pipeline from raw JSOC archive downloads to ISN predictions
- Two-phase training with progressive layer unfreezing on ResNet50
- Grad-CAM adapted for scalar regression tasks with quantitative XAI metrics (Deletion AUC, simulated IoU)
- Strictly chronological train/val/test splits to prevent data leakage
- ARIMA and LSTM baselines for comparison

---

## Dataset

### 1. SDO/HMI Continuum Images

| Property | Value |
|---|---|
| Archive | [JSOC Stanford — `hmi.Ic_45s`](http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_fetch) |
| Series | `hmi.Ic_45s` — 6173 Å continuum intensity filtergrams |
| Cadence used | 1 image/day at 12:00 UTC |
| Period | 2010-01-01 → 2022-12-31 |
| Native resolution | 4096 × 4096 px, 0.5″/pixel |
| Pre-processed size | 224 × 224 px, 8-bit PNG, channel-replicated RGB |
| Registration | Free — [register here](http://jsoc.stanford.edu/ajax/register_email.html) |
| Total size | ~4 GB after preprocessing |

### 2. SILSO International Sunspot Number v2.0

| Property | Value |
|---|---|
| Provider | [Royal Observatory of Belgium — SILSO](https://www.sidc.be/silso/) |
| Direct CSV download | [`SN_d_tot_V2.0.csv`](https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.csv) |
| Format | Semicolon-separated: `year;month;day;frac_year;ISN;error;n_obs;definitive` |
| License | Free for scientific use — CC BY-NC 4.0 |

### Dataset Splits

> Strictly chronological — **no data leakage** between splits.

| Split | Period | Samples |
|---|---|---|
| Train | 2010–2018 | 3,011 |
| Val | 2019 | 319 |
| Test | 2020–2022 | 901 |
| **Total** | **2010–2022** | **4,231** |

---

## Repository Structure

```
.
├── prepare_dataset.py   # JSOC download + SILSO pairing
├── dataset.py           # PyTorch Dataset & DataLoaders
├── model.py             # ResNet50ISN + WeightedMSELoss
├── train.py             # Two-phase training loop
├── evaluate.py          # Test evaluation + figure generation
├── gradcam.py           # Grad-CAM for scalar regression (Eqs 2–3)
├── xai_metrics.py       # Deletion AUC + simulated IoU
├── run_gradcam_vis.py   # Generate Grad-CAM overlays
├── baselines.py         # ARIMA + LSTM baselines
└── README.md
```

---

## Quickstart

### Requirements

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn statsmodels
```

| Requirement | Detail |
|---|---|
| JSOC access | Free registration: [jsoc.stanford.edu](http://jsoc.stanford.edu/ajax/register_email.html) |
| GPU | Single NVIDIA V100 (or equivalent) |
| Storage | ~4 GB for pre-processed images |
| SILSO ISN | [SN_d_tot_V2.0.csv](https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.csv) |



## Method

### Model

- **Backbone:** ResNet50 pretrained on ImageNet
- **Regression head:** Global Average Pooling → FC(2048, 512) → ReLU → Dropout(0.3) → FC(512, 1)
- **Loss:** Weighted MSE — upweights high-ISN solar maximum samples

### Grad-CAM for Regression

Standard Grad-CAM is adapted for scalar output regression. Gradients of the predicted ISN scalar with respect to the final convolutional feature map are used to produce spatially localized saliency maps (Equations 2–3 in the paper), highlighting active region candidates on the solar disk.

### XAI Metrics

- **Deletion AUC** — measures prediction degradation as salient pixels are progressively masked
- **Simulated IoU** — compares Grad-CAM activations against known active region locations

---





---

## License

This project is released for academic and research use. SDO/HMI data is provided by NASA/JSOC Stanford. SILSO ISN data is © Royal Observatory of Belgium, licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

---

## Acknowledgements

- [NASA SDO / JSOC Stanford](http://jsoc.stanford.edu/) for the HMI continuum image archive
- [Royal Observatory of Belgium — SILSO](https://www.sidc.be/silso/) for the International Sunspot Number v2.0
- CMR University, Bengaluru & KLEF, Hyderabad for institutional support
