# Decoding Keystrokes from sEMG Signals

**Team:** Yihe Xu, Zijing Zhu, Xiang Zhou, Chi En Chen

## Overview

This project explores **decoding keystrokes from unsegmented surface electromyography (sEMG)** signals.
Built on Meta's **emg2qwerty** dataset and codebase, we evaluate several neural architectures and experimental conditions to reduce **Character Error Rate (CER)**.

---

## Results

### Architecture Comparison

| Model              | Validation CER (%) | Test CER (%) |
| ------------------ | ------------------ | ------------ |
| **LSTM (Best)**    | **15.35**          | **15.40**    |
| CNN + BiLSTM       | 14.80              | 15.63        |
| Transformer        | 19.12              | 21.29        |
| TDSConv (Baseline) | 18.81              | 22.15        |
| RNN                | 53.41              | 49.64        |

---

## Key Findings

* **Data Augmentation** – Temporal augmentation improves LSTM performance (best **14.30% CER**).
* **Electrode Channels** – Performance remains stable down to **8 channels**, but degrades significantly at **4 channels**.
* **Sampling Rate** – Reducing **2000 Hz → 667 Hz** increases CER from **21.48% → 58.05%**.
* **Few-Shot Adaptation** – Cross-subject CER (**64.7%**) can be reduced to **24.1%** with a **5-minute calibration session**.

---


## Repository layout

```text
.
├── .github/workflows/         # CI and test workflow
├── Archive/                   # Archived notes / placeholders
├── config/                    # Hydra experiment configs
│   ├── cluster/               # Local / slurm launch presets
│   ├── decoder/               # Greedy / beam CTC decoding configs
│   ├── lr_scheduler/          # Scheduler presets
│   ├── model/                 # Model definitions and ablations
│   ├── optimizer/             # Optimizer presets
│   ├── transforms/            # Feature transform presets
│   └── user/                  # Train/val/test session lists
├── emg2qwerty/                # Core package
│   ├── data.py                # HDF5 dataset and collation
│   ├── decoder.py             # CTC decoders
│   ├── lightning.py           # Lightning modules + datamodule
│   ├── metrics.py             # CER metrics
│   ├── modules.py             # Neural model components
│   ├── train.py               # Main training entry point
│   ├── train_downsample.py    # Downsampled training variant
│   ├── transforms*.py         # Feature transforms and augmentations
│   └── tests/                 # Unit tests
├── models/lm/                 # Language-model placeholders
├── scripts/                   # Utility scripts
├── *.ipynb                    # Research notebooks / experiments
├── environment.yml            # Conda environment
├── requirements.txt           # Pip requirements
├── setup.py                   # Package install metadata
└── README.md
```

Key configuration files:

* `config/model/tds_conv_ctc.yaml` – Model hyperparameters
* `config/user/single_user.yaml` – Data split settings

---

## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate emg2qwerty
pip install -e .
```

Download and link the dataset:

```bash
cd ~
wget https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
tar -xvzf emg2qwerty-data-2021-08.tar.gz

ln -s ~/emg2qwerty-data-2021-08 ~/emg2qwerty/data
```

---

## Training

Train the personalized user model:

```bash
python -m emg2qwerty.train user="single_user" trainer.accelerator=gpu trainer.devices=1
```

---



