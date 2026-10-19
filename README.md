# SAGE Qwen Vision Model Trainer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Project:** Autonomous Visual AI Agent for Precision Agriculture  
**Dataset:** SAGE Agricultural Image Dataset (`IDK-Chaitanya/SAGE5class`)  
**Target Model:** Qwen2.5-VL-3B-Instruct  
**Fine-Tuning:** 4-bit NF4 QLoRA  
**First Target:** Tomato Disease Classification (8 classes)

---

## 1. Architecture Overview

```text
sage-qwen-trainer/
├── README.md                  # Development, Git, and Kaggle guide
├── requirements.txt           # Pinned Kaggle dependencies
├── .gitignore                 # Prevents large weights/images/secrets from Git
├── configs/
│   └── config.yaml            # Single source of truth control center
├── src/
│   ├── __init__.py
│   ├── config.py              # Config loader, deep merge, and hardware validator
│   ├── data_loader.py         # Selective HF download, metadata parser & leak detector
│   ├── image_preprocessing.py # Aspect-ratio preserving transforms & EXIF transpose
│   ├── model_loader.py        # /kaggle/input/ discovery, 4-bit NF4 & LoRA setup
│   ├── metrics.py             # Generative JSON parser, precision/recall/F1 & plots
│   ├── checkpoint_manager.py  # Checkpoint pruning, retention & interruption handler
│   ├── reporting.py           # Structured CSV/JSON metrics & model card generator
│   ├── evaluate.py            # Greedy generative inference & latency benchmarks
│   ├── train.py               # Multimodal collator, live progress & training loop
│   └── utils.py               # Hardware detection, seeds, Git tracker
├── scripts/
│   ├── validate_dataset.py    # CLI dataset integrity & split balance validation
│   ├── smoke_test.py          # Multimodal forward/backward pass verification
│   └── train.py               # CLI training entry point
├── kaggle/
│   └── kaggle_train.ipynb     # Step-by-step Kaggle notebook execution
└── tests/
    ├── test_config.py
    ├── test_data_loader.py
    └── test_image_preprocessing.py
```

---

## 2. Crop Reusability (Tomato -> Soybean, Rice, Wheat, Corn)

This repository is designed to train on **Tomato first**, and seamlessly switch to other crops without altering Python source code.

To switch crops, edit `configs/config.yaml`:

```yaml
project:
  crop: "Rice" # Options: "Tomato", "Soybean", "Rice", "Wheat", "Corn"
```

Or pass via command-line / environment:
```bash
python scripts/train.py --crop Rice
# or
export SAGE_CROP=Rice
```

The data access layer reads canonical class definitions directly from `metadata/selected_classes.json` and selectively downloads only images for the specified crop.

---

## 3. Local Development & Git Workflow (Laptop)

Development occurs on your local machine; training occurs on Kaggle GPU.

### Step 1: Initialize Git and Commit
Run in PowerShell / Terminal from the repository directory:
```bash
cd c:\Projects\SAGE5Class

# Initialize git repository
git init

# Stage all source code, tests, and configurations
git add .

# Create initial commit
git commit -m "feat: complete SAGE Qwen2.5-VL QLoRA training repository"

# Set main branch
git branch -M main

# Add your GitHub remote repository (replace with your actual URL)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/sage-qwen-trainer.git

# Push to GitHub
git push -u origin main
```

### Step 2: Running Unit Tests Locally
```bash
python -m unittest discover tests
```

---

## 4. Kaggle Cloud Execution Workflow

### Step 1: Create Notebook on Kaggle
1. Go to [Kaggle](https://www.kaggle.com/) and create a new Python Notebook.
2. Under **Notebook options**:
   - **Accelerator:** Select GPU (T4 x2 or P100 / L4 / A100).
   - **Internet:** Turn **ON** (required for GitHub and dataset downloads).

### Step 2: Attach Model Input
1. In the right panel under **Input -> Add Input**, search and add `Qwen2.5-VL-3B-Instruct` (or your attached model dataset).
2. The notebook's auto-discovery module scans `/kaggle/input/` for `config.json` and resolves the weights automatically.

### Step 3: Run the Notebook
Open `kaggle/kaggle_train.ipynb` or execute the cells:
1. Clone the repository:
   ```bash
   !git clone https://github.com/<YOUR_GITHUB_USERNAME>/sage-qwen-trainer.git /kaggle/working/sage-qwen-trainer
   %cd /kaggle/working/sage-qwen-trainer
   ```
2. Install dependencies:
   ```bash
   !pip install -q -r requirements.txt
   ```
3. Run the dataset validator:
   ```bash
   !python scripts/validate_dataset.py --config configs/config.yaml --crop Tomato
   ```
4. Run the forward/backward smoke test:
   ```bash
   !python scripts/smoke_test.py --config configs/config.yaml
   ```
5. Launch training:
   ```bash
   !python scripts/train.py --config configs/config.yaml
   ```

---

## 5. Training Outputs & Artifacts

All training artifacts are stored under `/kaggle/working/sage_runs/<crop>_YYYYMMDD_HHMMSS/`:

- `adapter/`: Trained PEFT LoRA adapter weights and processor files.
- `checkpoints/`: Resumable step checkpoints (pruned to `save_total_limit`).
- `batch_loss.csv`: Step-by-step training loss, running loss, learning rate, and VRAM.
- `epoch_metrics.csv`: Mean epoch training loss and validation scores.
- `classification_report.csv`: Per-class precision, recall, F1, and support.
- `confusion_matrix.png` & `confusion_matrix.csv`: Full confusion matrix heatmap.
- `predictions.csv`: Sample-by-sample predictions and inference latency.
- `run_summary.json`: Complete execution metadata, timing, and Git commit.
- `model_card.md`: Standardized model card documenting training and evaluation results.
