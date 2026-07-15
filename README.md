# PhysioFM: Optimization Diagnostics and Failure Analysis of Hierarchical Foundation Models in rPPG

![PhysioFM Header](paper/figures/clinical_motivation.png) *(Note: Add your final banner here)*

PhysioFM is a research framework designed to investigate the integration of large-scale Spatial Foundation Models (Swin-V2) with Spatio-Temporal Difference Stems (STDS) for remote Photoplethysmography (rPPG). 

Unlike standard architectures, this project focuses heavily on **isolating optimization bottlenecks**—specifically **Gradient Starvation** and **Objective Function Mismatch**—that lead to mode collapse in dual-branch hierarchical transformers.

---

## 📖 Research Dossier
The complete scientific findings, failure analyses, and optimization diagnostics of this project are compiled in the master dossier:
**[`PhysioFM_Research_Dossier.md`](PhysioFM_Research_Dossier.md)**

---

## 🗂️ Repository Structure

```text
PhysioFM_Codebase/
├── configs/             # YAML experiment hashes (EXP02 through EXP08)
├── datasets/            # Dataset parsing and caching logic
├── dossier_sections/    # Raw markdown chunks for the final publication
├── experiments/         # Checkpoints, logs, and diagnostic metrics
├── paper/               # Generated publication assets (figures/tables)
├── src/
│   ├── research/
│   │   ├── models/      # Neural architectures (Swin-V2, STDS, Fusion)
│   │   ├── training/    # Optimization loops & diagnostic monitors
│   │   └── evaluation/  # Validation metrics (MAE, RMSE, Pearson)
└── README.md
```

## 🛠️ Installation & Environment

This repository strictly enforces reproducibility.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/PhysioFM_Codebase.git
   cd PhysioFM_Codebase
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```
   *Required: Python 3.10.11, CUDA 11.8*

## 💾 Dataset Preparation

This project benchmarks on the **UBFC-rPPG Dataset**. 
Download the dataset and ensure it follows this exact structure:

```text
C:/Users/MNIT USER/Downloads/UBFC-rPPG Dataset/
├── subject1/
│   ├── vid.avi
│   └── ground_truth.txt
├── subject2/
...
```
*(Configure the absolute path in your YAML experiment files under `dataset_path`)*

## 🚀 Training

To execute a controlled experiment, pass the corresponding YAML configuration to the training engine:

```bash
# Run the baseline Swin-V2 model
python -m src.research.training.train --config configs/exp02_full_baseline.yaml

# Run the STDS architecture with Pre-Fusion LayerNorm
python -m src.research.training.train --config configs/exp06c_feature_norm.yaml
```

All metrics, model checkpoints, and diagnostic scatter plots will be securely isolated in the `experiments/` directory.

## 📊 Reproducibility
- **Random Seed:** Locked to `42` across PyTorch, Numpy, and Python Random.
- **Hardware Profile:** NVIDIA GeForce RTX 3060 (12GB VRAM), 32GB System RAM.
- **Average Training Time:** ~22 minutes per 100 epochs (Batch Size = 2).

## 📄 License
This codebase is released under the MIT License for academic and research purposes.