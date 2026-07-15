import os

base_dir = r"d:\Shashwat\Platform\ml_pipeline"
docs_dir = os.path.join(base_dir, "docs")
os.makedirs(docs_dir, exist_ok=True)

docs = {
    "architecture.md": "# System Architecture\nLiving documentation for the Physiological Foundation Model.",
    "datasets.md": "# Datasets\nDetails on versioning, formats, and acquisition for UBFC-rPPG, VIPL-HR, etc.",
    "preprocessing.md": "# Preprocessing\nDetails on video decoding, ROI extraction, and synchronization.",
    "experiments.md": "# Experiments\nGuidelines for running and logging experiments via WandB/MLflow.",
    "training.md": "# Training Framework\nHyperparameters, mixed precision, and checkpoint management.",
    "evaluation.md": "# Evaluation Framework\nMetrics (MAE, RMSE, Pearson), Bland-Altman plots, and profiling.",
    "reproducibility.md": "# Reproducibility\nSteps to perfectly reproduce baseline and novel architecture results.",
    "roadmap.md": "# Roadmap\nPhased rollout of the research pipeline."
}

for filename, content in docs.items():
    with open(os.path.join(docs_dir, filename), "w") as f:
        f.write(content)
        
print("Successfully created docs/ directory and living documentation files.")
