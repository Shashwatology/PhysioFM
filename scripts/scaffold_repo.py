import os

# Base directory
base_dir = r"d:\Shashwat\Platform\ml_pipeline"

# Directories to create
directories = [
    "configs",
    "datasets/UBFC-rPPG/raw",
    "datasets/UBFC-rPPG/processed",
    "datasets/UBFC-rPPG/metadata",
    "datasets/UBFC-rPPG/checksums",
    "datasets/UBFC-rPPG/splits",
    "datasets/VIPL-HR",
    "datasets/V4V",
    "datasets/iBVP",
    "experiments/exp001_baseline",
    "experiments/exp002_swin",
    "experiments/checkpoints",
    "experiments/logs",
    "experiments/tensorboard",
    "experiments/predictions",
    "experiments/reports",
    "paper/figures",
    "paper/tables",
    "paper/results",
    "paper/latex",
    "paper/references",
    "paper/appendix",
    "src/research/baselines/POS",
    "src/research/baselines/CHROM",
    "src/research/baselines/DeepPhys",
    "src/research/baselines/PhysNet",
    "src/research/models",
    "src/research/training",
    "src/research/evaluation",
    "src/research/preprocessing",
    "src/production/api",
    "src/production/inference",
    "src/production/deployment"
]

# Create directories
print("Creating directory structure...")
for d in directories:
    path = os.path.join(base_dir, d)
    os.makedirs(path, exist_ok=True)
    print(f"  Created {d}")

# Create config files
configs = {
    "baseline.yaml": "model_name: POS\nlearning_rate: 0.001\nepochs: 30",
    "dataset.yaml": "name: UBFC-rPPG\npath: ../datasets/UBFC-rPPG/processed\nbatch_size: 16",
    "train.yaml": "optimizer: AdamW\nweight_decay: 0.01\nloss: MSE",
    "fusion.yaml": "fusion_type: cross_attention\nembed_dim: 512\nheads: 8",
    "experiment.yaml": "project: Physiological-Foundation-Model\ntracking: wandb"
}
print("\nCreating YAML configs...")
for name, content in configs.items():
    with open(os.path.join(base_dir, "configs", name), "w") as f:
        f.write(content)

# Create registry files
print("\nCreating Experiment Registries...")
with open(os.path.join(base_dir, "experiments", "registry.csv"), "w") as f:
    f.write("Experiment,Dataset,Model,MAE,RMSE,Pearson,FPS,Epochs,Best Checkpoint\n")
    
with open(os.path.join(base_dir, "experiments", "leaderboard.csv"), "w") as f:
    f.write("Model,Dataset,Best MAE,Best RMSE,Best Pearson,Params (M)\n")

with open(os.path.join(base_dir, "experiments", "experiment_tracker.md"), "w") as f:
    f.write("# Experiment Tracker\n\nAutomated log of all research runs.")

# Create repo files
print("\nCreating repository dotfiles...")
with open(os.path.join(base_dir, "README.md"), "w") as f:
    f.write("# Physiological Foundation Model\n\nResearch repository for multimodal contactless physiological monitoring.")

with open(os.path.join(base_dir, "requirements.txt"), "w") as f:
    f.write("torch>=2.0.0\ntorchvision\ntimm\ntransformers\nwandb\nmlflow\nopencv-python\nmediapipe\npandas\nnumpy\nscipy\npyyaml\nmatplotlib\n")

with open(os.path.join(base_dir, "environment.yml"), "w") as f:
    f.write("name: physio-foundation\nchannels:\n  - pytorch\n  - defaults\ndependencies:\n  - python=3.10\n  - pip\n  - pip:\n    - -r requirements.txt\n")

with open(os.path.join(base_dir, ".gitignore"), "w") as f:
    f.write("venv/\n__pycache__/\n*.pyc\ndatasets/*/raw/\ndatasets/*/processed/\nexperiments/checkpoints/\nexperiments/tensorboard/\nexperiments/predictions/\n.env\nwandb/\nmlruns/\n")

with open(os.path.join(base_dir, "CHANGELOG.md"), "w") as f:
    f.write("# Changelog\n\n## [Unreleased]\n- Initial repository setup and scaffolding.\n")

print("\nScaffolding Complete!")
