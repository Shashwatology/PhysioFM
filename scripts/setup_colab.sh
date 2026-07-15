#!/bin/bash
# Setup script for Google Colab environments

echo "Setting up PhysioFM on Google Colab..."

# Colab has torch and torchvision pre-installed
pip install pandas PyYAML mediapipe opencv-python tqdm scipy wandb umap-learn

echo "Setup complete. Remember to mount Google Drive if your datasets are stored there:"
echo "from google.colab import drive"
echo "drive.mount('/content/drive')"
echo ""
echo "To run training:"
echo "!python src/research/training/train.py --config configs/physiofm.yaml"
