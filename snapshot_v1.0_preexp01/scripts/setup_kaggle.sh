#!/bin/bash
# Setup script for Kaggle environments

echo "Setting up PhysioFM on Kaggle..."

# Kaggle usually has PyTorch installed, but we make sure dependencies are met
pip install -r requirements.txt

# Optionally install dev dependencies if needed
# pip install -r requirements-dev.txt

echo "Setup complete. To run training:"
echo "python src/research/training/train.py --config configs/physiofm.yaml"
