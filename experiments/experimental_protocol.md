# Experimental Protocol

This document serves as the reproducible blueprint for all PhysioFM experiments. Any deviations from this protocol must be explicitly documented in the `experiment_manifest.md`.

## 1. Experimental Sequence
1. **Dataset Validation:** Run `dataset_validation.py` to confirm subject counts, FPS, missing frames, and Ground Truth integrity.
2. **Code Integrity Run (Stage 1):** Train on 1 Subject. Verify loss convergence, waveform output, latent embeddings, and checkpoint serialization.
3. **Stability Tuning (Stage 2):** Train on 5 Subjects. Evaluate stability over multiple epochs.
4. **Hyperparameter Sweep:** Conduct grid search across LR, Batch Size, and Sequence Length.
5. **Full Benchmark Run:** Execute on all 42 Subjects using 5-Fold Subject-Independent Cross-Validation.

## 2. Hardware and Environment
- **Operating System:** Windows 10/11
- **Deep Learning Framework:** PyTorch (with `timm` for vision backbones)
- **Precision:** FP16 Automatic Mixed Precision (AMP)
- **Random Seed:** Locked at `42` across `random`, `numpy`, and `torch` (CUDA deterministic mode enabled).

## 3. Training Protocol
- **Optimizer:** AdamW ($\beta_1=0.9$, $\beta_2=0.999$)
- **Learning Rate Schedule:** Cosine Annealing.
- **Early Stopping:** Triggered if Validation MAE does not improve for 15 consecutive epochs.
- **Objective Function:** Multi-task MSE (Heart Rate + BVP Waveform).

## 4. Evaluation and Statistical Tests
- **Primary Metrics:** Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Pearson Correlation ($r$).
- **Cross-Validation:** 5-Fold Subject-Independent (No frames from validation subjects exist in the training folds).
- **Significance Testing:** Paired Student's t-test ($p < 0.05$) to establish statistical superiority over baselines.
- **Explainability:** UMAP projections of latent space; Attention rollout for temporal contribution mapping.
