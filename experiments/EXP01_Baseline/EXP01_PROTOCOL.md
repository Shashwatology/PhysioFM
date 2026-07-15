# EXP01_PROTOCOL

## Metadata
- **Experiment ID**: EXP01_Baseline
- **Timestamp**: 2026-07-10 17:29:56
- **Config Hash**: dc8f4dedb48d4c3c556fa0e426c47d434b6d388316cbabf1f450fe73c7b14b69
- **Random Seed**: 42

## Environment
- **Python**: 3.11.9
- **PyTorch**: 2.7.1+cu118
- **GPU**: NVIDIA RTX A2000 12GB

## Purpose
To establish a completely frozen, baseline performance standard for the raw PhysioFM architecture without any structural modifications.

## Hypothesis (H1)
The current PhysioFM architecture can successfully learn meaningful physiological representations from a small subset of the UBFC-rPPG dataset without architectural modifications, producing stable optimization, decreasing validation error, and physiologically plausible waveform reconstruction.

## Scientific Motivation
Before iterating on architecture or loss functions, it is necessary to record the baseline capacity of the model on a tiny curated set to separate implementation bugs from theoretical capacity limits.

## Experiment Configuration
- **Dataset**: UBFC-rPPG (Raw)
- **Subject Split**: 5 subjects (first 5 indexed)
- **Epochs**: 50
- **Optimizer**: Adam (implied default)
- **Scheduler**: None (constant LR)
- **Learning Rate**: 0.001
- **Loss Function**: MSE (implied default)
- **Early Stopping**: Patience = 15

## Evaluation Metrics
- Mean Absolute Error (MAE) for Heart Rate (BPM)
- Pearson Correlation Coefficient for Heart Rate Agreement
- Training vs Validation Loss Trajectory
- UMAP Manifold Cohesion

## Expected Outputs
1. est_model.pth and last_model.pth
2. 	raining.log with strictly decreasing train loss.
3. isualization_report.md with 9 dual-format IEEE figures.
4. TensorBoard traces.

## Reviewer Checklist
- [ ] No NaNs during training.
- [ ] Stable gradient norms (no extreme spikes).
- [ ] Validation loss decreases alongside training loss.
- [ ] GPU memory utilization is stable (no leaks).
- [ ] Checkpoints are successfully written.
- [ ] Reconstructed waveform visibly resembles ground truth BVP.
- [ ] HR prediction shows positive correlation with targets.
- [ ] UMAP projection does not collapse to a single point.
