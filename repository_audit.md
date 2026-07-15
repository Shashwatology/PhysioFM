# Repository Audit

## 1. Codebase Structure
- **train.py**: Standard PyTorch training loop. Implements Early Stopping, gradient clipping, CSV logging, and DiagnosticMonitor hooks. Recently updated to include HR and WV tracking.
- **dataset.py / ubfc_dataset.py**: Successfully parses BVP traces and synchronizes them to RGB ROIs using `SignalSynchronizer`. Ground truth extraction operates correctly.
- **model.py / physio_fm.py**: The core architecture module. Connects SwinV2, STDS, Temporal Transformer, Fusion Layer, and Heads.
  > [!WARNING]
  > **Architectural Bottleneck**: In `physio_fm.py` Line 88, `latent_embedding = fused_seq_emb.mean(dim=1)` destroys the temporal dimension `T`. The `WaveformHead` is attached *after* this pooling, forcing it to reconstruct a `T`-length sequence from a single static vector.
- **configs**: Clean YAML structure. `exp06a_waveform.yaml` successfully loaded with `stds_fusion_mode: concat`.
- **loss functions**: Pure MSE for HR. `NegativePearsonLoss` added for Waveform supervision.
- **evaluation (metrics.py)**: Correctly computes MAE, RMSE, Pearson, CCC. Raises appropriate `NearConstantInputWarning` when predictions collapse.
- **reports**: Detailed failure analyses for EXP04, EXP04A, and EXP05A are present and reproducible.

## 2. Dependencies and Hardware
- CUDA 11.8 / PyTorch available and functional.
- GPU Memory utilization is stable (~528 MB). No OOM issues.

## 3. Data Integrity
- UBFC Dataset is correctly split (29 Train, 6 Val subjects).
- Random seeds are fixed, ensuring reproducibility.
