# Experiment Summary

| Experiment | Configuration | Pred Std (BPM) | Pred Entropy | Pearson | Head/STDS Ratio | Result |
|---|---|---|---|---|---|---|
| **EXP02** | Baseline (`add`) | ~0.0000 | 0.0000 | ~ -0.40 | > 100,000 | Collapse |
| **EXP03** | Diagnostics | 0.0000 | 0.0000 | ~ -0.40 | > 100,000 | Collapse |
| **EXP04** | Gated Residual ($\alpha=0$) | 0.0000 | 0.0000 | ~ -0.40 | Infinite (Gate=0) | Collapse |
| **EXP04A**| Gate Init ($\alpha=1$) | 0.0000 | 0.0000 | ~ -0.40 | > 100,000 | Collapse |
| **EXP05A**| `concat` fusion | 0.0000 | 0.0000 | ~ -0.42 | > 200,000 | Collapse |
| **EXP06A**| `concat` + Waveform Loss| 0.0000 | 0.0000 | ~ -0.46 | > 260,000 | Collapse |

## Metrics Breakdown
- **MAE / RMSE**: Consistently sits around 8.54 / 8.65, which corresponds exactly to the dataset mean HR.
- **Pearson / CCC**: Consistently negative or near zero. `NearConstantInputWarning` triggers constantly.
- **Prediction Variance / Entropy**: Effectively zero. The model predicts the same static number for every video.
- **Gradient Ratios**: Massive mismatch. The Head sees gradients in the thousands, Temporal Transformer in the hundreds, but Spatial STDS/Swin sees `1e-3` or lower.
- **GPU Usage / Time**: Constant ~528 MB memory and ~23s per epoch across all architectures.

**Conclusion**: Across all 6 experiments exploring Fusion (Gate, Add, Concat) and Supervision (HR MSE vs Waveform Pearson), the mode collapse remains unbroken. The gradients vanish structurally before reaching the spatial stems.
