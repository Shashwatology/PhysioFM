# Failure Report: EXP06B_PrePool_Smoke

## Diagnostic Summary
- **Intervention**: Moved the `WaveformHead` before the `mean(dim=1)` temporal pooling layer so that it predicts `(B, T)` directly from the full temporal sequence `(B, T, D)`.
- **Result**: Mode collapse persists. `PredStd` = 0.000012. `HR_Loss` = 218. `WV_Loss` = 0.85 (Pearson ~0.15).
- **Gradient Ratios**: `Head/STDS Ratio` = 18,919. Gradient to STDS remains microscopically small (`0.14`).

## Diagnosis
The architectural fix to the pooling bottleneck did not break the mode collapse. This proves that while the pre-pooling attachment is structurally necessary for temporal gradients to exist, there is a *second* bottleneck upstream that is destroying the dynamic signal before the loss can even optimize it.

By analyzing the `activation_statistics.csv`, I discovered a severe feature imbalance:
- `Swin_Stage1` activation std: **~2.30**
- `STDS` activation std: **~0.33**

Because `Swin` processes RGB frames directly, its output is heavily biased toward static spatial features (the face's geometry), which barely change across the 150 frames. `STDS` processes temporal differences (`x[t] - x[t-1]`), so its output contains the dynamic physiological micro-color changes. 

When these two streams are fused via `Concat + Linear` projection, the static `Swin` features mathematically overwhelm the dynamic `STDS` features by a factor of 7x. 
Consequently, the input to the `TemporalTransformer` is a virtually static sequence. A static sequence produces a flat waveform, which causes the `NegativePearsonLoss` to yield near-zero variance and tiny gradients, perpetuating the mode collapse.

## Decision
Reject Hypothesis 1 as the sole bottleneck. The true bottleneck is **Spatial Feature Dominance**. We must proceed to test Hypothesis 4 (Feature Normalization before Fusion).
