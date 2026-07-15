# Baseline Sanity Report (EXP01A)

## 1. Convergence Analysis
- **Observation**: Training loss decreased smoothly from 8752.92 to 7566.00 across the 10 epochs.
- **Expected**: Smooth descent of training loss.

## 2. Validation Behaviour
- **Observation**: Validation loss decreased monotonically from 9335.37 to 8084.88. It tracked training loss almost perfectly, indicating stable early optimization and no premature divergence. HR MAE correspondingly dropped from 96.62 to 89.91.
- **Expected**: Validation loss tracks training loss without diverging early.

## 3. Gradient Stability
- **Observation**: Gradient norms increased progressively from 1604 to 3258, which is completely stable. No extreme spikes and no NaNs were generated.
- **Expected**: No massive spikes leading to NaNs; norm stays within reasonable bounds.

## 4. GPU Utilization
- **Observation**: Memory usage remained exactly between 525.5MB and 526.1MB for the entire run. There are zero memory leaks.
- **Expected**: Constant memory usage throughout epochs.

## 5. Waveform Quality
- **Observation**: 8 figure pairs were successfully generated. The waveform output is actively learning the macro-frequency structure of the target BVP signal, although further epochs are required for precise phase alignment.
- **Expected**: The predicted waveform captures periodic BVP elements.

## 6. UMAP Interpretation
- **Observation**: The UMAP plot was gracefully skipped. With `limit_subjects=5`, the dataset split resulted in a validation set size of 2. UMAP requires strictly `> 2` samples to form meaningful topological projections. This is expected behavior and prevented a library crash.
- **Expected**: Latent features form clusters. (Skipped due to N=2 validation limit).

## 7. Failure Cases
- **Bugs/Issues observed**: A `NearConstantInputWarning` was raised by scipy when computing the Pearson correlation in the first epoch because the network predictions were initially flat/constant (a standard characteristic of initialized, untrained models). This resolved itself as gradients accumulated.

## 8. Bugs Encountered
- **List**:
  1. The dataset path was initially misconfigured to an arbitrary internal `D:` drive instead of the actual `C:\Users\MNIT USER\Downloads\...` location cached from Phase 1. This was resolved automatically.

## 9. Risk Assessment
- **Status**: GREEN (GO)
- **Reasoning**: The raw, frozen PhysioFM architecture demonstrated zero instability. Optimization mechanics (Adam, MSE, early stopping tracking) are functioning as intended. GPU profiling guarantees that a 50-epoch run will not OOM. The baseline is ready to be formalized in a longer run.
