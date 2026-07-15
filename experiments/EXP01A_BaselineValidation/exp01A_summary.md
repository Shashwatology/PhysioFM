# EXP01A: Baseline Validation Summary

## Objective
To validate optimization dynamics, infrastructure stability, and artifact generation of the raw PhysioFM architecture prior to a full-scale experiment.

## Configuration Parameters
- **Dataset**: UBFC-rPPG
- **Subjects**: S1-S5 (Limit=5)
- **Split**: Train=3, Val/Test=2
- **Seed**: 42
- **Epochs**: 10
- **Early Stopping**: Patience=15 (lower bound=10)
- **Mixed Precision**: OFF
- **Architecture**: FROZEN (No changes)

## Infrastructure Checks
- **Metrics Logged**: Train Loss, Val Loss, HR MAE, RMSE, Pearson Correlation, Learning Rate, Gradient Norm, GPU Memory, Epoch Time.
- **Artifacts Saved**: `best_model.pth`, `last_model.pth`, `epoch_05.pt`, `epoch_10.pt`, `training.log`, static tensors.
- **Visualization**: All required IEEE plots generated offline using `plot_results.py`.

## Results
- **Optimization**: Train loss and validation loss decreased monotonically, showing stable early optimization.
- **Stability**: No NaNs observed. Gradient norms remained stable.
- **Hardware**: GPU memory utilization stayed completely flat, indicating no memory leaks.

## Conclusion
The infrastructure for the experiment is functioning nominally. Optimization dynamics are stable in the early epochs. 
