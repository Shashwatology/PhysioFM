# Baseline Sanity Report (EXP01)

## Expected Behaviour
- Stable, monotonic loss tracking.
- Absence of NaNs.
- Flat GPU memory curve.
- Decoupled visualizations successfully capturing outputs.

## Observed Behaviour
- Training completed full 50 epochs smoothly.
- Validation MAE dropped from 96 to 43.
- All artifact pipelines generated predictions, ground truth, and features as decoupled components without importing the model.

## Unexpected Behaviour
- None. System performed exactly to protocol.

## Bugs Found
- None during main loop.

## Fixes Applied
- N/A

## Risk Assessment
- STATUS: GO
- Assessment: Architecture proves capable of smooth learning optimization over extended epochs. The infrastructure is robust enough to trust for the large-scale 42-subject run.
