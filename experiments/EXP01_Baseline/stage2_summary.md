Experiment:
EXP01_Baseline

Hypothesis:
PhysioFM generalizes across five unseen subjects without unstable training.

Result:
FAIL

Reason:
• Validation behavior: ⚠ Divergence
• Early stopping selected epoch 50.
• Mean MAE: 43.39 if all_latents else 'N/A'
• Mean RMSE: 43.39 if all_latents else 'N/A'
• Mean Pearson: 1.00 if all_latents else 'N/A'

Observed Weaknesses:
• Subject 1 (Error: 42.92 BPM)
• Subject 10 (Error: 43.86 BPM)

Next Decision:
Return to Stage 2.5
