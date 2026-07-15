Experiment:
EXP01A_BaselineValidation

Hypothesis:
PhysioFM generalizes across five unseen subjects without unstable training.

Result:
FAIL

Reason:
• Validation behavior: ⚠ Divergence
• Early stopping selected epoch 10.
• Mean MAE: 89.91 if all_latents else 'N/A'
• Mean RMSE: 89.92 if all_latents else 'N/A'
• Mean Pearson: -1.00 if all_latents else 'N/A'

Observed Weaknesses:
• Subject 1 (Error: 89.45 BPM)
• Subject 10 (Error: 90.38 BPM)

Next Decision:
Return to Stage 2.5
