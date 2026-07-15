Experiment:
EXP03_Diagnostics

Hypothesis:
PhysioFM generalizes across five unseen subjects without unstable training.

Result:
FAIL

Reason:
• Validation behavior: ⚠ Overfitting (Val Loss increasing)
• Early stopping selected epoch 37.
• Mean MAE: 8.54 if all_latents else 'N/A'
• Mean RMSE: 8.59 if all_latents else 'N/A'
• Mean Pearson: -0.17 if all_latents else 'N/A'

Observed Weaknesses:
• Subject 40 (Error: 9.19 BPM)
• Subject 41 (Error: 9.03 BPM)
• Subject 42 (Error: 8.23 BPM)
• Subject 43 (Error: 8.49 BPM)
• Subject 44 (Error: 6.74 BPM)
• Subject 45 (Error: 9.58 BPM)

Next Decision:
Return to Stage 2.5
