Experiment:
EXP06C_FeatureNorm

Hypothesis:
PhysioFM generalizes across five unseen subjects without unstable training.

Result:
FAIL

Reason:
• Validation behavior: ⚠ Overfitting (Val Loss increasing)
• Early stopping selected epoch 29.
• Mean MAE: 8.54 if all_latents else 'N/A'
• Mean RMSE: 8.58 if all_latents else 'N/A'
• Mean Pearson: -0.17 if all_latents else 'N/A'

Observed Weaknesses:
• Subject 40 (Error: 9.94 BPM)
• Subject 41 (Error: 8.28 BPM)
• Subject 42 (Error: 8.98 BPM)
• Subject 43 (Error: 7.74 BPM)
• Subject 44 (Error: 7.49 BPM)
• Subject 45 (Error: 8.83 BPM)

Next Decision:
Return to Stage 2.5
