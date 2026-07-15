Experiment:
EXP02_FullBaseline

Hypothesis:
PhysioFM generalizes across five unseen subjects without unstable training.

Result:
FAIL

Reason:
• Validation behavior: ⚠ Overfitting (Val Loss increasing)
• Early stopping selected epoch 30.
• Mean MAE: 8.54 if all_latents else 'N/A'
• Mean RMSE: 8.58 if all_latents else 'N/A'
• Mean Pearson: 0.11 if all_latents else 'N/A'

Observed Weaknesses:
• Subject 40 (Error: 9.92 BPM)
• Subject 41 (Error: 8.30 BPM)
• Subject 42 (Error: 8.96 BPM)
• Subject 43 (Error: 7.76 BPM)
• Subject 44 (Error: 7.47 BPM)
• Subject 45 (Error: 8.85 BPM)

Next Decision:
Return to Stage 2.5
