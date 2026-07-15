Experiment:
EXP05_Fusion_Concat_V2

Hypothesis:
PhysioFM generalizes across five unseen subjects without unstable training.

Result:
FAIL

Reason:
• Validation behavior: ⚠ Divergence
• Early stopping selected epoch 10.
• Mean MAE: 8.54 if all_latents else 'N/A'
• Mean RMSE: 8.63 if all_latents else 'N/A'
• Mean Pearson: -0.41 if all_latents else 'N/A'

Observed Weaknesses:
• Subject 40 (Error: 8.74 BPM)
• Subject 41 (Error: 9.49 BPM)
• Subject 42 (Error: 7.78 BPM)
• Subject 43 (Error: 8.94 BPM)
• Subject 44 (Error: 6.28 BPM)
• Subject 45 (Error: 10.03 BPM)

Next Decision:
Return to Stage 2.5
