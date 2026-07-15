Experiment:
EXP06A_Waveform_Supervision

Hypothesis:
PhysioFM generalizes across five unseen subjects without unstable training.

Result:
FAIL

Reason:
• Validation behavior: ⚠ Divergence
• Early stopping selected epoch 10.
• Mean MAE: 8.54 if all_latents else 'N/A'
• Mean RMSE: 8.65 if all_latents else 'N/A'
• Mean Pearson: -0.46 if all_latents else 'N/A'

Observed Weaknesses:
• Subject 40 (Error: 8.57 BPM)
• Subject 41 (Error: 9.65 BPM)
• Subject 42 (Error: 7.61 BPM)
• Subject 43 (Error: 9.11 BPM)
• Subject 44 (Error: 6.12 BPM)
• Subject 45 (Error: 10.20 BPM)

Next Decision:
Return to Stage 2.5
