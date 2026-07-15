Experiment:
EXP06B_PrePool_Smoke

Hypothesis:
PhysioFM generalizes across five unseen subjects without unstable training.

Result:
PASS

Reason:
• Validation behavior: Healthy convergence
• Early stopping selected epoch 5.
• Mean MAE: 8.54 if all_latents else 'N/A'
• Mean RMSE: 8.61 if all_latents else 'N/A'
• Mean Pearson: -0.41 if all_latents else 'N/A'

Observed Weaknesses:
• Subject 40 (Error: 10.45 BPM)
• Subject 41 (Error: 7.77 BPM)
• Subject 42 (Error: 9.49 BPM)
• Subject 43 (Error: 7.23 BPM)
• Subject 44 (Error: 8.00 BPM)
• Subject 45 (Error: 8.32 BPM)

Next Decision:
Proceed to Stage 3
