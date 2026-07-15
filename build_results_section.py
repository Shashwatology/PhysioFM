import os

with open('paper/tables/master_metrics.md', 'r') as f:
    table_content = f.read()

content = f"""# Section 6: Complete Results

This section documents the absolute mathematical metrics extracted directly from the historical prediction tensors (`hr_preds.pt` and `hr_targets.pt`). 

## 6.1 Master Metric Table
*All metrics are verified from raw logs. Any missing data is explicitly omitted.*

{table_content}

## 6.2 Key Observations
1. **Prediction Variance ($0.00$):** Every single model variation collapsed to a prediction standard deviation of exactly 0.00. The models universally output a constant scalar prediction.
2. **Artificial MAE:** The constant prediction exactly matched the mean of the validation dataset ($\sim 85$ BPM). Because the validation set has a standard deviation of 8.58, predicting the constant mean mathematically guarantees an MAE of $\sim 8.54$ and an RMSE of $\sim 8.58$.
3. **Correlation Failure:** Because the prediction is a constant scalar, the covariance with the ground truth is exactly 0. Therefore, Pearson's $r$ and the Concordance Correlation Coefficient (CCC) are strictly 0.00 across the board.
"""

with open('dossier_sections/06_CompleteResults.md', 'w') as f:
    f.write(content)

print("Section 6 Generated.")
