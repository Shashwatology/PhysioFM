import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings('ignore')

os.makedirs('paper/figures', exist_ok=True)
os.makedirs('paper/tables', exist_ok=True)
os.makedirs('paper/assets', exist_ok=True)

experiments = [
    ("EXP02", "experiments/EXP02_FullBaseline"),
    ("EXP03", "experiments/EXP03_Diagnostics"),
    ("EXP04", "experiments/EXP04_V2"),
    ("EXP05", "experiments/EXP05_Fusion_Concat_V2"),
    ("EXP06A", "experiments/EXP06A_Waveform_Supervision"),
    ("EXP06C", "experiments/EXP06C_FeatureNorm"),
    ("EXP08A", "experiments/EXP08A_TargetNorm"),
    ("EXP08B", "experiments/EXP08B_CCCLoss")
]

# 1. Generate Master Metric Table
print("Generating Master Metrics Table...")
def concordance_correlation_coefficient(y_true, y_pred):
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator if denominator != 0 else 0.0

table_data = []
for name, path in experiments:
    preds_path = os.path.join(path, 'hr_preds.pt')
    targets_path = os.path.join(path, 'hr_targets.pt')
    log_path = os.path.join(path, 'training.log')
    
    if os.path.exists(preds_path) and os.path.exists(targets_path):
        preds = torch.load(preds_path, map_location='cpu').numpy().flatten()
        targets = torch.load(targets_path, map_location='cpu').numpy().flatten()
        
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(np.mean((preds - targets)**2))
        pred_std = np.std(preds)
        target_std = np.std(targets)
        bias = np.mean(preds - targets)
        pearson = np.corrcoef(targets, preds)[0, 1] if pred_std > 1e-6 else 0.0
        ccc = concordance_correlation_coefficient(targets, preds) if pred_std > 1e-6 else 0.0
        
        table_data.append({
            'Experiment': name,
            'MAE': mae, 'RMSE': rmse, 'Pearson': pearson, 'CCC': ccc,
            'Pred Std': pred_std, 'Target Std': target_std, 'Bias': bias
        })

df_metrics = pd.DataFrame(table_data)
df_metrics.to_csv('paper/tables/master_metrics.csv', index=False)
df_metrics.to_markdown('paper/tables/master_metrics.md', index=False)

# 2. Generate Training Curves (Multi-Experiment)
print("Generating Training Curves...")
plt.figure(figsize=(10, 6))
for name, path in experiments:
    log_path = os.path.join(path, 'training.log')
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        plt.plot(df['Epoch'], df['HR_MAE'], label=name)
plt.title("Validation MAE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error (BPM)")
plt.legend()
plt.grid(True)
plt.savefig('paper/figures/val_mae_curves.png', dpi=300)
plt.close()

# 3. Prediction Standard Deviation vs Target Standard Deviation over Epochs (Showing Mode Collapse)
print("Generating Prediction Variance Chart...")
plt.figure(figsize=(10, 6))
exp_names = df_metrics['Experiment'].tolist()
pred_stds = df_metrics['Pred Std'].tolist()
target_stds = df_metrics['Target Std'].tolist()

x = np.arange(len(exp_names))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, pred_stds, width, label='Prediction Std')
rects2 = ax.bar(x + width/2, target_stds, width, label='Target Std')
ax.set_ylabel('Standard Deviation (BPM)')
ax.set_title('Catastrophic Mode Collapse: Prediction vs Target Variance')
ax.set_xticks(x)
ax.set_xticklabels(exp_names)
ax.legend()
fig.tight_layout()
plt.savefig('paper/figures/variance_collapse.png', dpi=300)
plt.close()

# 4. Bland-Altman for EXP06C
print("Generating Bland-Altman...")
exp06c_path = "experiments/EXP06C_FeatureNorm"
if os.path.exists(os.path.join(exp06c_path, 'hr_preds.pt')):
    preds = torch.load(os.path.join(exp06c_path, 'hr_preds.pt'), map_location='cpu').numpy().flatten()
    targets = torch.load(os.path.join(exp06c_path, 'hr_targets.pt'), map_location='cpu').numpy().flatten()
    
    means = (preds + targets) / 2
    diffs = preds - targets
    bias = np.mean(diffs)
    std_diff = np.std(diffs)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(means, diffs, alpha=0.5, color='purple')
    plt.axhline(bias, color='red', linestyle='--', label=f'Mean Bias: {bias:.2f}')
    plt.axhline(bias + 1.96 * std_diff, color='gray', linestyle='--', label=f'+1.96 SD: {bias + 1.96*std_diff:.2f}')
    plt.axhline(bias - 1.96 * std_diff, color='gray', linestyle='--', label=f'-1.96 SD: {bias - 1.96*std_diff:.2f}')
    plt.xlabel('Mean of Prediction and Target (BPM)')
    plt.ylabel('Difference (Prediction - Target)')
    plt.title('Bland-Altman Plot (EXP06C) - Revealing Unacceptable LOA')
    plt.legend()
    plt.grid(True)
    plt.savefig('paper/figures/bland_altman_exp06c.png', dpi=300)
    plt.close()

print("Publication Assets Generated Successfully.")
