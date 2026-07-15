import os
import torch
import numpy as np
from scipy import stats

def concordance_correlation_coefficient(y_true, y_pred):
    """Calculates the Concordance Correlation Coefficient (CCC)."""
    cor = np.corrcoef(y_true, y_pred)[0][1]
    
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    
    return numerator / denominator if denominator != 0 else 0.0

def compute_metrics(exp_dir):
    preds_path = os.path.join(exp_dir, 'hr_preds.pt')
    targets_path = os.path.join(exp_dir, 'hr_targets.pt')
    
    if not os.path.exists(preds_path) or not os.path.exists(targets_path):
        return None
        
    preds = torch.load(preds_path, map_location='cpu').numpy().flatten()
    targets = torch.load(targets_path, map_location='cpu').numpy().flatten()
    
    if "TargetNorm" in exp_dir:
        preds = preds * 10.0 + 85.0
    
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets)**2)))
    
    pred_std = float(np.std(preds))
    target_std = float(np.std(targets))
    
    if pred_std > 1e-6 and target_std > 1e-6:
        pearson = float(stats.pearsonr(preds, targets)[0])
    else:
        pearson = 0.0
        
    ccc = float(concordance_correlation_coefficient(targets, preds))
    if np.isnan(ccc):
        ccc = 0.0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Pearson': pearson,
        'CCC': ccc,
        'Pred Std': pred_std,
        'Target Std': target_std
    }

if __name__ == '__main__':
    experiments = [
        "experiments/EXP02_FullBaseline",
        "experiments/EXP03_Diagnostics",
        "experiments/EXP04_V2",
        "experiments/EXP05_Fusion_Concat_V2",
        "experiments/EXP06A_Waveform_Supervision",
        "experiments/EXP06C_FeatureNorm",
        "experiments/EXP08A_TargetNorm",
        "experiments/EXP08B_CCCLoss"
    ]
    
    print(f"{'Experiment':<35} | {'MAE':<7} | {'RMSE':<7} | {'Pearson':<7} | {'CCC':<7} | {'Pred Std':<8} | {'Target Std':<10}")
    print("-" * 105)
    
    for exp in experiments:
        metrics = compute_metrics(exp)
        exp_name = os.path.basename(exp)
        if metrics is None:
            print(f"{exp_name:<35} | Missing Data")
            continue
            
        print(f"{exp_name:<35} | {metrics['MAE']:<7.2f} | {metrics['RMSE']:<7.2f} | {metrics['Pearson']:<7.2f} | {metrics['CCC']:<7.2f} | {metrics['Pred Std']:<8.2f} | {metrics['Target Std']:<10.2f}")
