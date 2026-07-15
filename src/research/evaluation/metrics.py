import torch
import numpy as np
import scipy.stats as stats
from typing import Dict, Any

def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor, baseline_preds: torch.Tensor = None) -> Dict[str, Any]:
    """
    Calculates standard physiological monitoring metrics with rigorous statistical significance.
    Args:
        preds: (N,) tensor of PhysioFM predictions
        targets: (N,) tensor of ground truth values
        baseline_preds: (N,) tensor of baseline predictions (optional, for paired t-tests)
    Returns:
        Dict containing MAE, RMSE, Pearson Correlation, complete with Std Dev, 95% CIs, and p-values.
    """
    metrics = {}
    
    # Ensure CPU numpy for scipy
    preds_np = preds.detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten()
    
    errors = np.abs(preds_np - targets_np)
    sq_errors = (preds_np - targets_np) ** 2
    
    # 1. MAE (Mean ± Std, 95% CI)
    mae_mean = np.mean(errors)
    mae_std = np.std(errors, ddof=1)
    # 95% CI for the mean error
    ci_margin = stats.t.ppf(0.975, len(errors)-1) * (mae_std / np.sqrt(len(errors))) if len(errors) > 1 else 0
    metrics['MAE'] = mae_mean
    metrics['MAE_std'] = mae_std
    metrics['MAE_95CI'] = (mae_mean - ci_margin, mae_mean + ci_margin)
    
    # 2. RMSE (Mean)
    rmse_mean = np.sqrt(np.mean(sq_errors))
    metrics['RMSE'] = rmse_mean
    
    # 3. Pearson Correlation (r) & CCC
    if np.std(preds_np) == 0 or np.std(targets_np) == 0:
        pearson = 0.0
        p_val_corr = 1.0
        ccc = 0.0
    else:
        pearson, p_val_corr = stats.pearsonr(preds_np, targets_np)
        
        # Calculate CCC
        mean_true, mean_pred = np.mean(targets_np), np.mean(preds_np)
        var_true, var_pred = np.var(targets_np), np.var(preds_np)
        sd_true, sd_pred = np.std(targets_np), np.std(preds_np)
        
        numerator = 2 * pearson * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred)**2
        ccc = numerator / denominator if denominator > 0 else 0.0
        
    metrics['Pearson'] = pearson
    metrics['Pearson_p_value'] = p_val_corr
    metrics['CCC'] = ccc
    
    # 4. Statistical Tests against Baseline
    if baseline_preds is not None:
        base_np = baseline_preds.detach().cpu().numpy().flatten()
        base_errors = np.abs(base_np - targets_np)
        
        # Paired t-test on Absolute Errors (Tests if one model is significantly better)
        t_stat, p_val_ttest = stats.ttest_rel(errors, base_errors)
        metrics['vs_baseline_ttest_p_value'] = p_val_ttest
        
        # Wilcoxon signed-rank test (non-parametric, safer for highly skewed errors)
        try:
            w_stat, p_val_wilcoxon = stats.wilcoxon(errors, base_errors)
            metrics['vs_baseline_wilcoxon_p_value'] = p_val_wilcoxon
        except ValueError:
            metrics['vs_baseline_wilcoxon_p_value'] = 1.0 # E.g., if all differences are exactly zero
            
    return metrics

def bland_altman_analysis(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Computes Bland-Altman limits of agreement.
    """
    preds_np = preds.detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten()
    
    differences = preds_np - targets_np
    bias = np.mean(differences)
    sd = np.std(differences, ddof=1) if len(differences) > 1 else 0.0
    
    return {
        'bias': float(bias),
        'upper_loa': float(bias + 1.96 * sd),
        'lower_loa': float(bias - 1.96 * sd)
    }

class MemoryProfiler:
    """
    Simple wrapper to trace GPU memory peaks during inference.
    """
    @staticmethod
    def start():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    @staticmethod
    def get_peak_mb():
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0
