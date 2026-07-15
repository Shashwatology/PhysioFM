import os
import argparse
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# IEEE Publication Style Configuration
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True
})

def cohens_d(x, y):
    """Calculates Cohen's d for paired samples."""
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def bland_altman_plot(preds, targets, out_dir):
    """Generates a Bland-Altman plot."""
    means = (preds + targets) / 2
    diffs = preds - targets
    
    bias = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)
    
    loa_upper = bias + 1.96 * sd_diff
    loa_lower = bias - 1.96 * sd_diff
    
    plt.figure(figsize=(7, 5))
    plt.scatter(means, diffs, alpha=0.7, color="#0072B2", edgecolor='white', s=40)
    plt.axhline(bias, color='red', linestyle='-', linewidth=2, label=f'Bias: {bias:.2f}')
    plt.axhline(loa_upper, color='black', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {loa_upper:.2f}')
    plt.axhline(loa_lower, color='black', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {loa_lower:.2f}')
    
    plt.xlabel('Mean of Prediction and Target (BPM)')
    plt.ylabel('Difference (Prediction - Target) (BPM)')
    plt.title('Bland-Altman Plot')
    plt.legend()
    
    plt.savefig(os.path.join(out_dir, "bland_altman.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "bland_altman.pdf"), bbox_inches='tight')
    plt.close()
    
    return bias, loa_lower, loa_upper

def main():
    parser = argparse.ArgumentParser(description="Statistical Validation Suite (EXP07)")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing hr_preds.pt and hr_targets.pt")
    parser.add_argument("--out_dir", type=str, default="experiments/EXP07_Statistical_Validation")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    preds_path = os.path.join(args.source_dir, "hr_preds.pt")
    targets_path = os.path.join(args.source_dir, "hr_targets.pt")
    
    if not os.path.exists(preds_path) or not os.path.exists(targets_path):
        print(f"[ERROR] Could not find predictions in {args.source_dir}. Ensure training finished.")
        return
        
    preds = torch.load(preds_path, map_location='cpu').numpy().flatten()
    targets = torch.load(targets_path, map_location='cpu').numpy().flatten()
    
    if len(preds) != len(targets):
        print("[ERROR] Mismatched prediction and target lengths.")
        return
        
    print(f"Loaded {len(preds)} samples for statistical validation.")
    
    # 1. Bland-Altman
    bias, loa_lower, loa_upper = bland_altman_plot(preds, targets, args.out_dir)
    
    # 2. Cohen's d
    d_val = cohens_d(preds, targets)
    
    # 3. Significance Testing
    diffs = preds - targets
    
    # Check normality of differences
    stat_shapiro, p_shapiro = stats.shapiro(diffs)
    is_normal = p_shapiro > 0.05
    
    if is_normal:
        stat_test, p_test = stats.ttest_rel(preds, targets)
        test_name = "Paired t-test"
    else:
        stat_test, p_test = stats.wilcoxon(preds, targets)
        test_name = "Wilcoxon signed-rank test"
        
    # Generate Report
    report_path = os.path.join(args.out_dir, "statistical_report.md")
    with open(report_path, "w") as f:
        f.write("# EXP07: Statistical Validation Report\n\n")
        f.write(f"**Source Evaluation:** `{args.source_dir}`\n\n")
        
        f.write("## 1. Bland-Altman Analysis\n")
        f.write(f"- **Mean Bias:** {bias:.3f} BPM\n")
        f.write(f"- **Limits of Agreement (95%):** [{loa_lower:.3f}, {loa_upper:.3f}]\n\n")
        
        f.write("## 2. Effect Size (Cohen's d)\n")
        f.write(f"- **Cohen's d:** {d_val:.3f}\n")
        if abs(d_val) < 0.2:
            f.write("  - *Interpretation:* Negligible effect size (Excellent alignment).\n\n")
        elif abs(d_val) < 0.5:
            f.write("  - *Interpretation:* Small effect size.\n\n")
        else:
            f.write("  - *Interpretation:* Medium/Large effect size (Significant systematic bias).\n\n")
            
        f.write("## 3. Statistical Significance Testing\n")
        f.write(f"- **Normality Test (Shapiro-Wilk):** p = {p_shapiro:.4e} (Errors are {'Normal' if is_normal else 'Not Normal'})\n")
        f.write(f"- **Selected Test:** {test_name}\n")
        f.write(f"- **Test Statistic:** {stat_test:.4f}\n")
        f.write(f"- **P-Value:** {p_test:.4e}\n")
        if p_test < 0.05:
            f.write("  - *Conclusion:* Reject Null Hypothesis. There is a statistically significant difference between predictions and targets.\n")
        else:
            f.write("  - *Conclusion:* Fail to reject Null Hypothesis. There is no statistically significant difference between predictions and targets.\n")

    print(f"Validation complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
