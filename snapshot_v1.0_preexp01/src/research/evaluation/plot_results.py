import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from src.research.evaluation.metrics import bland_altman_analysis, calculate_metrics

def plot_loss_curve(exp_name: str):
    csv_path = f"experiments/logs/{exp_name}_loss.csv"
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="Epoch", y="Loss", marker="o")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(f"experiments/figures/{exp_name}_loss_curve.png", dpi=300)
    plt.close()

def plot_hr_scatter(preds: np.ndarray, targets: np.ndarray, exp_name: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.6, edgecolors='w')
    
    # Perfect agreement line
    min_val = min(np.min(targets), np.min(preds)) - 5
    max_val = max(np.max(targets), np.max(preds)) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel("Ground Truth HR (BPM)")
    plt.ylabel("Predicted HR (BPM)")
    plt.title("HR Prediction vs Ground Truth")
    plt.grid(True)
    plt.savefig(f"experiments/figures/{exp_name}_hr_scatter.png", dpi=300)
    plt.close()

def plot_bland_altman(preds: np.ndarray, targets: np.ndarray, exp_name: str):
    import torch
    ba_stats = bland_altman_analysis(torch.tensor(preds), torch.tensor(targets))
    
    means = (preds + targets) / 2
    diffs = preds - targets
    
    plt.figure(figsize=(8, 5))
    plt.scatter(means, diffs, alpha=0.6)
    plt.axhline(ba_stats['bias'], color='red', linestyle='-', label=f"Bias: {ba_stats['bias']:.2f}")
    plt.axhline(ba_stats['upper_loa'], color='red', linestyle='--', label=f"+1.96 SD: {ba_stats['upper_loa']:.2f}")
    plt.axhline(ba_stats['lower_loa'], color='red', linestyle='--', label=f"-1.96 SD: {ba_stats['lower_loa']:.2f}")
    
    plt.xlabel("Mean of Prediction and Ground Truth (BPM)")
    plt.ylabel("Difference (Prediction - Ground Truth)")
    plt.title("Bland-Altman Plot")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"experiments/figures/{exp_name}_bland_altman.png", dpi=300)
    plt.close()

def plot_waveform(waveforms: np.ndarray, exp_name: str):
    # Plot the first sample in the batch
    if len(waveforms) == 0:
        return
        
    wf = waveforms[0]
    plt.figure(figsize=(10, 3))
    plt.plot(wf, color='blue', label='Predicted Waveform')
    plt.xlabel("Frames")
    plt.ylabel("Amplitude")
    plt.title("Sample Reconstructed BVP Waveform")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"experiments/figures/{exp_name}_waveform_sample.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs("experiments/figures", exist_ok=True)
    
    # 1. Loss Curve
    plot_loss_curve(args.exp_name)
    
    # 2. HR Scatter & Bland Altman
    preds_file = f"experiments/predictions/{args.exp_name}_hr_preds.pt"
    targets_file = f"experiments/predictions/{args.exp_name}_hr_targets.pt"
    
    if os.path.exists(preds_file) and os.path.exists(targets_file):
        preds = torch.load(preds_file).numpy().flatten()
        targets = torch.load(targets_file).numpy().flatten()
        
        plot_hr_scatter(preds, targets, args.exp_name)
        plot_bland_altman(preds, targets, args.exp_name)
        
        # Calculate full metrics to console
        metrics = calculate_metrics(torch.tensor(preds), torch.tensor(targets))
        print(f"Metrics for {args.exp_name}:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
    
    # 3. Waveform
    wf_file = f"experiments/waveforms/{args.exp_name}_waveforms.pt"
    if os.path.exists(wf_file):
        waveforms = torch.load(wf_file).numpy()
        plot_waveform(waveforms, args.exp_name)
        
    print(f"All figures generated in experiments/figures/")
