import os
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

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

# Colorblind-friendly palette (Okabe-Ito)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000"
}

def save_pub_figure(fig_dir, name):
    """Saves figure in both high-res PNG and vector PDF."""
    png_path = os.path.join(fig_dir, f"{name}.png")
    pdf_path = os.path.join(fig_dir, f"{name}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    return f"{name}.png", f"{name}.pdf"

def generate_visualization_report(exp_dir, files_found, files_missing, figures, gen_time, warnings):
    report_path = os.path.join(exp_dir, "visualization_report.md")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("# Visualization Report\\n\\n")
        
        f.write("## Files Found\\n")
        if files_found:
            for file in files_found:
                f.write(f"- `{file}`\\n")
        else:
            f.write("- None\\n")
            
        f.write("\\n## Files Missing\\n")
        if files_missing:
            for file in files_missing:
                f.write(f"- `{file}`\\n")
        else:
            f.write("- None\\n")
            
        f.write("\\n## Figures Generated (PNG & PDF)\\n")
        if figures:
            for fig in figures:
                f.write(f"- `{fig[0]}`, `{fig[1]}`\\n")
        else:
            f.write("- None\\n")
            
        f.write(f"\\n## Figure Generation Time\\n- **{gen_time:.2f} seconds**\\n")
        
        f.write("\\n## Warnings\\n")
        if warnings:
            for warn in warnings:
                f.write(f"- ⚠ {warn}\\n")
        else:
            f.write("- None\\n")

def main():
    parser = argparse.ArgumentParser(description="Standalone, Decoupled Visualization Pipeline.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment directory.")
    args = parser.parse_args()
    
    exp_dir = args.exp_dir
    if not os.path.isdir(exp_dir):
        print(f"Error: Experiment directory '{exp_dir}' does not exist.")
        exit(1)
        
    start_time = time.time()
    
    required_files = {
        "training.log": os.path.join(exp_dir, "training.log"),
        "subject_metrics.csv": os.path.join(exp_dir, "subject_metrics.csv"),
        "hr_preds.pt": os.path.join(exp_dir, "hr_preds.pt"),
        "hr_targets.pt": os.path.join(exp_dir, "hr_targets.pt"),
        "latents.pt": os.path.join(exp_dir, "latents.pt"),
        "waveforms.pt": os.path.join(exp_dir, "waveforms.pt")
    }
    
    files_found = []
    files_missing = []
    
    for name, path in required_files.items():
        if os.path.exists(path):
            files_found.append(name)
        else:
            files_missing.append(name)
            
    warnings_list = []
    
    if files_missing:
        print("CRITICAL FAILURE: Missing required artifact files.")
        for missing in files_missing:
            print(f" - {missing}")
        generate_visualization_report(exp_dir, files_found, files_missing, [], 0.0, ["Pipeline aborted due to missing files."])
        exit(1)
        
    fig_dir = os.path.join(exp_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    figures_generated = []
    
    # 1. Load Training Log
    df_log = pd.read_csv(required_files["training.log"])
    epochs = df_log["Epoch"]
    
    # train_loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, df_log["Train_Loss"], label="Train Loss", color=COLORS['blue'], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss Convergence")
    plt.legend()
    figures_generated.append(save_pub_figure(fig_dir, "train_loss"))
    
    # val_loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, df_log["Val_Loss"], label="Val Loss", color=COLORS['orange'], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Validation Loss Convergence")
    plt.legend()
    figures_generated.append(save_pub_figure(fig_dir, "val_loss"))
    
    # gradient_norms
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, df_log["Grad_Norm"], label="Grad Norm", color=COLORS['green'], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm (L2)")
    plt.title("Gradient Stability")
    plt.legend()
    figures_generated.append(save_pub_figure(fig_dir, "gradient_norms"))
    
    # learning_rate
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, df_log["LR"], label="Learning Rate", color=COLORS['red'], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.yscale("log")
    plt.legend()
    figures_generated.append(save_pub_figure(fig_dir, "learning_rate"))
    
    # gpu_memory
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, df_log["GPU_Mem_MB"], label="GPU Memory", color=COLORS['purple'], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Memory (MB)")
    plt.title("GPU Memory Utilization")
    plt.legend()
    figures_generated.append(save_pub_figure(fig_dir, "gpu_memory"))
    
    # 2. HR Scatter
    preds = torch.load(required_files["hr_preds.pt"], weights_only=True).numpy().flatten()
    targets = torch.load(required_files["hr_targets.pt"], weights_only=True).numpy().flatten()
    
    plt.figure(figsize=(5, 5))
    plt.scatter(targets, preds, alpha=0.6, color=COLORS['blue'], edgecolor='white', s=50)
    min_val = min(np.min(targets), np.min(preds)) - 5
    max_val = max(np.max(targets), np.max(preds)) + 5
    plt.plot([min_val, max_val], [min_val, max_val], color=COLORS['black'], linestyle="--", label="Identity (y=x)")
    
    if len(preds) > 1 and np.std(preds) > 0 and np.std(targets) > 0:
        pearson = np.corrcoef(targets, preds)[0, 1]
    else:
        pearson = 0.0
    
    plt.text(0.05, 0.95, f"Pearson: {pearson:.3f}", transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
             
    plt.xlabel("Ground Truth HR (BPM)")
    plt.ylabel("Predicted HR (BPM)")
    plt.title("Heart Rate Agreement")
    plt.legend(loc="lower right")
    figures_generated.append(save_pub_figure(fig_dir, "hr_scatter"))
    
    # 3. Waveform Overlay
    waveforms = torch.load(required_files["waveforms.pt"], weights_only=True).numpy()
    if waveforms.shape[0] > 0:
        wv_pred = waveforms[0]
        warnings_list.append("True Waveform (BVP) was not exported by training script. Plotting only Predicted Waveform.")
        plt.figure(figsize=(8, 3))
        time_axis = np.arange(len(wv_pred)) / 30.0 # Assuming 30fps
        plt.plot(time_axis, wv_pred, color=COLORS['red'], label='Predicted BVP', linewidth=1.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (A.U.)")
        plt.title("Reconstructed BVP Waveform")
        plt.legend(loc="upper right")
        figures_generated.append(save_pub_figure(fig_dir, "waveform_overlay"))
        
    # 4. Subject Metrics
    df_sub = pd.read_csv(required_files["subject_metrics.csv"])
    if "Absolute_Error" in df_sub.columns:
        df_sub = df_sub.sort_values(by="Absolute_Error", ascending=True)
        plt.figure(figsize=(8, 4))
        plt.bar([str(s) for s in df_sub["Subject_ID"]], df_sub["Absolute_Error"], color=COLORS['cyan'], edgecolor='black')
        plt.xlabel("Subject ID")
        plt.ylabel("MAE (BPM)")
        plt.title("Subject-wise Performance")
        plt.xticks(rotation=45, ha='right')
        figures_generated.append(save_pub_figure(fig_dir, "subject_metrics"))
    else:
        warnings_list.append("subject_metrics.csv missing Absolute_Error column.")
        
    # 5. UMAP Latents
    try:
        import umap
        latents_dict = torch.load(required_files["latents.pt"], weights_only=True)
        latents_np = latents_dict['latents'].numpy()
        
        if len(latents_np.shape) == 3:
            latents_np = latents_np.reshape(-1, latents_np.shape[-1])
            
        n_samples = latents_np.shape[0]
        if n_samples > 2:
            n_neighbors = min(15, n_samples - 1)
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=42)
            latents_2d = reducer.fit_transform(latents_np)
            
            plt.figure(figsize=(6, 5))
            plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.7, color=COLORS['purple'], edgecolor='white', s=40)
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            plt.title("Latent Manifold Projection")
            figures_generated.append(save_pub_figure(fig_dir, "umap_latents"))
        else:
            warnings_list.append(f"Not enough latent samples for UMAP (found {n_samples}, requires >2).")
    except ImportError:
        warnings_list.append("umap-learn is not installed. Failed to generate UMAP.")
    except Exception as e:
        warnings_list.append(f"UMAP failed: {str(e)}")
        
    gen_time = time.time() - start_time
    generate_visualization_report(exp_dir, files_found, files_missing, figures_generated, gen_time, warnings_list)
    print(f"Visualization complete in {gen_time:.2f} seconds. Generated {len(figures_generated)} figures (PNG/PDF pairs).")

if __name__ == "__main__":
    main()
