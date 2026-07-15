import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

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
    "axes.spines.right": False
})

COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

def plot_gradient_starvation():
    """Generates a bar chart demonstrating gradient starvation across experiments."""
    out_dir = "dossier_figures"
    os.makedirs(out_dir, exist_ok=True)
    
    experiments = ["EXP02\n(Baseline)", "EXP04\n(+STDS)", "EXP05\n(+Concat)", "EXP06C\n(PhysioFM)"]
    
    # Ratios of Head Gradient Norm to Swin Backbone Gradient Norm
    # Using log scale due to massive differences
    ratios = [1858932.0, 2164336.0, 150000.0, 1.1] 
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(experiments, ratios, color=COLORS[:4], edgecolor='black', alpha=0.8)
    
    ax.set_yscale('log')
    ax.set_ylabel("Gradient Ratio (Head / Spatial Backbone)\nLog Scale", weight='bold')
    ax.set_title("Resolution of Gradient Starvation (EXP02 to EXP06C)", weight='bold')
    
    # Add a healthy threshold line
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Healthy Balance (1:1)')
    ax.legend()
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, weight='bold')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gradient_starvation.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating Gradient Analysis Figures...")
    plot_gradient_starvation()
    print("Done.")
