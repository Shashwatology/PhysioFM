import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('paper/figures', exist_ok=True)

experiments = [
    ("EXP02", "experiments/EXP02_FullBaseline"),
    ("EXP03", "experiments/EXP03_Diagnostics"),
    ("EXP04", "experiments/EXP04_V2"),
    ("EXP05", "experiments/EXP05_Fusion_Concat_V2"),
    ("EXP06A", "experiments/EXP06A_Waveform_Supervision"),
    ("EXP06C", "experiments/EXP06C_FeatureNorm")
]

# 3. Training Curves
def plot_metric(metric_col, title, ylabel, filename, log_scale=False):
    plt.figure(figsize=(10, 6))
    plotted = False
    for name, path in experiments:
        log_path = os.path.join(path, 'training.log')
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            if metric_col in df.columns:
                plt.plot(df['Epoch'], df[metric_col], label=name)
                plotted = True
    if plotted:
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        if log_scale:
            plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'paper/figures/{filename}', dpi=300)
    plt.close()

plot_metric('Train_Loss', 'Training Loss Convergence', 'Loss', 'training_loss.png')
plot_metric('Val_Loss', 'Validation Loss Convergence', 'Loss', 'validation_loss.png')
plot_metric('HR_MAE', 'Validation MAE', 'MAE (BPM)', 'validation_mae.png')
plot_metric('HR_RMSE', 'Validation RMSE', 'RMSE (BPM)', 'validation_rmse.png')
plot_metric('Pearson', 'Pearson Correlation', 'r', 'validation_pearson.png')
plot_metric('Learning_Rate', 'Learning Rate Schedule', 'LR', 'learning_rate.png')
plot_metric('Train_WV_Loss', 'Waveform Loss (Train)', 'Loss', 'waveform_loss.png', log_scale=True)

# 5. Ablation Bar Chart (EXP02 -> EXP04 -> EXP05 -> EXP06C)
try:
    df_metrics = pd.read_csv('paper/tables/master_metrics.csv')
    ablation_df = df_metrics[df_metrics['Experiment'].isin(['EXP02', 'EXP04', 'EXP05', 'EXP06C'])].copy()
    cat_type = pd.CategoricalDtype(categories=['EXP02', 'EXP04', 'EXP05', 'EXP06C'], ordered=True)
    ablation_df['Experiment'] = ablation_df['Experiment'].astype(cat_type)
    ablation_df = ablation_df.sort_values('Experiment')
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(ablation_df))
    plt.bar(x, ablation_df['MAE'], color='royalblue', alpha=0.8, label='Validation MAE')
    plt.xticks(x, ablation_df['Experiment'])
    plt.title('Ablation Study: Validation MAE Evolution')
    plt.xlabel('Experiment Version')
    plt.ylabel('Mean Absolute Error (BPM)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('paper/figures/ablation_bar_chart.png', dpi=300)
    plt.close()
except:
    pass

print("Final assets generated.")
