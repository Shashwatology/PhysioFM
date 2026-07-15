import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('paper/figures', exist_ok=True)

experiments = [
    ("EXP02", "experiments/EXP02_FullBaseline"),
    ("EXP04", "experiments/EXP04_V2"),
    ("EXP05", "experiments/EXP05_Fusion_Concat_V2"),
    ("EXP06C", "experiments/EXP06C_FeatureNorm"),
    ("EXP08B", "experiments/EXP08B_CCCLoss")
]

# 5. Ablation Bar Chart (EXP02 -> EXP04 -> EXP05 -> EXP06C -> EXP08B)
print("Generating Ablation Bar Chart...")
try:
    df_metrics = pd.read_csv('paper/tables/master_metrics.csv')
    # Filter only ablation subset
    ablation_df = df_metrics[df_metrics['Experiment'].isin(['EXP02', 'EXP04', 'EXP05', 'EXP06C', 'EXP08B'])].copy()
    # Sort according to chronological order
    cat_type = pd.CategoricalDtype(categories=['EXP02', 'EXP04', 'EXP05', 'EXP06C', 'EXP08B'], ordered=True)
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
except Exception as e:
    print(f"Skipping ablation chart generation: {e}")

# 6. Expanded Training Curves (LR, Waveform Loss)
print("Generating Expanded Training Curves...")
plt.figure(figsize=(10, 6))
for name, path in experiments:
    log_path = os.path.join(path, 'training.log')
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        if 'Learning_Rate' in df.columns:
            plt.plot(df['Epoch'], df['Learning_Rate'], label=name)
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.legend()
plt.grid(True)
plt.savefig('paper/figures/learning_rate_schedule.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for name, path in experiments:
    log_path = os.path.join(path, 'training.log')
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        if 'Train_WV_Loss' in df.columns:
            plt.plot(df['Epoch'], df['Train_WV_Loss'], label=name)
plt.title("Train Waveform Loss Convergence")
plt.xlabel("Epoch")
plt.ylabel("Waveform Loss")
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.savefig('paper/figures/waveform_loss_curves.png', dpi=300)
plt.close()

print("Extended Assets Generated.")
