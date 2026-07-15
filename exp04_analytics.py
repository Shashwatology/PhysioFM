import os
import pandas as pd
import numpy as np
import scipy.stats as stats

exp_dir = 'experiments/EXP04_V2'
out_dir = 'C:/Users/MNIT USER/.gemini/antigravity-ide/brain/c0100c4e-932e-4971-9545-c6717d1e59d7'

def main():
    if not os.path.exists(exp_dir):
        print(f"Directory {exp_dir} not found!")
        return
        
    train_log = pd.read_csv(os.path.join(exp_dir, 'training.log'))
    grad_dyn = pd.read_csv(os.path.join(exp_dir, 'gradient_dynamics.csv'))
    collapse = pd.read_csv(os.path.join(exp_dir, 'collapse_dynamics.csv'))
    
    # Merge logs
    gold_df = pd.merge(train_log, grad_dyn, on='Epoch', how='inner')
    gold_df = pd.merge(gold_df, collapse, on='Epoch', how='inner')
    
    # Compute Ratios
    gold_df['Avg_Swin_Grad'] = gold_df[['Swin_Stage1', 'Swin_Stage2', 'Swin_Stage3', 'Swin_Stage4']].mean(axis=1)
    gold_df['Ratio_Head_to_Swin'] = gold_df['Head'] / (gold_df['Avg_Swin_Grad'] + 1e-9)
    gold_df['Ratio_Head_to_STDS'] = gold_df['Head'] / (gold_df['STDS'] + 1e-9)
    
    # Reorder columns
    cols = ['Epoch', 'Train_Loss', 'Val_Loss', 'Pred_Mean', 'Pred_Std', 'Pred_Entropy', 'Target_Std', 'Latent_Std',
            'Head', 'STDS', 'Fusion', 'Temporal', 'Swin_Stage4', 'Swin_Stage3', 'Swin_Stage2', 'Swin_Stage1',
            'Avg_Swin_Grad', 'Ratio_Head_to_Swin', 'Ratio_Head_to_STDS']
    
    # We might not have all columns if some were not logged, let's filter
    cols = [c for c in cols if c in gold_df.columns]
    gold_df[cols].to_csv(os.path.join(out_dir, 'exp04_gold_dynamics.csv'), index=False)
    
    print("EXP04 Analytics generated successfully.")
    
    # Let's print out the last epoch's crucial stats for evaluation
    last_row = gold_df.iloc[-1]
    print(f"--- EXP04 Final Epoch ({last_row['Epoch']}) Stats ---")
    print(f"Pred Std: {last_row['Pred_Std']:.6f}")
    print(f"Pred Entropy: {last_row['Pred_Entropy']:.2f}")
    print(f"Target Std: {last_row['Target_Std']:.6f}")
    print(f"STDS Grad: {last_row['STDS']:.2f}")
    print(f"Avg Swin Grad: {last_row['Avg_Swin_Grad']:.2f}")
    print(f"Head Grad: {last_row['Head']:.2f}")
    print(f"Ratio Head/Swin: {last_row.get('Ratio_Head_to_Swin', 0):.2f}")
    print(f"Ratio Head/STDS: {last_row.get('Ratio_Head_to_STDS', 0):.2f}")

if __name__ == '__main__':
    main()
