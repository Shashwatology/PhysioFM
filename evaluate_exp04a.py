import pandas as pd
import os

alphas = ['0_0', '0_1', '0_5', '1_0']

print(f"{'Alpha':<8} | {'PredStd':<8} | {'Pearson':<8} | {'GradSTDS':<12} | {'GradAlpha':<12} | {'Head/Swin':<10}")
print("-" * 65)

for a in alphas:
    path = f"experiments/EXP04A_Alpha_{a}/comprehensive_metrics.csv"
    if not os.path.exists(path):
        print(f"Missing {a}")
        continue
    df = pd.read_csv(path)
    if len(df) == 0:
        continue
    
    last_epoch = df.iloc[-1]
    
    pred_std = last_epoch['PredStd']
    pearson = last_epoch['Pearson']
    grad_stds = last_epoch['GradSTDS']
    grad_alpha = last_epoch['GradAlpha']
    head_swin = last_epoch['Head/Swin Ratio']
    
    print(f"{a:<8} | {pred_std:<8.4f} | {pearson:<8.4f} | {grad_stds:<12.2e} | {grad_alpha:<12.2e} | {head_swin:<10.2f}")

print("\nEvaluation against Early Failure / Success rules:")
print("Early Failure: PredStd < 0.05 AND grad(STDS) ~ 0 AND grad(alpha) ~ 0")
print("Early Success: PredStd > 1.0 AND Pearson > 0.2")
