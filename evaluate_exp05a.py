import pandas as pd
import os

path = "experiments/EXP05_Fusion_Concat_V2/comprehensive_metrics.csv"
if not os.path.exists(path):
    print("File not found.")
    exit(1)

df = pd.read_csv(path)
last_epoch = df.iloc[-1]

print("=== EXP05A (10-Epoch Concat Screening) Evaluation ===")
print(f"Epoch: {last_epoch['Epoch']}")
print(f"Prediction Std: {last_epoch['PredStd']:.6f} BPM")
print(f"Latent Std: {last_epoch['LatentStd']:.6f}")
print(f"Prediction Entropy: {last_epoch['PredictionEntropy']:.6f}")
print(f"Pearson: {last_epoch['Pearson']:.4f}")
print("---")
print(f"Grad(STDS): {last_epoch['GradSTDS']:.2e}")
print(f"Head/STDS Ratio: {last_epoch['Head/STDS Ratio']:.2f}")
print(f"Head/Swin Ratio: {last_epoch['Head/Swin Ratio']:.2f}")
print("---")
print("Evaluation:")
if last_epoch['PredStd'] < 1.0 or last_epoch['Pearson'] < 0.2:
    print("Result: COLLAPSE PERSISTS. Intervention failed.")
else:
    print("Result: SUCCESS. Collapse broken.")
