import os
import torch
import numpy as np
import pandas as pd
from src.research.models.physio_fm import PhysioFM

def run_diagnostics():
    exp_dir = 'experiments/EXP02_FullBaseline'
    
    print("=== 1. PREDICTION DISTRIBUTION ANALYSIS ===")
    preds = np.load(os.path.join(exp_dir, 'predictions.npy')).flatten()
    targs = np.load(os.path.join(exp_dir, 'ground_truth.npy')).flatten()
    print(f"Predictions -> Mean: {np.mean(preds):.4f}, Std: {np.std(preds):.6f}, Min: {np.min(preds):.4f}, Max: {np.max(preds):.4f}")
    print(f"Targets     -> Mean: {np.mean(targs):.4f}, Std: {np.std(targs):.6f}, Min: {np.min(targs):.4f}, Max: {np.max(targs):.4f}")
    
    print("\n=== 2. LATENT REPRESENTATION ANALYSIS ===")
    latents = np.load(os.path.join(exp_dir, 'latent_features.npy'))
    print(f"Latent Shape: {latents.shape}")
    feature_var = np.var(latents, axis=0)
    print(f"Mean Feature Variance: {np.mean(feature_var):.6f}")
    print(f"Max Feature Variance: {np.max(feature_var):.6f}")
    
    # Pairwise L2 Distances
    N = latents.shape[0]
    distances = []
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(latents[i] - latents[j])
            distances.append(dist)
    if distances:
        print(f"Pairwise Distances -> Mean: {np.mean(distances):.6f}, Min: {np.min(distances):.6f}, Max: {np.max(distances):.6f}")
    
    print("\n=== 3. OPTIMIZATION ANALYSIS ===")
    df = pd.read_csv(os.path.join(exp_dir, 'epoch_metrics.csv'))
    print(f"Initial Train Loss: {df['Train_Loss'].iloc[0]:.4f}")
    print(f"Final Train Loss (Epoch {df['Epoch'].iloc[-1]}): {df['Train_Loss'].iloc[-1]:.4f}")
    print(f"Mean Global Grad Norm: {df['Grad_Norm'].mean():.4f}")
    print(f"Max Global Grad Norm: {df['Grad_Norm'].max():.4f}")
    
    print("\n=== 4. LAYER-WISE GRADIENT ANALYSIS ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhysioFM(embed_dim=256, seq_len=150)
    
    ckpt_path = os.path.join(exp_dir, 'best_model.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    
    model = model.to(device)
    model.train()
    
    dummy_input = torch.randn(1, 150, 3, 112, 112, device=device)
    dummy_target = torch.randn(1, 1, device=device)
    
    outputs = model(dummy_input)
    loss = torch.nn.functional.mse_loss(outputs['hr'], dummy_target)
    loss.backward()
    
    def get_norm(module):
        total_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    print(f"Head (hr_head) Grad Norm: {get_norm(model.hr_head):.6f}")
    print(f"Fusion (fusion) Grad Norm: {get_norm(model.fusion):.6f}")
    print(f"Temporal (rgb_temporal) Grad Norm: {get_norm(model.rgb_temporal):.6f}")
    print(f"Spatial (rgb_spatial) Grad Norm: {get_norm(model.rgb_spatial):.6f}")

if __name__ == '__main__':
    run_diagnostics()
