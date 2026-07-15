import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import numpy as np
import time

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_target_grad_norm(model):
    """Computes the gradient norm specifically for the temporal transformer and fusion layers."""
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None and ('transformer' in name or 'fusion' in name):
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def run_loss_audit(config_path="configs/exp06c_featurenorm.yaml", epochs=10):
    config = load_config(config_path)
    print(f"--- Launching Loss-Scale Audit (Config: {config_path}) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from src.research.models.physio_fm import PhysioFM
    model = PhysioFM(
        embed_dim=config.get('embed_dim', 256), 
        seq_len=config.get('seq_len', 150),
        stds_alpha_init=config.get('stds_alpha_init', 0.0),
        stds_fusion_mode=config.get('stds_fusion_mode', 'concat')
    ).to(device)
    
    from src.research.preprocessing.ubfc_dataset import UBFCDataset
    db_path = config.get('dataset_path', "datasets/UBFC-rPPG/raw")
    train_dataset = UBFCDataset(root_dir=db_path, split='train')
    
    # Fast Audit: Limit to 3 subjects so epochs take seconds instead of minutes
    train_dataset.samples = train_dataset.samples[:3]
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 2), shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.get('learning_rate', 1e-3), weight_decay=1e-4)
    criterion = nn.MSELoss()
    from src.research.training.losses import NegativePearsonLoss
    neg_pearson_loss = NegativePearsonLoss().to(device)
    
    wv_loss_weight = config.get('waveform_loss_weight', 1.0)
    
    print(f"{'Epoch':<6} | {'HR Loss':<10} | {'WV Loss':<10} | {'Grad L_hr':<12} | {'Grad L_wv':<12} | {'Ratio (HR/WV)':<15}")
    print("-" * 80)
    
    audit_results = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        epoch_hr_loss = 0.0
        epoch_wv_loss = 0.0
        
        epoch_grad_hr = 0.0
        epoch_grad_wv = 0.0
        
        for batch_idx, (videos, targets) in enumerate(train_loader):
            videos = videos.to(device)
            target_hr = targets['hr'].view(-1, 1).to(device)
            target_bvp = targets['bvp'].to(device)
            
            outputs = model(videos, thm_video=None)
            
            hr_loss = criterion(outputs['hr'], target_hr)
            wv_loss = neg_pearson_loss(outputs['waveform'], target_bvp)
            
            # 1. Backward for HR Loss ONLY
            optimizer.zero_grad()
            hr_loss.backward(retain_graph=True)
            grad_hr = get_target_grad_norm(model)
            model.zero_grad()
            
            # 2. Backward for Waveform Loss ONLY
            scaled_wv_loss = wv_loss_weight * wv_loss
            scaled_wv_loss.backward(retain_graph=True)
            grad_wv = get_target_grad_norm(model)
            model.zero_grad()
            
            # 3. Final Backward and Step
            total_loss = hr_loss + scaled_wv_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            epoch_hr_loss += hr_loss.item()
            epoch_wv_loss += wv_loss.item()
            epoch_grad_hr += grad_hr
            epoch_grad_wv += grad_wv
            
        avg_hr_loss = epoch_hr_loss / len(train_loader)
        avg_wv_loss = epoch_wv_loss / len(train_loader)
        avg_grad_hr = epoch_grad_hr / len(train_loader)
        avg_grad_wv = epoch_grad_wv / len(train_loader)
        
        ratio = avg_grad_hr / avg_grad_wv if avg_grad_wv > 0 else float('inf')
        
        print(f"{epoch:<6} | {avg_hr_loss:<10.4f} | {avg_wv_loss:<10.4f} | {avg_grad_hr:<12.4f} | {avg_grad_wv:<12.4f} | {ratio:<15.2f}")
        
        audit_results.append({
            'epoch': epoch,
            'hr_loss': avg_hr_loss,
            'wv_loss': avg_wv_loss,
            'grad_hr': avg_grad_hr,
            'grad_wv': avg_grad_wv,
            'ratio': ratio
        })
        
    print("-" * 80)
    
    # Save audit report
    with open("experiments/loss_scale_audit.md", "w") as f:
        f.write("# Loss-Scale Gradient Audit\\n\\n")
        f.write(f"Configuration: EXP06C (FeatureNorm + Waveform Loss, λ={wv_loss_weight})\\n\\n")
        f.write("| Epoch | HR Loss | WV Loss (Unscaled) | $$||\\nabla \\mathcal{L}_{hr}||$$ | $$||\\nabla \\mathcal{L}_{wv}||$$ | Ratio |\\n")
        f.write("|---|---|---|---|---|---|\\n")
        for res in audit_results:
            f.write(f"| {res['epoch']} | {res['hr_loss']:.4f} | {res['wv_loss']:.4f} | {res['grad_hr']:.4f} | {res['grad_wv']:.4f} | {res['ratio']:.2f} |\\n")
            
        final_ratio = audit_results[-1]['ratio']
        if final_ratio > 10.0:
            f.write(f"\\n**Conclusion:** A severe gradient imbalance exists. The MSE Loss gradients overpower the Waveform Loss gradients by a factor of {final_ratio:.2f}. The network collapses to the mean because the Waveform Loss is mathematically underweighted.\\n")
        else:
            f.write(f"\\n**Conclusion:** The gradients are balanced (Ratio: {final_ratio:.2f}). Scaling is NOT the root cause.\\n")

if __name__ == "__main__":
    run_loss_audit()
