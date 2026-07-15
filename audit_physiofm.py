import torch
import torch.nn as nn
from src.research.models.physio_fm import PhysioFM

def analyze_tensor(name, t):
    if t is None:
        return f"{name}: None"
    mean = t.mean().item()
    std = t.std().item()
    norm = t.norm(2).item()
    return f"{name:<20} | Mean: {mean:>10.6f} | Std: {std:>10.6f} | Norm: {norm:>10.6f}"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=== PHYSIOFM ROOT CAUSE AUDIT ===")
    
    # 1. Initialize model
    model = PhysioFM(embed_dim=256, seq_len=150, stds_alpha_init=1.0)
    model.to(device)
    model.train()
    
    print("\n--- Model initialized (Pretrained Swin loaded) ---")
    
    # Create dummy input (B, T, C, H, W)
    B, T, C, H, W = 1, 30, 3, 128, 128
    dummy_video = torch.randn(B, T, C, H, W, device=device)
    dummy_target = torch.randn(B, 1, device=device) * 10 + 70 # Fake HR around 70 BPM
    
    print("\n--- Forward Pass Analysis ---")
    # We will hook into the model to extract intermediate features
    features = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                features[name] = output[0].detach()
            else:
                features[name] = output.detach()
        return hook
        
    model.rgb_spatial.register_forward_hook(hook_fn('Swin_Output'))
    if hasattr(model, 'rgb_stds'):
        model.rgb_stds.register_forward_hook(hook_fn('STDS_Output'))
        
    outputs = model(dummy_video, thm_video=None)
    
    print(analyze_tensor('Swin Output', features.get('Swin_Output')))
    print(analyze_tensor('STDS Output', features.get('STDS_Output')))
    print(analyze_tensor('Final Latent', outputs['latent']))
    print(analyze_tensor('HR Prediction', outputs['hr']))
    
    print("\n--- Backward Pass (Gradient) Analysis ---")
    criterion = nn.MSELoss()
    loss = criterion(outputs['hr'], dummy_target)
    loss.backward()
    
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Group by top-level module
            module_name = name.split('.')[0]
            if module_name not in grad_norms:
                grad_norms[module_name] = 0.0
            grad_norms[module_name] += param.grad.data.norm(2).item() ** 2
            
    print(f"{'Module':<20} | {'Grad Norm'}")
    print("-" * 40)
    for k, v in grad_norms.items():
        print(f"{k:<20} | {v ** 0.5:>10.6e}")
        
    # Analyze STDS specifically
    stds_grad = 0.0
    for name, param in model.named_parameters():
        if 'stds' in name and param.grad is not None:
            stds_grad += param.grad.data.norm(2).item() ** 2
    print(f"\nTotal STDS Grad Norm: {stds_grad ** 0.5:>10.6e}")
    
    print("\n=== AUDIT COMPLETE ===")

if __name__ == '__main__':
    main()
