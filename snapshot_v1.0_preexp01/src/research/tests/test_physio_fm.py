import torch
import torch.nn as nn
from src.research.models.physio_fm import PhysioFM
import time

def test_model():
    print("Initializing PhysioFM...")
    # B=2, T=30 (shorter for quick test), C=3, H=64, W=64
    batch_size = 2
    seq_len = 30
    embed_dim = 256
    
    # We use a smaller embedding dim and swin_v2_t for speed in this test
    # Note: swin_v2_t head expects a certain resolution. For ImageNet it's 256x256. 
    # But for a quick tensor shape test, we'll try 64x64 or just skip if torchvision complains.
    # Actually, standard Swin needs spatial dimensions that are multiples of 32. 64x64 is fine.
    
    model = PhysioFM(embed_dim=embed_dim, seq_len=seq_len)
    model.train() # Set to train to test gradient flow
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Create dummy RGB video tensor (B, T, C, H, W)
    print(f"Creating dummy tensor shape: ({batch_size}, {seq_len}, 3, 256, 256)")
    # Using 256x256 to ensure Swin V2 doesn't complain about window size
    dummy_rgb = torch.randn(batch_size, seq_len, 3, 256, 256)
    
    # Forward Pass
    print("Running Forward Pass...")
    start_time = time.time()
    outputs = model(dummy_rgb, thm_video=None)
    forward_time = time.time() - start_time
    print(f"Forward Pass Time: {forward_time:.4f} seconds")
    
    # Verify Shapes
    assert outputs['hr'].shape == (batch_size, 1), f"HR shape mismatch: {outputs['hr'].shape}"
    assert outputs['rr'].shape == (batch_size, 1), f"RR shape mismatch: {outputs['rr'].shape}"
    assert outputs['waveform'].shape == (batch_size, seq_len), f"Waveform shape mismatch: {outputs['waveform'].shape}"
    assert outputs['latent'].shape == (batch_size, embed_dim), f"Latent shape mismatch: {outputs['latent'].shape}"
    print("All output tensor shapes verified successfully!")
    
    # Backward Pass (Gradient Flow)
    print("Running Backward Pass to verify gradient flow...")
    dummy_target_hr = torch.randn(batch_size, 1)
    loss_fn = nn.MSELoss()
    loss = loss_fn(outputs['hr'], dummy_target_hr)
    
    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time
    print(f"Backward Pass Time: {backward_time:.4f} seconds")
    
    # Verify gradients exist for Swin Backbone and Temporal Engine
    has_grad_spatial = any(p.grad is not None for p in model.rgb_spatial.parameters())
    has_grad_temporal = any(p.grad is not None for p in model.rgb_temporal.parameters())
    has_grad_head = any(p.grad is not None for p in model.hr_head.parameters())
    
    assert has_grad_spatial, "No gradient flow to Spatial Backbone!"
    assert has_grad_temporal, "No gradient flow to Temporal Engine!"
    assert has_grad_head, "No gradient flow to Task Head!"
    print("Gradient flow verified through the entire architecture!")
    
    print("\n--- VERIFICATION COMPLETE ---")
    print("PhysioFM is ready for integration into the training loop.")

if __name__ == "__main__":
    test_model()
