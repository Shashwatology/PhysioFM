import torch
from src.research.models.physio_fm import PhysioFM

print("Instantiating model...")
model = PhysioFM(embed_dim=256, seq_len=150, stds_fusion_mode='concat')
print("Model created.")

B, T, C, H, W = 1, 30, 3, 128, 128
dummy_video = torch.randn(B, T, C, H, W)
print("Running forward pass...")
out = model(dummy_video)
print("Forward pass successful. Output HR shape:", out['hr'].shape)
