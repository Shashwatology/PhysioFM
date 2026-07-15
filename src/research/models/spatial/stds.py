import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTemporalDifferenceStem(nn.Module):
    """
    PhysioFM v2 Module: Spatial-Temporal Difference Stem (STDS)
    
    A lightweight, parallel fast-pathway designed to explicitly bypass 
    spatial gradient starvation. It computes temporal differences and 
    extracts micro-color physiological features directly from the raw pixels.
    """
    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()
        
        # Lightweight 3D CNN to extract features from temporal differences
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.global_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # Project to the shared embedding dimension
        self.proj = nn.Linear(64, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input video of shape (B, T, C, H, W)
        Returns:
            torch.Tensor: Physiological features of shape (B, T, embed_dim)
        """
        B, T, C, H, W = x.shape
        
        # 1. Compute Temporal Differences
        # diff[t] = x[t] - x[t-1]
        diff = torch.zeros_like(x)
        diff[:, 1:] = x[:, 1:] - x[:, :-1]
        
        # Pytorch 3D CNNs expect (B, C, T, H, W)
        diff = diff.permute(0, 2, 1, 3, 4).contiguous()
        
        # 2. Extract Micro-Color Features
        features = self.conv1(diff)
        features = self.bn1(features)
        features = self.relu1(features)
        features = self.pool1(features)
        
        features = self.conv2(features)
        features = self.bn2(features)
        features = self.relu2(features)
        
        # 3. Global Spatial Pooling
        # Shape becomes (B, 64, T, 1, 1)
        features = self.global_pool(features)
        
        # Reshape to (B, T, 64)
        features = features.view(B, 64, T).permute(0, 2, 1).contiguous()
        
        # 4. Project to Embed Dim
        out = self.proj(features) # (B, T, embed_dim)
        
        return out
