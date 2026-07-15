import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseSpatialEncoder(nn.Module, ABC):
    """
    Abstract Base Class for Spatial Encoders.
    Ensures that all backbones (Swin, ResNet, etc.) conform to the same interface
    for easy ablation studies.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the spatial encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)
                B = Batch Size
                T = Temporal Sequence Length
                C = Channels
                H, W = Spatial Dimensions
                
        Returns:
            torch.Tensor: Spatial embeddings of shape (B, T, embed_dim)
        """
        pass
