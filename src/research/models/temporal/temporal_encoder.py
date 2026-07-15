import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseTemporalEncoder(nn.Module, ABC):
    """
    Abstract Base Class for Temporal Encoders.
    Ensures that all temporal backbones (TimeSformer, LSTM, TCN, etc.) conform to the same interface.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the temporal encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, embed_dim)
                B = Batch Size
                T = Temporal Sequence Length
                
        Returns:
            torch.Tensor: Temporal embeddings of shape (B, T, embed_dim)
        """
        pass
