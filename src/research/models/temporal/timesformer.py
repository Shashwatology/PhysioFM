import torch
import torch.nn as nn
import math
from .temporal_encoder import BaseTemporalEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is (B, T, D)
        T = x.size(1)
        x = x + self.pe[:T, :].unsqueeze(0)
        return x

class TemporalTransformer(BaseTemporalEncoder):
    """
    Temporal Transformer (1D TimeSformer variant).
    Operates strictly across the temporal axis to isolate pulse frequencies
    from the spatial embeddings provided by the spatial backbone.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, num_layers: int = 4, max_seq_len: int = 500):
        super().__init__(embed_dim)
        
        # Positional Encoding to retain temporal order
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)
        
        # Standard Transformer Encoder operating across T
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape (B, T, embed_dim)
        Returns:
            torch.Tensor: Temporally enriched embeddings of shape (B, T, embed_dim)
        """
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply Temporal Self-Attention
        out = self.transformer_encoder(x)
        
        return out
