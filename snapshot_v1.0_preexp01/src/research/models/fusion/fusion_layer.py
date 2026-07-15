import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    """
    Modality Fusion Layer.
    Currently acts as an Identity Pass-Through since only RGB data is available.
    Designed as a placeholder to be swapped out for true Cross-Attention 
    when Thermal or NIR datasets become available.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, rgb_emb: torch.Tensor, thermal_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            rgb_emb (torch.Tensor): Primary RGB embeddings (B, T, D)
            thermal_emb (torch.Tensor, optional): Secondary thermal embeddings (B, T, D)
            
        Returns:
            torch.Tensor: Unified Physiological Representation (B, T, D)
        """
        # In this phase, we strictly rely on the RGB pipeline.
        # Thermal is ignored even if provided, serving as extreme modality dropout.
        return rgb_emb
