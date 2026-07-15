import torch
import torch.nn as nn
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from .spatial_encoder import BaseSpatialEncoder

class SwinV2Encoder(BaseSpatialEncoder):
    """
    Swin Transformer V2 Spatial Backbone.
    Extracts high-fidelity spatial embeddings from individual frames.
    """
    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__(embed_dim)
        
        # Load Swin V2 model
        if pretrained:
            weights = Swin_V2_T_Weights.IMAGENET1K_V1
            self.backbone = swin_v2_t(weights=weights)
        else:
            self.backbone = swin_v2_t(weights=None)
            
        # Extract features instead of classifying
        # In torchvision Swin, the output of the feature extractor is (B, 768)
        self.feature_dim = self.backbone.head.in_features
        
        # Remove the classification head
        self.backbone.head = nn.Identity()
        
        # Projection layer to map to our unified embed_dim
        self.proj = nn.Linear(self.feature_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape (B, T, C, H, W)
        Returns:
            torch.Tensor: Spatial embeddings of shape (B, T, embed_dim)
        """
        B, T, C, H, W = x.shape
        
        # Merge Batch and Time dimensions for standard 2D CNN/Swin processing
        # Swin expects (B*T, C, H, W)
        x_reshaped = x.view(B * T, C, H, W)
        
        # Forward pass through spatial backbone
        features = self.backbone(x_reshaped) # Output: (B*T, feature_dim)
        
        # Project to unified embedding dimension
        embeddings = self.proj(features) # Output: (B*T, embed_dim)
        
        # Separate Batch and Time dimensions
        embeddings = embeddings.view(B, T, self.embed_dim)
        
        return embeddings
