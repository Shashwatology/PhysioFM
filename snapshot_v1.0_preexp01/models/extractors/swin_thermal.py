import torch
import torch.nn as nn
import timm

class SwinThermalExtractor(nn.Module):
    """
    Swin Transformer V2 customized for single-channel (or false color 3-channel) 
    thermal imaging to extract respiratory and temperature asymmetry features.
    """
    def __init__(self, pretrained=True, in_channels=1, embed_dim=512):
        super(SwinThermalExtractor, self).__init__()
        
        self.backbone = timm.create_model(
            'swinv2_tiny_window16_256', 
            pretrained=pretrained, 
            in_chans=in_channels,
            num_classes=0, 
            global_pool='avg'
        )
        
        backbone_dim = self.backbone.num_features
        
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input Thermal tensor of shape (B, 1, H, W)
        Returns:
            Tensor: Extracted spatial features of shape (B, embed_dim)
        """
        features = self.backbone(x)
        embeddings = self.proj(features)
        return embeddings

if __name__ == "__main__":
    model = SwinThermalExtractor(pretrained=False, in_channels=1)
    dummy_input = torch.randn(2, 1, 256, 256)
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")  # Expected: (2, 512)
