import torch
import torch.nn as nn
import timm

class SwinRGBExtractor(nn.Module):
    """
    Swin Transformer V2 for extracting cardiac and pulse (rPPG) features 
    from RGB video frames.
    """
    def __init__(self, pretrained=True, embed_dim=512):
        super(SwinRGBExtractor, self).__init__()
        
        # Load a Swin-V2-Tiny model from timm
        # We remove the classification head as this is just a feature extractor
        self.backbone = timm.create_model(
            'swinv2_tiny_window16_256', 
            pretrained=pretrained, 
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
            x (Tensor): Input RGB tensor of shape (B, C, H, W)
        Returns:
            Tensor: Extracted spatial features of shape (B, embed_dim)
        """
        # With num_classes=0 and global_pool='avg', backbone returns (B, C)
        features = self.backbone(x)
        embeddings = self.proj(features)
        
        return embeddings

if __name__ == "__main__":
    # Quick test
    model = SwinRGBExtractor(pretrained=False)
    dummy_input = torch.randn(2, 3, 256, 256)
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")  # Expected: (2, 512)
