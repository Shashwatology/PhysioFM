import torch
import torch.nn as nn
from transformers import TimesformerModel, TimesformerConfig

class TimeSformerMotionExtractor(nn.Module):
    """
    TimeSformer model for extracting spatiotemporal motion features 
    (activity, posture, falls, micro-movements) directly from video sequences.
    """
    def __init__(self, pretrained=True, seq_len=64, embed_dim=512):
        super(TimeSformerMotionExtractor, self).__init__()
        
        if pretrained:
            # We use a base TimeSformer pre-trained on Kinetics-400
            self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
        else:
            config = TimesformerConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                num_frames=seq_len,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
            )
            self.backbone = TimesformerModel(config)
            
        backbone_dim = 768
        
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input video tensor of shape (B, T, C, H, W)
        Returns:
            Tensor: Extracted spatiotemporal motion embedding of shape (B, embed_dim)
        """
        # TimeSformer expects shape (B, T, C, H, W) 
        outputs = self.backbone(x)
        
        # The first token is the CLS token which aggregates the global video context
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        embeddings = self.proj(cls_token)
        return embeddings

if __name__ == "__main__":
    # Note: Requires transformers library
    model = TimeSformerMotionExtractor(pretrained=False, seq_len=16)
    dummy_input = torch.randn(1, 16, 3, 224, 224)
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")  # Expected: (1, 512)
