import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """
    Dynamically fuses RGB (cardiac) and Thermal (respiratory) embeddings using Cross-Attention.
    Allows one modality to 'query' the other, effectively weighting reliability based on context.
    """
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        
        # Self-attention for RGB context
        self.rgb_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # Self-attention for Thermal context
        self.thm_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention: Thermal queries RGB
        self.cross_attn_thm_q_rgb_kv = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # Cross-attention: RGB queries Thermal
        self.cross_attn_rgb_q_thm_kv = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Final projection to the unified Physiological Representation
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm_final = nn.LayerNorm(embed_dim)

    def forward(self, rgb_emb, thm_emb):
        """
        Args:
            rgb_emb (Tensor): RGB spatial/temporal embedding (B, Seq, Dim) or (B, Dim)
            thm_emb (Tensor): Thermal spatial/temporal embedding (B, Seq, Dim) or (B, Dim)
        Returns:
            Tensor: Fused Physiological Representation (B, embed_dim)
        """
        # Ensure we have a sequence dimension (e.g. B, 1, Dim) if input is 2D
        if rgb_emb.dim() == 2:
            rgb_emb = rgb_emb.unsqueeze(1)
            thm_emb = thm_emb.unsqueeze(1)
            
        # 1. Self Attention Context
        rgb_self, _ = self.rgb_self_attn(rgb_emb, rgb_emb, rgb_emb)
        rgb_emb = self.norm1(rgb_emb + rgb_self)
        
        thm_self, _ = self.thm_self_attn(thm_emb, thm_emb, thm_emb)
        thm_emb = self.norm2(thm_emb + thm_self)
        
        # 2. Cross Attention
        # RGB queries Thermal (RGB wants to know about respiration/temp)
        rgb_fused, _ = self.cross_attn_rgb_q_thm_kv(rgb_emb, thm_emb, thm_emb)
        
        # Thermal queries RGB (Thermal wants to know about pulse/cardiac)
        thm_fused, _ = self.cross_attn_thm_q_rgb_kv(thm_emb, rgb_emb, rgb_emb)
        
        # 3. Concatenation and Projection
        combined = torch.cat([rgb_fused, thm_fused], dim=-1) # (B, Seq, Dim*2)
        
        physio_rep = self.ffn(combined)
        physio_rep = self.norm_final(physio_rep)
        
        # Pool sequence dimension if needed to output (B, Dim)
        physio_rep = physio_rep.mean(dim=1) 
        
        return physio_rep

if __name__ == "__main__":
    model = CrossAttentionFusion(embed_dim=512)
    dummy_rgb = torch.randn(4, 10, 512) # Batch 4, Seq 10, Dim 512
    dummy_thm = torch.randn(4, 10, 512)
    out = model(dummy_rgb, dummy_thm)
    print(f"Fused physiological representation shape: {out.shape}") # Expected: (4, 512)
