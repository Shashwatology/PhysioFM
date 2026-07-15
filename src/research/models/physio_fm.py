import torch
import torch.nn as nn

from .spatial.swin_v2 import SwinV2Encoder
from .spatial.stds import SpatialTemporalDifferenceStem
from .temporal.timesformer import TemporalTransformer
from .fusion.fusion_layer import FusionLayer
from .heads.task_heads import HeartRateHead, RespirationHead, WaveformHead

class PhysioFM(nn.Module):
    """
    Multimodal Physiological Foundation Model (PhysioFM v2).
    
    A highly modular architecture designed to learn a central physiological 
    latent representation from video data. It incorporates the Spatial-Temporal 
    Difference Stem (STDS) to bypass spatial gradient starvation.
    """
    def __init__(self, embed_dim: int = 512, seq_len: int = 150, stds_alpha_init: float = 0.0, stds_fusion_mode: str = 'concat'):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.stds_fusion_mode = stds_fusion_mode.lower()
        
        # 1. Spatial Backbones
        self.rgb_spatial = SwinV2Encoder(embed_dim=embed_dim, pretrained=True)
        self.rgb_stds = SpatialTemporalDifferenceStem(in_channels=3, embed_dim=embed_dim)
        
        # Fusion strategy
        if self.stds_fusion_mode == 'gate':
            self.stds_alpha = nn.Parameter(torch.tensor(stds_alpha_init))
        elif self.stds_fusion_mode == 'concat':
            self.swin_norm = nn.LayerNorm(embed_dim)
            self.stds_norm = nn.LayerNorm(embed_dim)
            self.stds_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        # Thermal spatial backbone placeholder (omitted to save memory for now)
        # self.thm_spatial = ResNetEncoder(...) 
        
        # 2. Temporal Engines
        self.rgb_temporal = TemporalTransformer(embed_dim=embed_dim, max_seq_len=seq_len)
        self.thm_temporal = TemporalTransformer(embed_dim=embed_dim, max_seq_len=seq_len)
        
        # 3. Fusion Layer
        self.fusion = FusionLayer(embed_dim=embed_dim)
        
        # 4. Task Heads
        self.hr_head = HeartRateHead(embed_dim=embed_dim)
        self.rr_head = RespirationHead(embed_dim=embed_dim)
        self.wv_head = WaveformHead(embed_dim=embed_dim, seq_len=seq_len)
        
    def forward(self, rgb_video: torch.Tensor, thm_video: torch.Tensor = None):
        """
        Args:
            rgb_video (torch.Tensor): RGB input (B, T, C, H, W)
            thm_video (torch.Tensor, optional): Thermal input (B, T, C, H, W)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing task predictions.
        """
        B, T, C, H, W = rgb_video.shape
        
        # A. Spatial Extraction
        rgb_swin_emb = self.rgb_spatial(rgb_video) # (B, T, D)
        rgb_stds_emb = self.rgb_stds(rgb_video) # (B, T, D)
        
        # Integration
        if self.stds_fusion_mode == 'gate':
            rgb_spatial_emb = rgb_swin_emb + (self.stds_alpha * rgb_stds_emb)
        elif self.stds_fusion_mode == 'add':
            rgb_spatial_emb = rgb_swin_emb + rgb_stds_emb
        elif self.stds_fusion_mode == 'concat':
            norm_swin = self.swin_norm(rgb_swin_emb)
            norm_stds = self.stds_norm(rgb_stds_emb)
            rgb_spatial_emb = self.stds_proj(torch.cat([norm_swin, norm_stds], dim=-1))
        else:
            raise ValueError(f"Unknown fusion mode {self.stds_fusion_mode}")
        
        # Modality Dropout: If thermal is missing, we create a zero-tensor placeholder
        if thm_video is None:
            thm_spatial_emb = torch.zeros_like(rgb_spatial_emb)
        else:
            pass # thm_spatial_emb = self.thm_spatial(thm_video)
            
        # B. Temporal Dynamics
        rgb_temporal_emb = self.rgb_temporal(rgb_spatial_emb) # (B, T, D)
        thm_temporal_emb = self.thm_temporal(thm_spatial_emb) # (B, T, D)
        
        # C. Fusion (Currently Identity Pass-Through for RGB)
        fused_seq_emb = self.fusion(rgb_temporal_emb, thm_temporal_emb) # (B, T, D)
        
        # D. Central Physiological Representation (Pool across time)
        # We mean-pool the sequence to get a single global context vector per video
        latent_embedding = fused_seq_emb.mean(dim=1) # (B, D)
        
        # E. Independent Task Heads
        hr_pred = self.hr_head(latent_embedding)
        rr_pred = self.rr_head(latent_embedding)
        wv_pred = self.wv_head(fused_seq_emb).squeeze(-1) # Waveform regresses from the full sequence (B, T)
        
        return {
            'hr': hr_pred,
            'rr': rr_pred,
            'waveform': wv_pred,
            'latent': latent_embedding,
            'fused_seq_emb_shape': tuple(fused_seq_emb.shape),
            'temporal_time_std': rgb_temporal_emb.std(dim=1).mean()
        }
