import torch
import torch.nn as nn

from .spatial.swin_v2 import SwinV2Encoder
from .temporal.timesformer import TemporalTransformer
from .fusion.fusion_layer import FusionLayer
from .heads.task_heads import HeartRateHead, RespirationHead, WaveformHead

class PhysioFM(nn.Module):
    """
    Multimodal Physiological Foundation Model (PhysioFM).
    
    A highly modular architecture designed to learn a central physiological 
    latent representation from video data. It supports Modality Dropout,
    allowing it to train on purely RGB datasets (like UBFC-rPPG) while
    remaining ready for future Thermal or NIR modalities.
    """
    def __init__(self, embed_dim: int = 512, seq_len: int = 150):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # 1. Spatial Backbones
        self.rgb_spatial = SwinV2Encoder(embed_dim=embed_dim, pretrained=True)
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
        rgb_spatial_emb = self.rgb_spatial(rgb_video) # (B, T, D)
        
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
        wv_pred = self.wv_head(latent_embedding) # Waveform regresses from the global latent context
        
        return {
            'hr': hr_pred,
            'rr': rr_pred,
            'waveform': wv_pred,
            'latent': latent_embedding
        }
