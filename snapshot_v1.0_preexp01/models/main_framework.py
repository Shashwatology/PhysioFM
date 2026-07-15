import torch
import torch.nn as nn

# Assuming the sub-modules exist in the PYTHONPATH
from models.extractors.swin_rgb import SwinRGBExtractor
from models.extractors.swin_thermal import SwinThermalExtractor
from models.extractors.timesformer_motion import TimeSformerMotionExtractor
from models.fusion.cross_attention import CrossAttentionFusion
from models.reasoning.clinical_engine import ClinicalReasoningEngine

class MultimodalPhysiologicalFramework(nn.Module):
    """
    The orchestrator tying the entire Multimodal Physiological Reasoning Framework together.
    Note: Detection and Segmentation are typically handled in a pre-processing pipeline
    to save VRAM, so this framework expects cropped and masked ROI inputs.
    """
    def __init__(self, embed_dim=512):
        super(MultimodalPhysiologicalFramework, self).__init__()
        
        # 1. Unimodal Extractors (Pre-trained and optionally frozen via StageRunner)
        self.rgb_extractor = SwinRGBExtractor(pretrained=True, embed_dim=embed_dim)
        self.thm_extractor = SwinThermalExtractor(pretrained=True, in_channels=1, embed_dim=embed_dim)
        self.mot_extractor = TimeSformerMotionExtractor(pretrained=True, seq_len=8, embed_dim=embed_dim)
        
        # 2. Multimodal Fusion
        self.fusion_layer = CrossAttentionFusion(embed_dim=embed_dim)
        
        # 3. Clinical Reasoning
        self.clinical_engine = ClinicalReasoningEngine(embed_dim=embed_dim, num_risk_classes=1)

    def forward(self, rgb_video, thm_video, full_body_video):
        """
        Args:
            rgb_video (Tensor): Cropped Face/Chest RGB video (B, T, C, H, W)
            thm_video (Tensor): Cropped Face/Chest Thermal video (B, T, 1, H, W)
            full_body_video (Tensor): Full scene video for motion tracking (B, T, C, H, W)
        Returns:
            Tensor: Final risk prediction
            dict: Intermediate states for explainability
        """
        # Collapse batch and time dimensions for 2D CNN backbones
        B, T, C_rgb, H, W = rgb_video.shape
        _, _, C_thm, _, _ = thm_video.shape
        
        rgb_flat = rgb_video.view(B*T, C_rgb, H, W)
        thm_flat = thm_video.view(B*T, C_thm, H, W)
        
        # Extract Spatial Features
        rgb_feats = self.rgb_extractor(rgb_flat) # (B*T, Dim)
        thm_feats = self.thm_extractor(thm_flat) # (B*T, Dim)
        
        # Unflatten to sequence
        rgb_seq = rgb_feats.view(B, T, -1)
        thm_seq = thm_feats.view(B, T, -1)
        
        # Extract Spatiotemporal Motion Features (TimeSformer handles 5D input natively)
        mot_feats = self.mot_extractor(full_body_video) # (B, Dim)
        
        # Fuse RGB and Thermal to create the physiological representation
        # (This combines heart rate and respiratory data)
        vital_rep = self.fusion_layer(rgb_seq, thm_seq) # (B, Dim)
        
        # Combine with overall body motion (posture/agitation)
        # For simplicity, we add them, but this could be another attention layer
        physio_rep = vital_rep + mot_feats
        
        # Pass through the Clinical Reasoning Engine
        risk_score, xai_states = self.clinical_engine(physio_rep)
        
        return risk_score, xai_states
