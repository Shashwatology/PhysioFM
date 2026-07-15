import torch
import sys
import os

# Add the ml_pipeline directory to the path so we can import from models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.main_framework import MultimodalPhysiologicalFramework

def run_test():
    print("Initializing the Multimodal Physiological Framework...")
    
    try:
        # Disable pretrained weights for a fast local test without downloading massive checkpoints
        model = MultimodalPhysiologicalFramework(embed_dim=512)
        
        # We need to monkey-patch the initialization for the test to avoid downloading if they were hardcoded to True
        # Since they are hardcoded to True in main_framework.py (self.rgb_extractor = SwinRGBExtractor(pretrained=True...)), 
        # this might trigger a download. We will let it run.
        
        print("Model initialized successfully.")
        
        # Batch Size 2, Sequence Length 10 (frames)
        B, T = 1, 4
        print(f"Generating dummy inputs for Batch Size: {B}, Sequence Length: {T}...")
        dummy_rgb = torch.randn(B, T, 3, 256, 256)
        dummy_thm = torch.randn(B, T, 1, 256, 256)
        dummy_body = torch.randn(B, 8, 3, 224, 224) 
        
        print("Executing forward pass through the unified pipeline...")
        
        # Note: Depending on the exact TimeSformer config, it might enforce seq_len=64 strictly during forward if not careful.
        # We initialized mot_extractor with seq_len=64.
        
        risk_score, xai_states = model(dummy_rgb, dummy_thm, dummy_body)
        
        print("\n--- Forward Pass Successful ---")
        print(f"Output Risk Score Shape: {risk_score.shape} (Expected: {B}, 1)")
        print(f"Output Cardio State Shape: {xai_states['cardio_state'].shape} (Expected: {B}, 256)")
        print(f"Output Activity State Shape: {xai_states['activity_state'].shape} (Expected: {B}, 256)")
        
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    run_test()
