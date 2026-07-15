import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.extractors.swin_rgb import SwinRGBExtractor
from models.extractors.timesformer_motion import TimeSformerMotionExtractor

def test_vision():
    print("Testing Vision Extractors...")
    
    print("1. Testing SwinRGBExtractor (Face/Chest rPPG)...")
    try:
        # pretrained=False to avoid massive downloads
        rgb_model = SwinRGBExtractor(pretrained=False, embed_dim=512)
        dummy_rgb = torch.randn(2, 3, 256, 256)
        out_rgb = rgb_model(dummy_rgb)
        print(f"   [SUCCESS] SwinRGB Output shape: {out_rgb.shape} (Expected: 2, 512)")
    except Exception as e:
        print(f"   [FAILED] SwinRGB error: {e}")

    print("\n2. Testing TimeSformerMotionExtractor (Full body posture)...")
    try:
        mot_model = TimeSformerMotionExtractor(pretrained=False, seq_len=8, embed_dim=512)
        dummy_video = torch.randn(1, 8, 3, 224, 224)
        out_mot = mot_model(dummy_video)
        print(f"   [SUCCESS] TimeSformer Output shape: {out_mot.shape} (Expected: 1, 512)")
    except Exception as e:
        print(f"   [FAILED] TimeSformer error: {e}")

if __name__ == "__main__":
    test_vision()
