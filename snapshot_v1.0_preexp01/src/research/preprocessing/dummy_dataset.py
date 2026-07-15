import torch
import numpy as np
from typing import Dict, Any
from .base_dataset import BasePhysiologicalDataset

class DummySyntheticDataset(BasePhysiologicalDataset):
    """
    A synthetic dataset matching the expected tensor shapes of real datasets.
    Used STRICTLY for pipeline validation and debugging, NOT for training models.
    """
    
    def __init__(self, root_dir: str = "", split: str = 'train', transform=None, 
                 num_samples: int = 10, frames: int = 300, h: int = 64, w: int = 64):
        self.num_samples = num_samples
        self.frames = frames
        self.h = h
        self.w = w
        super().__init__(root_dir, split, transform)

    def _build_index(self):
        """Populates dummy sample indices."""
        for i in range(self.num_samples):
            self.samples.append({
                'video': f'dummy_video_{i}.avi',
                'id': i
            })

    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """
        Generates a random tensor representing extracted ROI frames.
        Shape: (T, C, H, W) -> (300, 3, 64, 64)
        """
        # Simulating normalized RGB frames [-1, 1]
        return torch.randn(self.frames, 3, self.h, self.w)

    def _load_ground_truth(self, sample_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Generates synthetic Blood Volume Pulse (BVP), Heart Rate (HR), and Resp Rate (RR).
        """
        t = np.linspace(0, 10, self.frames) # 10 seconds
        
        # Synthetic BVP: Sine wave at ~1.2 Hz (72 BPM) with some noise
        bvp = np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.1, self.frames)
        
        # Static ground truth values for the window
        hr = torch.tensor([72.0], dtype=torch.float32)
        rr = torch.tensor([16.0], dtype=torch.float32)
        
        return {
            'bvp': torch.tensor(bvp, dtype=torch.float32),
            'hr': hr,
            'rr': rr
        }

if __name__ == "__main__":
    # Quick test
    ds = DummySyntheticDataset(num_samples=2)
    video, targets = ds[0]
    print(f"Video Shape: {video.shape}")
    print(f"BVP Shape: {targets['bvp'].shape}")
    print(f"HR: {targets['hr'].item()}")
