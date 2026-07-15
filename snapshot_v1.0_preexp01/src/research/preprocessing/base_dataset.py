import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
from torch.utils.data import Dataset

class BasePhysiologicalDataset(Dataset, ABC):
    """
    Abstract base class for all physiological monitoring datasets.
    Enforces a unified pipeline: load raw -> preprocess (ROI) -> normalize -> return tensor.
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Args:
            root_dir (str): Path to the processed dataset directory.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Must be populated by the child class in _build_index()
        self.samples = [] 
        self._build_index()
        
    @abstractmethod
    def _build_index(self):
        """
        Parses the dataset directory and populates self.samples with dictionaries
        containing paths to video and ground truth data for the specific split.
        Example: self.samples.append({'video': path, 'hr': path, 'rr': path})
        """
        pass

    @abstractmethod
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """
        Loads and decodes a video into a PyTorch Tensor of shape (T, C, H, W).
        """
        pass

    @abstractmethod
    def _load_ground_truth(self, sample_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Loads the ground truth physiological signals (e.g., HR, RR, BVP).
        Returns a dictionary of 1D Tensors.
        """
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            video_tensor: (T, C, H, W)
            target_dict: Dict containing 'bvp', 'hr', 'rr' etc.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_info = self.samples[idx]
        
        # 1. Load Data
        video_tensor = self._load_video_frames(sample_info['video'])
        targets = self._load_ground_truth(sample_info)
        
        # 2. Apply Transforms (Augmentation / Normalization)
        if self.transform:
            video_tensor = self.transform(video_tensor)
            
        return video_tensor, targets
