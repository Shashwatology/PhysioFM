import os
import numpy as np
import torch
from typing import Dict, Any

from src.research.preprocessing.base_dataset import BasePhysiologicalDataset
from src.research.preprocessing.video_loader import VideoDecoder
from src.research.preprocessing.roi import ROIExtractor
from src.research.preprocessing.synchronize import SignalSynchronizer
from src.research.preprocessing.augment import VideoAugmentationPipeline

class UBFCDataset(BasePhysiologicalDataset):
    """
    UBFC-rPPG Dataset implementation.
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None, 
                 target_fps: int = 30, target_size: tuple = (64, 64)):
        self.target_fps = target_fps
        self.target_size = target_size
        
        # Initialize preprocessing modules
        self.video_decoder = VideoDecoder(target_fps=target_fps)
        self.roi_extractor = ROIExtractor(target_size=target_size)
        self.synchronizer = SignalSynchronizer(target_fps=target_fps)
        
        # Default augmentation pipeline if none provided
        if transform is None:
            self.transform = VideoAugmentationPipeline(target_size=target_size, mode=split)
        else:
            self.transform = transform
            
        super().__init__(root_dir, split, self.transform)

    def _build_index(self):
        """
        Parses the ubfc_inventory.csv to build the sample list.
        """
        import csv
        meta_path = os.path.join(self.root_dir, '..', 'metadata', 'ubfc_inventory.csv')
        if not os.path.exists(meta_path):
            # Fallback for dynamic indexing if manager hasn't run
            self._fallback_indexing()
            return
            
        valid_subjects = []
        with open(meta_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Valid'] == 'True':
                    valid_subjects.append(row['Subject_ID'])
        
        # Simple train/val/test split based on Subject IDs
        subjects = sorted(valid_subjects)
        n = len(subjects)
        
        if self.split == 'train':
            split_subjects = subjects[:int(0.7*n)]
        elif self.split == 'val':
            split_subjects = subjects[int(0.7*n):int(0.85*n)]
        else:
            split_subjects = subjects[int(0.85*n):]
            
        for subj in split_subjects:
            subj_dir = os.path.join(self.root_dir, subj)
            self.samples.append({
                'video': os.path.join(subj_dir, 'vid.avi'),
                'gt': os.path.join(subj_dir, 'ground_truth.txt'),
                'id': subj
            })
            
    def _fallback_indexing(self):
        import glob
        subjects = glob.glob(os.path.join(self.root_dir, 'subject*'))
        for subj_dir in subjects:
            self.samples.append({
                'video': os.path.join(subj_dir, 'vid.avi'),
                'gt': os.path.join(subj_dir, 'ground_truth.txt'),
                'id': os.path.basename(subj_dir)
            })

    def _load_video_frames(self, video_path: str, start_idx: int = 0, seq_len: int = 150) -> torch.Tensor:
        """
        Loads the video, extracts ROIs for only a specific window of frames.
        """
        # Load raw BGR frames up to the required sequence length
        end_idx = start_idx + seq_len
        raw_frames = self.video_decoder.extract_frames(video_path, max_frames=end_idx)
        
        # Truncate to save massive CPU time on ROI extraction
        raw_frames = raw_frames[start_idx:end_idx]
        
        # If video is shorter than seq_len, pad by repeating the last frame
        if len(raw_frames) < seq_len:
            pad_len = seq_len - len(raw_frames)
            last_frame = raw_frames[-1:]
            raw_frames = np.concatenate([raw_frames, np.repeat(last_frame, pad_len, axis=0)], axis=0)
        
        roi_frames = []
        for frame in raw_frames:
            # Extract ROI using MediaPipe (or fallback crop)
            roi = self.roi_extractor.extract_face(frame)
            roi_frames.append(roi)
            
        roi_tensor = torch.tensor(np.array(roi_frames), dtype=torch.float32) 
        # Output shape: (T, H, W, 3)
        return roi_tensor

    def _load_ground_truth(self, sample_info: Dict[str, Any], start_idx: int = 0, seq_len: int = 150) -> Dict[str, torch.Tensor]:
        """
        Loads the ground truth BVP and HR from the txt file.
        """
        gt_data = np.loadtxt(sample_info['gt'])
        bvp = gt_data[0, :]
        hr = gt_data[1, :]
        timestamps = gt_data[2, :]
        
        # Synchronize traces to target FPS just in case
        target_timestamps = np.arange(timestamps[0], timestamps[-1], 1.0 / self.target_fps)
        bvp_synced = self.synchronizer.synchronize(bvp, timestamps, target_timestamps)
        hr_synced = self.synchronizer.synchronize(hr, timestamps, target_timestamps)
        
        # Truncate to window
        end_idx = min(start_idx + seq_len, len(bvp_synced))
        bvp_window = bvp_synced[start_idx:end_idx]
        hr_window = hr_synced[start_idx:end_idx]
        
        # Pad if necessary
        if len(bvp_window) < seq_len:
            pad_len = seq_len - len(bvp_window)
            bvp_window = np.pad(bvp_window, (0, pad_len), 'edge')
            hr_window = np.pad(hr_window, (0, pad_len), 'edge')
            
        # Normalize BVP using zscore
        bvp_norm = self.synchronizer.normalize(bvp_window, method='zscore')
        
        # The target for a window is often the mean HR of that window
        mean_hr = np.mean(hr_window)
        
        return {
            'bvp': torch.tensor(bvp_norm, dtype=torch.float32),
            'hr': torch.tensor([mean_hr], dtype=torch.float32),
            'hr_trace': torch.tensor(hr_window, dtype=torch.float32),
            'id': int(sample_info['id'].replace('subject', '')) if 'subject' in sample_info['id'] else 0
        }
