import torch
import torchvision.transforms as T
import random

class VideoAugmentationPipeline:
    """
    Data augmentation for spatiotemporal physiological datasets.
    Ensures spatial transformations are consistent across all frames in a window.
    """
    def __init__(self, target_size=(64, 64), mode='train'):
        self.mode = mode
        self.target_size = target_size
        
        # Spatial transforms (must be applied identically to all frames in T)
        self.spatial_transform = T.Compose([
            T.Resize(self.target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_tensor: (T, H, W, C) numpy array or tensor of RGB frames
        Returns:
            augmented: (T, C, H, W) normalized tensor
        """
        T_frames, H, W, C = video_tensor.shape
        
        # Apply standard torchvision transform to each frame
        # In a highly optimized pipeline, you'd batch this as (T, C, H, W)
        processed_frames = []
        for i in range(T_frames):
            frame = video_tensor[i]
            # Convert numpy to PIL or directly use tensor transforms
            # Assuming frame is already a tensor here for simplicity
            if not isinstance(frame, torch.Tensor):
                frame = torch.from_numpy(frame)
                
            if frame.shape[-1] == 3:
                frame = frame.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
                
            if frame.max() > 1.0:
                frame = frame.float() / 255.0
            
            # Apply normalizations
            frame = T.functional.resize(frame, self.target_size, antialias=True)
            frame = T.functional.normalize(frame, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            processed_frames.append(frame)
            
        stacked = torch.stack(processed_frames) # (T, C, H, W)
        
        if self.mode == 'train':
            # Temporal Augmentations
            stacked = self._apply_temporal_augmentations(stacked)
            
        return stacked
        
    def _apply_temporal_augmentations(self, video: torch.Tensor) -> torch.Tensor:
        """
        Applies temporal masking or reverse playback for augmentation.
        Args:
            video: (T, C, H, W)
        Returns:
            video: augmented (T, C, H, W)
        """
        # Example: Random time-reverse (data augmentation for rPPG)
        if random.random() > 0.5:
            video = torch.flip(video, dims=[0])
        return video
