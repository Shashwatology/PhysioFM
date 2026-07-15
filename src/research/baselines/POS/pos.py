import torch
import numpy as np

class POSBaseline:
    """
    Plane-Orthogonal to Skin (POS) rPPG algorithm.
    Reference: Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). 
    Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering.
    """
    
    def __init__(self, fps: int = 30, window_size: int = 45):
        """
        Args:
            fps (int): Frames per second of the input video.
            window_size (int): Size of the sliding window for projection.
        """
        self.fps = fps
        self.window_size = window_size
        
    def extract_bvp(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: Tensor of shape (T, C, H, W) where C=3 (RGB)
        Returns:
            bvp: 1D Tensor of shape (T,) containing the estimated BVP signal.
        """
        # Ensure tensor is CPU numpy for fast mathematical processing
        if frames.is_cuda:
            frames = frames.cpu()
        
        # Spatial pooling: Calculate mean RGB values per frame
        # Shape: (T, 3)
        mean_rgb = frames.mean(dim=(2, 3)).numpy()
        
        T = mean_rgb.shape[0]
        bvp = np.zeros(T)
        
        # Sliding window projection
        for i in range(T - self.window_size + 1):
            window = mean_rgb[i : i + self.window_size] # (W, 3)
            
            # Temporal normalization
            mean_window = np.mean(window, axis=0)
            std_window = np.std(window, axis=0)
            
            # Avoid division by zero
            normalized_window = (window - mean_window) / (mean_window + 1e-6)
            
            # Projection based on POS algorithm
            # X = 3Rn - 2Gn
            # Y = 1.5Rn + Gn - 1.5Bn
            Rn, Gn, Bn = normalized_window[:, 0], normalized_window[:, 1], normalized_window[:, 2]
            X = 3 * Rn - 2 * Gn
            Y = 1.5 * Rn + Gn - 1.5 * Bn
            
            # Alpha tuning
            alpha = np.std(X) / (np.std(Y) + 1e-6)
            
            # Projected signal h
            h = X - alpha * Y
            
            # Add to the continuous BVP signal (overlap-add method simplified)
            # In a true overlap-add, we would window 'h' and add it. 
            # Here we take the final point or running mean.
            bvp[i : i + self.window_size] += (h - np.mean(h))
            
        return torch.tensor(bvp, dtype=torch.float32)
