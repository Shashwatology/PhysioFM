import torch
import torch.nn as nn

class BaseTaskHead(nn.Module):
    """
    Abstract Base Class for independent downstream task heads.
    """
    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Central physiological embedding (B, D)
        Returns:
            torch.Tensor: Regressed values (B, output_dim)
        """
        return self.net(x)

class HeartRateHead(BaseTaskHead):
    """
    Predicts the mean Heart Rate (BPM) from the latent embedding.
    """
    def __init__(self, embed_dim: int):
        super().__init__(embed_dim, output_dim=1)

class RespirationHead(BaseTaskHead):
    """
    Predicts the mean Respiration Rate (RPM) from the latent embedding.
    Placeholder until V4V data arrives.
    """
    def __init__(self, embed_dim: int):
        super().__init__(embed_dim, output_dim=1)

class WaveformHead(nn.Module):
    """
    Reconstructs the continuous BVP waveform.
    Outputs sequence of length T.
    """
    def __init__(self, embed_dim: int, seq_len: int = 150):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1) # Outputs a 1D trace over the time window per frame
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Central physiological embedding (B, D)
        Returns:
            torch.Tensor: BVP Waveform (B, T)
        """
        return self.net(x)
