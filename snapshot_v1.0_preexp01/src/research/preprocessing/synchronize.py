import numpy as np
from scipy.interpolate import interp1d

class SignalSynchronizer:
    """
    Synchronizes signals from different modalities (e.g. BVP, Respiration, Video) 
    that may have different sampling rates.
    """
    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps
        
    def synchronize(self, signal: np.ndarray, original_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
        """
        Interpolates a 1D signal to align with a new set of target timestamps.
        Args:
            signal: (N,) numpy array containing the raw signal.
            original_timestamps: (N,) numpy array of timestamps for the raw signal in seconds.
            target_timestamps: (M,) numpy array of desired timestamps in seconds.
        Returns:
            synced_signal: (M,) numpy array interpolated to the target timestamps.
        """
        # Create an interpolation function
        # Using cubic interpolation for smooth physiological signals if possible, else linear
        kind = 'cubic' if len(signal) > 3 else 'linear'
        
        # Ensure timestamps are strictly increasing
        sort_idx = np.argsort(original_timestamps)
        original_timestamps = original_timestamps[sort_idx]
        signal = signal[sort_idx]
        
        # Interpolate
        interpolator = interp1d(original_timestamps, signal, kind=kind, bounds_error=False, fill_value="extrapolate")
        synced_signal = interpolator(target_timestamps)
        
        return synced_signal

    def normalize(self, signal: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalizes a physiological signal.
        Args:
            signal: (N,) numpy array
            method: 'zscore' or 'minmax'
        Returns:
            normalized_signal: (N,) numpy array
        """
        if method == 'zscore':
            mean = np.mean(signal)
            std = np.std(signal)
            if std == 0:
                return np.zeros_like(signal)
            return (signal - mean) / std
            
        elif method == 'minmax':
            min_val = np.min(signal)
            max_val = np.max(signal)
            if max_val - min_val == 0:
                return np.zeros_like(signal)
            return (signal - min_val) / (max_val - min_val)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
