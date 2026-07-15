import unittest
import torch
import numpy as np

# Adjust imports according to the structure
from src.research.preprocessing.dummy_dataset import DummySyntheticDataset
from src.research.preprocessing.roi import ROIExtractor
from src.research.preprocessing.synchronize import SignalSynchronizer

class TestPreprocessingPipeline(unittest.TestCase):
    
    def test_dummy_dataset_shapes(self):
        """Tests that the dataset abstraction loads the expected tensor shapes."""
        dataset = DummySyntheticDataset(num_samples=2, frames=150, h=32, w=32)
        self.assertEqual(len(dataset), 2)
        
        video_tensor, targets = dataset[0]
        
        # Check video shape: (T, C, H, W)
        self.assertEqual(video_tensor.shape, (150, 3, 32, 32))
        
        # Check target dictionary keys
        self.assertIn('bvp', targets)
        self.assertIn('hr', targets)
        self.assertIn('rr', targets)
        
        # Check BVP shape matches time dimension
        self.assertEqual(targets['bvp'].shape[0], 150)
        
    def test_roi_extraction_fallback(self):
        """Tests ROI extractor fallback mechanism on random noise frames."""
        extractor = ROIExtractor(target_size=(64, 64))
        
        # Create a dummy RGB frame (H, W, C)
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Should fallback to center crop because it's random noise (no face)
        roi = extractor.extract_face(dummy_frame)
        
        # Output should match target size
        self.assertEqual(roi.shape, (64, 64, 3))
        
    def test_signal_synchronization(self):
        """Tests the interpolation logic of SignalSynchronizer."""
        synchronizer = SignalSynchronizer(target_fps=30.0)
        
        # Dummy signal sampled at 10Hz for 1 second (10 points)
        original_timestamps = np.linspace(0, 1, 10)
        signal = np.sin(2 * np.pi * original_timestamps)
        
        # We want to resample it to 30Hz for 1 second (30 points)
        target_timestamps = np.linspace(0, 1, 30)
        
        synced_signal = synchronizer.synchronize(signal, original_timestamps, target_timestamps)
        
        self.assertEqual(synced_signal.shape[0], 30)

if __name__ == '__main__':
    unittest.main()
