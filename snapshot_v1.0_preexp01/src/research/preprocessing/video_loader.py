import cv2
import torch
import numpy as np

class VideoDecoder:
    """
    Decodes video files into PyTorch Tensors.
    """
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps

    def extract_frames(self, video_path: str, max_frames: int = None) -> np.ndarray:
        """
        Reads a video file and returns an array of frames.
        Args:
            video_path: Absolute path to the video file.
            max_frames: Optional hard limit on frames to extract.
        Returns:
            frames: (T, H, W, 3) BGR numpy array.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
            
        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frames.append(frame)
            count += 1
            if max_frames and count >= max_frames:
                break
                
        cap.release()
        return np.array(frames)
