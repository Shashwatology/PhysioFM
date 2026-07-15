import cv2
import mediapipe as mp
import numpy as np
import torch

class VideoPreprocessor:
    """
    Handles preprocessing of raw video feeds into the format expected by the 
    Multimodal Physiological Framework (Cropped Tensors).
    """
    def __init__(self, target_size=(256, 256), sequence_length=8):
        self.target_size = target_size
        self.sequence_length = sequence_length
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, # 0 for short-range (webcam)
            min_detection_confidence=0.5
        )

    def extract_face_sequence(self, video_path_or_cam=0):
        """
        Reads a video stream, detects the face, crops it, and returns a 
        PyTorch tensor sequence ready for the SwinRGBExtractor.
        """
        cap = cv2.VideoCapture(video_path_or_cam)
        if not cap.isOpened():
            print("Warning: Cannot open webcam/video. Falling back to generating a dummy face sequence for testing...")
            return self._generate_dummy_sequence()

        frames = []
        
        while len(frames) < self.sequence_length:
            success, image = cap.read()
            if not success:
                break
                
            # MediaPipe expects RGB images
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image_rgb)
            
            if results.detections:
                # Take the first detected face
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                
                # Convert relative bounding box to absolute pixel coordinates
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                             
                # Add some padding around the face (for rPPG context like cheeks/forehead)
                padding = int(max(w, h) * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(iw, x + w + padding)
                y2 = min(ih, y + h + padding)
                
                face_crop = image_rgb[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # Resize to the exact input size expected by Swin (256x256)
                    face_resized = cv2.resize(face_crop, self.target_size)
                    frames.append(face_resized)
            else:
                # If no face is found, we can duplicate the last frame or skip
                if frames:
                    frames.append(frames[-1])
                    
        cap.release()
        
        if len(frames) < self.sequence_length:
            print("Warning: Not enough frames extracted. Falling back to dummy sequence...")
            return self._generate_dummy_sequence()
            
        # Convert list of arrays (H, W, C) to a single numpy array (T, H, W, C)
        frames_np = np.array(frames, dtype=np.float32)
        
        # Normalize pixel values to [0, 1]
        frames_np = frames_np / 255.0
        
        # Convert to PyTorch Tensor and rearrange dims to (T, C, H, W)
        tensor_sequence = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
        
        # Add Batch dimension (B, T, C, H, W)
        batch_tensor = tensor_sequence.unsqueeze(0)
        
        return batch_tensor

    def _generate_dummy_sequence(self):
        """Generates a dummy tensor shaped (1, T, 3, H, W) for testing."""
        T = self.sequence_length
        H, W = self.target_size
        return torch.randn(1, T, 3, H, W)

if __name__ == "__main__":
    print("Initializing Video Preprocessor...")
    preprocessor = VideoPreprocessor(sequence_length=8)
    
    print("Attempting to capture 8 frames from the default webcam (0)...")
    # Change to a video file path if you don't have a webcam connected
    tensor_batch = preprocessor.extract_face_sequence(0)
    
    if tensor_batch is not None:
        print(f"Success! Extracted Tensor Shape: {tensor_batch.shape}")
        print("This tensor is now ready to be fed into the SwinRGBExtractor.")
