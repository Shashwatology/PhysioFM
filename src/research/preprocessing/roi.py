import cv2
import numpy as np
try:
    import mediapipe as mp
except ImportError:
    mp = None

class ROIExtractor:
    """
    Extracts Region of Interest (ROI) from video frames.
    Currently utilizes MediaPipe Face Mesh as a rapid prototype baseline.
    Architected to be easily swappable with RT-DETR for robust clinical scenarios.
    """
    def __init__(self, target_size=(64, 64), expand_ratio=1.2):
        self.target_size = target_size
        self.expand_ratio = expand_ratio
        
        # Initialize MediaPipe Face Detection (Prototype)
        self.detector = None
        try:
            if mp is not None and hasattr(mp, 'solutions'):
                self.mp_face_detection = mp.solutions.face_detection
                self.detector = self.mp_face_detection.FaceDetection(
                    model_selection=1, # 1 is for far-range
                    min_detection_confidence=0.5
                )
        except Exception:
            pass

        
    def extract_face(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts and resizes the facial ROI from a single frame.
        Args:
            frame: (H, W, 3) BGR numpy array
        Returns:
            roi: (target_size[0], target_size[1], 3) RGB numpy array
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        results = None
        if self.detector is not None:
            try:
                results = self.detector.process(image_rgb)
            except:
                pass
        
        if results and getattr(results, 'detections', None):
            # Take the first face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Apply expansion ratio
            x_center = x_min + width // 2
            y_center = y_min + height // 2
            
            new_width = int(width * self.expand_ratio)
            new_height = int(height * self.expand_ratio)
            
            x1 = max(0, x_center - new_width // 2)
            y1 = max(0, y_center - new_height // 2)
            x2 = min(w, x_center + new_width // 2)
            y2 = min(h, y_center + new_height // 2)
            
            roi = image_rgb[y1:y2, x1:x2]
        else:
            # Fallback: return center crop if no face detected
            cy, cx = h // 2, w // 2
            roi = image_rgb[max(0, cy-50):min(h, cy+50), max(0, cx-50):min(w, cx+50)]
            
        # Resize to target expected tensor size
        # Check if roi is empty due to out of bounds
        if roi.size == 0:
            roi = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        else:
            roi = cv2.resize(roi, self.target_size, interpolation=cv2.INTER_AREA)
        return roi
