import sys
print(f'Python version: {sys.version}')
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA version (PyTorch): {torch.version.cuda}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU Name: {torch.cuda.get_device_name(0)}')
        print(f'Device Count: {torch.cuda.device_count()}')
except ImportError as e:
    print(f'PyTorch ImportError: {e}')

try:
    import cv2
    print(f'OpenCV version: {cv2.__version__}')
except ImportError as e:
    print(f'OpenCV ImportError: {e}')

try:
    import mediapipe as mp
    print(f'MediaPipe version: {mp.__version__}')
except ImportError as e:
    print(f'MediaPipe ImportError: {e}')

try:
    import timm
    print(f'timm version: {timm.__version__}')
except ImportError as e:
    print(f'timm ImportError: {e}')

