import sys
import os

print('--- Python Environment ---')
print('Python version:', sys.version)

print('\n--- Package Verification ---')
try:
    import torch
    print('PyTorch installed successfully.')
    print('torch.__version__:', torch.__version__)
    print('torch.version.cuda:', torch.version.cuda)
    is_avail = torch.cuda.is_available()
    print('torch.cuda.is_available():', is_avail)
    if is_avail:
        print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))
except Exception as e:
    print(f'PyTorch Exception: {type(e).__name__}: {e}')

try:
    import cv2
    print('OpenCV imported successfully.')
except Exception as e:
    print(f'OpenCV Exception: {type(e).__name__}: {e}')

try:
    import mediapipe as mp
    print('MediaPipe imported successfully.')
except Exception as e:
    print(f'MediaPipe Exception: {type(e).__name__}: {e}')

try:
    import timm
    print('timm imported successfully.')
except Exception as e:
    print(f'timm Exception: {type(e).__name__}: {e}')

try:
    import numpy
    print('numpy imported successfully.')
except Exception as e:
    print(f'numpy Exception: {type(e).__name__}: {e}')

try:
    import pandas
    print('pandas imported successfully.')
except Exception as e:
    print(f'pandas Exception: {type(e).__name__}: {e}')

try:
    import matplotlib
    print('matplotlib imported successfully.')
except Exception as e:
    print(f'matplotlib Exception: {type(e).__name__}: {e}')

