# PyTorch Environment Diagnostic (V2)

## 1. System Information
- **OS Platform:** Windows-10-10.0.26100-SP0
- **Architecture:** 64-bit
- **Python Version:** Python 3.11.9
- **Pip Version:** pip 26.1.2 from F:\INTERN DATA\Shashwat\PhysioFM_Codebase\.venv\Lib\site-packages\pip (python 3.11)
- **Where Python:** C:\Users\MNIT USER\AppData\Local\Microsoft\WindowsApps\python.exe
- **Where Pip:** C:\Users\MNIT USER\AppData\Local\Programs\Python\Python379\Scripts\pip.exe
C:\Users\MNIT USER\AppData\Local\Microsoft\WindowsApps\pip.exe

## 2. Hardware and Drivers
- **GPU Driver Version:** 527.48
- **Supported CUDA Runtime (nvidia-smi):** 12.0
- **Detected GPU:** Unknown

## 3. Toolchains & Dependencies
- **NVCC Version:**
```
FAILED: [WinError 2] The system cannot find the file specified
```
- **Visual C++ Redistributable (System32):** msvcp140.dll: Found, vcruntime140.dll: Found, vcruntime140_1.dll: Found

## 4. Environment State
- **Installed Packages (`pip list`):**
```
Package               Version
--------------------- -----------
absl-py               2.5.0
anyio                 4.14.1
certifi               2026.6.17
cffi                  2.1.0
click                 8.4.2
colorama              0.4.6
contourpy             1.3.3
cycler                0.12.1
filelock              3.29.7
flatbuffers           25.12.19
fonttools             4.63.0
fsspec                2026.6.0
h11                   0.16.0
hf-xet                1.5.1
httpcore              1.0.9
httpx                 0.28.1
huggingface_hub       1.23.0
idna                  3.18
Jinja2                3.1.6
kiwisolver            1.5.0
MarkupSafe            3.0.3
matplotlib            3.11.0
mediapipe             0.10.35
mpmath                1.3.0
networkx              3.6.1
numpy                 2.4.6
opencv-contrib-python 5.0.0.93
opencv-python         5.0.0.93
packaging             26.2
pandas                3.0.3
pillow                12.3.0
pip                   26.1.2
pycparser             3.0
pyparsing             3.3.2
python-dateutil       2.9.0.post0
PyYAML                6.0.3
safetensors           0.8.0
scipy                 1.17.1
setuptools            83.0.0
six                   1.17.0
sounddevice           0.5.5
sympy                 1.14.0
timm                  1.0.27
torchvision           0.28.0
tqdm                  4.68.4
typing_extensions     4.16.0
tzdata                2026.2
wheel                 0.47.0
```

## 5. Root Cause Analysis
The `WinError 1114` (DLL initialization failed) for `c10.dll` typically stems from either:
1. **Missing MSVC Runtimes** (which provide C++ standard library implementations required by PyTorch). We verified these are `msvcp140.dll: Found, vcruntime140.dll: Found, vcruntime140_1.dll: Found`.
2. **Incorrect PyTorch Wheel**. The standard PyPI wheel on Windows is missing specific CUDA runtime DLLs (e.g. `cuDNN` or `cublas`) if the system does not have the CUDA Toolkit installed natively. Or, the user downloaded the CPU-only wheel which may have incompatibilities with the C++ runtime. 
The PyTorch `122MB` size indicates it likely installed a non-CUDA or missing-DLL bundle.

## 6. Official Recommended Installation Command
Based on the CUDA Driver Version (12.0), the correct command to install PyTorch with bundled CUDA binaries (which removes the dependency on system-level CUDA toolkits) is:

```bash
# Example for CUDA 12.1 (adjust based on detected CUDA version)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
*(Wait for review before running)*

## 7. Resolution Record
- **Driver Version:** 527.48
- **GPU Model:** NVIDIA RTX A2000
- **Chosen CUDA Runtime:** cu118 (CUDA 11.8)
- **Reasoning:** Driver 527.48 supports a maximum CUDA runtime of 12.0. The official PyTorch builds target `cu118`, `cu121`, or `cu124`. Because `cu121` and `cu124` require driver version >= 528.33, they are strictly incompatible with the current host driver. Therefore, `cu118` is the latest natively supported PyTorch CUDA bundle for this hardware environment.
