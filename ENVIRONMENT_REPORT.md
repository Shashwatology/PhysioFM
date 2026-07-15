# Environment Report

## Python & Hardware
- **Python Version:** 3.13.14
- **Virtual Environment Location:** `f:\INTERN DATA\Shashwat\PhysioFM_Codebase\.venv`
- **CUDA Version:** N/A (Blocked by PyTorch failure)
- **GPU:** NVIDIA RTX A2000 12 GB
- **VRAM:** 12 GB

## Package Verification
- **PyTorch:** `FAIL` - PyPI wheel for Python 3.13 crashes.
- **timm:** `FAIL` - Cascading failure due to PyTorch.
- **MediaPipe:** `PASS`
- **OpenCV:** `PASS`
- **NumPy & Pandas:** `PASS`

## Datasets
**UBFC-rPPG**
- **Location:** `C:\Users\MNIT USER\Downloads\UBFC-rPPG Dataset`
- **Subjects:** 42 
- **Status:** PASS

**PURE**
- **Status:** UNREGISTERED (Incorrect NLP dataset was downloaded).

## Exact Blocker & Recommended Fix
| Issue | Details | Exact Recommended Fix |
|---|---|---|
| **PyTorch WinError 1114** | `c10.dll` fails to initialize. We successfully installed the Microsoft Visual C++ 2015-2022 Redistributable and performed a clean re-installation of PyTorch, but the error persists. This confirms the experimental PyPI `torch-2.13.0-cp313` wheel is broken on Windows. | **Downgrade to Python 3.11.** Python 3.13 is too new for stable PyTorch CUDA support on Windows. |
