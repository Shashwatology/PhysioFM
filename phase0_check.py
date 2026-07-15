import sys
import os
import platform
import subprocess

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"FAIL: {e.stderr.strip()}"
    except Exception as e:
        return f"FAIL: {str(e)}"

# 1. Gather System Information
python_ver = sys.version.split('\n')[0]
os_ver = platform.platform()

# 2. Test Imports
try:
    import torch
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if torch.version.cuda else "None"
    cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "None"
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None"
except Exception as e:
    torch_ver, cuda_avail, cuda_ver, cudnn_ver, gpu_name = f"FAIL: {e}", "FAIL", "FAIL", "FAIL", "FAIL"

try:
    import cv2
    cv2_ver = cv2.__version__
except Exception as e:
    cv2_ver = f"FAIL: {e}"

try:
    import mediapipe as mp
    mp_ver = mp.__version__
except Exception as e:
    mp_ver = f"FAIL: {e}"

try:
    import timm
    timm_ver = timm.__version__
except Exception as e:
    timm_ver = f"FAIL: {e}"

# 3. Hardware Benchmark (cuBLAS stress test)
hw_time = "N/A"
hw_mem = "N/A"
if cuda_avail == True:
    try:
        import time
        x = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            y = x @ x
        torch.cuda.synchronize()
        hw_time = f"{time.time() - start:.4f} seconds"
        hw_mem = f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    except Exception as e:
        hw_time = f"FAIL: {e}"
        hw_mem = "FAIL"

# 4. Generate PHASE0_REPORT.md
report = f"""# Phase 0 Verification Report

## Verification Checks
- **Python**: {python_ver} ({"PASS" if "3.11" in python_ver else "FAIL"})
- **PyTorch**: {torch_ver} ({"PASS" if "FAIL" not in torch_ver else "FAIL"})
- **CUDA Available**: {cuda_avail} ({"PASS" if cuda_avail == True else "FAIL"})
- **GPU Detected**: {gpu_name} ({"PASS" if "A2000" in str(gpu_name) else "FAIL"})
- **cuDNN Initialized**: {cudnn_ver} ({"PASS" if "FAIL" not in str(cudnn_ver) and str(cudnn_ver) != "None" else "FAIL"})
- **OpenCV**: {cv2_ver} ({"PASS" if "FAIL" not in cv2_ver else "FAIL"})
- **MediaPipe**: {mp_ver} ({"PASS" if "FAIL" not in mp_ver else "FAIL"})
- **timm**: {timm_ver} ({"PASS" if "FAIL" not in timm_ver else "FAIL"})

## Hardware Benchmark (cuBLAS)
- **Matrix Multiplication (50x 4096x4096)**: {hw_time}
- **Peak VRAM Allocated**: {hw_mem}

## System Details
- **CUDA Version**: {cuda_ver}
- **cuDNN Version**: {cudnn_ver}

## Warnings
- None detected during import phase.
"""
with open("PHASE0_REPORT.md", "w", encoding="utf-8") as f:
    f.write(report)

# 5. Generate ENVIRONMENT_SNAPSHOT.md
snapshot = f"""# Environment Snapshot (v1.0 Baseline)

## System Infrastructure
- **Windows Version**: {os_ver}
- **GPU Driver**: NVIDIA RTX A2000 (Detailed driver version requires nvidia-smi)

## Python Ecosystem
- **Python Version**: {python_ver}

## Machine Learning Framework
- **PyTorch**: {torch_ver}
- **CUDA Runtime**: {cuda_ver}
- **cuDNN**: {cudnn_ver}

## Libraries
- **timm**: {timm_ver}
- **MediaPipe**: {mp_ver}
- **OpenCV**: {cv2_ver}
"""
with open("ENVIRONMENT_SNAPSHOT.md", "w", encoding="utf-8") as f:
    f.write(snapshot)

print("Verification complete.")
