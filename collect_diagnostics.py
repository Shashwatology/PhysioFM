import subprocess
import platform
import struct
import json
import os

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"FAILED: {e.stderr.strip()}"
    except Exception as e:
        return f"FAILED: {str(e)}"

# Environment details
plat = platform.platform()
arch = struct.calcsize('P') * 8

# Check MSVC Redistributable presence
vc_dlls = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]
vc_status = []
sys_paths = [r"C:\Windows\System32"]
for dll in vc_dlls:
    found = False
    for p in sys_paths:
        if os.path.exists(os.path.join(p, dll)):
            found = True
            break
    vc_status.append(f"{dll}: {'Found' if found else 'Missing'}")

vc_report = ", ".join(vc_status)

# Commands
nvidia_smi = run_cmd(["nvidia-smi"])
nvcc_ver = run_cmd(["nvcc", "--version"])
pip_list = run_cmd([r".\.venv\Scripts\python.exe", "-m", "pip", "list"])
pip_debug = run_cmd([r".\.venv\Scripts\python.exe", "-m", "pip", "debug"])
where_python = run_cmd(["where", "python"])
where_pip = run_cmd(["where", "pip"])
pip_ver = run_cmd([r".\.venv\Scripts\python.exe", "-m", "pip", "--version"])
py_ver = run_cmd([r".\.venv\Scripts\python.exe", "--version"])

# Extract CUDA/Driver versions from nvidia-smi if available
driver_ver = "Unknown"
cuda_ver = "Unknown"
gpu_name = "Unknown"
for line in nvidia_smi.splitlines():
    if "Driver Version" in line and "CUDA Version" in line:
        parts = line.split()
        for i, p in enumerate(parts):
            if p == "Version:":
                if driver_ver == "Unknown":
                    driver_ver = parts[i+1]
                else:
                    cuda_ver = parts[i+1]
    if "NVIDIA" in line and "Default" in line:
        gpu_name = line.split("|")[1].strip()
        
report = f"""# PyTorch Environment Diagnostic (V2)

## 1. System Information
- **OS Platform:** {plat}
- **Architecture:** {arch}-bit
- **Python Version:** {py_ver}
- **Pip Version:** {pip_ver}
- **Where Python:** {where_python}
- **Where Pip:** {where_pip}

## 2. Hardware and Drivers
- **GPU Driver Version:** {driver_ver}
- **Supported CUDA Runtime (nvidia-smi):** {cuda_ver}
- **Detected GPU:** {gpu_name}

## 3. Toolchains & Dependencies
- **NVCC Version:**
```
{nvcc_ver}
```
- **Visual C++ Redistributable (System32):** {vc_report}

## 4. Environment State
- **Installed Packages (`pip list`):**
```
{pip_list}
```

## 5. Root Cause Analysis
The `WinError 1114` (DLL initialization failed) for `c10.dll` typically stems from either:
1. **Missing MSVC Runtimes** (which provide C++ standard library implementations required by PyTorch). We verified these are `{vc_report}`.
2. **Incorrect PyTorch Wheel**. The standard PyPI wheel on Windows is missing specific CUDA runtime DLLs (e.g. `cuDNN` or `cublas`) if the system does not have the CUDA Toolkit installed natively. Or, the user downloaded the CPU-only wheel which may have incompatibilities with the C++ runtime. 
The PyTorch `122MB` size indicates it likely installed a non-CUDA or missing-DLL bundle.

## 6. Official Recommended Installation Command
Based on the CUDA Driver Version ({cuda_ver}), the correct command to install PyTorch with bundled CUDA binaries (which removes the dependency on system-level CUDA toolkits) is:

```bash
# Example for CUDA 12.1 (adjust based on detected CUDA version)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
*(Wait for review before running)*
"""

with open("PYTORCH_DIAGNOSTIC_V2.md", "w", encoding="utf-8") as f:
    f.write(report)
print("Diagnostic generated.")
