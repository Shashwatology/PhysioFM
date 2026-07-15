import os
import sys
import torch
import platform
import subprocess
import yaml
import time
from typing import Dict, Any

def get_git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except Exception:
        return "Not available (Not a git repository or git not installed)"

def generate_config_hash(exp_folder: str, config: Dict[str, Any]):
    import hashlib
    import json
    
    hasher = hashlib.sha256()
    
    # Hash the config itself
    config_str = json.dumps(config, sort_keys=True)
    hasher.update(config_str.encode('utf-8'))
    
    # Hash core files
    core_files = [
        "src/research/training/train.py",
        "src/research/preprocessing/ubfc_dataset.py",
        "src/research/models/physio_fm.py"
    ]
    
    for fpath in core_files:
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                hasher.update(f.read())
                
    hash_hex = hasher.hexdigest()
    hash_file = os.path.join(exp_folder, "experiment_config_hash.txt")
    with open(hash_file, "w") as f:
        f.write(f"SHA256: {hash_hex}\\nFiles Hashed: config.yaml, {', '.join(core_files)}\\n")
        
    return hash_hex

def generate_reproducibility_report(exp_folder: str, config: Dict[str, Any], seed: int = 42):
    """
    Auto-generates reproducibility.md for a given experiment run.
    """
    os.makedirs(exp_folder, exist_ok=True)
    
    # Attempt to load timm version
    try:
        import timm
        timm_version = timm.__version__
    except ImportError:
        timm_version = "Not installed"
        
    gpu_info = "No GPU Available"
    if torch.cuda.is_available():
        gpu_info = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)"
        
    md_content = f"""# Reproducibility Report

## Experiment Details
- **Experiment Name**: {config.get('exp_name', 'Unknown')}
- **Run Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Random Seed**: {seed}

## Environment Specifications
- **OS**: {platform.system()} {platform.release()} ({platform.version()})
- **Python Version**: {sys.version.split(' ')[0]}
- **PyTorch Version**: {torch.__version__}
- **Torchvision Version**: {torch.version.cuda if torch.version.cuda else 'CPU Only'}
- **Timm Version**: {timm_version}
- **Git Commit**: {get_git_commit()}

## Hardware
- **CPU**: {platform.processor()}
- **GPU**: {gpu_info}
- **CUDA Version**: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}

## Configuration
```yaml
{yaml.dump(config, default_flow_style=False)}
```

> **Note**: This report was auto-generated to strictly adhere to Phase 5 reproducibility standards. 
"""

    report_path = os.path.join(exp_folder, "reproducibility.md")
    with open(report_path, "w") as f:
        f.write(md_content)
        
    hash_val = generate_config_hash(exp_folder, config)
    print(f"Reproducibility report generated at {report_path}")
    print(f"Experiment Config Hash: {hash_val}")

def set_seed(seed: int = 42):
    """
    Locks random seeds for strict reproducibility.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
