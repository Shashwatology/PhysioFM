# Installation Log

## Phase 0 - Infrastructure Package Upgrade
**Command:** `.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel`

**Output:**
```
Requirement already satisfied: pip in f:\intern data\shashwat\physiofm_codebase\.venv\lib\site-packages (24.0)
Collecting pip
  Using cached pip-26.1.2-py3-none-any.whl.metadata (4.6 kB)
Requirement already satisfied: setuptools in f:\intern data\shashwat\physiofm_codebase\.venv\lib\site-packages (83.0.0)
Collecting wheel
  Using cached wheel-0.47.0-py3-none-any.whl.metadata (2.3 kB)
Requirement already satisfied: packaging>=24.0 in f:\intern data\shashwat\physiofm_codebase\.venv\lib\site-packages (from wheel) (26.2)
Using cached pip-26.1.2-py3-none-any.whl (1.8 MB)
Using cached wheel-0.47.0-py3-none-any.whl (32 kB)
Installing collected packages: wheel, pip
  Attempting uninstall: pip
    Found existing installation: pip 24.0
    Uninstalling pip-24.0:
      Successfully uninstalled pip-24.0
Successfully installed pip wheel
```

## Phase 0 - Requirements Installation
**Command:** `.\.venv\Scripts\python.exe -m pip --version` and `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

*(Currently running)*

## Phase 0 - Requirements Installation (Continued)

```
pip 26.1.2 from F:\INTERN DATA\Shashwat\PhysioFM_Codebase\.venv\Lib\site-packages\pip (python 3.11)

Collecting torch>=2.0.0 (from -r requirements.txt (line 1))
  Using cached torch-2.13.0-cp311-cp311-win_amd64.whl.metadata (39 kB)
Collecting torchvision>=0.15.0 (from -r requirements.txt (line 2))
  Using cached torchvision-0.28.0-cp311-cp311-win_amd64.whl.metadata (5.6 kB)
Requirement already satisfied: numpy>=1.24.0 in .\.venv\Lib\site-packages (from -r requirements.txt (line 3)) (2.4.6)
Collecting pandas>=2.0.0 (from -r requirements.txt (line 4))
  Using cached pandas-3.0.3-cp311-cp311-win_amd64.whl.metadata (19 kB)
Requirement already satisfied: PyYAML>=6.0 in .\.venv\Lib\site-packages (from -r requirements.txt (line 5)) (6.0.3)
Collecting mediapipe>=0.10.0 (from -r requirements.txt (line 6))
  Using cached mediapipe-0.10.35-py3-none-win_amd64.whl.metadata (9.8 kB)
Requirement already satisfied: opencv-python>=4.8.0 in .\.venv\Lib\site-packages (from -r requirements.txt (line 7)) (5.0.0.93)
Requirement already satisfied: tqdm>=4.65.0 in .\.venv\Lib\site-packages (from -r requirements.txt (line 8)) (4.68.4)
Requirement already satisfied: scipy>=1.10.0 in .\.venv\Lib\site-packages (from -r requirements.txt (line 9)) (1.17.1)
Requirement already satisfied: filelock in .\.venv\Lib\site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (3.29.7)
Requirement already satisfied: typing-extensions>=4.10.0 in .\.venv\Lib\site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (4.16.0)
Requirement already satisfied: setuptools>=77.0.3 in .\.venv\Lib\site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (83.0.0)
Requirement already satisfied: sympy>=1.13.3 in .\.venv\Lib\site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in .\.venv\Lib\site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (3.6.1)
Requirement already satisfied: jinja2 in .\.venv\Lib\site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in .\.venv\Lib\site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (2026.6.0)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in .\.venv\Lib\site-packages (from torchvision>=0.15.0->-r requirements.txt (line 2)) (12.3.0)
Requirement already satisfied: python-dateutil>=2.8.2 in .\.venv\Lib\site-packages (from pandas>=2.0.0->-r requirements.txt (line 4)) (2.9.0.post0)
Requirement already satisfied: tzdata in .\.venv\Lib\site-packages (from pandas>=2.0.0->-r requirements.txt (line 4)) (2026.2)
Requirement already satisfied: absl-py~=2.3 in .\.venv\Lib\site-packages (from mediapipe>=0.10.0->-r requirements.txt (line 6)) (2.5.0)
Requirement already satisfied: certifi in .\.venv\Lib\site-packages (from mediapipe>=0.10.0->-r requirements.txt (line 6)) (2026.6.17)
Collecting sounddevice~=0.5 (from mediapipe>=0.10.0->-r requirements.txt (line 6))
  Using cached sounddevice-0.5.5-py3-none-win_amd64.whl.metadata (1.4 kB)
Requirement already satisfied: flatbuffers~=25.9 in .\.venv\Lib\site-packages (from mediapipe>=0.10.0->-r requirements.txt (line 6)) (25.12.19)
Requirement already satisfied: opencv-contrib-python in .\.venv\Lib\site-packages (from mediapipe>=0.10.0->-r requirements.txt (line 6)) (5.0.0.93)
Collecting matplotlib (from mediapipe>=0.10.0->-r requirements.txt (line 6))
  Using cached matplotlib-3.11.0-cp311-cp311-win_amd64.whl.metadata (80 kB)
Requirement already satisfied: cffi in .\.venv\Lib\site-packages (from sounddevice~=0.5->mediapipe>=0.10.0->-r requirements.txt (line 6)) (2.1.0)
Requirement already satisfied: colorama in .\.venv\Lib\site-packages (from tqdm>=4.65.0->-r requirements.txt (line 8)) (0.4.6)
Requirement already satisfied: six>=1.5 in .\.venv\Lib\site-packages (from python-dateutil>=2.8.2->pandas>=2.0.0->-r requirements.txt (line 4)) (1.17.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in .\.venv\Lib\site-packages (from sympy>=1.13.3->torch>=2.0.0->-r requirements.txt (line 1)) (1.3.0)
Requirement already satisfied: pycparser in .\.venv\Lib\site-packages (from cffi->sounddevice~=0.5->mediapipe>=0.10.0->-r requirements.txt (line 6)) (3.0)
Requirement already satisfied: MarkupSafe>=2.0 in .\.venv\Lib\site-packages (from jinja2->torch>=2.0.0->-r requirements.txt (line 1)) (3.0.3)
Requirement already satisfied: contourpy>=1.0.1 in .\.venv\Lib\site-packages (from matplotlib->mediapipe>=0.10.0->-r requirements.txt (line 6)) (1.3.3)
Requirement already satisfied: cycler>=0.10 in .\.venv\Lib\site-packages (from matplotlib->mediapipe>=0.10.0->-r requirements.txt (line 6)) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in .\.venv\Lib\site-packages (from matplotlib->mediapipe>=0.10.0->-r requirements.txt (line 6)) (4.63.0)
Requirement already satisfied: kiwisolver>=1.3.1 in .\.venv\Lib\site-packages (from matplotlib->mediapipe>=0.10.0->-r requirements.txt (line 6)) (1.5.0)
Requirement already satisfied: packaging>=20.0 in .\.venv\Lib\site-packages (from matplotlib->mediapipe>=0.10.0->-r requirements.txt (line 6)) (26.2)
Requirement already satisfied: pyparsing>=3 in .\.venv\Lib\site-packages (from matplotlib->mediapipe>=0.10.0->-r requirements.txt (line 6)) (3.3.2)
Using cached torch-2.13.0-cp311-cp311-win_amd64.whl (122.0 MB)
Using cached torchvision-0.28.0-cp311-cp311-win_amd64.whl (3.8 MB)
Using cached pandas-3.0.3-cp311-cp311-win_amd64.whl (9.9 MB)
Using cached mediapipe-0.10.35-py3-none-win_amd64.whl (10.9 MB)
Using cached sounddevice-0.5.5-py3-none-win_amd64.whl (365 kB)
Using cached matplotlib-3.11.0-cp311-cp311-win_amd64.whl (9.3 MB)
Installing collected packages: torch, sounddevice, pandas, matplotlib, torchvision, mediapipe

Successfully installed matplotlib mediapipe pandas sounddevice torch torchvision

```
