# 8. Supplementary Engineering Appendix

## 8.1 GitHub Repository Structure
```text
repo/
├── configs/             # YAML configurations
├── datasets/            # Secure tensor cache
├── docs/                # Architecture notes
├── experiments/         # Checkpoints and logs
├── paper/               
│   ├── figures/         # Generated graphs
│   └── tables/          # Markdown tables
├── src/
│   ├── research/
│   │   ├── models/      # Neural architectures
│   │   ├── training/    # Optimization loops
│   │   └── evaluation/  # Validation metrics
└── README.md
```

## 8.2 Repository Statistics
- **Experiments Executed:** 8 tracked versions
- **Modularity:** 25+ decoupled modules
- **Scale:** 10,000+ Lines of Code
- **Reproducibility:** 100% Deterministic (Config Hashing)

## 8.3 Reproducibility Checklist
| Component | Value | Status |
|---|---|---|
| Python Version | 3.10.11 | ✅ Locked |
| PyTorch | 2.0.1+cu118 | ✅ Locked |
| CUDA | 11.8 | ✅ Locked |
| Random Seed | 42 | ✅ Fixed (PyTorch, Numpy, Random) |
| Dataset Split Hash | `e3b0c442` | ✅ Validated |
| Avg Training Time | 22 min / 100 epochs | ✅ |
| GPU | NVIDIA GeForce RTX 3060 | ✅ |
| System RAM | 32 GB | ✅ |


## 8.4 Hyperparameter Grid
| Hyperparameter | Value | Description |
|---|---|---|
| Learning Rate | 0.001 | AdamW optimizer base learning rate |
| Batch Size | 2 | Constrained by 3D Swin-V2 memory footprint |
| Optimizer | AdamW | Weight decay = 0.01 |
| Scheduler | CosineAnnealing | T_max = 50 |
| Waveform $\lambda$ | 1.0 | Weight for negative Pearson loss |
| Embed Dimension | 256 | Swin-V2 output projection dimension |
| Transformer Layers | 2 | Temporal alignment depth |
| Attention Heads | 4 | Temporal multi-head attention |
| Dropout | 0.3 | Applied to temporal transformer MLP |
| Sequence Length | 150 | Frames per window (5.0 seconds @ 30 FPS) |

