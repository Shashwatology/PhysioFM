# Publication Tables

## Table I: Dataset Summary
| Dataset | Modality | Subjects | Total Videos | Lighting Conditions | Motion | Resolution |
|---|---|---|---|---|---|---|
| UBFC-rPPG | RGB | 42 | 42 | Natural | Low | 640x480 |
| VIPL-HR | RGB+NIR | 107 | 2,378 | Controlled+Dark | High | Varies |
| iBVP | RGB+Thermal| 52 | 300+ | Variable | Moderate | Varies |

## Table II: Implementation Details
| Hyperparameter | Value | Description |
|---|---|---|
| Spatial Backbone | Swin-V2-Tiny | Pretrained on ImageNet |
| Temporal Engine | TimeSformer | 1D Self-Attention |
| Input Resolution | 64x64 | Cropped Face ROI |
| Sequence Length | 150 Frames | 5 Seconds @ 30 FPS |
| Optimizer | AdamW | Weight decay 1e-4 |
| Learning Rate | 1e-4 | Cosine Annealing |
| Batch Size | 4 | Mixed Precision (FP16) |

## Table III: Comparison with State-of-the-Art on UBFC-rPPG
| Method | Modality | MAE (bpm) ↓ | RMSE (bpm) ↓ | Pearson (r) ↑ |
|---|---|---|---|---|
| POS (Wang et al.) | RGB | ~2.50 | ~3.80 | ~0.85 |
| DeepPhys (Chen et al.) | RGB | ~1.80 | ~2.20 | ~0.90 |
| PhysNet (Yu et al.) | RGB | ~1.10 | ~1.40 | ~0.94 |
| **PhysioFM (Ours)** | **RGB** | **TBD** | **TBD** | **TBD** |

## Table IV: Architectural Ablation Study (UBFC-rPPG)
| Configuration | Temporal Mod. | MAE (bpm) ↓ | RMSE (bpm) ↓ |
|---|---|---|---|
| ConvNeXt Baseline | LSTM | TBD | TBD |
| Swin-V2 | LSTM | TBD | TBD |
| Swin-V2 | None | TBD | TBD |
| **Swin-V2 (PhysioFM)** | **TimeSformer** | **TBD** | **TBD** |

## Table V: Model Efficiency and Computational Cost
| Model | Parameters | GFLOPs | Peak GPU Mem | Latency (ms) | FPS |
|---|---|---|---|---|---|
| DeepPhys | ~2M | ~15 | ~1.2 GB | ~5 | ~200 |
| PhysNet | ~1.2M | ~20 | ~1.5 GB | ~8 | ~120 |
| **PhysioFM** | **~34M** | **TBD** | **TBD** | **TBD** | **TBD** |
