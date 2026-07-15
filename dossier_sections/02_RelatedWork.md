# 2. Related Work

Recent advancements in remote physiological measurement have heavily utilized 3D Convolutional Neural Networks and, more recently, Vision Transformers. However, integration of spatial Foundation Models into rPPG is largely unexplored due to optimization difficulties.

## 2.1 Literature Comparison

| Model | Backbone | Dataset | HR | RR | Waveform | Transformer | Novelty |
|---|---|---|---|---|---|---|---|
| DeepPhys | 2D CNN | PURE | ✅ | ❌ | ✅ | ❌ | First Deep Learning rPPG |
| PhysNet | 3D CNN | OBF | ✅ | ✅ | ✅ | ❌ | Spatiotemporal Convolutions |
| PhysFormer | ViT | UBFC | ✅ | ✅ | ✅ | ✅ | First pure Transformer rPPG |
| EffPhys | CNN | UBFC | ✅ | ❌ | ✅ | ❌ | Efficient mobile architecture |
| RhythmFormer| ViT | VIPL | ✅ | ✅ | ✅ | ✅ | Spatial-temporal attention |
| PhysMamba | Mamba | UBFC | ✅ | ❌ | ✅ | ❌ | State Space Models |
| ME-rPPG | CNN | UBFC | ✅ | ❌ | ❌ | ❌ | Micro-expression focus |
| PulseFormer | ViT | UBFC | ✅ | ❌ | ✅ | ✅ | Multi-scale Transformer |
| **PhysioFM (Ours)** | **Swin-V2 + STDS** | **UBFC** | **✅** | **✅** | **✅** | **✅** | **Gradient bottleneck isolation** |

