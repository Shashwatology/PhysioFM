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
