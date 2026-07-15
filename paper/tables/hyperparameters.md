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
