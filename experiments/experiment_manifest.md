# PhysioFM Experiment Manifest

This document serves as the central research notebook for the PhysioFM project. Every executed experiment must be documented here for strict scientific tracking.

| Exp ID | Objective | Dataset | Seed | Architecture | Optimizer | LR | Batch | Epochs | Expected Outcome | Actual Outcome | Conclusion | Paper Figure Generated | Paper Table Updated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Stage_0.5` | Validate dataset health, check frame alignment and missing GT. | UBFC-rPPG | 42 | N/A | N/A | N/A | N/A | N/A | Dataset should contain 42 valid subjects with synchronized BVP/HR. | *Pending* | *Pending* | dataset_stats.png | Table I |
| `Stage_1.0` | Verify Code Infrastructure. Ensure loss decreases and checkpoints/latents save. | UBFC-rPPG | 42 | Swin-V2 + TimeSformer | AdamW | 1e-4 | 2 | 2 | Loss should decrease; latents, waveforms, preds exported properly. | *Pending* | *Pending* | None | None |
| `Stage_2.0` | Tuning convergence and stability on 5 subjects. | UBFC-rPPG | 42 | Swin-V2 + TimeSformer | AdamW | 1e-4 | 4 | 10 | Model should learn generalizable physiological representations. | *Pending* | *Pending* | loss_curve.png | None |
