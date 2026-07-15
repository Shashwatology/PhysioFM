# EXP06A Failure Report: Waveform Supervision

## Objective
To determine if modifying the supervision signal (from pure HR MSE to a combined HR MSE + Waveform Negative Pearson Loss) breaks the persistent mode collapse in the PhysioFM v2 architecture, while preserving the `Concat + Linear` fusion module and all other hyper-parameters.

## Results against Acceptance Criteria

| Metric | Threshold | Result (Epoch 10) | Pass/Fail |
|--------|-----------|--------|-----------|
| **Prediction Std** | > 1.0 BPM | ~0.00004 BPM | ❌ FAIL |
| **Prediction Entropy** | > 0.0 | 0.0000 | ❌ FAIL |
| **Pearson Correlation** | > 0.2 | ~ -0.46 | ❌ FAIL |
| **Head/STDS Grad Ratio** | Decreasing | > 263,000 | ❌ FAIL |
| **Head/Swin Grad Ratio** | Decreasing | > 5,720,000 | ❌ FAIL |

## Diagnostic Analysis
The network successfully minimizes the `HR_MSE` loss (dropping to ~75.7 on epoch 9), achieving an MAE of `8.54` (the dataset mean). However, the `WV_Loss` remains stuck at `0.914` (which equals a Pearson correlation of `0.086`). The gradients still fail to propagate to the spatial backbones.

### Structural Root Cause Identified
Despite adding waveform supervision, the model failed to learn dynamic phase alignment. An audit of `physio_fm.py` reveals the exact bottleneck preventing waveform supervision from working:

```python
# D. Central Physiological Representation (Pool across time)
latent_embedding = fused_seq_emb.mean(dim=1) # (B, D)

# E. Independent Task Heads
hr_pred = self.hr_head(latent_embedding)
wv_pred = self.wv_head(latent_embedding)
```

The temporal dimension `T` is **mean-pooled and destroyed** *before* the embedding reaches the `WaveformHead`. `WaveformHead` is subsequently forced to project a static, time-agnostic `(B, D)` vector back into a temporal sequence `(B, seq_len)`.

Because the prediction does not depend on specific temporal frame features, the gradients from `WaveformLoss` flow backward as a `(B, D)` vector. When backpropagating through `mean(dim=1)`, this gradient is simply duplicated across all `T` frames identically. **The Temporal Transformer receives a flat, uniform gradient across time, which provides zero phase-alignment signal.**

## Conclusion
Waveform supervision cannot function correctly if the temporal representation is pooled into a static vector before the waveform is predicted. The hypothesis that "changing the supervision objective alone mitigates mode collapse" is **falsified under the current structural constraint**.

## Next Steps
Per the strict instructions ("IF EXP06A FAILS... Explain why. Do NOT redesign architecture. STOP."), I am halting further experiments. Mode collapse cannot be broken without an architectural redesign of how the temporal sequence connects to the task heads.
