# Remaining Hypotheses for Mode Collapse

## Hypothesis 4: Spatial Feature Dominance (Feature Imbalance at Fusion)
- **Probability**: 95%
- **Supporting Evidence**: `EXP06B_PrePool_Smoke` fixed the pooling bottleneck but gradients still vanished at the `Concat` fusion layer. `activation_statistics.csv` reveals that `Swin` features have an std of ~2.3, while `STDS` features have an std of ~0.33. Because `Swin` dominates the concatenated vector by a factor of 7x, and `Swin` produces essentially static spatial embeddings of the face, the resulting sequence `rgb_spatial_emb` is static. A static sequence entering `TemporalTransformer` forces the `WaveformHead` to predict a flat line, producing zero variance and tiny gradients, perpetuating mode collapse.
- **Contradicting Evidence**: None. 
- **Complexity**: Low. Requires applying `nn.LayerNorm` to `rgb_swin_emb` and `rgb_stds_emb` *before* concatenation.
- **Expected Improvement**: Normalizing the features will put the dynamic temporal micro-color changes from STDS on equal footing with the static spatial features from Swin, finally presenting a dynamic sequence to the Temporal Transformer.

## Hypothesis 3: Gradient Scaling / Optimizer Instability
- **Probability**: 5%
- **Supporting Evidence**: `HR_MSE` gradients are inherently larger than `NegativePearsonLoss` gradients.
- **Contradicting Evidence**: Even if `HR_MSE` is larger, `WaveformHead` still calculates a gradient of ~300 for `TemporalTransformer`. The gradient completely dies *before* reaching the spatial stems, implying the attention is saturated or features are static, pointing to Hypothesis 4.
- **Complexity**: Low. Requires changing loss weights.
- **Expected Improvement**: Minimal without architectural fix.
