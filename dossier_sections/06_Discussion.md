# 6. Discussion: Unraveling Optimization Failures

Why did the architectures fail, and what does it teach us about physiological ML?

## 6.1 Why did STDS fail initially?
STDS was designed to extract micro-motion, but when combined with Swin-V2, its gradients vanished. The initialized variance of Swin-V2 features was astronomically larger than the freshly initialized STDS 3D CNN, leading to a "magnitude dominance" where the Linear fusion layer completely ignored the STDS inputs.

## 6.2 Why did Gating (Add) fail?
In EXP04, we attempted to add the STDS features as a gate ($f_{swin} + lpha f_{stds}$). This failed because the addition operation requires the two tensors to operate on the same absolute scale, which was violated by the frozen vs trainable dynamic.

## 6.3 Why did LayerNorm help?
Pre-Fusion Layer Normalization mathematically forced both stems to zero-mean and unit-variance *before* concatenation. This equalized the distribution scales, which substantially improved gradient propagation down both branches, dropping the Gradient Ratio from $10^6:1$ to $1.1:1$.

## 6.4 Why did waveform supervision alone fail?
Even with stable gradient flow, the architecture collapsed. The absolute magnitude of the Heart Rate MSE loss ($\sim 100$) dwarfed the Waveform Pearson loss ($\sim 1$). The optimizer took the path of least resistance: it predicted the dataset mean to minimize the massive MSE penalty, ignoring the temporal phase entirely.
