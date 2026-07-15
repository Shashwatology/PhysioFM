# PhysioFM: Executive Summary

**Date:** July 14, 2026
**Subject:** Final Architectural Assessment & Failure Autopsy

## The Objective
The PhysioFM project was initiated to build a Physiological Foundation Model capable of extracting Heart Rate (HR) and continuous Blood Volume Pulse (BVP) from RGB video by combining the spatial representation power of a frozen Foundation Model (Swin-V2) with a dynamic Spatio-Temporal Difference Stem (STDS) and a Temporal Transformer.

## The Architectural Triumph (Solving Gradient Starvation)
In early iterations (EXP02, EXP04, EXP05), the architecture failed due to **Gradient Starvation**. The immense embedding scale of the pre-trained Swin-V2 model fundamentally overpowered the smaller, dynamic STDS branch during feature fusion. The layer-wise gradient flow diagnostic revealed a fatal imbalance of $1,000,000 : 1$. 

In EXP06C, we pioneered **Pre-Fusion Layer Normalization**. By independently mathematically enforcing zero-mean and unit-variance on both spatial stems prior to fusion, we successfully restored the gradient flow, bringing the gradient ratio down to a perfect $1.1 : 1$. This structural intervention is a genuinely novel contribution to hierarchical foundation modeling.

## The Mathematical Failure (Objective Function Mismatch)
Despite solving the structural gradient blockage, the EXP06C architecture suffered catastrophic mode collapse, predicting a constant scalar value equal to the dataset mean ($\sim 85$ BPM). 

A strict Loss-Scale Audit was executed. The audit revealed that the absolute magnitude of the Mean Absolute Error (MSE) loss ($\sim 100$) completely dwarfed the auxiliary Waveform correlation loss ($\sim 1$) by a ratio of **240:1**. 

Because the Waveform Loss (which enforces temporal phase and variance) was mathematically insignificant compared to the MSE, the optimizer ignored it. To rapidly minimize the massive MSE penalty, the network took the shortest mathematical path: predicting the constant dataset mean. 

## Scientific Conclusion
PhysioFM has successfully resolved the spatial-temporal fusion bottleneck but is currently paralyzed by an objective function scaling error. 

The metrics appear artificially excellent (e.g., Wilcoxon $p = 0.8437$, Mean Bias = $0.260$ BPM) because predicting the dataset mean perfectly aligns the central tendencies. However, the true Prediction Standard Deviation is exactly $0.00$, the Concordance Correlation Coefficient is $0.00$, and the 95% Limits of Agreement are an unacceptable $\pm 18$ BPM. 

**Recommendation:** Do not publish as a physiological extractor. Proceed immediately to EXP08 to replace the absolute MSE loss with a scale-invariant Concordance Correlation (CCC) loss.
