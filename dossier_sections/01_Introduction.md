# 1. Introduction

The estimation of human physiological signals, specifically Heart Rate (HR) and Blood Volume Pulse (BVP), using remote Photoplethysmography (rPPG) from RGB video remains challenging due to lighting variations, subject movement, and the inherent low signal-to-noise ratio. Current architectures relying on traditional Convolutional Neural Networks (CNNs) and standard Vision Transformers struggle with the temporal alignment of micro-color changes over long sequences.

## 1.1 Clinical Motivation
Remote physiological monitoring enables contactless, continuous vital sign tracking, which is essential for telehealth, neonatal care, and triaging in critical environments.
![Clinical Motivation](paper/figures/clinical_motivation.png)
*Figure 1: The target workflow of PhysioFM. Raw video from consumer-grade cameras is processed through the Spatio-Temporal model to output clinical-grade vital signs to a hospital dashboard.*

## 1.2 The Bottleneck of Hierarchical Transformers
We hypothesized that the immense spatial representation power of a frozen Foundation Model (Swin-V2) could be leveraged for micro-physiological extraction when paired with specialized Temporal layers. 

However, we systematically identified and isolated a critical optimization bottleneck: **Gradient Starvation**. When massive spatial embeddings are hierarchically fused with lightweight temporal modules, the temporal gradients vanish, forcing catastrophic mode collapse.

## 1.3 Contributions
1. We systematically identify and isolate optimization bottlenecks responsible for mode collapse in transformer-based rPPG.
2. We propose a Spatio-Temporal Difference Stem (STDS) to isolate temporal phase shifts independent of the spatial Foundation Model.
3. We introduce Pre-Fusion Layer Normalization, an optimization strategy that substantially improves stable gradient propagation between dual stems.
