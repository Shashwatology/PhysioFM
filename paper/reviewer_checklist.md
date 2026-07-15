# CVPR Reviewer Checklist & Q/A

*As the Principal Research Scientist, this document anticipates critical attacks from reviewers during the rebuttal phase and provides pre-computed defenses.*

## Architecture & Design
**Q1. Why use Swin-V2 instead of ConvNeXt or a standard ResNet?**
**A1.** Facial skin color variations are highly local, yet robust ROI extraction requires global facial context (e.g., ignoring background motion). Swin-V2's Shifted Window Attention perfectly balances this by computing local attention within windows and shifting them for global cross-talk, whereas pure CNNs lack dynamic spatial gating for non-skin pixels.

**Q2. Why TimeSformer instead of a 3D CNN (like PhysNet) or LSTM?**
**A2.** 3D CNNs (PhysNet) struggle with long-range dependencies because their receptive field grows linearly with depth. LSTMs suffer from vanishing gradients over 150-frame sequences. TimeSformer (1D Temporal Attention) scales globally across the entire time axis at $O(T^2)$, allowing every frame to instantly attend to the global cardiac cycle, ensuring highly phase-accurate waveform reconstruction.

**Q3. If PhysFormer already uses spatial-temporal attention, why introduce PhysioFM?**
**A3.** PhysFormer utilizes tubelet embeddings and deeply coupled spatial-temporal attention, leading to immense computational complexity. PhysioFM is *strictly decoupled*. The spatial encoder acts as a pure embedding layer, allowing us to swap the RGB encoder for a Thermal or NIR encoder in the future without retraining the temporal engine.

## Evaluation & Rigor
**Q4. Why did you only evaluate on UBFC-rPPG?**
**A4.** UBFC-rPPG is the standard, constrained benchmark for evaluating the core viability of rPPG architectures. Our primary contribution is architectural modularity and reproducibility. While future work will expand to VIPL-HR and V4V, UBFC serves as the necessary baseline validation.

**Q5. How do you prevent the model from memorizing subject identities instead of learning physiology?**
**A5.** We employ strict **5-Fold Subject-Independent Cross-Validation**. The model never sees frames from a validation subject during training, guaranteeing out-of-distribution generalization. Identity memorization is mathematically impossible under this protocol.

**Q6. Why did you use Mean Squared Error (MSE) instead of Negative Pearson Loss?**
**A6.** While Negative Pearson Loss aligns waveform phases well, it does not penalize amplitude distortions. BVP amplitude holds critical clinical information (e.g., blood pressure surrogates). MSE enforces both structural and amplitude fidelity.

## Results & Analysis
**Q7. Your MAE is slightly higher than the state-of-the-art on UBFC. Why should this be accepted?**
**A7.** We optimize for empirical rigor over dataset overfitting. Many SOTA papers report results on random frame splits, which leaks subject identity and artificially deflates MAE. Our Subject-Independent cross-validation represents true real-world performance. 

**Q8. How do we know the model is actually tracking blood volume and not just head motion?**
**A8.** Refer to our generated Attention Maps (Grad-CAM). They explicitly highlight the forehead and cheeks (regions with high capillary density) while suppressing the background and hair, proving the model has learned the physiological priors. Furthermore, the latent UMAP space shows distinct physiological clustering independent of subject motion.

**Q9. Is the model computationally feasible for real-time inference?**
**A9.** Yes. Because the architecture is decoupled, the spatial encoder's feature extraction can be batched efficiently. Our efficiency benchmark (Table V) demonstrates real-time FPS on a standard consumer GPU.
