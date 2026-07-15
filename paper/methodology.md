# Methodology

## 3.1 Overall Architecture
PhysioFM proposes a decoupled spatial-temporal architecture for physiological representation learning. Given a video input tensor $\mathbf{V} \in \mathbb{R}^{T \times C \times H \times W}$, where $T$ is the number of frames, $C$ is the number of channels, and $H, W$ are the spatial dimensions, the model processes the video to extract underlying physiological signals. The architecture consists of a Spatial Encoder, a Temporal Engine, and task-specific regression heads.

![Architecture Diagram](../figures/architecture.svg)

## 3.2 Spatial Encoder (Swin-V2)
To extract robust spatial features representing subtle color changes in the facial region, we employ a Swin-V2 Transformer. The video is processed frame-by-frame:
$$ \mathbf{F}_s = 	ext{SwinV2}(\mathbf{V}) \in \mathbb{R}^{T \times D_s} $$
where $D_s$ is the spatial embedding dimension. We utilize shifted-window attention to capture both local skin-tone variations and global facial contexts.

## 3.3 Temporal Engine (TimeSformer)
Traditional methods aggregate temporal information using 3D CNNs or RNNs, which struggle with long-range dependencies. PhysioFM utilizes a 1D Temporal Self-Attention mechanism (TimeSformer) to analyze the sequence of spatial embeddings:
$$ \mathbf{Z} = 	ext{LayerNorm}(\mathbf{F}_s) $$
$$ \mathbf{F}_t = 	ext{Softmax}\left( rac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} ight) \mathbf{V} + \mathbf{Z} $$
where $\mathbf{Q, K, V}$ are projections of $\mathbf{Z}$. The output latent representation $\mathbf{L} \in \mathbb{R}^{T \times D_t}$ captures the rhythmic physiological oscillations.

## 3.4 Multi-Task Optimization
We employ independent task heads to decode the latent space. The objective function minimizes the Mean Squared Error (MSE) across the target tasks:
$$ \mathcal{L}_{total} = \lambda_1 || \mathbf{\hat{y}}_{hr} - \mathbf{y}_{hr} ||^2_2 + \lambda_2 || \mathbf{\hat{y}}_{bvp} - \mathbf{y}_{bvp} ||^2_2 $$

---

# Experimental Settings

## 4.1 Datasets
- **UBFC-rPPG**: Used as the primary benchmark for validation. The dataset contains 42 videos recorded under natural lighting conditions.
- **Preprocessing**: We employ MediaPipe FaceMesh to extract dynamic facial ROIs. Videos are temporally cropped to $T=150$ frames (5 seconds at 30 FPS).

## 4.2 Implementation Details
The model is implemented in PyTorch. The Swin-V2 backbone is initialized with ImageNet pre-trained weights to accelerate convergence.
- **Optimizer**: AdamW optimizer with a base learning rate of $1 	imes 10^{-4}$ and weight decay of $1 	imes 10^{-4}$.
- **Hardware**: All experiments are conducted on an NVIDIA RTX GPU environment with FP16 mixed-precision training.
- **Evaluation**: We utilize a 5-Fold Subject-Level Cross-Validation strategy to ensure rigorous evaluation and prevent subject identity memorization.
