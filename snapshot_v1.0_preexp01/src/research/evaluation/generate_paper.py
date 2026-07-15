import os

def generate_paper_draft(report_dir: str):
    """
    Generates the skeleton and content for the Methodology and Experimental Setup sections.
    """
    os.makedirs(report_dir, exist_ok=True)
    
    # SVG Architecture Diagram
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400" width="100%" height="100%">
    <!-- Definitions for arrows and gradients -->
    <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#333" />
        </marker>
        <linearGradient id="spatial" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#4facfe" />
            <stop offset="100%" stop-color="#00f2fe" />
        </linearGradient>
        <linearGradient id="temporal" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#f093fb" />
            <stop offset="100%" stop-color="#f5576c" />
        </linearGradient>
    </defs>

    <!-- Video Input -->
    <rect x="50" y="150" width="80" height="100" rx="10" fill="#e0e0e0" stroke="#333" stroke-width="2"/>
    <text x="90" y="200" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">Input Video</text>
    <text x="90" y="220" font-family="Arial" font-size="12" text-anchor="middle">(T, C, H, W)</text>

    <!-- Arrow 1 -->
    <line x1="130" y1="200" x2="180" y2="200" stroke="#333" stroke-width="3" marker-end="url(#arrow)" />

    <!-- Spatial Encoder -->
    <rect x="190" y="120" width="120" height="160" rx="15" fill="url(#spatial)" stroke="#333" stroke-width="2"/>
    <text x="250" y="190" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#fff">Swin-V2</text>
    <text x="250" y="210" font-family="Arial" font-size="12" text-anchor="middle" fill="#fff">Spatial Encoder</text>

    <!-- Arrow 2 -->
    <line x1="310" y1="200" x2="360" y2="200" stroke="#333" stroke-width="3" marker-end="url(#arrow)" />

    <!-- Temporal Encoder -->
    <rect x="370" y="120" width="120" height="160" rx="15" fill="url(#temporal)" stroke="#333" stroke-width="2"/>
    <text x="430" y="190" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#fff">TimeSformer</text>
    <text x="430" y="210" font-family="Arial" font-size="12" text-anchor="middle" fill="#fff">Temporal Engine</text>

    <!-- Arrow 3 -->
    <line x1="490" y1="200" x2="540" y2="200" stroke="#333" stroke-width="3" marker-end="url(#arrow)" />

    <!-- Feature Fusion & Heads -->
    <rect x="550" y="100" width="100" height="60" rx="10" fill="#4caf50" stroke="#333" stroke-width="2"/>
    <text x="600" y="135" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#fff">HR Head</text>

    <rect x="550" y="170" width="100" height="60" rx="10" fill="#ff9800" stroke="#333" stroke-width="2"/>
    <text x="600" y="205" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#fff">RR Head</text>

    <rect x="550" y="240" width="100" height="60" rx="10" fill="#9c27b0" stroke="#333" stroke-width="2"/>
    <text x="600" y="275" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#fff">BVP Head</text>

    <path d="M 490 200 L 520 200 L 520 130 L 540 130" fill="none" stroke="#333" stroke-width="3" marker-end="url(#arrow)"/>
    <path d="M 490 200 L 520 200 L 520 270 L 540 270" fill="none" stroke="#333" stroke-width="3" marker-end="url(#arrow)"/>

</svg>"""

    with open(os.path.join(report_dir, "figures", "architecture.svg"), "w") as f:
        f.write(svg_content)
        
    md_content = """# Methodology

## 3.1 Overall Architecture
PhysioFM proposes a decoupled spatial-temporal architecture for physiological representation learning. Given a video input tensor $\mathbf{V} \in \mathbb{R}^{T \\times C \\times H \\times W}$, where $T$ is the number of frames, $C$ is the number of channels, and $H, W$ are the spatial dimensions, the model processes the video to extract underlying physiological signals. The architecture consists of a Spatial Encoder, a Temporal Engine, and task-specific regression heads.

![Architecture Diagram](../figures/architecture.svg)

## 3.2 Spatial Encoder (Swin-V2)
To extract robust spatial features representing subtle color changes in the facial region, we employ a Swin-V2 Transformer. The video is processed frame-by-frame:
$$ \mathbf{F}_s = \text{SwinV2}(\mathbf{V}) \in \mathbb{R}^{T \\times D_s} $$
where $D_s$ is the spatial embedding dimension. We utilize shifted-window attention to capture both local skin-tone variations and global facial contexts.

## 3.3 Temporal Engine (TimeSformer)
Traditional methods aggregate temporal information using 3D CNNs or RNNs, which struggle with long-range dependencies. PhysioFM utilizes a 1D Temporal Self-Attention mechanism (TimeSformer) to analyze the sequence of spatial embeddings:
$$ \mathbf{Z} = \text{LayerNorm}(\mathbf{F}_s) $$
$$ \mathbf{F}_t = \text{Softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V} + \mathbf{Z} $$
where $\mathbf{Q, K, V}$ are projections of $\mathbf{Z}$. The output latent representation $\mathbf{L} \in \mathbb{R}^{T \\times D_t}$ captures the rhythmic physiological oscillations.

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
- **Optimizer**: AdamW optimizer with a base learning rate of $1 \times 10^{-4}$ and weight decay of $1 \times 10^{-4}$.
- **Hardware**: All experiments are conducted on an NVIDIA RTX GPU environment with FP16 mixed-precision training.
- **Evaluation**: We utilize a 5-Fold Subject-Level Cross-Validation strategy to ensure rigorous evaluation and prevent subject identity memorization.
"""

    with open(os.path.join(report_dir, "methodology.md"), "w", encoding='utf-8') as f:
        f.write(md_content)

if __name__ == "__main__":
    os.makedirs(r"paper\figures", exist_ok=True)
    generate_paper_draft("paper")
    print("Paper methodology draft and SVG generated successfully.")
