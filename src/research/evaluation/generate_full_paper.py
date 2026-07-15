import os

def generate_svgs(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall Pipeline
    svg_1 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 200" width="100%" height="100%">
    <rect x="50" y="50" width="120" height="100" fill="#e3f2fd" stroke="#1e88e5" stroke-width="2"/>
    <text x="110" y="105" font-family="Arial" font-size="14" text-anchor="middle">Input Video</text>
    <line x1="170" y1="100" x2="220" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <rect x="220" y="50" width="120" height="100" fill="#e8f5e9" stroke="#43a047" stroke-width="2"/>
    <text x="280" y="95" font-family="Arial" font-size="14" text-anchor="middle">Face Detection &amp;</text>
    <text x="280" y="115" font-family="Arial" font-size="14" text-anchor="middle">ROI Extraction</text>
    <line x1="340" y1="100" x2="390" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <rect x="390" y="50" width="150" height="100" fill="#fff3e0" stroke="#fb8c00" stroke-width="2"/>
    <text x="465" y="95" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">PhysioFM</text>
    <text x="465" y="115" font-family="Arial" font-size="12" text-anchor="middle">(Spatial-Temporal)</text>
    <line x1="540" y1="100" x2="590" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <rect x="590" y="50" width="120" height="100" fill="#f3e5f5" stroke="#8e24aa" stroke-width="2"/>
    <text x="650" y="95" font-family="Arial" font-size="14" text-anchor="middle">Multi-Task</text>
    <text x="650" y="115" font-family="Arial" font-size="14" text-anchor="middle">Predictions</text>
</svg>"""
    with open(os.path.join(output_dir, "fig1_overall_pipeline.svg"), "w") as f: f.write(svg_1)

    # 2. PhysioFM Architecture
    svg_2 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 300" width="100%" height="100%">
    <rect x="100" y="50" width="150" height="200" fill="#e1f5fe" stroke="#0288d1" stroke-width="2"/>
    <text x="175" y="150" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">Swin-V2</text>
    <text x="175" y="170" font-family="Arial" font-size="12" text-anchor="middle">Spatial Encoder</text>
    <line x1="250" y1="150" x2="320" y2="150" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <rect x="320" y="50" width="150" height="200" fill="#fce4ec" stroke="#c2185b" stroke-width="2"/>
    <text x="395" y="150" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">TimeSformer</text>
    <text x="395" y="170" font-family="Arial" font-size="12" text-anchor="middle">Temporal Engine</text>
    <line x1="470" y1="150" x2="540" y2="150" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <rect x="540" y="50" width="150" height="200" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
    <text x="615" y="140" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">HR / RR / Waveform</text>
    <text x="615" y="160" font-family="Arial" font-size="12" text-anchor="middle">Decoupled Task Heads</text>
</svg>"""
    with open(os.path.join(output_dir, "fig2_physiofm.svg"), "w") as f: f.write(svg_2)

    # 3. Training Pipeline
    svg_3 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 200" width="100%" height="100%">
    <text x="400" y="100" font-family="Arial" font-size="20" text-anchor="middle">Training Pipeline Diagram (To Be Finalized)</text>
</svg>"""
    with open(os.path.join(output_dir, "fig3_training_pipeline.svg"), "w") as f: f.write(svg_3)

    # 4. Evaluation Workflow
    svg_4 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 200" width="100%" height="100%">
    <text x="400" y="100" font-family="Arial" font-size="20" text-anchor="middle">Evaluation Workflow (To Be Finalized)</text>
</svg>"""
    with open(os.path.join(output_dir, "fig4_eval_workflow.svg"), "w") as f: f.write(svg_4)

    # 5. Error Analysis Pipeline
    svg_5 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 200" width="100%" height="100%">
    <text x="400" y="100" font-family="Arial" font-size="20" text-anchor="middle">Error Analysis Pipeline (To Be Finalized)</text>
</svg>"""
    with open(os.path.join(output_dir, "fig5_error_analysis.svg"), "w") as f: f.write(svg_5)


def generate_paper_text(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    md = """# PhysioFM: A Modular Spatial–Temporal Physiological Representation Network for Contactless Video-Based Cardiopulmonary Monitoring

## 1. Introduction
Contactless physiological monitoring using remote photoplethysmography (rPPG) has rapidly evolved, transitioning from traditional signal processing heuristics to sophisticated deep learning frameworks. By analyzing subtle micro-color variations in the human face caused by the cardiac cycle, video-based cardiopulmonary monitoring enables continuous assessment of Heart Rate (HR), Respiratory Rate (RR), and Blood Volume Pulse (BVP) without obtrusive wearable sensors.

Despite significant advancements, current methods predominantly rely on highly coupled 3D Convolutional Neural Networks (3D-CNNs) or monolithic architectures that restrict adaptability across diverse modalities (e.g., Thermal, NIR) and limit interpretability. 

In this work, we introduce **PhysioFM**, a decoupled spatial-temporal representation network. By explicitly separating spatial feature extraction (using a vision transformer backbone, Swin-V2) from temporal aggregation (using a 1D Temporal Self-Attention engine, TimeSformer), PhysioFM learns robust, modular physiological representations.

**Contributions:**
1. We propose a fully decoupled spatial-temporal architecture that natively supports interchangeable modality encoders.
2. We establish a highly reproducible, rigorous evaluation framework utilizing 5-Fold Subject-Level Cross-Validation on the UBFC-rPPG benchmark.
3. We present an extensive error analysis pipeline mapping performance against heart rate ranges, ensuring rigorous calibration.

## 2. Related Work
Early rPPG methods, such as POS (Wang et al., 2016) and CHROM (De Haan et al., 2013), relied on skin-orthogonal mathematical projections. While effective under controlled lighting, they lacked robustness against significant motion and varying illumination.

The advent of deep learning introduced CNN-based end-to-end regressors. **DeepPhys** (Chen et al., 2018) popularized the 2D attention mechanism using normalized frame differences, while **PhysNet** (Yu et al., 2019) utilized 3D spatio-temporal convolutions to learn robust representations from raw RGB sequences. However, 3D convolutions scale poorly with sequence length and struggle to capture very long-range physiological dependencies.

Recently, Transformer-based methods like **PhysFormer** (Yu et al., 2022) have demonstrated the efficacy of self-attention in capturing long-range temporal context. Similarly, **EfficientPhys** (Liu et al., 2023) focused on reducing the computational overhead of vision transformers. 

**How PhysioFM differs:** Unlike PhysNet's monolithic 3D volume processing or PhysFormer's deeply integrated spatial-temporal attention, PhysioFM strictly decouples the operations. The spatial encoder (Swin-V2) acts purely as an embedding extractor, passing high-dimensional semantic tokens to the Temporal Engine (TimeSformer). This modularity allows for independent optimization, arbitrary sequence lengths, and zero-shot multi-modal scalability.

## 3. Methodology

![Overall Pipeline](figures/fig1_overall_pipeline.svg)
*Figure 1: Overall Pipeline of PhysioFM.*

### 3.1 Spatial Encoder (Swin-V2)
The raw video tensor $\mathbf{V} \in \mathbb{R}^{T \\times C \\times H \\times W}$ is processed frame-by-frame. We employ a Swin-V2 architecture with Shifted Window Attention. For a local window of features $\mathbf{X}$, the attention is computed as:
$$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}} + \mathbf{B}\right)\mathbf{V} $$
where $\mathbf{B}$ is the relative position bias. The resulting spatial embedding sequence is $\mathbf{F}_s \in \mathbb{R}^{T \\times D_s}$.

### 3.2 Temporal Engine (TimeSformer)
To capture rhythmic oscillations, we employ a 1D Temporal Self-Attention block. We apply LayerNorm followed by Multi-Head Self Attention (MSA) over the sequence length $T$:
$$ \mathbf{Z} = \text{LayerNorm}(\mathbf{F}_s) $$
$$ \mathbf{F}_t = \text{MSA}(\mathbf{Z}) + \mathbf{F}_s $$
A Feed-Forward Network (FFN) with GELU activation processes the attention outputs:
$$ \mathbf{L} = \text{FFN}(\text{LayerNorm}(\mathbf{F}_t)) + \mathbf{F}_t $$
The output $\mathbf{L} \in \mathbb{R}^{T \\times D_t}$ represents the physiological latent space.

### 3.3 Multi-Task Objective
The latent space is decoded by independent multi-layer perceptrons into HR, RR, and the continuous BVP waveform. The model is optimized using AdamW via a multi-task Mean Squared Error (MSE) objective:
$$ \mathcal{L}_{total} = \lambda_1 || \mathbf{\hat{y}}_{hr} - \mathbf{y}_{hr} ||^2_2 + \lambda_2 \sum_{t=1}^{T} || \mathbf{\hat{y}}_{bvp}^{(t)} - \mathbf{y}_{bvp}^{(t)} ||^2_2 $$

## 4. Experimental Settings

### 4.1 Datasets and Preprocessing
We evaluate PhysioFM on the **UBFC-rPPG** dataset (42 subjects). Video frames are processed dynamically via MediaPipe FaceMesh to extract highly stable Region of Interest (ROI) bounding boxes. Sequences are padded or cropped to $T=150$ (5 seconds at 30 FPS).

### 4.2 Implementation Details
- **Optimizer**: AdamW with $\\beta_1=0.9$, $\\beta_2=0.999$, and weight decay of $1 \\times 10^{-4}$.
- **Learning Rate**: $1 \\times 10^{-4}$ with a Cosine Annealing scheduler.
- **Hardware**: FP16 Automatic Mixed Precision on an NVIDIA GPU environment.
- **Evaluation**: We utilize a rigorous 5-Fold Subject-Level Cross-Validation to ensure true out-of-distribution generalization.
"""

    with open(os.path.join(output_dir, "manuscript_draft.md"), "w", encoding='utf-8') as f:
        f.write(md)
        
if __name__ == "__main__":
    generate_svgs(r"paper\figures")
    generate_paper_text(r"paper")
    print("Paper generated successfully.")
