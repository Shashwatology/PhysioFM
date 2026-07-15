import os

base_dir = r"d:\Shashwat\Platform\ml_pipeline"
src_models_dir = os.path.join(base_dir, "src", "research", "models")
docs_dir = os.path.join(base_dir, "docs", "modules")

# Directory structure
dirs_to_create = [
    os.path.join(src_models_dir, "spatial"),
    os.path.join(src_models_dir, "temporal"),
    os.path.join(src_models_dir, "fusion"),
    os.path.join(src_models_dir, "heads"),
    docs_dir
]

for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)

# Module Documentation Templates
docs = {
    "spatial_encoder.md": "# Spatial Encoder Module\n**Input:** `(B, T, C, H, W)`\n**Output:** `(B, T, D)` spatial embedding\n**Mathematical Formulation:** Swin V2 / CNN feature extraction.\n**Complexity:** Highly dependent on resolution.\n**Rationale:** Pluggable interface for robust feature extraction across modalities.",
    "temporal_encoder.md": "# Temporal Encoder Module\n**Input:** `(B, T, D)`\n**Output:** `(B, D)` or `(B, T, D)` depending on sequence pooling.\n**Mathematical Formulation:** Divided Space-Time Attention (TimeSformer).\n**Complexity:** $O(T^2)$ for self-attention.\n**Rationale:** Capturing subtle pulse frequency dynamics across the time dimension.",
    "fusion_layer.md": "# Fusion Layer\n**Input:** Multimodal embeddings `(B, T, D)`\n**Output:** `(B, T, D)` fused embedding.\n**Mathematical Formulation:** Identity Pass-Through (currently), Cross-Attention (future).\n**Rationale:** Decouples modalities cleanly for missing data scenarios.",
    "task_heads.md": "# Task Heads\n**Input:** Central Physiological Embedding `(B, D)`\n**Output:** Continuous Regressions (HR, RR).\n**Rationale:** Keeps the core foundation model entirely agnostic to the downstream task."
}

for filename, content in docs.items():
    with open(os.path.join(docs_dir, filename), "w") as f:
        f.write(content)

# Core Python files scaffold
py_files = [
    os.path.join(src_models_dir, "spatial", "spatial_encoder.py"),
    os.path.join(src_models_dir, "spatial", "swin_v2.py"),
    os.path.join(src_models_dir, "temporal", "temporal_encoder.py"),
    os.path.join(src_models_dir, "temporal", "timesformer.py"),
    os.path.join(src_models_dir, "fusion", "fusion_layer.py"),
    os.path.join(src_models_dir, "heads", "task_heads.py"),
    os.path.join(src_models_dir, "physio_fm.py")
]

for py_file in py_files:
    if not os.path.exists(py_file):
        with open(py_file, "w") as f:
            f.write("# Scaffolded PhysioFM Module\n")

print("Successfully scaffolded PhysioFM architecture and documentation.")
