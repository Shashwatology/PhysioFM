# Spatial Encoder Module
**Input:** `(B, T, C, H, W)`
**Output:** `(B, T, D)` spatial embedding
**Mathematical Formulation:** Swin V2 / CNN feature extraction.
**Complexity:** Highly dependent on resolution.
**Rationale:** Pluggable interface for robust feature extraction across modalities.