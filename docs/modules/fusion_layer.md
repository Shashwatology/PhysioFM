# Fusion Layer
**Input:** Multimodal embeddings `(B, T, D)`
**Output:** `(B, T, D)` fused embedding.
**Mathematical Formulation:** Identity Pass-Through (currently), Cross-Attention (future).
**Rationale:** Decouples modalities cleanly for missing data scenarios.