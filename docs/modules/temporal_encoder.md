# Temporal Encoder Module
**Input:** `(B, T, D)`
**Output:** `(B, D)` or `(B, T, D)` depending on sequence pooling.
**Mathematical Formulation:** Divided Space-Time Attention (TimeSformer).
**Complexity:** $O(T^2)$ for self-attention.
**Rationale:** Capturing subtle pulse frequency dynamics across the time dimension.