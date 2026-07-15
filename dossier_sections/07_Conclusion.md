# 7. Conclusion

This paper presents a systematic diagnostic analysis of optimization bottlenecks in hierarchical physiological transformers. While the final model exhibits mode collapse, we successfully isolated and resolved the critical "Gradient Starvation" bottleneck using Pre-Fusion Layer Normalization. Our findings provide a clear roadmap for the community: future dual-branch rPPG models must strictly normalize latent distributions before fusion, and rely on scale-invariant losses (like CCC) to prevent objective function collapse.
