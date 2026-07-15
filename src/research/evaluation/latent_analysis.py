import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score

def visualize_latent_space(latents: np.ndarray, hr_targets: np.ndarray, subject_ids: np.ndarray, exp_name: str):
    """
    Analyzes the (B, D) Latent Physiological Embedding using UMAP and t-SNE.
    """
    os.makedirs("experiments/figures", exist_ok=True)
    
    print(f"Running UMAP and t-SNE on {latents.shape[0]} embeddings of dim {latents.shape[1]}")
    
    # 1. UMAP
    reducer_umap = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    umap_emb = reducer_umap.fit_transform(latents)
    
    # 2. t-SNE
    # Perplexity must be less than number of samples
    perplexity = min(30, max(1, latents.shape[0] - 1))
    reducer_tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, metric='cosine')
    tsne_emb = reducer_tsne.fit_transform(latents)
    
    # Discretize HR for coloring
    hr_bins = [0, 70, 100, 200]
    hr_labels = ["Resting (<70)", "Elevated (70-100)", "High (>100)"]
    hr_categories = pd.cut(hr_targets, bins=hr_bins, labels=hr_labels)
    
    # Quantify Clustering (using HR categories as proxy for physiological state)
    # Mask out NaNs
    valid_mask = ~hr_categories.isna()
    if np.sum(valid_mask) > 1 and len(np.unique(hr_categories[valid_mask])) > 1:
        sil_score = silhouette_score(latents[valid_mask], hr_categories[valid_mask])
        db_score = davies_bouldin_score(latents[valid_mask], hr_categories[valid_mask])
        print(f"Silhouette Score (HR clustering): {sil_score:.4f}")
        print(f"Davies-Bouldin Index: {db_score:.4f}")
    
    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # UMAP by HR
    sns.scatterplot(x=umap_emb[:, 0], y=umap_emb[:, 1], hue=hr_targets, palette="coolwarm", ax=axes[0, 0])
    axes[0, 0].set_title("UMAP by Heart Rate")
    
    # UMAP by Subject
    sns.scatterplot(x=umap_emb[:, 0], y=umap_emb[:, 1], hue=subject_ids, palette="tab20", ax=axes[0, 1], legend=False)
    axes[0, 1].set_title("UMAP by Subject ID")
    
    # t-SNE by HR
    sns.scatterplot(x=tsne_emb[:, 0], y=tsne_emb[:, 1], hue=hr_targets, palette="coolwarm", ax=axes[1, 0])
    axes[1, 0].set_title("t-SNE by Heart Rate")
    
    # t-SNE by Subject
    sns.scatterplot(x=tsne_emb[:, 0], y=tsne_emb[:, 1], hue=subject_ids, palette="tab20", ax=axes[1, 1], legend=False)
    axes[1, 1].set_title("t-SNE by Subject ID")
    
    plt.tight_layout()
    save_path = f"experiments/figures/{exp_name}_latent_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"Latent analysis saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()
    
    latent_file = f"experiments/latent/{args.exp_name}_latents.pt"
    if not os.path.exists(latent_file):
        print(f"File {latent_file} not found.")
        exit(1)
        
    data = torch.load(latent_file)
    latents = data['latents'].numpy()
    hr_targets = data['hr_targets'].numpy().flatten()
    subject_ids = np.array(data['subject_ids'])
    
    visualize_latent_space(latents, hr_targets, subject_ids, args.exp_name)
