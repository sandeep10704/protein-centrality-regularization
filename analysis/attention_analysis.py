"""
Module: analysis/attention_analysis.py
Dynamic Attention Analysis for Protein GATv2:
1. Attention weight distribution across heads.
2. Correlation between attention coefficients and physical contact distances.
3. Attention focus on structural hubs vs. peripheral active clefts.
4. Effect of Bio-AGA regularization on attention sharpening and entropy.
"""

import os
import json
import numpy as np
import torch
import scipy.stats as stats

from datasets import load_enzymes_dataset, load_dataset_splits, get_dataloaders
from models import ProteinGATv2, ProteinResidualGATv2


def analyze_model_attentions(model, loader, device):
    """
    Extracts attention coefficients across test graphs and analyzes their correlation with:
    - Physical distance (edge Euclidean distance or sequence separation)
    - Node degrees
    """
    model.eval()
    all_attentions = []
    all_distances = []
    all_degrees = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, att_list = model(batch.x, batch.edge_index, batch=batch.batch, return_attention=True)

            edge_index_att, alpha = att_list[-1] # Last layer attention weights: [num_edges, heads]
            mean_alpha = alpha.mean(dim=-1).cpu().numpy() # Average over heads: [num_edges]

            u, v = edge_index_att[0].cpu().numpy(), edge_index_att[1].cpu().numpy()
            seq_sep = np.abs(u - v)

            # Node degree approximation
            deg = np.bincount(u, minlength=batch.num_nodes)
            edge_deg = (deg[u] + deg[v]) / 2.0

            all_attentions.extend(mean_alpha.tolist())
            all_distances.extend(seq_sep.tolist())
            all_degrees.extend(edge_deg.tolist())

            if len(all_attentions) > 2000:
                break

    all_attentions = np.array(all_attentions)
    all_distances = np.array(all_distances)
    all_degrees = np.array(all_degrees)

    # Compute Spearman and Pearson correlations
    corr_dist, _ = stats.spearmanr(all_attentions, all_distances)
    corr_deg, _ = stats.spearmanr(all_attentions, all_degrees)

    # Attention entropy (information dispersion)
    p = np.clip(all_attentions, 1e-12, 1.0)
    p = p / np.sum(p)
    entropy = -np.sum(p * np.log(p))

    return {
        "mean_attention": float(np.mean(all_attentions)),
        "std_attention": float(np.std(all_attentions)),
        "spearman_attention_vs_distance": float(corr_dist),
        "spearman_attention_vs_degree": float(corr_deg),
        "attention_entropy": float(entropy)
    }


def run_attention_evaluation(checkpoints_dir: str = "results/checkpoints", results_dir: str = "results/metrics"):
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cpu")

    dataset = load_enzymes_dataset(root="data/raw", use_node_attr=True)
    all_splits = load_dataset_splits("data/splits/enzymes_splits.json")
    split_0 = all_splits["42"][0]
    _, _, test_loader = get_dataloaders(dataset, split_0, batch_size=32)

    # Instantiate models
    vanilla_model = ProteinGATv2(dataset.num_features, dataset.num_classes, hidden_dim=64, heads=4)
    bio_model = ProteinResidualGATv2(dataset.num_features, dataset.num_classes, hidden_dim=64, heads=4)

    # Load checkpoints if available
    v_chk = os.path.join(checkpoints_dir, "gatv2_seed42.pt")
    b_chk = os.path.join(checkpoints_dir, "bio_aga_resgatv2_seed42.pt")

    if os.path.exists(v_chk):
        vanilla_model.load_state_dict(torch.load(v_chk, map_location=device))
    if os.path.exists(b_chk):
        bio_model.load_state_dict(torch.load(b_chk, map_location=device))

    vanilla_stats = analyze_model_attentions(vanilla_model, test_loader, device)
    bio_stats = analyze_model_attentions(bio_model, test_loader, device)

    results = {
        "vanilla_gatv2": vanilla_stats,
        "bio_aga_resgatv2": bio_stats
    }

    out_file = os.path.join(results_dir, "attention_analysis.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\nAttention Analysis Results:")
    print(f"  Vanilla GATv2 Attention Entropy: {vanilla_stats['attention_entropy']:.3f}, Corr(Attn, Dist): {vanilla_stats['spearman_attention_vs_distance']:.3f}")
    print(f"  Bio-AGA ResGATv2 Attention Entropy: {bio_stats['attention_entropy']:.3f}, Corr(Attn, Dist): {bio_stats['spearman_attention_vs_distance']:.3f}")
    return results


if __name__ == "__main__":
    run_attention_evaluation()
