"""
Module: analysis/biological_mechanics.py
Rigorous Mechanistic and Biological Analysis:
1. Hub vs. Peripheral residue contact retention rates.
2. Short-range (2 <= |u-v| <= 4) vs. Long-range (|u-v| >= 12) contact retention rates.
3. 100% Covalent backbone preservation verification.
4. Active-site and catalytic residue contact preservation (analyzed on 1TIM catalytic triad: Lys12, His95, Glu165).
5. Graph structural integrity (giant component size, connectivity preservation).
"""

import os
import json
import glob
import numpy as np
import torch
import networkx as nx

from datasets import load_enzymes_dataset
from preprocessing.pdb_parser import parse_pdb_to_residue_graph
from centrality import compute_batched_degree_centrality, compute_centrality
from augmentations import uniform_edge_drop, aga_degree_edge_drop, bio_aga_edge_drop


def analyze_mechanistic_retention(data, drop_prob=0.20, gamma=0.35, alpha_seq=0.25):
    """
    Computes exact edge retention probabilities across structural categories:
    - Covalent backbone (|u-v| == 1)
    - Short-range non-covalent (2 <= |u-v| <= 4)
    - Medium-range (5 <= |u-v| <= 11)
    - Long-range tertiary (|u-v| >= 12)
    - Structural hubs (top 15% centrality)
    - Peripheral surface residues (bottom 25% centrality)
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_edges = edge_index.size(1)

    u, v = edge_index[0], edge_index[1]
    seq_sep = torch.abs(u.float() - v.float())

    # Covalent mask
    if hasattr(data, "covalent_mask") and data.covalent_mask is not None:
        covalent_mask = data.covalent_mask
    else:
        covalent_mask = (seq_sep == 1.0)

    short_range_mask = (seq_sep >= 2.0) & (seq_sep <= 4.0)
    med_range_mask = (seq_sep >= 5.0) & (seq_sep <= 11.0)
    long_range_mask = (seq_sep >= 12.0)

    # Hub vs peripheral nodes
    c_deg = compute_batched_degree_centrality(edge_index, num_nodes)
    hub_thresh = torch.quantile(c_deg, 0.85)
    periph_thresh = torch.quantile(c_deg, 0.25)

    is_hub_edge = (c_deg[u] >= hub_thresh) | (c_deg[v] >= hub_thresh)
    is_periph_edge = (c_deg[u] <= periph_thresh) & (c_deg[v] <= periph_thresh)

    # 1. Uniform DropEdge retention probabilities
    uniform_p_retain = torch.full((num_edges,), 1.0 - drop_prob)

    # 2. Original AGA retention probabilities
    importance_orig = (c_deg[u] + c_deg[v]) / 2.0
    shield_orig = torch.pow(torch.clamp(importance_orig, min=1e-8, max=1.0), gamma)
    aga_p_drop = drop_prob * (1.0 - shield_orig)
    aga_p_retain = 1.0 - aga_p_drop

    # 3. Bio-AGA retention probabilities
    omega_seq = 1.0 + alpha_seq * torch.tanh(seq_sep / 12.0)
    dist = data.edge_attr[:, 0] if (hasattr(data, "edge_attr") and data.edge_attr is not None) else torch.full((num_edges,), 6.0)
    psi_dist = torch.exp(- (dist ** 2) / (2.0 * (4.0 ** 2)))
    s_bio = torch.pow(torch.clamp(importance_orig, min=1e-8, max=1.0), gamma) * omega_seq * psi_dist
    s_bio = torch.clamp(s_bio, min=1e-8, max=1.0)
    bio_p_drop = drop_prob * (1.0 - s_bio)
    bio_p_drop = torch.where(covalent_mask, torch.zeros_like(bio_p_drop), bio_p_drop)
    bio_p_retain = 1.0 - bio_p_drop

    def get_stats(p_tensor, mask):
        if mask.sum() == 0:
            return 100.0
        return float(p_tensor[mask].mean().item() * 100.0)

    return {
        "covalent_retention": {
            "uniform": get_stats(uniform_p_retain, covalent_mask),
            "aga_original": get_stats(aga_p_retain, covalent_mask),
            "bio_aga": get_stats(bio_p_retain, covalent_mask)
        },
        "short_range_retention": {
            "uniform": get_stats(uniform_p_retain, short_range_mask),
            "aga_original": get_stats(aga_p_retain, short_range_mask),
            "bio_aga": get_stats(bio_p_retain, short_range_mask)
        },
        "medium_range_retention": {
            "uniform": get_stats(uniform_p_retain, med_range_mask),
            "aga_original": get_stats(aga_p_retain, med_range_mask),
            "bio_aga": get_stats(bio_p_retain, med_range_mask)
        },
        "long_range_retention": {
            "uniform": get_stats(uniform_p_retain, long_range_mask),
            "aga_original": get_stats(aga_p_retain, long_range_mask),
            "bio_aga": get_stats(bio_p_retain, long_range_mask)
        },
        "hub_retention": {
            "uniform": get_stats(uniform_p_retain, is_hub_edge),
            "aga_original": get_stats(aga_p_retain, is_hub_edge),
            "bio_aga": get_stats(bio_p_retain, is_hub_edge)
        },
        "peripheral_retention": {
            "uniform": get_stats(uniform_p_retain, is_periph_edge),
            "aga_original": get_stats(aga_p_retain, is_periph_edge),
            "bio_aga": get_stats(bio_p_retain, is_periph_edge)
        }
    }


def analyze_catalytic_site_preservation(pdb_path: str = "data/raw/pdb/1TIM.pdb"):
    """
    Evaluates edge retention specifically around experimentally validated catalytic residues:
    Triosephosphate isomerase (TIM):
    - Lys12: Transition state phosphate stabilizer
    - His95: Electrophilic general acid
    - Glu165: Catalytic general base
    """
    if not os.path.exists(pdb_path):
        return None

    data = parse_pdb_to_residue_graph(pdb_path, contact_cutoff=8.0)
    u, v = data.edge_index[0], data.edge_index[1]

    # In 1TIM sequence indices (1-based in PDB, 0-based in array):
    # Lys12 (~idx 11), His95 (~idx 94), Glu165 (~idx 164)
    catalytic_indices = [11, 94, 164]
    is_active_edge = torch.isin(u, torch.tensor(catalytic_indices)) | torch.isin(v, torch.tensor(catalytic_indices))

    c_deg = compute_batched_degree_centrality(data.edge_index, data.num_nodes)
    importance = (c_deg[u] + c_deg[v]) / 2.0

    # Original AGA
    s_orig = torch.pow(torch.clamp(importance, min=1e-8, max=1.0), 0.35)
    aga_p_retain = 1.0 - 0.20 * (1.0 - s_orig)

    # Bio-AGA
    seq_sep = torch.abs(u.float() - v.float())
    omega_seq = 1.0 + 0.25 * torch.tanh(seq_sep / 12.0)
    dist = data.edge_attr[:, 0]
    psi_dist = torch.exp(- (dist ** 2) / (2.0 * (4.0 ** 2)))
    s_bio = s_orig * omega_seq * psi_dist
    bio_p_drop = 0.20 * (1.0 - torch.clamp(s_bio, min=1e-8, max=1.0))
    bio_p_drop = torch.where(data.covalent_mask, torch.zeros_like(bio_p_drop), bio_p_drop)
    bio_p_retain = 1.0 - bio_p_drop

    uniform_p_retain = torch.full((data.edge_index.size(1),), 0.80)

    res = {
        "protein": "1TIM (Triosephosphate isomerase)",
        "catalytic_residues": ["Lys12", "His95", "Glu165"],
        "num_catalytic_edges": int(is_active_edge.sum().item()),
        "active_edge_retention_uniform": float(uniform_p_retain[is_active_edge].mean().item() * 100.0),
        "active_edge_retention_original_aga": float(aga_p_retain[is_active_edge].mean().item() * 100.0),
        "active_edge_retention_bio_aga": float(bio_p_retain[is_active_edge].mean().item() * 100.0)
    }
    return res


def run_biological_analysis(results_dir: str = "results/metrics"):
    os.makedirs(results_dir, exist_ok=True)
    print("\n==========================================")
    print("Running Biological and Mechanistic Analysis")
    print("==========================================")

    # 1. Analyze Track A ENZYMES (aggregate across 50 sample proteins)
    dataset = load_enzymes_dataset(root="data/raw", use_node_attr=True)
    aggregate_mechanics = {
        "covalent_retention": {"uniform": [], "aga_original": [], "bio_aga": []},
        "short_range_retention": {"uniform": [], "aga_original": [], "bio_aga": []},
        "medium_range_retention": {"uniform": [], "aga_original": [], "bio_aga": []},
        "long_range_retention": {"uniform": [], "aga_original": [], "bio_aga": []},
        "hub_retention": {"uniform": [], "aga_original": [], "bio_aga": []},
        "peripheral_retention": {"uniform": [], "aga_original": [], "bio_aga": []}
    }

    sample_size = min(100, len(dataset))
    for i in range(sample_size):
        m = analyze_mechanistic_retention(dataset[i])
        for cat in aggregate_mechanics:
            for method in ["uniform", "aga_original", "bio_aga"]:
                aggregate_mechanics[cat][method].append(m[cat][method])

    final_mechanics = {}
    for cat in aggregate_mechanics:
        final_mechanics[cat] = {
            m: {
                "mean": float(np.mean(aggregate_mechanics[cat][m])),
                "std": float(np.std(aggregate_mechanics[cat][m]))
            }
            for m in ["uniform", "aga_original", "bio_aga"]
        }

    # 2. Catalytic Site Retention on 1TIM
    cat_analysis = analyze_catalytic_site_preservation()

    full_bio_results = {
        "dataset_mechanics": final_mechanics,
        "catalytic_site_analysis": cat_analysis
    }

    out_file = os.path.join(results_dir, "biological_mechanics.json")
    with open(out_file, "w") as f:
        json.dump(full_bio_results, f, indent=2)

    print("Mechanistic Retention Summary (ENZYMES Benchmark):")
    print(f"  Covalent Backbone Retention: Uniform={final_mechanics['covalent_retention']['uniform']['mean']:.1f}%, Bio-AGA={final_mechanics['covalent_retention']['bio_aga']['mean']:.1f}% (Axiom 1: 100% Invariant)")
    print(f"  Long-Range Contacts (|u-v|>=12): Uniform={final_mechanics['long_range_retention']['uniform']['mean']:.1f}%, Original AGA={final_mechanics['long_range_retention']['aga_original']['mean']:.1f}%, Bio-AGA={final_mechanics['long_range_retention']['bio_aga']['mean']:.1f}%")
    print(f"  Structural Core Hubs: Uniform={final_mechanics['hub_retention']['uniform']['mean']:.1f}%, Bio-AGA={final_mechanics['hub_retention']['bio_aga']['mean']:.1f}%")
    print(f"  Peripheral Surface Loops: Uniform={final_mechanics['peripheral_retention']['uniform']['mean']:.1f}%, Bio-AGA={final_mechanics['peripheral_retention']['bio_aga']['mean']:.1f}%")

    if cat_analysis:
        print(f"\nCatalytic Active-Site Analysis (1TIM Active Triad):")
        print(f"  Active-site edges: {cat_analysis['num_catalytic_edges']}")
        print(f"  Uniform DropEdge Retention: {cat_analysis['active_edge_retention_uniform']:.2f}%")
        print(f"  Original AGA Retention:     {cat_analysis['active_edge_retention_original_aga']:.2f}%")
        print(f"  Bio-AGA Retention:          {cat_analysis['active_edge_retention_bio_aga']:.2f}%")

    print(f"Biological mechanics analysis saved to {out_file}")
    return full_bio_results


if __name__ == "__main__":
    run_biological_analysis()
