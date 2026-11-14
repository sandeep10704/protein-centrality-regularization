"""
Module: augmentations/aga_original.py
Method A: Original AGA-Degree Regularization independently adapted for batched protein graphs.
Calculates per-graph normalized degree centrality and applies the non-linear protective shield (gamma=0.35).
"""

import torch
from centrality.degree import compute_batched_degree_centrality


def aga_degree_edge_drop(data, drop_prob: float = 0.20, gamma: float = 0.35, eps: float = 1e-8):
    """
    Original Anti-Graph-Attack (AGA) Degree Centrality Regularization on batched protein graphs.

    Mathematical formulation:
        C(v) = d(v) / (max_{w in V_g} d(w) + eps)
        I(u, v) = (C(u) + C(v)) / 2.0
        S(u, v) = clamp(I(u, v), eps, 1.0)^gamma
        P_drop(u, v) = drop_prob * (1.0 - S(u, v))
        P_retain(u, v) = 1.0 - P_drop(u, v)
        mask(u, v) ~ Bernoulli(P_retain(u, v))

    Args:
        data (torch_geometric.data.Data or Batch): Input protein graph or batch.
        drop_prob (float): Maximum base drop probability for peripheral edges. Default: 0.20.
        gamma (float): Protective shield concavity exponent. Default: 0.35.
        eps (float): Numerical stability epsilon.

    Returns:
        torch_geometric.data.Data or Batch: Augmented graph with centrality-protected edges.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_edges = edge_index.size(1)

    if drop_prob <= 0.0 or num_edges == 0:
        return data.clone()

    batch = data.batch if hasattr(data, "batch") and data.batch is not None else None

    # 1. Batch-aware degree centrality calculation (normalized strictly per protein graph)
    c_deg = compute_batched_degree_centrality(edge_index, num_nodes, batch=batch, eps=eps)

    # 2. Edge importance calculation
    u, v = edge_index[0], edge_index[1]
    importance = (c_deg[u] + c_deg[v]) / 2.0

    # 3. Non-linear protective shield
    shield = torch.pow(torch.clamp(importance, min=eps, max=1.0), gamma)

    # 4. Adaptive drop and retention probabilities
    p_drop = drop_prob * (1.0 - shield)
    p_retain = 1.0 - p_drop

    # 5. Stochastic retention mask
    mask = torch.rand(num_edges, device=edge_index.device) < p_retain

    # Guarantee at least some edges remain to avoid empty message-passing collapses
    if mask.sum() == 0:
        mask[torch.randint(0, num_edges, (1,), device=edge_index.device)] = True

    data_aug = data.clone()
    data_aug.edge_index = edge_index[:, mask]

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data_aug.edge_attr = data.edge_attr[mask]

    return data_aug
