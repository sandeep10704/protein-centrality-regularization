"""
Centrality computation suite for protein structural graphs.
Provides batch-aware Degree, Betweenness, Closeness, and Eigenvector centralities.
"""

import torch
from .degree import compute_batched_degree_centrality
from .betweenness import compute_batched_betweenness_centrality
from .closeness import compute_batched_closeness_centrality
from .eigenvector import compute_batched_eigenvector_centrality


def compute_centrality(edge_index, num_nodes, batch=None, metric="degree"):
    """
    Unified dispatcher for graph centrality metrics on batched protein graphs.
    """
    metric = metric.lower()
    if metric == "degree":
        return compute_batched_degree_centrality(edge_index, num_nodes, batch)
    elif metric == "betweenness":
        return compute_batched_betweenness_centrality(edge_index, num_nodes, batch)
    elif metric == "closeness":
        return compute_batched_closeness_centrality(edge_index, num_nodes, batch)
    elif metric == "eigenvector":
        return compute_batched_eigenvector_centrality(edge_index, num_nodes, batch)
    elif metric == "random":
        # Random centrality baseline for controlled ablation
        return torch.rand(num_nodes, device=edge_index.device)
    else:
        raise ValueError(f"Unknown centrality metric '{metric}'. Expected degree, betweenness, closeness, eigenvector, random.")


__all__ = [
    "compute_batched_degree_centrality",
    "compute_batched_betweenness_centrality",
    "compute_batched_closeness_centrality",
    "compute_batched_eigenvector_centrality",
    "compute_centrality",
]
