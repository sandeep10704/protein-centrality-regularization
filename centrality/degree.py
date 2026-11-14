"""
Module: centrality/degree.py
Computes batch-aware normalized node degree centrality for protein graphs.
Strictly calculates maximum degree per individual graph to prevent minibatch scale distortion.
"""

import torch
from torch_geometric.utils import degree, scatter


def compute_batched_degree_centrality(edge_index: torch.Tensor, num_nodes: int, batch: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes per-graph normalized degree centrality for batched protein graphs.

    Args:
        edge_index (torch.Tensor): [2, num_edges] edge index tensor.
        num_nodes (int): Total number of nodes in the batch.
        batch (torch.Tensor, optional): [num_nodes] batch assignment vector.
        eps (float): Numerical stability epsilon.

    Returns:
        torch.Tensor: [num_nodes] normalized degree centrality in [0, 1].
    """
    if num_nodes == 0:
        return torch.empty(0, device=edge_index.device)

    # 1. Compute raw degree per node (in-degree / total degree for undirected)
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)

    # 2. If no batch vector is provided, normalize over the single graph
    if batch is None:
        deg_max = deg.max()
        return deg / (deg_max + eps) if deg_max > 0 else deg

    # 3. Compute maximum degree per graph g in the batch
    # scatter with reduce='max' finds max degree for each batch index
    max_deg_per_graph = scatter(deg, batch, dim=0, reduce='max')  # shape: [num_graphs]
    
    # Broadcast per-graph max back to each individual node
    node_max_deg = max_deg_per_graph[batch]  # shape: [num_nodes]

    # Normalize
    deg_norm = deg / (node_max_deg + eps)
    return torch.clamp(deg_norm, min=0.0, max=1.0)
