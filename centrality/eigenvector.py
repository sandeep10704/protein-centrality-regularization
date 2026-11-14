"""
Module: centrality/eigenvector.py
Computes per-graph normalized eigenvector centrality for protein graphs.
Eigenvector centrality captures residues connected to high-degree structural clusters.
"""

import torch
import networkx as nx
import numpy as np


def compute_batched_eigenvector_centrality(edge_index: torch.Tensor, num_nodes: int, batch: torch.Tensor = None, eps: float = 1e-8, max_iter: int = 500) -> torch.Tensor:
    """
    Computes per-graph normalized eigenvector centrality for batched protein graphs.

    Args:
        edge_index (torch.Tensor): [2, num_edges] edge index tensor.
        num_nodes (int): Total number of nodes in the batch.
        batch (torch.Tensor, optional): [num_nodes] batch assignment vector.
        eps (float): Numerical stability epsilon.
        max_iter (int): Power iteration limit.

    Returns:
        torch.Tensor: [num_nodes] normalized eigenvector centrality in [0, 1].
    """
    if num_nodes == 0:
        return torch.empty(0, device=edge_index.device)

    device = edge_index.device
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    num_graphs = int(batch.max().item()) + 1 if num_nodes > 0 else 0
    c_eigen = torch.zeros(num_nodes, dtype=torch.float, device=device)

    edge_index_cpu = edge_index.detach().cpu().numpy()
    batch_cpu = batch.detach().cpu().numpy()

    for g in range(num_graphs):
        node_indices = np.where(batch_cpu == g)[0]
        if len(node_indices) == 0:
            continue

        if len(node_indices) <= 2:
            c_eigen[node_indices] = 1.0
            continue

        local_map = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}
        mask = np.isin(edge_index_cpu[0], node_indices) & np.isin(edge_index_cpu[1], node_indices)
        g_edges_src = [local_map[u] for u in edge_index_cpu[0][mask]]
        g_edges_dst = [local_map[v] for v in edge_index_cpu[1][mask]]

        G = nx.Graph()
        G.add_nodes_from(range(len(node_indices)))
        G.add_edges_from(zip(g_edges_src, g_edges_dst))

        try:
            eigen_dict = nx.eigenvector_centrality(G, max_iter=max_iter, tol=1e-4)
            eigen_vals = np.array([eigen_dict[i] for i in range(len(node_indices))], dtype=np.float32)
            e_max = eigen_vals.max()
            if e_max > 0:
                eigen_vals = eigen_vals / (e_max + eps)
            c_eigen[node_indices] = torch.from_numpy(eigen_vals).to(device)
        except Exception:
            # If power iteration does not converge, fallback to degree centrality
            from torch_geometric.utils import degree
            local_deg = np.array([d for _, d in G.degree()], dtype=np.float32)
            d_max = local_deg.max()
            if d_max > 0:
                local_deg = local_deg / (d_max + eps)
            c_eigen[node_indices] = torch.from_numpy(local_deg).to(device)

    return torch.clamp(c_eigen, min=0.0, max=1.0)
