"""
Module: centrality/closeness.py
Computes per-graph normalized closeness centrality for protein graphs.
Closeness reflects the topological proximity of a residue to the rest of the protein fold.
"""

import torch
import networkx as nx
import numpy as np


def compute_batched_closeness_centrality(edge_index: torch.Tensor, num_nodes: int, batch: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes per-graph normalized closeness centrality for batched protein graphs.

    Args:
        edge_index (torch.Tensor): [2, num_edges] edge index tensor.
        num_nodes (int): Total number of nodes in the batch.
        batch (torch.Tensor, optional): [num_nodes] batch assignment vector.
        eps (float): Numerical stability epsilon.

    Returns:
        torch.Tensor: [num_nodes] normalized closeness centrality in [0, 1].
    """
    if num_nodes == 0:
        return torch.empty(0, device=edge_index.device)

    device = edge_index.device
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    num_graphs = int(batch.max().item()) + 1 if num_nodes > 0 else 0
    c_close = torch.zeros(num_nodes, dtype=torch.float, device=device)

    edge_index_cpu = edge_index.detach().cpu().numpy()
    batch_cpu = batch.detach().cpu().numpy()

    for g in range(num_graphs):
        node_indices = np.where(batch_cpu == g)[0]
        if len(node_indices) == 0:
            continue

        if len(node_indices) <= 2:
            c_close[node_indices] = 1.0
            continue

        local_map = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}
        mask = np.isin(edge_index_cpu[0], node_indices) & np.isin(edge_index_cpu[1], node_indices)
        g_edges_src = [local_map[u] for u in edge_index_cpu[0][mask]]
        g_edges_dst = [local_map[v] for v in edge_index_cpu[1][mask]]

        G = nx.Graph()
        G.add_nodes_from(range(len(node_indices)))
        G.add_edges_from(zip(g_edges_src, g_edges_dst))

        try:
            close_dict = nx.closeness_centrality(G)
            close_vals = np.array([close_dict[i] for i in range(len(node_indices))], dtype=np.float32)
            c_max = close_vals.max()
            if c_max > 0:
                close_vals = close_vals / (c_max + eps)
            c_close[node_indices] = torch.from_numpy(close_vals).to(device)
        except Exception:
            c_close[node_indices] = 0.5

    return torch.clamp(c_close, min=0.0, max=1.0)
