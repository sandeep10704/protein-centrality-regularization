"""
Module: centrality/betweenness.py
Computes per-graph normalized betweenness centrality for protein graphs.
Betweenness identifies bottleneck residues critical for allosteric and inter-domain communication.
"""

import torch
import networkx as nx
import numpy as np


def compute_batched_betweenness_centrality(edge_index: torch.Tensor, num_nodes: int, batch: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes per-graph normalized betweenness centrality for batched protein graphs.

    Args:
        edge_index (torch.Tensor): [2, num_edges] edge index tensor.
        num_nodes (int): Total number of nodes in the batch.
        batch (torch.Tensor, optional): [num_nodes] batch assignment vector.
        eps (float): Numerical stability epsilon.

    Returns:
        torch.Tensor: [num_nodes] normalized betweenness centrality in [0, 1].
    """
    if num_nodes == 0:
        return torch.empty(0, device=edge_index.device)

    device = edge_index.device
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    num_graphs = int(batch.max().item()) + 1 if num_nodes > 0 else 0
    c_betw = torch.zeros(num_nodes, dtype=torch.float, device=device)

    edge_index_cpu = edge_index.detach().cpu().numpy()
    batch_cpu = batch.detach().cpu().numpy()

    for g in range(num_graphs):
        node_indices = np.where(batch_cpu == g)[0]
        if len(node_indices) == 0:
            continue

        if len(node_indices) <= 2:
            # Trivial graph: all nodes receive equal normalized centrality
            c_betw[node_indices] = 1.0
            continue

        # Map global node indices to local [0, N_g - 1]
        local_map = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}
        
        # Filter edges for graph g
        mask = np.isin(edge_index_cpu[0], node_indices) & np.isin(edge_index_cpu[1], node_indices)
        g_edges_src = [local_map[u] for u in edge_index_cpu[0][mask]]
        g_edges_dst = [local_map[v] for v in edge_index_cpu[1][mask]]

        G = nx.Graph()
        G.add_nodes_from(range(len(node_indices)))
        G.add_edges_from(zip(g_edges_src, g_edges_dst))

        # Compute betweenness
        try:
            betw_dict = nx.betweenness_centrality(G, normalized=True)
            betw_vals = np.array([betw_dict[i] for i in range(len(node_indices))], dtype=np.float32)
            b_max = betw_vals.max()
            if b_max > 0:
                betw_vals = betw_vals / (b_max + eps)
            c_betw[node_indices] = torch.from_numpy(betw_vals).to(device)
        except Exception:
            # Fallback to degree if computation encounters non-convergence
            c_betw[node_indices] = 0.5

    return torch.clamp(c_betw, min=0.0, max=1.0)
