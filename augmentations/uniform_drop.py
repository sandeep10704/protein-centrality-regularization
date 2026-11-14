"""
Module: augmentations/uniform_drop.py
Independent baseline implementation of Uniform DropEdge (Rong et al., ICLR 2020) for protein graphs.
Supports PyG Data and Batch objects, preserving aligned edge attributes.
"""

import torch


def uniform_edge_drop(data, drop_prob: float = 0.20):
    """
    Uniformly drops edges with probability drop_prob across the entire graph.

    Args:
        data (torch_geometric.data.Data or Batch): Input protein graph or batch.
        drop_prob (float): Base drop probability. Default: 0.20.

    Returns:
        Augmented graph with uniformly sampled edges and aligned edge attributes.
    """
    edge_index = data.edge_index
    num_edges = edge_index.size(1)

    if drop_prob <= 0.0 or num_edges == 0:
        return data.clone()

    retain_prob = 1.0 - drop_prob
    mask = torch.rand(num_edges, device=edge_index.device) < retain_prob

    # Guarantee at least some edges remain to avoid empty message-passing collapses
    if mask.sum() == 0:
        mask[torch.randint(0, num_edges, (1,), device=edge_index.device)] = True

    data_aug = data.clone()
    data_aug.edge_index = edge_index[:, mask]

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data_aug.edge_attr = data.edge_attr[mask]

    return data_aug
