"""
Unit tests for batched graph centralities: Degree, Betweenness, Closeness, Eigenvector.
Tests per-graph normalization and disconnected/trivial edge cases.
"""

import pytest
import torch
from torch_geometric.data import Data, Batch
from centrality.degree import compute_batched_degree_centrality
from centrality.betweenness import compute_batched_betweenness_centrality
from centrality.closeness import compute_batched_closeness_centrality
from centrality.eigenvector import compute_batched_eigenvector_centrality


def test_degree_centrality_batch_isolation():
    """
    Critical Test: Ensure maximum degree normalization is computed STRICTLY per graph,
    NOT globally across the concatenated minibatch.
    """
    # Graph 1: Star graph with 4 nodes, hub has degree 3, leaves have degree 1
    # max_deg_g1 = 3 -> hub = 1.0, leaves = 1/3
    edge_index1 = torch.tensor([[0, 0, 0, 1, 2, 3],
                                [1, 2, 3, 0, 0, 0]], dtype=torch.long)
    g1 = Data(edge_index=edge_index1, num_nodes=4)

    # Graph 2: Complete graph with 5 nodes, all nodes have degree 4
    # max_deg_g2 = 4 -> all nodes = 1.0
    u, v = [], []
    for i in range(5):
        for j in range(5):
            if i != j:
                u.append(i)
                v.append(j)
    edge_index2 = torch.tensor([u, v], dtype=torch.long)
    g2 = Data(edge_index=edge_index2, num_nodes=5)

    batch_obj = Batch.from_data_list([g1, g2])

    c_deg = compute_batched_degree_centrality(batch_obj.edge_index, batch_obj.num_nodes, batch=batch_obj.batch)

    assert c_deg.shape == (9,)
    # In graph 1 (nodes 0..3): node 0 is hub -> 1.0; leaves -> 1/3 ~ 0.333
    assert torch.isclose(c_deg[0], torch.tensor(1.0), atol=1e-3)
    assert torch.isclose(c_deg[1], torch.tensor(1.0 / 3.0), atol=1e-3)

    # In graph 2 (nodes 4..8): all nodes must be 1.0 (since max degree in graph 2 is 4)
    # If it were wrongly normalized across the whole batch, it would be 4/4 = 1.0 as well,
    # but let's check a leaf from g1: if g2's max (4) leaked into g1, leaf would be 1/4 = 0.25!
    assert not torch.isclose(c_deg[1], torch.tensor(0.25), atol=1e-3)
    assert torch.all(c_deg[4:9] == 1.0)


def test_betweenness_centrality():
    # Path graph 0 - 1 - 2 - 3 - 4
    # Central node 2 should have highest betweenness
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    c_betw = compute_batched_betweenness_centrality(edge_index, num_nodes=5)
    assert c_betw.shape == (5,)
    assert c_betw[2] == 1.0
    assert c_betw[0] < c_betw[2]
    assert c_betw[4] < c_betw[2]


def test_closeness_and_eigenvector_centralities():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                                [1, 0, 2, 1, 0, 2]], dtype=torch.long) # Triangle
    c_close = compute_batched_closeness_centrality(edge_index, num_nodes=3)
    c_eigen = compute_batched_eigenvector_centrality(edge_index, num_nodes=3)

    assert torch.all(c_close > 0.0)
    assert torch.all(c_eigen > 0.0)
    assert torch.all(c_close <= 1.0)
    assert torch.all(c_eigen <= 1.0)


def test_disconnected_edge_case():
    # 2 isolated nodes
    edge_index = torch.empty((2, 0), dtype=torch.long)
    c_deg = compute_batched_degree_centrality(edge_index, num_nodes=2)
    c_betw = compute_batched_betweenness_centrality(edge_index, num_nodes=2)
    assert c_deg.shape == (2,)
    assert c_betw.shape == (2,)
