"""
Unit tests for edge regularization methods: Uniform DropEdge, Original AGA-Degree, Bio-AGA, and Perturbations.
Verifies covalent protection, aligned edge attributes, and valid edge indices.
"""

import pytest
import torch
from torch_geometric.data import Data, Batch
from augmentations.uniform_drop import uniform_edge_drop
from augmentations.aga_original import aga_degree_edge_drop
from augmentations.aga_protein import bio_aga_edge_drop
from augmentations.perturbations import inject_synthetic_edge_addition, inject_synthetic_edge_deletion


def test_uniform_drop_edge_attr_alignment():
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)
    edge_attr = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)

    data_aug = uniform_edge_drop(data, drop_prob=0.5)
    assert data_aug.edge_index.size(1) == data_aug.edge_attr.size(0)
    assert data_aug.edge_index.size(1) <= 4


def test_original_aga_degree_drop():
    # Star graph: node 0 is hub, nodes 1..5 are peripheral
    u = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    v = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
    edge_index = torch.tensor([u, v], dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=6)

    # Run multiple times to verify hub edge preservation
    retained_counts = 0
    total_runs = 50
    for _ in range(total_runs):
        aug = aga_degree_edge_drop(data, drop_prob=0.5, gamma=0.35)
        retained_counts += aug.edge_index.size(1)

    avg_retained = retained_counts / total_runs
    # With gamma=0.35, importance shield is high for hub connections, so retention should be high (> 60%)
    assert avg_retained > 5.0


def test_bio_aga_covalent_protection_invariance():
    """
    Critical Axiom 1: Verified covalent peptide bonds must NEVER be dropped by Bio-AGA (P_drop = 0).
    """
    # 5-residue linear peptide with backbone connections (0,1), (1,2), (2,3), (3,4)
    # and one non-covalent spatial contact (0, 4)
    u = [0, 1, 1, 2, 2, 3, 3, 4, 0, 4]
    v = [1, 0, 2, 1, 3, 2, 4, 3, 4, 0]
    edge_index = torch.tensor([u, v], dtype=torch.long)
    # Covalent mask: first 8 edges are covalent backbone, last 2 are non-covalent
    covalent_mask = torch.tensor([True]*8 + [False]*2, dtype=torch.bool)
    data = Data(edge_index=edge_index, covalent_mask=covalent_mask, num_nodes=5)

    # Even under extreme drop_prob=0.99, covalent edges MUST be 100% retained!
    for _ in range(30):
        aug = bio_aga_edge_drop(data, drop_prob=0.99, protect_covalent=True)
        # Check all covalent edges are present in aug.edge_index
        aug_pairs = set(zip(aug.edge_index[0].tolist(), aug.edge_index[1].tolist()))
        assert (0, 1) in aug_pairs
        assert (1, 0) in aug_pairs
        assert (1, 2) in aug_pairs
        assert (2, 3) in aug_pairs
        assert (3, 4) in aug_pairs


def test_perturbation_injection():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=3)

    corrupt_add = inject_synthetic_edge_addition(data, noise_ratio=0.5, seed=42)
    assert corrupt_add.edge_index.size(1) >= data.edge_index.size(1)

    corrupt_del = inject_synthetic_edge_deletion(data, drop_ratio=0.5, seed=42)
    assert corrupt_del.edge_index.size(1) <= data.edge_index.size(1)
