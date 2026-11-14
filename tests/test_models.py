"""
Unit tests for model architectures: GCN, GAT, GATv2, and Residual GATv2.
Verifies batched variable-sized graph processing, forward pass, backward gradient propagation, and attention weights.
"""

import pytest
import torch
from torch_geometric.data import Data, Batch
from models import ProteinGCN, ProteinGAT, ProteinGATv2, ProteinResidualGATv2


def create_mock_batch():
    # Graph 1: 4 nodes, 8 edges
    x1 = torch.randn(4, 21)
    e1 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
    y1 = torch.tensor([0], dtype=torch.long)
    g1 = Data(x=x1, edge_index=e1, y=y1)

    # Graph 2: 6 nodes, 10 edges
    x2 = torch.randn(6, 21)
    e2 = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0, 1, 2, 3, 4]], dtype=torch.long)
    y2 = torch.tensor([3], dtype=torch.long)
    g2 = Data(x=x2, edge_index=e2, y=y2)

    return Batch.from_data_list([g1, g2])


def test_models_forward_backward():
    batch = create_mock_batch()
    num_classes = 6
    num_features = 21

    models = [
        ProteinGCN(num_features=num_features, num_classes=num_classes, hidden_dim=32, num_layers=2),
        ProteinGAT(num_features=num_features, num_classes=num_classes, hidden_dim=32, heads=2, num_layers=2),
        ProteinGATv2(num_features=num_features, num_classes=num_classes, hidden_dim=32, heads=2, num_layers=2),
        ProteinResidualGATv2(num_features=num_features, num_classes=num_classes, hidden_dim=32, heads=2, num_layers=2)
    ]

    for model in models:
        model.train()
        logits = model(batch.x, batch.edge_index, batch=batch.batch)

        assert logits.shape == (2, num_classes), f"{model.__class__.__name__} output shape mismatch"

        loss = torch.nn.functional.cross_entropy(logits, batch.y)
        loss.backward()

        # Check all gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name} in {model.__class__.__name__}"


def test_gatv2_attention_weights():
    batch = create_mock_batch()
    model = ProteinGATv2(num_features=21, num_classes=6, hidden_dim=32, heads=2, num_layers=2)
    model.eval()

    logits, attentions = model(batch.x, batch.edge_index, batch=batch.batch, return_attention=True)
    assert logits.shape == (2, 6)
    assert len(attentions) == 1
    edge_index_att, alpha = attentions[0]
    assert alpha.dim() == 2 # [num_edges, heads]
    assert alpha.size(1) == 2
