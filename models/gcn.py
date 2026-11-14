"""
Module: models/gcn.py
Baseline Graph Convolutional Network (GCN) for protein graph classification (Kipf & Welling, ICLR 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.pooling import ProteinPooling


class ProteinGCN(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        pooling: str = "mean_max"
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Intermediate layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.pool = ProteinPooling(hidden_dim, mode=pooling)

        self.classifier = nn.Sequential(
            nn.Linear(self.pool.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        graph_emb = self.pool(h, batch)
        logits = self.classifier(graph_emb)
        return logits
