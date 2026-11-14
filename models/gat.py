"""
Module: models/gat.py
Standard Graph Attention Network (GAT) for protein graph classification (Veličković et al., ICLR 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from models.pooling import ProteinPooling


class ProteinGAT(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        pooling: str = "mean_max",
        edge_dim: int = None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Layer 1
        self.convs.append(GATConv(num_features, hidden_dim // heads, heads=heads, dropout=dropout, edge_dim=edge_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Intermediate layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.pool = ProteinPooling(hidden_dim, mode=pooling)

        self.classifier = nn.Sequential(
            nn.Linear(self.pool.output_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        for i in range(self.num_layers):
            if edge_attr is not None:
                h = self.convs[i](h, edge_index, edge_attr=edge_attr)
            else:
                h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        graph_emb = self.pool(h, batch)
        logits = self.classifier(graph_emb)
        return logits
