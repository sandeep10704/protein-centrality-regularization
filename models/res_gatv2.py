"""
Module: models/res_gatv2.py
Noise-resilient Residual GATv2 for protein structural graphs.
Equipped with linear residual skip connections (highways) to preserve residue-specific identity signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from models.pooling import ProteinPooling


class ProteinResidualGATv2(nn.Module):
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
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.edge_dim = edge_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.residuals = nn.ModuleList()

        # Layer 1
        self.convs.append(GATv2Conv(num_features, hidden_dim // heads, heads=heads, dropout=dropout, edge_dim=edge_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.residuals.append(nn.Linear(num_features, hidden_dim))

        # Intermediate layers
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.residuals.append(nn.Linear(hidden_dim, hidden_dim))

        self.pool = ProteinPooling(hidden_dim, mode=pooling)

        self.classifier = nn.Sequential(
            nn.Linear(self.pool.output_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch=None, edge_attr=None, return_attention=False):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        attentions = []

        for i in range(self.num_layers):
            h_in = h
            if return_attention and i == (self.num_layers - 1):
                h_conv, att = self.convs[i](h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                attentions.append(att)
            else:
                if edge_attr is not None:
                    h_conv = self.convs[i](h, edge_index, edge_attr=edge_attr)
                else:
                    h_conv = self.convs[i](h, edge_index)

            # Residual highway connection
            h = h_conv + self.residuals[i](h_in)
            h = self.bns[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        graph_emb = self.pool(h, batch)
        logits = self.classifier(graph_emb)

        if return_attention:
            return logits, attentions
        return logits
