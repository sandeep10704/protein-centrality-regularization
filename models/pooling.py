"""
Module: models/pooling.py
Unified global graph pooling utilities for variable-sized protein graphs in minibatches.
Supports mean, max, combined mean+max, and attention-weighted pooling.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention


class ProteinPooling(nn.Module):
    """
    Graph pooling module aggregating residue/SSE representations into graph-level embeddings.
    """
    def __init__(self, hidden_dim: int, mode: str = "mean_max"):
        super().__init__()
        self.mode = mode.lower()
        self.hidden_dim = hidden_dim

        if self.mode == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.att_pool = GlobalAttention(gate_nn=gate_nn)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Pools node embeddings x into graph embeddings based on batch vector.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.mode == "mean":
            return global_mean_pool(x, batch)
        elif self.mode == "max":
            return global_max_pool(x, batch)
        elif self.mode == "mean_max":
            mean_p = global_mean_pool(x, batch)
            max_p = global_max_pool(x, batch)
            return torch.cat([mean_p, max_p], dim=-1)
        elif self.mode == "attention":
            return self.att_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}. Choose from 'mean', 'max', 'mean_max', 'attention'.")

    @property
    def output_dim(self) -> int:
        if self.mode == "mean_max":
            return self.hidden_dim * 2
        return self.hidden_dim
