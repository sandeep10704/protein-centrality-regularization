"""
Module: datasets/protein_dataset.py
Manages Track A (ENZYMES) and Track B (PDB) datasets, caching structural centralities, and creating DataLoaders.
"""

import os
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from centrality import compute_batched_degree_centrality, compute_centrality


def load_enzymes_dataset(root: str = "data/raw", use_node_attr: bool = True):
    """
    Loads the TUDataset ENZYMES benchmark dataset.
    600 enzyme structures across 6 Enzyme Commission top-level classes.
    """
    dataset = TUDataset(root=root, name="ENZYMES", use_node_attr=use_node_attr)
    return dataset


def precompute_dataset_centralities(dataset, metric: str = "degree"):
    """
    Precomputes and attaches normalized node centrality to each graph in the dataset to eliminate online overhead.
    """
    cached_dataset = []
    print(f"Precomputing {metric} centrality for {len(dataset)} protein graphs...")
    for data in dataset:
        data_copy = data.clone()
        c = compute_centrality(data_copy.edge_index, data_copy.num_nodes, batch=None, metric=metric)
        setattr(data_copy, f"centrality_{metric}", c)
        cached_dataset.append(data_copy)
    return cached_dataset


def get_dataloaders(dataset, split: dict, batch_size: int = 32, num_workers: int = 0):
    """
    Constructs PyTorch Geometric DataLoaders for train, validation, and test splits.
    """
    train_data = [dataset[i] for i in split["train"]]
    val_data = [dataset[i] for i in split["val"]]
    test_data = [dataset[i] for i in split["test"]]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
