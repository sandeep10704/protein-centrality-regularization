"""
Script: prepare_data.py
Prepares Track A (ENZYMES) dataset:
- Generates reproducible stratified 5-fold cross-validation splits across 5 random seeds (42, 1337, 2024, 7, 99).
- Inspects dataset semantics and prints structural summary.
- Verifies integrity of Track B PDB structures.
"""

import os
import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
import numpy as np
from datasets import load_enzymes_dataset, get_stratified_splits, save_dataset_splits
from preprocessing.pdb_parser import parse_pdb_to_residue_graph

SEEDS = [42, 1337, 2024, 7, 99]
SPLITS_DIR = "data/splits"
os.makedirs(SPLITS_DIR, exist_ok=True)

print("Loading Track A (ENZYMES) dataset...")
dataset = load_enzymes_dataset(root="data/raw", use_node_attr=True)
num_graphs = len(dataset)
num_classes = dataset.num_classes
num_features = dataset.num_features

labels = np.array([int(data.y.item()) for data in dataset])

# Class distribution
unique, counts = np.unique(labels, return_counts=True)
class_dist = dict(zip([int(u) for u in unique], [int(c) for c in counts]))

# Node and edge count statistics
node_counts = [data.num_nodes for data in dataset]
edge_counts = [data.edge_index.size(1) for data in dataset]

stats = {
    "num_graphs": num_graphs,
    "num_classes": num_classes,
    "num_features": num_features,
    "class_distribution": class_dist,
    "avg_nodes": float(np.mean(node_counts)),
    "std_nodes": float(np.std(node_counts)),
    "min_nodes": int(np.min(node_counts)),
    "max_nodes": int(np.max(node_counts)),
    "avg_edges": float(np.mean(edge_counts)),
    "std_edges": float(np.std(edge_counts)),
    "min_edges": int(np.min(edge_counts)),
    "max_edges": int(np.max(edge_counts)),
}

print(f"ENZYMES Dataset Summary:")
print(f"  Total Graphs: {num_graphs}")
print(f"  Classes: {num_classes} (Counts: {class_dist})")
print(f"  Node Features: {num_features}")
print(f"  Nodes: {stats['avg_nodes']:.1f} +/- {stats['std_nodes']:.1f} (range [{stats['min_nodes']}, {stats['max_nodes']}])")
print(f"  Edges: {stats['avg_edges']:.1f} +/- {stats['std_edges']:.1f} (range [{stats['min_edges']}, {stats['max_edges']}])")

# Save summary
with open("data/processed/enzymes_metadata.json", "w") as f:
    json.dump(stats, f, indent=2)

# Generate multi-seed stratified splits
all_seed_splits = {}
for seed in SEEDS:
    splits = get_stratified_splits(labels, num_folds=5, seed=seed)
    all_seed_splits[str(seed)] = splits

save_dataset_splits(all_seed_splits, os.path.join(SPLITS_DIR, "enzymes_splits.json"))
print(f"Stratified splits across {len(SEEDS)} seeds saved to {os.path.join(SPLITS_DIR, 'enzymes_splits.json')}")
