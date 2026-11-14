"""
Module: datasets/splits.py
Generates reproducible, stratified K-fold cross-validation and multi-seed splits for protein graphs.
Strictly eliminates data leakage between train, validation, and test subsets.
"""

import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def get_stratified_splits(labels: np.ndarray, num_folds: int = 5, seed: int = 42):
    """
    Returns list of (train_indices, val_indices, test_indices) tuples.
    Uses 5-fold cross-validation with an 80/10/10 or standard train/val/test split.
    """
    labels = np.asarray(labels)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = []

    indices = np.arange(len(labels))

    for train_val_idx, test_idx in skf.split(indices, labels):
        # From train_val, split off 10% for validation
        train_val_labels = labels[train_val_idx]
        train_sub, val_sub = train_test_split(
            train_val_idx,
            test_size=0.125, # 0.125 of 80% = 10% of total
            stratify=train_val_labels,
            random_state=seed
        )
        splits.append({
            "train": train_sub.tolist(),
            "val": val_sub.tolist(),
            "test": test_idx.tolist()
        })

    return splits


def save_dataset_splits(splits_dict: dict, save_path: str = "data/splits/splits.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(splits_dict, f, indent=2)


def load_dataset_splits(load_path: str = "data/splits/splits.json"):
    with open(load_path, "r") as f:
        return json.load(f)
