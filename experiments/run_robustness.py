"""
Module: experiments/run_robustness.py
Experiment F: Rigorous Robustness and Structural Noise Stress-Testing.
Evaluates resilience of Vanilla GATv2, Uniform DropEdge, Original AGA-Degree, Proposed Bio-AGA, and Bio-AGA + ResGATv2
under:
1. Random edge additions (spurious / adversarial false positive contacts from 0% to 40%).
2. Random edge deletions (missing contacts from 0% to 30%).
3. Long-range contact perturbation (non-local tertiary contact corruption).
"""

import os
import json
import time
import argparse
import numpy as np
import torch

from datasets import load_enzymes_dataset, load_dataset_splits, get_dataloaders
from models import ProteinGATv2, ProteinResidualGATv2
from augmentations import (
    uniform_edge_drop,
    aga_degree_edge_drop,
    bio_aga_edge_drop,
    inject_synthetic_edge_addition,
    inject_synthetic_edge_deletion,
    inject_long_range_contact_perturbation
)
from training.trainer import ProteinTrainer


def run_experiment_f(
    dataset_root: str = "data/raw",
    splits_path: str = "data/splits/enzymes_splits.json",
    results_dir: str = "results/metrics",
    checkpoints_dir: str = "results/checkpoints",
    device_name: str = None
):
    os.makedirs(results_dir, exist_ok=True)
    if device_name is not None:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Running Experiment F (Robustness Stress-Test) on device: {device}")

    dataset = load_enzymes_dataset(root=dataset_root, use_node_attr=True)
    all_splits = load_dataset_splits(splits_path)

    seeds = [42, 1337, 2024]
    noise_levels = [0.0, 0.10, 0.20, 0.30, 0.40]

    models_to_test = [
        {"key": "vanilla", "name": "Vanilla GATv2 (No Defense)", "model_cls": ProteinGATv2, "reg": None, "res": False},
        {"key": "uniform", "name": "Uniform DropEdge Baseline", "model_cls": ProteinGATv2,
         "reg": lambda b: uniform_edge_drop(b, drop_prob=0.20), "res": False},
        {"key": "aga_degree", "name": "Original AGA-Degree", "model_cls": ProteinGATv2,
         "reg": lambda b: aga_degree_edge_drop(b, drop_prob=0.20, gamma=0.35), "res": False},
        {"key": "bio_aga", "name": "Proposed Bio-AGA", "model_cls": ProteinGATv2,
         "reg": lambda b: bio_aga_edge_drop(b, drop_prob=0.20, gamma=0.35, protect_covalent=True), "res": False},
        {"key": "bio_aga_res", "name": "Bio-AGA + ResGATv2", "model_cls": ProteinResidualGATv2,
         "reg": lambda b: bio_aga_edge_drop(b, drop_prob=0.20, gamma=0.35, protect_covalent=True), "res": True},
    ]

    robustness_results = {
        "metadata": {
            "title": "Experiment F: Robustness Stress-Testing Under Structural Noise",
            "noise_levels": noise_levels,
            "seeds": seeds,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        },
        "edge_addition_noise": {},
        "edge_deletion_noise": {},
        "long_range_perturbation": {}
    }

    # First, train models on clean data (or load pre-trained checkpoints if available)
    trained_models = {}
    print("\n--- Training / Loading Models on Benchmark Splits ---")
    for m_cfg in models_to_test:
        m_key = m_cfg["key"]
        trained_models[m_key] = []

        for s in seeds:
            torch.manual_seed(s)
            split_0 = all_splits[str(s)][0]
            train_loader, val_loader, test_loader = get_dataloaders(dataset, split_0, batch_size=32)

            if m_cfg["res"]:
                model = ProteinResidualGATv2(num_features=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=64, heads=4)
            else:
                model = ProteinGATv2(num_features=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=64, heads=4)

            chk_path = os.path.join(checkpoints_dir, f"robustness_{m_key}_s{s}.pt")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            trainer = ProteinTrainer(model=model, optimizer=optimizer, device=device, regularizer_fn=m_cfg["reg"])

            res = trainer.fit(train_loader, val_loader, test_loader, epochs=70, patience=20, model_name=f"robustness_{m_key}_s{s}.pt")
            trained_models[m_key].append(trainer)

    # 1. Stress Test: Edge Addition Noise (0% to 40%)
    print("\n--- Stress Test 1: Spurious Edge Additions (0% to 40%) ---")
    for m_cfg in models_to_test:
        m_key = m_cfg["key"]
        robustness_results["edge_addition_noise"][m_key] = {"name": m_cfg["name"], "trajectory": {}}

        for nl in noise_levels:
            accs = []
            for s_idx, s in enumerate(seeds):
                split_0 = all_splits[str(s)][0]
                _, _, test_loader = get_dataloaders(dataset, split_0, batch_size=32)
                trainer = trained_models[m_key][s_idx]

                perturb_fn = lambda b, noise=nl: inject_synthetic_edge_addition(b, noise_ratio=noise, seed=s)
                eval_metrics = trainer.evaluate(test_loader, perturb_fn=perturb_fn)
                accs.append(eval_metrics["accuracy"] * 100.0)

            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            robustness_results["edge_addition_noise"][m_key]["trajectory"][str(nl)] = {
                "noise_ratio": nl,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc
            }
            print(f"  {m_cfg['name'][:22]:22s} | Noise: {nl*100:2.0f}% -> Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%")

    # 2. Stress Test: Edge Deletion Noise (0% to 30%)
    print("\n--- Stress Test 2: Edge Deletion (Missing Contacts 0% to 30%) ---")
    del_levels = [0.0, 0.10, 0.20, 0.30]
    for m_cfg in models_to_test:
        m_key = m_cfg["key"]
        robustness_results["edge_deletion_noise"][m_key] = {"name": m_cfg["name"], "trajectory": {}}

        for dl in del_levels:
            accs = []
            for s_idx, s in enumerate(seeds):
                split_0 = all_splits[str(s)][0]
                _, _, test_loader = get_dataloaders(dataset, split_0, batch_size=32)
                trainer = trained_models[m_key][s_idx]

                perturb_fn = lambda b, drop=dl: inject_synthetic_edge_deletion(b, drop_ratio=drop, seed=s)
                eval_metrics = trainer.evaluate(test_loader, perturb_fn=perturb_fn)
                accs.append(eval_metrics["accuracy"] * 100.0)

            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            robustness_results["edge_deletion_noise"][m_key]["trajectory"][str(dl)] = {
                "drop_ratio": dl,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc
            }
            print(f"  {m_cfg['name'][:22]:22s} | Deletion: {dl*100:2.0f}% -> Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%")

    out_file = os.path.join(results_dir, "exp_f_robustness.json")
    with open(out_file, "w") as f:
        json.dump(robustness_results, f, indent=2)
    print(f"\nExperiment F completed. Results saved to {out_file}")

    return robustness_results


if __name__ == "__main__":
    run_experiment_f()
