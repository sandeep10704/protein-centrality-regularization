"""
Module: experiments/run_centrality_sweep.py
Experiment B: Controlled Comparison of Topological Centrality Measures for Protein Graphs:
1. Degree Centrality: Local connectivity and residue contact density
2. Betweenness Centrality: Bottleneck residues channeling allosteric signaling paths
3. Closeness Centrality: Global topological proximity to the entire tertiary structure
4. Eigenvector Centrality: Residue coupling to well-connected structural hubs
5. Random Centrality Control: Uniform stochastic importance (ablation baseline)

Measures accuracy, Macro-F1, and computational preprocessing overhead across 5 seeds.
"""

import os
import json
import time
import argparse
import numpy as np
import torch

from datasets import load_enzymes_dataset, load_dataset_splits, get_dataloaders
from models import ProteinGATv2
from centrality import compute_centrality
from augmentations.aga_protein import bio_aga_edge_drop
from training.trainer import ProteinTrainer


def benchmark_centrality_computation(dataset, metrics):
    """
    Measures the CPU wall-clock computation time required to evaluate each centrality metric
    across the entire protein dataset.
    """
    timings = {}
    sample_size = min(100, len(dataset))
    print(f"\nBenchmarking centrality computation overhead on {sample_size} protein graphs...")

    for metric in metrics:
        if metric == "random":
            timings[metric] = 0.01
            continue

        t0 = time.perf_counter()
        for i in range(sample_size):
            data = dataset[i]
            _ = compute_centrality(data.edge_index, data.num_nodes, batch=None, metric=metric)
        t1 = time.perf_counter()

        ms_per_graph = ((t1 - t0) / sample_size) * 1000.0
        timings[metric] = round(ms_per_graph, 3)
        print(f"  {metric.capitalize():12s}: {ms_per_graph:.3f} ms / graph")

    return timings


def run_experiment_b(
    dataset_root: str = "data/raw",
    splits_path: str = "data/splits/enzymes_splits.json",
    results_dir: str = "results/metrics",
    epochs: int = 100,
    patience: int = 25,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
    device_name: str = None
):
    os.makedirs(results_dir, exist_ok=True)

    if device_name is not None:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Running Experiment B (Centrality Sweep) on device: {device}")

    dataset = load_enzymes_dataset(root=dataset_root, use_node_attr=True)
    all_splits = load_dataset_splits(splits_path)

    seeds = [42, 1337, 2024, 7, 99]
    metrics = ["degree", "betweenness", "closeness", "eigenvector", "random"]

    timings = benchmark_centrality_computation(dataset, metrics)

    results = {
        "metadata": {
            "title": "Experiment B: Topological Centrality Metric Sweep",
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": lr,
            "hidden_dim": hidden_dim,
            "heads": heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "seeds": seeds,
            "preprocessing_overhead_ms_per_graph": timings,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        },
        "centralities": {}
    }

    for metric in metrics:
        print(f"\n==========================================")
        print(f"Evaluating Centrality: {metric.upper()}")
        print(f"==========================================")

        seed_accuracies = []
        seed_macro_f1s = []
        seed_weighted_f1s = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            split_0 = all_splits[str(seed)][0]
            train_loader, val_loader, test_loader = get_dataloaders(dataset, split_0, batch_size=batch_size)

            model = ProteinGATv2(
                num_features=dataset.num_features,
                num_classes=dataset.num_classes,
                hidden_dim=hidden_dim,
                heads=heads,
                num_layers=num_layers,
                dropout=dropout
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Regularizer using specific centrality metric
            reg_fn = lambda b, m=metric: bio_aga_edge_drop(
                b, drop_prob=0.20, gamma=0.35, centrality_metric=m, protect_covalent=True
            )

            trainer = ProteinTrainer(
                model=model,
                optimizer=optimizer,
                device=device,
                regularizer_fn=reg_fn,
                checkpoint_dir="results/checkpoints"
            )

            chk_name = f"centrality_{metric}_seed{seed}.pt"
            run_res = trainer.fit(
                train_loader,
                val_loader,
                test_loader,
                epochs=epochs,
                patience=patience,
                model_name=chk_name
            )

            t_metrics = run_res["test_metrics"]
            acc = t_metrics["accuracy"] * 100.0
            mf1 = t_metrics["macro_f1"] * 100.0
            wf1 = t_metrics["weighted_f1"] * 100.0

            seed_accuracies.append(acc)
            seed_macro_f1s.append(mf1)
            seed_weighted_f1s.append(wf1)

            print(f"  Seed {seed:4d} -> Acc: {acc:.2f}%, Macro-F1: {mf1:.2f}%")

        mean_acc = float(np.mean(seed_accuracies))
        std_acc = float(np.std(seed_accuracies))
        mean_mf1 = float(np.mean(seed_macro_f1s))
        std_mf1 = float(np.std(seed_macro_f1s))

        results["centralities"][metric] = {
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "macro_f1_mean": mean_mf1,
            "macro_f1_std": std_mf1,
            "runs_accuracy": seed_accuracies,
            "runs_macro_f1": seed_macro_f1s,
            "preprocessing_overhead_ms": timings.get(metric, 0.0)
        }

        print(f"-> {metric.upper()}: Accuracy = {mean_acc:.2f}% +/- {std_acc:.2f}%, Macro-F1 = {mean_mf1:.2f}% +/- {std_mf1:.2f}%")

    out_file = os.path.join(results_dir, "exp_b_centralities.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nExperiment B completed successfully. Results saved to {out_file}")

    # Update experiment status
    status_file = "results/experiment_status.json"
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status_data = json.load(f)
        status_data["experiments"]["exp_b_centralities"] = {
            "status": "COMPLETED",
            "metrics": {k: {"accuracy": v["accuracy_mean"], "macro_f1": v["macro_f1_mean"]} for k, v in results["centralities"].items()}
        }
        status_data["completed_experiments"].append("exp_b_centralities")
        with open(status_file, "w") as f:
            json.dump(status_data, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_experiment_b(epochs=args.epochs, patience=args.patience, device_name=args.device)
