"""
Module: experiments/run_baselines.py
Experiment A: Comprehensive Baseline Evaluation across 7 Model Architectures:
1. Baseline 1: Protein GCN (Kipf & Welling, ICLR 2017)
2. Baseline 2: Standard Protein GAT (Veličković et al., ICLR 2018)
3. Baseline 3: Dynamic GATv2 (Brody et al., ICLR 2022)
4. Baseline 4: GATv2 + Uniform DropEdge (Rong et al., ICLR 2020)
5. Method A:   GATv2 + Original AGA-Degree (Direct Cora Formulation adaptation)
6. Method B:   GATv2 + Proposed Bio-AGA (Structurally & Geometrically Grounded)
7. Enhanced:   Bio-AGA + Residual GATv2 (Noise-Resilient Dynamic Highway Architecture)

Evaluated across 5 random seeds (42, 1337, 2024, 7, 99) using stratified splits.
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from datasets import load_enzymes_dataset, load_dataset_splits, get_dataloaders
from models import ProteinGCN, ProteinGAT, ProteinGATv2, ProteinResidualGATv2
from augmentations import uniform_edge_drop, aga_degree_edge_drop, bio_aga_edge_drop
from training.trainer import ProteinTrainer
from evaluation.metrics import measure_inference_latency


def run_experiment_a(
    dataset_root: str = "data/raw",
    splits_path: str = "data/splits/enzymes_splits.json",
    results_dir: str = "results/metrics",
    checkpoints_dir: str = "results/checkpoints",
    epochs: int = 100,
    patience: int = 25,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
    device_name: str = None,
    model_name: str = "all"
):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    if device_name is not None:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # On Apple Silicon CPU is 50x faster than MPS for small PyG graphs
        device = torch.device("cpu")

    print(f"Running Experiment A on device: {device}")

    dataset = load_enzymes_dataset(root=dataset_root, use_node_attr=True)
    all_splits = load_dataset_splits(splits_path)

    seeds = [42, 1337, 2024, 7, 99]

    # Models and regularizations configuration
    configs = [
        {"key": "gcn", "name": "GCN Baseline", "model_class": ProteinGCN, "reg_fn": None, "use_res": False},
        {"key": "gat", "name": "Standard GAT", "model_class": ProteinGAT, "reg_fn": None, "use_res": False},
        {"key": "gatv2", "name": "Dynamic GATv2", "model_class": ProteinGATv2, "reg_fn": None, "use_res": False},
        {"key": "uniform_dropedge", "name": "GATv2 + Uniform DropEdge", "model_class": ProteinGATv2,
         "reg_fn": lambda b: uniform_edge_drop(b, drop_prob=0.20), "use_res": False},
        {"key": "aga_degree", "name": "GATv2 + Original AGA-Degree", "model_class": ProteinGATv2,
         "reg_fn": lambda b: aga_degree_edge_drop(b, drop_prob=0.20, gamma=0.35), "use_res": False},
        {"key": "bio_aga", "name": "GATv2 + Proposed Bio-AGA", "model_class": ProteinGATv2,
         "reg_fn": lambda b: bio_aga_edge_drop(b, drop_prob=0.20, gamma=0.35, alpha_seq=0.25, protect_covalent=True), "use_res": False},
        {"key": "bio_aga_resgatv2", "name": "Proposed Bio-AGA + ResGATv2", "model_class": ProteinResidualGATv2,
         "reg_fn": lambda b: bio_aga_edge_drop(b, drop_prob=0.20, gamma=0.35, alpha_seq=0.25, protect_covalent=True), "use_res": True},
    ]

    if model_name != "all":
        selected = [c for c in configs if c["key"] == model_name]
        if not selected:
            valid_keys = [c["key"] for c in configs]
            raise ValueError(f"Unknown model key '{model_name}'. Valid options: {valid_keys}")
        configs = selected

    results = {
        "metadata": {
            "title": "Experiment A: Baseline Model Suite on ENZYMES",
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": lr,
            "hidden_dim": hidden_dim,
            "heads": heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "seeds": seeds,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        },
        "models": {}
    }

    for cfg in configs:
        model_key = cfg["key"]
        print(f"\n==========================================")
        print(f"Evaluating: {cfg['name']} ({model_key})")
        print(f"==========================================")

        seed_accuracies = []
        seed_macro_f1s = []
        seed_weighted_f1s = []
        seed_runtimes = []
        seed_latencies = []
        param_count = 0

        for seed in seeds:
            # Determinism
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Use fold 0 for each seed's stratified split
            split_0 = all_splits[str(seed)][0]
            train_loader, val_loader, test_loader = get_dataloaders(dataset, split_0, batch_size=batch_size)

            # Model instance
            if cfg["model_class"] == ProteinGCN:
                model = ProteinGCN(
                    num_features=dataset.num_features,
                    num_classes=dataset.num_classes,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )
            elif cfg["model_class"] == ProteinGAT:
                model = ProteinGAT(
                    num_features=dataset.num_features,
                    num_classes=dataset.num_classes,
                    hidden_dim=hidden_dim,
                    heads=heads,
                    num_layers=num_layers,
                    dropout=dropout
                )
            elif cfg["model_class"] == ProteinGATv2:
                model = ProteinGATv2(
                    num_features=dataset.num_features,
                    num_classes=dataset.num_classes,
                    hidden_dim=hidden_dim,
                    heads=heads,
                    num_layers=num_layers,
                    dropout=dropout
                )
            elif cfg["model_class"] == ProteinResidualGATv2:
                model = ProteinResidualGATv2(
                    num_features=dataset.num_features,
                    num_classes=dataset.num_classes,
                    hidden_dim=hidden_dim,
                    heads=heads,
                    num_layers=num_layers,
                    dropout=dropout
                )

            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            trainer = ProteinTrainer(
                model=model,
                optimizer=optimizer,
                device=device,
                regularizer_fn=cfg["reg_fn"],
                checkpoint_dir=checkpoints_dir
            )

            chk_name = f"{model_key}_seed{seed}.pt"
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
            runtime = t_metrics["training_time_seconds"]

            # Inference latency
            latency = measure_inference_latency(model, test_loader, device=device)

            seed_accuracies.append(acc)
            seed_macro_f1s.append(mf1)
            seed_weighted_f1s.append(wf1)
            seed_runtimes.append(runtime)
            seed_latencies.append(latency)

            print(f"  Seed {seed:4d} -> Acc: {acc:.2f}%, Macro-F1: {mf1:.2f}%, Time: {runtime:.1f}s, Latency: {latency:.2f}ms/graph")

        mean_acc = float(np.mean(seed_accuracies))
        std_acc = float(np.std(seed_accuracies))
        mean_mf1 = float(np.mean(seed_macro_f1s))
        std_mf1 = float(np.std(seed_macro_f1s))
        mean_wf1 = float(np.mean(seed_weighted_f1s))
        std_wf1 = float(np.std(seed_weighted_f1s))
        mean_time = float(np.mean(seed_runtimes))
        mean_latency = float(np.mean(seed_latencies))

        results["models"][model_key] = {
            "name": cfg["name"],
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "macro_f1_mean": mean_mf1,
            "macro_f1_std": std_mf1,
            "weighted_f1_mean": mean_wf1,
            "weighted_f1_std": std_wf1,
            "runs_accuracy": seed_accuracies,
            "runs_macro_f1": seed_macro_f1s,
            "num_parameters": param_count,
            "training_time_mean_seconds": mean_time,
            "inference_latency_ms": mean_latency
        }

        print(f"-> {cfg['name']}: Accuracy = {mean_acc:.2f}% +/- {std_acc:.2f}%, Macro-F1 = {mean_mf1:.2f}% +/- {std_mf1:.2f}%")

    out_file = os.path.join(results_dir, "exp_a_baselines.json")
    if os.path.exists(out_file) and model_name != "all":
        try:
            with open(out_file, "r") as f:
                existing_data = json.load(f)
            if "models" in existing_data:
                existing_data["models"].update(results["models"])
                results = existing_data
        except Exception:
            pass
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nExperiment A completed successfully. Results saved to {out_file}")

    # Update experiment status
    status_file = "results/experiment_status.json"
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status_data = json.load(f)
        status_data["experiments"]["exp_a_baselines"] = {
            "status": "COMPLETED",
            "metrics": {k: {"accuracy": v["accuracy_mean"], "macro_f1": v["macro_f1_mean"]} for k, v in results["models"].items()}
        }
        if "exp_a_baselines" not in status_data["completed_experiments"]:
            status_data["completed_experiments"].append("exp_a_baselines")
        with open(status_file, "w") as f:
            json.dump(status_data, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment A: Comprehensive Baseline Evaluation on ENZYMES")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs per run")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', etc.)")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "gatv2", "uniform_dropedge", "aga_degree", "bio_aga", "bio_aga_resgatv2", "gcn", "gat"],
                        help="Specific baseline model to run (default: 'all' for complete suite)")
    args = parser.parse_args()

    run_experiment_a(epochs=args.epochs, patience=args.patience, device_name=args.device, model_name=args.model)
