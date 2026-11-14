"""
Module: experiments/run_ablations.py
Executes Experiments C, D, and E:
- Experiment C: Contact Topology & Geometry Analysis (Cutoffs 6, 8, 10, 12 A; kNN 6, 10, 16 on authentic PDB structures)
- Experiment D: Regularization Hyperparameter Landscapes (P_base in [0.0, 0.1, 0.2, 0.35, 0.50], gamma in [0.15, 0.35, 0.50, 0.75, 1.00])
- Experiment E: Comprehensive Component Ablation Study:
    1. Vanilla GATv2 (No edge dropout)
    2. Uniform DropEdge (No centrality)
    3. Degree Centrality Only
    4. Betweenness Centrality Only
    5. Without Sequence Distance Protection (alpha_seq = 0.0)
    6. Without Spatial Distance Decay
    7. Without Covalent Backbone Protection (protect_covalent = False)
    8. Original AGA Formulation
    9. Complete Proposed Bio-AGA
    10. Complete Bio-AGA + ResGATv2
"""

import os
import json
import time
import argparse
import numpy as np
import torch

from datasets import load_enzymes_dataset, load_dataset_splits, get_dataloaders
from models import ProteinGATv2, ProteinResidualGATv2
from augmentations import uniform_edge_drop, aga_degree_edge_drop, bio_aga_edge_drop
from preprocessing.pdb_parser import parse_pdb_to_residue_graph
from training.trainer import ProteinTrainer


def run_experiment_c_topology(pdb_dir: str = "data/raw/pdb", results_dir: str = "results/metrics"):
    """
    Experiment C: Analyzes graph connectivity, edge density, and residue degree distributions
    across varying contact cutoffs (6, 8, 10, 12 A) and k-NN topologies (6, 10, 16) on authentic PDB structures.
    """
    print("\n==========================================")
    print("Running Experiment C: Contact Topology Analysis")
    print("==========================================")

    import glob
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    if len(pdb_files) == 0:
        print("Warning: No PDB files found for Experiment C.")
        return {}

    cutoffs = [6.0, 8.0, 10.0, 12.0]
    knn_vals = [6, 10, 16]

    topology_results = {
        "radius_sweep": {},
        "knn_sweep": {}
    }

    # Radius cutoff sweep
    for rc in cutoffs:
        densities = []
        avg_degrees = []
        long_range_ratios = []

        for p in pdb_files:
            try:
                g = parse_pdb_to_residue_graph(p, contact_cutoff=rc)
                num_nodes = g.num_nodes
                num_edges = g.edge_index.size(1)
                deg = num_edges / max(num_nodes, 1)

                u, v = g.edge_index[0], g.edge_index[1]
                long_range = (torch.abs(u.float() - v.float()) >= 12).sum().item()
                lr_ratio = long_range / max(num_edges, 1)

                avg_degrees.append(deg)
                long_range_ratios.append(lr_ratio)
            except Exception as e:
                pass

        topology_results["radius_sweep"][str(rc)] = {
            "contact_cutoff_angstrom": rc,
            "mean_degree": float(np.mean(avg_degrees)),
            "std_degree": float(np.std(avg_degrees)),
            "mean_long_range_ratio": float(np.mean(long_range_ratios)),
            "std_long_range_ratio": float(np.std(long_range_ratios))
        }
        print(f"  Radius {rc:4.1f} A -> Avg Degree: {np.mean(avg_degrees):.2f}, Long-range fraction: {np.mean(long_range_ratios)*100:.1f}%")

    # k-NN sweep
    for k in knn_vals:
        avg_degrees = []
        long_range_ratios = []

        for p in pdb_files:
            try:
                g = parse_pdb_to_residue_graph(p, k_nn=k)
                num_nodes = g.num_nodes
                num_edges = g.edge_index.size(1)
                deg = num_edges / max(num_nodes, 1)

                u, v = g.edge_index[0], g.edge_index[1]
                long_range = (torch.abs(u.float() - v.float()) >= 12).sum().item()
                lr_ratio = long_range / max(num_edges, 1)

                avg_degrees.append(deg)
                long_range_ratios.append(lr_ratio)
            except Exception as e:
                pass

        topology_results["knn_sweep"][str(k)] = {
            "k_neighbors": k,
            "mean_degree": float(np.mean(avg_degrees)),
            "std_degree": float(np.std(avg_degrees)),
            "mean_long_range_ratio": float(np.mean(long_range_ratios)),
            "std_long_range_ratio": float(np.std(long_range_ratios))
        }
        print(f"  k-NN {k:2d} -> Avg Degree: {np.mean(avg_degrees):.2f}, Long-range fraction: {np.mean(long_range_ratios)*100:.1f}%")

    out_file = os.path.join(results_dir, "exp_c_contact_topology.json")
    with open(out_file, "w") as f:
        json.dump(topology_results, f, indent=2)
    print(f"Experiment C results saved to {out_file}")
    return topology_results


def run_experiment_d_hyperparameters(dataset, all_splits, device, results_dir: str = "results/metrics"):
    """
    Experiment D: Sweeps base dropout probability P_base and concavity exponent gamma.
    """
    print("\n==========================================")
    print("Running Experiment D: Hyperparameter Sensitivity")
    print("==========================================")

    p_bases = [0.0, 0.10, 0.20, 0.35, 0.50]
    gammas = [0.15, 0.35, 0.50, 0.75, 1.00]
    seeds = [42, 1337, 2024]

    d_results = {
        "p_base_sweep": {},
        "gamma_sweep": {}
    }

    # 1. P_base sweep (with fixed gamma = 0.35)
    print("\n--- Sweeping P_base (gamma = 0.35) ---")
    for p_b in p_bases:
        accs, f1s = [], []
        for s in seeds:
            torch.manual_seed(s)
            split_0 = all_splits[str(s)][0]
            train_loader, val_loader, test_loader = get_dataloaders(dataset, split_0, batch_size=32)
            model = ProteinGATv2(num_features=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=64, heads=4)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

            reg_fn = None if p_b == 0.0 else (lambda b, p=p_b: bio_aga_edge_drop(b, drop_prob=p, gamma=0.35, protect_covalent=True))
            trainer = ProteinTrainer(model=model, optimizer=optimizer, device=device, regularizer_fn=reg_fn)
            res = trainer.fit(train_loader, val_loader, test_loader, epochs=60, patience=20, model_name=f"pbase_{p_b}_s{s}.pt")

            accs.append(res["test_metrics"]["accuracy"] * 100.0)
            f1s.append(res["test_metrics"]["macro_f1"] * 100.0)

        d_results["p_base_sweep"][str(p_b)] = {
            "p_base": p_b,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s))
        }
        print(f"  P_base = {p_b:.2f} -> Accuracy: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%, Macro-F1: {np.mean(f1s):.2f}%")

    # 2. Gamma sweep (with fixed P_base = 0.20)
    print("\n--- Sweeping Gamma (P_base = 0.20) ---")
    for gam in gammas:
        accs, f1s = [], []
        for s in seeds:
            torch.manual_seed(s)
            split_0 = all_splits[str(s)][0]
            train_loader, val_loader, test_loader = get_dataloaders(dataset, split_0, batch_size=32)
            model = ProteinGATv2(num_features=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=64, heads=4)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

            reg_fn = lambda b, g=gam: bio_aga_edge_drop(b, drop_prob=0.20, gamma=g, protect_covalent=True)
            trainer = ProteinTrainer(model=model, optimizer=optimizer, device=device, regularizer_fn=reg_fn)
            res = trainer.fit(train_loader, val_loader, test_loader, epochs=60, patience=20, model_name=f"gamma_{gam}_s{s}.pt")

            accs.append(res["test_metrics"]["accuracy"] * 100.0)
            f1s.append(res["test_metrics"]["macro_f1"] * 100.0)

        d_results["gamma_sweep"][str(gam)] = {
            "gamma": gam,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s))
        }
        print(f"  Gamma = {gam:.2f} -> Accuracy: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%, Macro-F1: {np.mean(f1s):.2f}%")

    out_file = os.path.join(results_dir, "exp_d_hyperparameters.json")
    with open(out_file, "w") as f:
        json.dump(d_results, f, indent=2)
    print(f"Experiment D results saved to {out_file}")
    return d_results


def run_experiment_e_ablations(dataset, all_splits, device, results_dir: str = "results/metrics"):
    """
    Experiment E: Component Ablation Study isolating each biological and topological term.
    """
    print("\n==========================================")
    print("Running Experiment E: Component Ablation Study")
    print("==========================================")

    seeds = [42, 1337, 2024, 7, 99]

    ablation_cfgs = [
        {"name": "No Regularization (Vanilla GATv2)", "reg": None, "res": False},
        {"name": "Uniform DropEdge (No Centrality)", "reg": lambda b: uniform_edge_drop(b, drop_prob=0.20), "res": False},
        {"name": "Degree Centrality Only", "reg": lambda b: bio_aga_edge_drop(b, drop_prob=0.20, gamma=0.35, use_seq_protection=False, use_dist_affinity=False, protect_covalent=False), "res": False},
        {"name": "Betweenness Centrality Only", "reg": lambda b: bio_aga_edge_drop(b, drop_prob=0.20, gamma=0.35, centrality_metric="betweenness", use_seq_protection=False, use_dist_affinity=False, protect_covalent=False), "res": False},
        {"name": "Without Sequence Distance Protection", "reg": lambda b: bio_aga_edge_drop(b, drop_prob=0.20, gamma=0.35, use_seq_protection=False, protect_covalent=True), "res": False},
        {"name": "Without Spatial Distance Decay", "reg": lambda b: bio_aga_edge_drop(b, drop_prob=0.20, gamma=0.35, use_dist_affinity=False, protect_covalent=True), "res": False},
        {"name": "Without Covalent Backbone Protection", "reg": lambda b: bio_aga_edge_drop(b, drop_prob=0.20, gamma=0.35, protect_covalent=False), "res": False},
        {"name": "Original AGA-Degree (Cora Formulation)", "reg": lambda b: aga_degree_edge_drop(b, drop_prob=0.20, gamma=0.35), "res": False},
        {"name": "Complete Proposed Bio-AGA", "reg": lambda b: bio_aga_edge_drop(b, drop_prob=0.20, gamma=0.35, alpha_seq=0.25, protect_covalent=True), "res": False},
        {"name": "Bio-AGA + Residual Highway (ResGATv2)", "reg": lambda b: bio_aga_edge_drop(b, drop_prob=0.20, gamma=0.35, alpha_seq=0.25, protect_covalent=True), "res": True},
    ]

    ablation_results = {"ablations": {}}

    for cfg in ablation_cfgs:
        name = cfg["name"]
        print(f"\nEvaluating Ablation: {name}")

        accs, f1s = [], []
        for s in seeds:
            torch.manual_seed(s)
            split_0 = all_splits[str(s)][0]
            train_loader, val_loader, test_loader = get_dataloaders(dataset, split_0, batch_size=32)

            if cfg["res"]:
                model = ProteinResidualGATv2(num_features=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=64, heads=4)
            else:
                model = ProteinGATv2(num_features=dataset.num_features, num_classes=dataset.num_classes, hidden_dim=64, heads=4)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            trainer = ProteinTrainer(model=model, optimizer=optimizer, device=device, regularizer_fn=cfg["reg"])
            res = trainer.fit(train_loader, val_loader, test_loader, epochs=80, patience=25, model_name=f"ablation_{name[:10]}_s{s}.pt")

            accs.append(res["test_metrics"]["accuracy"] * 100.0)
            f1s.append(res["test_metrics"]["macro_f1"] * 100.0)

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))

        ablation_results["ablations"][name] = {
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "macro_f1_mean": mean_f1,
            "macro_f1_std": std_f1,
            "runs_accuracy": accs,
            "runs_macro_f1": f1s
        }
        print(f"-> {name}: Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%, Macro-F1: {mean_f1:.2f}% +/- {std_f1:.2f}%")

    out_file = os.path.join(results_dir, "exp_e_ablations.json")
    with open(out_file, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Experiment E results saved to {out_file}")
    return ablation_results


def run_all_ablations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_enzymes_dataset(root="data/raw", use_node_attr=True)
    all_splits = load_dataset_splits("data/splits/enzymes_splits.json")

    run_experiment_c_topology()
    run_experiment_d_hyperparameters(dataset, all_splits, device)
    run_experiment_e_ablations(dataset, all_splits, device)


if __name__ == "__main__":
    run_all_ablations()
