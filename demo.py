"""
Script: demo.py
Final Model Demonstration Pipeline for College Course Project Evaluation.

Demonstrates end-to-end execution in ~10 seconds:
1. Dataset Loading (ENZYMES and authentic PDB structures).
2. Mini-batch Graph Construction (PyG Batch tensor).
3. Node and Edge Feature Extraction.
4. Batch-Aware Centrality Computation (Degree, Closeness, Eigenvector).
5. Original AGA-Degree Edge Regularization.
6. Proposed Bio-AGA Edge Regularization (Covalent immunity + Sequence protection).
7. Live GATv2 Training Pass (3 demonstration epochs).
8. Model Checkpoint Loading (Bio-AGA + ResGATv2).
9. Full Test Evaluation & Classification Metrics (Accuracy, Macro-F1, Confusion Matrix).
"""

import sys
import os
import time
import torch
import numpy as np

# Ensure local modules are found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_enzymes_dataset, load_dataset_splits, get_dataloaders
from preprocessing.pdb_parser import parse_pdb_to_residue_graph
from centrality import compute_centrality
from augmentations import aga_degree_edge_drop, bio_aga_edge_drop
from models import ProteinGATv2, ProteinResidualGATv2
from evaluation.metrics import compute_classification_metrics


def print_step(step_num: int, title: str):
    print("\n" + "=" * 70)
    print(f" STEP {step_num}: {title.upper()}")
    print("=" * 70)


def run_demo():
    print("\n" + "#" * 70)
    print("  DEEP LEARNING COURSE PROJECT DEMONSTRATION")
    print("  Centrality-Aware Edge Regularization for GATv2 on Protein Graphs")
    print("#" * 70)

    device = torch.device("cpu")

    # ---------------------------------------------------------
    # STEP 1: Dataset Loading
    # ---------------------------------------------------------
    print_step(1, "Loading Protein Graph Datasets")
    print("Loading Track A Benchmark: TUDataset ENZYMES...")
    enzymes = load_enzymes_dataset(root="data/raw", use_node_attr=True)
    print(f"-> Successfully loaded {len(enzymes)} enzyme tertiary structures across {enzymes.num_classes} EC classes.")

    pdb_sample_path = "data/raw/pdb/1TIM.pdb"
    if os.path.exists(pdb_sample_path):
        print(f"Loading Track B Authentic Structure: Triosephosphate Isomerase ({pdb_sample_path})...")
        pdb_graph = parse_pdb_to_residue_graph(pdb_sample_path, contact_cutoff=8.0)
        print(f"-> Parsed PDB Residue Graph: {pdb_graph.num_nodes} residues, {pdb_graph.edge_index.size(1)} contacts, {pdb_graph.covalent_mask.sum().item()} covalent bonds.")

    # ---------------------------------------------------------
    # STEP 2 & 3: Batch Construction & Feature Extraction
    # ---------------------------------------------------------
    print_step(2, "Constructing Graph Batch and Extracting Features")
    splits = load_dataset_splits("data/splits/enzymes_splits.json")
    split_0 = splits["42"][0]
    train_loader, val_loader, test_loader = get_dataloaders(enzymes, split_0, batch_size=4)

    batch = next(iter(train_loader))
    print(f"Constructed PyG Batch of {batch.num_graphs} protein graphs:")
    print(f"  Total Batch Nodes:       {batch.num_nodes}")
    print(f"  Total Batch Edges:       {batch.edge_index.size(1)}")
    print(f"  Node Feature Matrix (X): shape {list(batch.x.shape)} (21 continuous/one-hot features)")
    print(f"  Target Classes (Y):      {batch.y.squeeze().tolist()} (Enzyme Commission top-level labels)")
    print(f"  Batch Assignment Vector: first 15 nodes -> {batch.batch[:15].tolist()}...")

    # ---------------------------------------------------------
    # STEP 4: Centrality Computation
    # ---------------------------------------------------------
    print_step(3, "Batch-Aware Centrality Computation")
    print("Computing per-graph normalized centralities (strictly isolated per protein):")

    c_deg = compute_centrality(batch.edge_index, batch.num_nodes, batch=batch.batch, metric="degree")
    c_close = compute_centrality(batch.edge_index, batch.num_nodes, batch=batch.batch, metric="closeness")
    c_eigen = compute_centrality(batch.edge_index, batch.num_nodes, batch=batch.batch, metric="eigenvector")

    print(f"  Degree Centrality (min/mean/max):      {c_deg.min():.2f} / {c_deg.mean():.2f} / {c_deg.max():.2f}")
    print(f"  Closeness Centrality (min/mean/max):   {c_close.min():.2f} / {c_close.mean():.2f} / {c_close.max():.2f}")
    print(f"  Eigenvector Centrality (min/mean/max): {c_eigen.min():.2f} / {c_eigen.mean():.2f} / {c_eigen.max():.2f}")

    # ---------------------------------------------------------
    # STEP 5 & 6: Applying Edge Regularization
    # ---------------------------------------------------------
    print_step(4, "Applying Centrality-Aware Edge Regularization")
    print(f"Original Edge Count in Minibatch: {batch.edge_index.size(1)} edges")

    # Method A: Original AGA-Degree
    batch_aga = aga_degree_edge_drop(batch, drop_prob=0.25, gamma=0.35)
    print(f"-> Original AGA-Degree (P_base=0.25, gamma=0.35):  {batch_aga.edge_index.size(1)} edges retained ({batch_aga.edge_index.size(1)/batch.edge_index.size(1)*100:.1f}%)")

    # Method B: Proposed Bio-AGA
    batch_bio = bio_aga_edge_drop(batch, drop_prob=0.25, gamma=0.35, alpha_seq=0.25, protect_covalent=True)
    print(f"-> Proposed Bio-AGA (Covalent Immune + Seq Shield): {batch_bio.edge_index.size(1)} edges retained ({batch_bio.edge_index.size(1)/batch.edge_index.size(1)*100:.1f}%)")

    # ---------------------------------------------------------
    # STEP 7: Live Model Training Demonstration
    # ---------------------------------------------------------
    print_step(5, "Live GATv2 Training Pass (3 Quick Demonstration Epochs)")
    model = ProteinGATv2(num_features=enzymes.num_features, num_classes=enzymes.num_classes, hidden_dim=32, heads=2, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    demo_loader, _, _ = get_dataloaders(enzymes, split_0, batch_size=32)

    model.train()
    for epoch in range(1, 4):
        t0 = time.time()
        total_loss = 0.0
        for b in demo_loader:
            b_aug = bio_aga_edge_drop(b, drop_prob=0.20, gamma=0.35)
            optimizer.zero_grad()
            out = model(b_aug.x, b_aug.edge_index, batch=b_aug.batch)
            loss = criterion(out, b.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * b.num_graphs

        epoch_loss = total_loss / len(split_0["train"])
        dt = time.time() - t0
        print(f"  Epoch {epoch}/3 -> Cross-Entropy Loss: {epoch_loss:.4f}  (Time: {dt:.2f}s)")

    # ---------------------------------------------------------
    # STEP 8 & 9: Checkpoint Loading & Test Evaluation
    # ---------------------------------------------------------
    print_step(6, "Loading Best Checkpoint and Evaluating on Full Test Set")
    chk_path = "results/checkpoints/bio_aga_resgatv2_seed42.pt"

    eval_model = ProteinResidualGATv2(num_features=enzymes.num_features, num_classes=enzymes.num_classes, hidden_dim=64, heads=4, num_layers=3)

    if os.path.exists(chk_path):
        print(f"Loading trained weights from: {chk_path}")
        eval_model.load_state_dict(torch.load(chk_path, map_location=device))
    else:
        print("Trained checkpoint not found, evaluating live model.")
        eval_model = model

    eval_model.eval()
    _, _, test_eval_loader = get_dataloaders(enzymes, split_0, batch_size=32)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for b in test_eval_loader:
            out = eval_model(b.x, b.edge_index, batch=b.batch)
            preds = out.argmax(dim=-1)
            all_preds.extend(preds.tolist())
            all_targets.extend(b.y.squeeze().tolist())

    metrics = compute_classification_metrics(np.array(all_targets), np.array(all_preds))

    print("\n" + "*" * 50)
    print("  FINAL EVALUATION RESULTS ON TEST SET (120 PROTEINS)")
    print("*" * 50)
    print(f"  Accuracy:         {metrics['accuracy'] * 100.0:.2f}%")
    print(f"  Macro-F1 Score:   {metrics['macro_f1'] * 100.0:.2f}%")
    print(f"  Weighted-F1:      {metrics['weighted_f1'] * 100.0:.2f}%")
    print(f"  Macro Precision:  {metrics['macro_precision'] * 100.0:.2f}%")
    print(f"  Macro Recall:     {metrics['macro_recall'] * 100.0:.2f}%")
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    conf_mat = np.array(metrics['confusion_matrix'])
    print("     EC1  EC2  EC3  EC4  EC5  EC6")
    for idx, row in enumerate(conf_mat):
        print(f"EC{idx+1}: " + "  ".join(f"{val:3d}" for val in row))

    print("\n" + "#" * 70)
    print("  DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    run_demo()
