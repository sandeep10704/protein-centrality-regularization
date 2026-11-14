"""
Script: evaluate_checkpoint.py
Evaluates a saved PyTorch model checkpoint on the test set of the ENZYMES benchmark.
Computes Accuracy, Macro-F1, Weighted-F1, Precision, Recall, and Confusion Matrix.
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_enzymes_dataset, load_dataset_splits, get_dataloaders
from models import ProteinGCN, ProteinGAT, ProteinGATv2, ProteinResidualGATv2
from evaluation.metrics import compute_classification_metrics, measure_inference_latency


MODEL_CLASSES = {
    "gcn": ProteinGCN,
    "gat": ProteinGAT,
    "gatv2": ProteinGATv2,
    "uniform_dropedge": ProteinGATv2,
    "aga_degree": ProteinGATv2,
    "bio_aga": ProteinGATv2,
    "bio_aga_resgatv2": ProteinResidualGATv2,
    "resgatv2": ProteinResidualGATv2,
}


def evaluate(
    checkpoint_path: str,
    model_type: str = "bio_aga_resgatv2",
    dataset_root: str = "data/raw",
    splits_path: str = "data/splits/enzymes_splits.json",
    seed: int = 42,
    fold: int = 0,
    batch_size: int = 32,
    device_name: str = "cpu"
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(device_name)
    print(f"Loading dataset and split (Seed {seed}, Fold {fold})...")
    dataset = load_enzymes_dataset(root=dataset_root, use_node_attr=True)
    all_splits = load_dataset_splits(splits_path)
    split = all_splits[str(seed)][fold]

    _, _, test_loader = get_dataloaders(dataset, split, batch_size=batch_size)

    model_cls = MODEL_CLASSES.get(model_type, ProteinResidualGATv2)
    model = model_cls(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_dim=64,
        heads=4,
        num_layers=3
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch=batch.batch)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch.y.squeeze().cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    metrics = compute_classification_metrics(y_true, y_pred)
    latency = measure_inference_latency(model, test_loader, device)

    print("\n" + "=" * 50)
    print("  CHECKPOINT EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Model Type:        {model_type}")
    print(f"  Checkpoint:        {checkpoint_path}")
    print(f"  Test Graphs:       {len(y_true)}")
    print(f"  Accuracy:          {metrics['accuracy'] * 100.0:.2f}%")
    print(f"  Macro-F1 Score:    {metrics['macro_f1'] * 100.0:.2f}%")
    print(f"  Weighted-F1:       {metrics['weighted_f1'] * 100.0:.2f}%")
    print(f"  Macro Precision:   {metrics['macro_precision'] * 100.0:.2f}%")
    print(f"  Macro Recall:      {metrics['macro_recall'] * 100.0:.2f}%")
    print(f"  Inference Latency: {latency:.3f} ms/graph")
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    conf_mat = np.array(metrics['confusion_matrix'])
    print("     EC1  EC2  EC3  EC4  EC5  EC6")
    for idx, row in enumerate(conf_mat):
        print(f"EC{idx+1}: " + "  ".join(f"{val:3d}" for val in row))
    print("=" * 50)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint on the ENZYMES test set.")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/bio_aga_resgatv2_seed42.pt",
                        help="Path to .pt checkpoint file")
    parser.add_argument("--model", type=str, default="bio_aga_resgatv2", choices=list(MODEL_CLASSES.keys()),
                        help="Model architecture type")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for stratified split")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu', 'cuda')")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        model_type=args.model,
        seed=args.seed,
        fold=args.fold,
        device_name=args.device
    )
