"""
Module: evaluation/metrics.py
Comprehensive evaluation metric suite for protein graph classification.
Computes Accuracy, Macro-F1, Weighted-F1, Precision, Recall, and Confusion Matrix.
"""

import time
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> dict:
    """
    Computes rigorous classification metrics across classes.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "confusion_matrix": conf_mat
    }
    return metrics


def measure_inference_latency(model, loader, device, num_warmup: int = 5, num_eval: int = 20):
    """
    Measures mean inference latency per protein graph in milliseconds.
    """
    model.eval()
    total_graphs = 0
    total_time = 0.0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            if i < num_warmup:
                _ = model(batch.x, batch.edge_index, batch=batch.batch)
                continue

            t0 = time.perf_counter()
            _ = model(batch.x, batch.edge_index, batch=batch.batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()

            total_time += (t1 - t0)
            total_graphs += batch.num_graphs

            if i >= (num_warmup + num_eval):
                break

    latency_per_graph_ms = (total_time / max(total_graphs, 1)) * 1000.0
    return latency_per_graph_ms
