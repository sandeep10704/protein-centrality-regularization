"""
Evaluation module for model performance and efficiency benchmarks.
"""

from .metrics import compute_classification_metrics, measure_inference_latency

__all__ = [
    "compute_classification_metrics",
    "measure_inference_latency",
]
