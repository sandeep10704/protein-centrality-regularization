"""
Training package for protein graph classification.
"""

from .trainer import ProteinTrainer, count_parameters

__all__ = [
    "ProteinTrainer",
    "count_parameters",
]
