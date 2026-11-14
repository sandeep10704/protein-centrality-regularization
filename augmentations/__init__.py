"""
Augmentation and regularization package for protein structural graphs.
Includes Uniform DropEdge, Original AGA-Degree, Protein-Specific Bio-AGA, and Structural Perturbations.
"""

from .uniform_drop import uniform_edge_drop
from .aga_original import aga_degree_edge_drop
from .aga_protein import bio_aga_edge_drop
from .perturbations import (
    inject_synthetic_edge_addition,
    inject_synthetic_edge_deletion,
    inject_long_range_contact_perturbation,
)

__all__ = [
    "uniform_edge_drop",
    "aga_degree_edge_drop",
    "bio_aga_edge_drop",
    "inject_synthetic_edge_addition",
    "inject_synthetic_edge_deletion",
    "inject_long_range_contact_perturbation",
]
