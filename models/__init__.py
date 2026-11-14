"""
Models package for protein graph classification.
Provides ProteinGCN, ProteinGAT, ProteinGATv2, and ProteinResidualGATv2.
"""

from .gcn import ProteinGCN
from .gat import ProteinGAT
from .gatv2 import ProteinGATv2
from .res_gatv2 import ProteinResidualGATv2
from .pooling import ProteinPooling

__all__ = [
    "ProteinGCN",
    "ProteinGAT",
    "ProteinGATv2",
    "ProteinResidualGATv2",
    "ProteinPooling",
]
