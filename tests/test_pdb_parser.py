"""
Unit tests for Track B PDB parsing and residue contact graph construction.
"""

import os
import tempfile
import pytest
import torch
from preprocessing.pdb_parser import parse_pdb_to_residue_graph
from preprocessing.protein_features import compute_residue_features

# Minimal authentic PDB content for a 4-residue peptide (ALA-GLY-SER-LEU)
SAMPLE_PDB_CONTENT = """HEADER    TEST PEPTIDE                            10-NOV-26   xxxx
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       2.000   1.000   1.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.500   2.000   1.000  1.00 20.00           C
ATOM      4  O   ALA A   1       2.000   3.000   1.000  1.00 20.00           O
ATOM      5  N   GLY A   2       3.800   1.800   1.000  1.00 20.00           N
ATOM      6  CA  GLY A   2       4.500   2.900   1.000  1.00 20.00           C
ATOM      7  C   GLY A   2       5.000   3.000   2.000  1.00 20.00           C
ATOM      8  O   GLY A   2       4.500   3.500   3.000  1.00 20.00           O
ATOM      9  N   SER A   3       6.000   2.500   2.000  1.00 20.00           N
ATOM     10  CA  SER A   3       6.800   3.200   2.800  1.00 20.00           C
ATOM     11  C   SER A   3       8.000   3.000   2.000  1.00 20.00           C
ATOM     12  O   SER A   3       8.500   4.000   1.500  1.00 20.00           O
ATOM     13  N   LEU A   4       8.500   1.800   2.000  1.00 20.00           N
ATOM     14  CA  LEU A   4       9.800   1.500   1.500  1.00 20.00           C
ATOM     15  C   LEU A   4      10.500   2.800   1.200  1.00 20.00           C
ATOM     16  O   LEU A   4      11.500   2.900   0.500  1.00 20.00           O
END
"""


def test_pdb_parsing():
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.write(SAMPLE_PDB_CONTENT)
        temp_pdb_path = f.name

    try:
        data = parse_pdb_to_residue_graph(temp_pdb_path, contact_cutoff=8.0)
        assert data.num_nodes == 4
        assert data.x.shape == (4, 28)
        assert data.edge_attr.dim() == 2
        assert data.edge_attr.size(1) == 18 # 1 dist + 16 RBF + 1 seq sep
        assert data.covalent_mask is not None
        assert data.covalent_mask.sum() > 0
    finally:
        if os.path.exists(temp_pdb_path):
            os.remove(temp_pdb_path)


def test_residue_features():
    feat_ala = compute_residue_features("ALA")
    feat_arg = compute_residue_features("ARG")
    feat_asp = compute_residue_features("ASP")

    assert feat_ala.shape == (28,)
    # ARG has positive charge (+1), ASP has negative charge (-1)
    assert feat_arg[21] == 1.0 # Charge index
    assert feat_asp[21] == -1.0
