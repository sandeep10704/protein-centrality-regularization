"""
Module: preprocessing/protein_features.py
Computes standardized physicochemical and structural features for protein residues.
Includes Kyte-Doolittle hydropathy, formal charge, molecular weight, vdW volume, pI, and aromaticity.
"""

import numpy as np
import torch

# Canonical 20 amino acids
AMINO_ACIDS = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# 1. Kyte-Doolittle Hydropathy Index
HYDROPATHY = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}

# 2. Net formal charge at physiological pH (7.4)
FORMAL_CHARGE = {
    'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.1, 'ASP': -1.0, 'GLU': -1.0,
    'ALA': 0.0, 'CYS': 0.0, 'GLN': 0.0, 'GLY': 0.0, 'ILE': 0.0,
    'LEU': 0.0, 'MET': 0.0, 'ASN': 0.0, 'PRO': 0.0, 'SER': 0.0,
    'THR': 0.0, 'VAL': 0.0, 'TRP': 0.0, 'TYR': 0.0, 'PHE': 0.0
}

# 3. Residue Molecular Weight (Da)
MOL_WEIGHT = {
    'GLY': 75.07, 'ALA': 89.09, 'SER': 105.09, 'PRO': 115.13, 'VAL': 117.15,
    'THR': 119.12, 'CYS': 121.16, 'ILE': 131.17, 'LEU': 131.17, 'ASN': 132.12,
    'ASP': 133.10, 'GLN': 146.15, 'LYS': 146.19, 'GLU': 147.13, 'MET': 149.21,
    'HIS': 155.16, 'PHE': 165.19, 'ARG': 174.20, 'TYR': 181.19, 'TRP': 204.23
}

# 4. Van der Waals Volume (A^3)
VDW_VOLUME = {
    'GLY': 48.0, 'ALA': 67.0, 'SER': 73.0, 'CYS': 86.0, 'THR': 93.0,
    'ASP': 91.0, 'PRO': 90.0, 'VAL': 105.0, 'ASN': 96.0, 'GLU': 109.0,
    'GLN': 114.0, 'HIS': 118.0, 'LEU': 124.0, 'ILE': 124.0, 'MET': 124.0,
    'LYS': 135.0, 'PHE': 135.0, 'TYR': 141.0, 'ARG': 148.0, 'TRP': 163.0
}

# 5. Isoelectric point (pI)
PI_VALUES = {
    'ASP': 2.77, 'GLU': 3.22, 'CYS': 5.07, 'TYR': 5.66, 'PHE': 5.48,
    'TRP': 5.89, 'MET': 5.74, 'VAL': 5.96, 'SER': 5.68, 'LEU': 5.98,
    'ILE': 6.02, 'THR': 6.16, 'GLY': 5.97, 'ALA': 6.00, 'PRO': 6.30,
    'GLN': 5.65, 'ASN': 5.41, 'HIS': 7.59, 'LYS': 9.74, 'ARG': 10.76
}

# 6. Hydrogen bond donors and acceptors (side chain)
HBOND_PROPS = {
    'ALA': (0, 0), 'CYS': (1, 0), 'ASP': (1, 2), 'GLU': (1, 2), 'PHE': (0, 0),
    'GLY': (0, 0), 'HIS': (1, 1), 'ILE': (0, 0), 'LYS': (2, 0), 'LEU': (0, 0),
    'MET': (0, 1), 'ASN': (2, 2), 'PRO': (0, 0), 'GLN': (2, 2), 'ARG': (4, 0),
    'SER': (1, 1), 'THR': (1, 1), 'VAL': (0, 0), 'TRP': (1, 0), 'TYR': (1, 1)
}

# 7. Aromaticity
AROMATIC_AAS = {'PHE', 'TYR', 'TRP', 'HIS'}


def compute_residue_features(res_name: str) -> np.ndarray:
    """
    Computes a 28-dimensional standardized residue feature vector:
    - 20-dim one-hot amino acid identity
    - 1-dim normalized hydropathy in [-1, 1]
    - 1-dim formal charge in [-1, 1]
    - 1-dim normalized molecular weight in [0, 1]
    - 1-dim normalized vdW volume in [0, 1]
    - 1-dim normalized pI in [0, 1]
    - 2-dim normalized HB donor/acceptor counts in [0, 1]
    - 1-dim aromaticity flag {0, 1}
    """
    res_name = res_name.upper()

    # One-hot encoding (20 dims)
    one_hot = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    if res_name in AA_TO_IDX:
        one_hot[AA_TO_IDX[res_name]] = 1.0

    # Physicochemical features (8 dims)
    hydro = HYDROPATHY.get(res_name, 0.0) / 4.5
    charge = FORMAL_CHARGE.get(res_name, 0.0)
    mw = (MOL_WEIGHT.get(res_name, 120.0) - 75.0) / (205.0 - 75.0)
    vdw = (VDW_VOLUME.get(res_name, 100.0) - 48.0) / (163.0 - 48.0)
    pi = (PI_VALUES.get(res_name, 6.0) - 2.7) / (10.8 - 2.7)
    donors, acceptors = HBOND_PROPS.get(res_name, (0, 0))
    donors_norm = donors / 4.0
    acceptors_norm = acceptors / 2.0
    aromatic = 1.0 if res_name in AROMATIC_AAS else 0.0

    physchem = np.array([hydro, charge, mw, vdw, pi, donors_norm, acceptors_norm, aromatic], dtype=np.float32)

    return np.concatenate([one_hot, physchem])


def rbf_distance_encoding(distances: torch.Tensor, num_rbf: int = 16, d_min: float = 0.0, d_max: float = 20.0) -> torch.Tensor:
    """
    Expands scalar distances using Gaussian Radial Basis Functions (RBF).
    """
    mu = torch.linspace(d_min, d_max, num_rbf, device=distances.device)
    sigma = (d_max - d_min) / num_rbf
    return torch.exp(-0.5 * ((distances.unsqueeze(-1) - mu) / sigma) ** 2)
