"""
Module: preprocessing/pdb_parser.py
Track B: High-fidelity parser converting authentic PDB structures into residue-level contact graphs.
Handles multi-chain proteins, chain breaks, non-standard residues, missing atoms, and coordinate geometries.
"""

import os
import torch
import numpy as np
from Bio.PDB import PDBParser, Polypeptide
from torch_geometric.data import Data
from scipy.spatial import KDTree

from .protein_features import compute_residue_features, rbf_distance_encoding


def parse_pdb_to_residue_graph(
    pdb_path: str,
    contact_cutoff: float = 8.0,
    k_nn: int = None,
    num_rbf: int = 16,
    chain_break_cutoff: float = 4.5
) -> Data:
    """
    Parses a PDB file and constructs a residue-level interaction graph (RIN).

    Args:
        pdb_path (str): Path to .pdb file.
        contact_cutoff (float): Spatial distance threshold in Angstroms for contact edges. Default: 8.0 A.
        k_nn (int, optional): If set, builds a k-nearest neighbor contact graph instead of radius graph.
        num_rbf (int): Number of Gaussian RBF distance basis functions. Default: 16.
        chain_break_cutoff (float): Maximum C-alpha distance (A) between sequential residues before declaring a chain break.

    Returns:
        torch_geometric.data.Data: Graph containing:
            - x: [num_residues, 28] residue features (one-hot + physicochemical)
            - edge_index: [2, num_edges] directed edge connectivity
            - edge_attr: [num_edges, 18] (1-dim distance + 16-dim RBF + 1-dim seq separation)
            - pos: [num_residues, 3] C-alpha coordinates
            - covalent_mask: [num_edges] boolean mask identifying genuine peptide bonds
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)

    residues = []
    ca_coords = []
    res_names = []
    chain_ids = []

    # Use first model (Model 1) to handle both X-ray and NMR ensembles correctly
    model = structure[0]
    for chain in model:
            c_id = chain.get_id()
            for res in chain:
                # Filter heteroatoms and water molecules
                if res.get_id()[0] != ' ':
                    continue

                # Verify C-alpha atom exists
                if 'CA' in res:
                    ca = res['CA']
                    res_name = res.get_resname().strip()
                    residues.append(res)
                    ca_coords.append(ca.get_coord())
                    res_names.append(res_name)
                    chain_ids.append(c_id)

    num_residues = len(residues)
    if num_residues == 0:
        raise ValueError(f"No valid amino acid residues with C-alpha coordinates found in {pdb_path}")

    ca_coords = np.array(ca_coords, dtype=np.float32)

    # 1. Compute Node Features (28-dim per residue)
    node_features = np.array([compute_residue_features(r) for r in res_names], dtype=np.float32)
    x = torch.from_numpy(node_features)
    pos = torch.from_numpy(ca_coords)

    # 2. Build Spatial Adjacency
    tree = KDTree(ca_coords)
    edge_pairs = set()

    if k_nn is not None and k_nn > 0:
        # k-Nearest Neighbors graph
        k_val = min(k_nn + 1, num_residues)
        _, indices = tree.query(ca_coords, k=k_val)
        for i in range(num_residues):
            for neighbor in indices[i][1:]: # Skip self
                edge_pairs.add((i, neighbor))
                edge_pairs.add((neighbor, i))
    else:
        # Radius contact graph
        pairs = tree.query_pairs(r=contact_cutoff)
        for i, j in pairs:
            edge_pairs.add((i, j))
            edge_pairs.add((j, i))

    # 3. Add Covalent Backbone Connections (Respecting Chain Boundaries and Chain Breaks)
    covalent_edges = set()
    for i in range(num_residues - 1):
        # Must be in same chain
        if chain_ids[i] == chain_ids[i + 1]:
            d_ca = np.linalg.norm(ca_coords[i] - ca_coords[i + 1])
            # Peptide bond check: C-alpha to C-alpha distance is typically 3.8 A; > 4.5 A indicates a chain break
            if d_ca <= chain_break_cutoff:
                edge_pairs.add((i, i + 1))
                edge_pairs.add((i + 1, i))
                covalent_edges.add((i, i + 1))
                covalent_edges.add((i + 1, i))

    if len(edge_pairs) == 0:
        # Fallback if sparse: connect adjacent
        for i in range(num_residues - 1):
            edge_pairs.add((i, i + 1))
            edge_pairs.add((i + 1, i))

    src_list = [p[0] for p in edge_pairs]
    dst_list = [p[1] for p in edge_pairs]
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # 4. Compute Edge Attributes
    u, v = edge_index[0], edge_index[1]
    distances = torch.norm(pos[u] - pos[v], dim=-1) # [num_edges]
    rbf_dist = rbf_distance_encoding(distances, num_rbf=num_rbf, d_min=0.0, d_max=contact_cutoff * 1.5) # [num_edges, num_rbf]

    seq_sep = torch.abs(u.float() - v.float()).unsqueeze(-1) # [num_edges, 1]
    # For residues across different chains, assign maximum sequence separation
    is_diff_chain = torch.tensor([chain_ids[i] != chain_ids[j] for i, j in zip(u.tolist(), v.tolist())], dtype=torch.bool)
    seq_sep[is_diff_chain] = 100.0

    edge_attr = torch.cat([distances.unsqueeze(-1), rbf_dist, seq_sep], dim=-1) # [num_edges, 1 + num_rbf + 1] = 18 dims

    # 5. Covalent Mask
    covalent_mask = torch.tensor([(p in covalent_edges) for p in zip(u.tolist(), v.tolist())], dtype=torch.bool)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        covalent_mask=covalent_mask,
        num_nodes=num_residues
    )
    return data
