"""
Module: augmentations/aga_protein.py
Method B: Novel Protein-Specific Bio-AGA Regularization.
Integrates:
1. Permanent covalent backbone protection (P_drop = 0 for peptide bonds).
2. Generalized multi-centrality shielding (Degree, Betweenness, Closeness, Eigenvector).
3. Non-local sequence distance protection (shielding long-range tertiary contacts).
4. Physical spatial distance affinity weighting.
"""

import torch
from centrality import compute_centrality


def bio_aga_edge_drop(
    data,
    drop_prob: float = 0.20,
    gamma: float = 0.35,
    centrality_metric: str = "degree",
    alpha_seq: float = 0.25,
    tau_seq: float = 12.0,
    sigma_dist: float = 4.0,
    protect_covalent: bool = True,
    use_seq_protection: bool = True,
    use_dist_affinity: bool = True,
    eps: float = 1e-8,
    cached_centrality: torch.Tensor = None
):
    """
    Proposed Method: Biologically & Structurally Grounded Centrality Regularization (Bio-AGA).

    Mathematical Formulation:
        For edge (u, v):
        If covalent backbone:
            P_drop(u, v) = 0.0 (if protect_covalent is True)
        Else:
            I(u, v) = (C(u) + C(v)) / 2.0
            Omega_seq(u, v) = 1.0 + alpha_seq * tanh(|u - v| / tau_seq)
            Psi_dist(u, v) = exp(-d(u, v)^2 / (2 * sigma_dist^2))
            S_bio(u, v) = clamp(I(u, v), eps, 1.0)^gamma * Omega_seq * Psi_dist
            S_final(u, v) = clamp(S_bio(u, v), eps, 1.0)
            P_drop(u, v) = drop_prob * (1.0 - S_final(u, v))
            P_retain(u, v) = 1.0 - P_drop(u, v)

    Args:
        data: PyG Data or Batch object.
        drop_prob: Base maximum edge drop probability. Default: 0.20.
        gamma: Sub-linear concavity exponent. Default: 0.35.
        centrality_metric: 'degree', 'betweenness', 'closeness', 'eigenvector'. Default: 'degree'.
        alpha_seq: Long-range sequence protection gain. Default: 0.25.
        tau_seq: Characteristic sequence separation scale. Default: 12.0.
        sigma_dist: Characteristic spatial interaction length. Default: 4.0.
        protect_covalent: Whether to permanently shield covalent peptide bonds. Default: True.
        use_seq_protection: Whether to apply sequence separation boost. Default: True.
        use_dist_affinity: Whether to apply spatial distance decay. Default: True.
        eps: Numerical epsilon.
        cached_centrality: Precomputed node centrality tensor (optional).

    Returns:
        Augmented PyG Data or Batch object with preserved covalent bonds and regularized contacts.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_edges = edge_index.size(1)

    if drop_prob <= 0.0 or num_edges == 0:
        return data.clone()

    batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
    device = edge_index.device
    u, v = edge_index[0], edge_index[1]

    # 1. Centrality Shielding
    if cached_centrality is not None:
        c = cached_centrality
    else:
        c = compute_centrality(edge_index, num_nodes, batch=batch, metric=centrality_metric)

    importance = (c[u] + c[v]) / 2.0
    shield_centrality = torch.pow(torch.clamp(importance, min=eps, max=1.0), gamma)

    # 2. Sequence Distance Protection Shield (Omega_seq)
    if use_seq_protection:
        # Check if residues belong to same chain / graph
        if batch is not None:
            same_graph = (batch[u] == batch[v])
        else:
            same_graph = torch.ones(num_edges, dtype=torch.bool, device=device)

        seq_sep = torch.abs(u.float() - v.float())
        omega_seq = 1.0 + alpha_seq * torch.tanh(seq_sep / tau_seq)
        # Only apply sequence protection within the same protein graph
        omega_seq = torch.where(same_graph, omega_seq, torch.tensor(1.0, device=device))
    else:
        omega_seq = 1.0

    # 3. Spatial Distance Affinity Factor (Psi_dist)
    if use_dist_affinity and hasattr(data, "pos") and data.pos is not None:
        pos_u = data.pos[u]
        pos_v = data.pos[v]
        dist = torch.norm(pos_u - pos_v, dim=-1)
        psi_dist = torch.exp(- (dist ** 2) / (2.0 * (sigma_dist ** 2)))
    elif use_dist_affinity and hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.dim() >= 2:
        # First column of edge_attr is spatial distance if encoded
        dist = data.edge_attr[:, 0]
        psi_dist = torch.exp(- (dist ** 2) / (2.0 * (sigma_dist ** 2)))
    else:
        psi_dist = 1.0

    # 4. Combined Bio-Shield
    s_bio = shield_centrality * omega_seq * psi_dist
    s_final = torch.clamp(s_bio, min=eps, max=1.0)

    # 5. Adaptive Drop & Retention Probabilities
    p_drop = drop_prob * (1.0 - s_final)

    # 6. Covalent Backbone Protection (Axiom 1)
    if protect_covalent:
        # Check if explicit covalent attribute exists
        if hasattr(data, "covalent_mask") and data.covalent_mask is not None:
            is_covalent = data.covalent_mask
        else:
            # Fallback: sequential adjacency within same graph
            seq_diff = torch.abs(u.long() - v.long())
            if batch is not None:
                is_covalent = (seq_diff == 1) & (batch[u] == batch[v])
            else:
                is_covalent = (seq_diff == 1)

        p_drop = torch.where(is_covalent, torch.tensor(0.0, device=device), p_drop)

    p_retain = 1.0 - p_drop

    # 7. Stochastic Sampling
    mask = torch.rand(num_edges, device=device) < p_retain

    if mask.sum() == 0:
        mask[torch.randint(0, num_edges, (1,), device=device)] = True

    data_aug = data.clone()
    data_aug.edge_index = edge_index[:, mask]

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data_aug.edge_attr = data.edge_attr[mask]

    return data_aug
