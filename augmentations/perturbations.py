"""
Module: augmentations/perturbations.py
Controlled structural graph perturbations and noise injection for robustness stress-testing.
Distinguishes synthetic graph perturbations from structural coordinate uncertainty.
"""

import torch


def inject_synthetic_edge_addition(data, noise_ratio: float = 0.20, seed: int = None):
    """
    Simulates adversarial or spurious edge corruption by randomly adding non-physical contacts.
    Only connects residues within the same protein graph in a batch.
    """
    if noise_ratio <= 0.0:
        return data.clone()

    if seed is not None:
        torch.manual_seed(seed)

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_edges = edge_index.size(1)
    device = edge_index.device
    batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(num_nodes, dtype=torch.long, device=device)

    num_noisy = int(num_edges * noise_ratio)
    if num_noisy == 0:
        return data.clone()

    # Sample random node pairs
    u = torch.randint(0, num_nodes, (num_noisy * 3,), device=device)
    v = torch.randint(0, num_nodes, (num_noisy * 3,), device=device)

    # Filter to pairs within the same graph and not self-loops
    valid = (batch[u] == batch[v]) & (u != v)
    u_valid = u[valid][:num_noisy]
    v_valid = v[valid][:num_noisy]

    if len(u_valid) == 0:
        return data.clone()

    noisy_edges = torch.stack([u_valid, v_valid], dim=0)

    data_corrupt = data.clone()
    data_corrupt.edge_index = torch.cat([edge_index, noisy_edges], dim=1)

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        # Pad edge attributes with average feature vector for synthetic edges
        mean_attr = data.edge_attr.mean(dim=0, keepdim=True).repeat(noisy_edges.size(1), 1)
        data_corrupt.edge_attr = torch.cat([data.edge_attr, mean_attr], dim=0)

    return data_corrupt


def inject_synthetic_edge_deletion(data, drop_ratio: float = 0.20, seed: int = None):
    """
    Simulates random loss of contact information (e.g., disordered regions or cryo-EM low resolution).
    """
    if drop_ratio <= 0.0:
        return data.clone()

    if seed is not None:
        torch.manual_seed(seed)

    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    retain_prob = 1.0 - drop_ratio
    mask = torch.rand(num_edges, device=edge_index.device) < retain_prob

    if mask.sum() == 0:
        mask[0] = True

    data_corrupt = data.clone()
    data_corrupt.edge_index = edge_index[:, mask]
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data_corrupt.edge_attr = data.edge_attr[mask]

    return data_corrupt


def inject_long_range_contact_perturbation(data, corruption_ratio: float = 0.30, min_seq_sep: int = 12, seed: int = None):
    """
    Selectively corrupts critical long-range tertiary contacts (|u - v| >= min_seq_sep).
    Tests whether the model's fold prediction degrades when non-local structural constraints are damaged.
    """
    if corruption_ratio <= 0.0:
        return data.clone()

    if seed is not None:
        torch.manual_seed(seed)

    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    device = edge_index.device
    u, v = edge_index[0], edge_index[1]

    seq_sep = torch.abs(u.float() - v.float())
    is_long_range = (seq_sep >= min_seq_sep)

    # Randomly drop a fraction of long-range contacts
    drop_long_range = (torch.rand(num_edges, device=device) < corruption_ratio) & is_long_range
    retain_mask = ~drop_long_range

    if retain_mask.sum() == 0:
        retain_mask[0] = True

    data_corrupt = data.clone()
    data_corrupt.edge_index = edge_index[:, retain_mask]
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data_corrupt.edge_attr = data.edge_attr[retain_mask]

    return data_corrupt
