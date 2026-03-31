import torch
import torch.nn.functional as F

# important: inputs must be projected embeddings
# shapes must be aligned: (N,D)
# And A and A_knn must be node aligned (same node ordering across views)

def gcpal_pair_loss(
    z1,
    z2,
    A=None,
    A_knn=None,
    temperature=0.5,
    eps=1e-8,
    debug=False,
):
    """
    GCPAL-style contrastive loss between two graph views.

    Args:
        z1, z2: (N, D) node embeddings (must already be projected)
        A: (N, N) adjacency matrix
        A_knn: (N, N) KNN adjacency matrix
        temperature: float
        debug: if True, returns intermediate tensors

    Returns:
        scalar loss (or dict if debug=True)
    """
    if z1.shape != z2.shape:
        raise ValueError(f"Shape mismatch: {z1.shape} vs {z2.shape}")

    N = z1.shape[0]
    device = z1.device

    # Normalize (cosine similarity)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # --- Build positive mask MP = A ∪ A_knn ∪ I ---
    if A is not None:
        A_pos = (A > 0).to(device=device, dtype=torch.bool)
    else:
        A_pos = torch.zeros((N, N), device=device, dtype=torch.bool)

    if A_knn is not None:
        A_knn_pos = (A_knn > 0).to(device=device, dtype=torch.bool)
    else:
        A_knn_pos = torch.zeros((N, N), device=device, dtype=torch.bool)

    eye = torch.eye(N, device=device, dtype=torch.bool)

    pos_mask = A_pos | A_knn_pos | eye   # (N, N)

    # --- Similarity matrix ---
    logits = (z1 @ z2.T) / temperature   # (N, N)

    def directional_loss(logits, pos_mask):
        # denominator: all nodes
        log_denom = torch.logsumexp(logits, dim=1)  # (N,)

        # numerator: only positives
        neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
        masked_logits = torch.where(pos_mask, logits, neg_inf)

        log_num = torch.logsumexp(masked_logits, dim=1)  # (N,)

        # safety (shouldn't happen due to identity)
        log_num = torch.where(torch.isfinite(log_num), log_num, torch.zeros_like(log_num))

        loss_vec = -(log_num - log_denom)  # (N,)
        return loss_vec.mean(), log_num, log_denom

    # forward direction
    loss_12, log_num_12, log_denom_12 = directional_loss(logits, pos_mask)

    # reverse direction
    logits_T = logits.T
    loss_21, log_num_21, log_denom_21 = directional_loss(logits_T, pos_mask)

    loss = 0.5 * (loss_12 + loss_21)

    if debug:
        return {
            "loss": loss,
            "logits": logits,
            "pos_mask": pos_mask,
            "log_num_12": log_num_12,
            "log_denom_12": log_denom_12,
            "log_num_21": log_num_21,
            "log_denom_21": log_denom_21,
        }

    return loss


def gcpal_pretrain_loss(
    z1,
    z2,
    z_knn,
    A=None,
    A_knn=None,
    temperature=0.5,
    lam=0.1,
    debug=False,
):
    """
    Full GCPAL pretraining loss.

    Args:
        z1, z2: random augmented views
        z_knn: KNN graph view
        lam: weighting parameter

    Returns:
        scalar loss (or dict if debug=True)
    """

    loss_random = gcpal_pair_loss(
        z1, z2,
        A=A,
        A_knn=A_knn,
        temperature=temperature,
    )

    loss_knn = gcpal_pair_loss(
        z2, z_knn,
        A=A,
        A_knn=A_knn,
        temperature=temperature,
    )

    total = lam * loss_random + (1 - lam) * loss_knn

    if debug:
        return {
            "total_loss": total,
            "loss_random": loss_random,
            "loss_knn": loss_knn,
        }

    return total
