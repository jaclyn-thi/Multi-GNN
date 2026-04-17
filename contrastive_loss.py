import torch
import torch.nn.functional as F

# important: inputs must be projected embeddings
# shapes must be aligned: (N,D)
# And A and A_knn must be node aligned (same node ordering across views)


def contrastive_loss(
    z1,
    z2,
    A=None,
    temperature=0.5,
    eps=1e-8,
    debug=False,
):
    """
    Contrastive loss between two graph views.

    Args:
        z1, z2: (N, D) node embeddings (projected)
        A: (N, N) adjacency matrix (forward edges only)
        temperature: float

    Returns:
        scalar loss (or debug dict)
    """

    if z1.shape != z2.shape:
        raise ValueError(f"Shape mismatch: {z1.shape} vs {z2.shape}")

    N = z1.shape[0]
    device = z1.device

    if A is not None:
        assert A.shape[0] == N, "Adjacency must match embedding size"

    # --- Normalize ---
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # --- Positive mask: A ∪ I ---
    if A is not None:
        A_pos = (A > 0).to(device=device, dtype=torch.bool)
    else:
        A_pos = torch.zeros((N, N), device=device, dtype=torch.bool)

    eye = torch.eye(N, device=device, dtype=torch.bool)

    pos_mask = A_pos | eye  # (N, N)

    # --- Similarity ---
    logits = (z1 @ z2.T) / temperature

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=logits.dtype)

    def directional_loss(logits, pos_mask):
        # denominator: all nodes
        log_denom = torch.logsumexp(logits, dim=1)

        # numerator: positives only
        masked_logits = torch.where(pos_mask, logits, neg_inf)
        log_num = torch.logsumexp(masked_logits, dim=1)

        # safety (should not happen due to identity)
        log_num = torch.where(torch.isfinite(log_num), log_num, torch.zeros_like(log_num))

        loss_vec = -(log_num - log_denom)
        return loss_vec.mean(), log_num, log_denom

    # forward
    loss_12, log_num_12, log_denom_12 = directional_loss(logits, pos_mask)

    # reverse
    loss_21, log_num_21, log_denom_21 = directional_loss(logits.T, pos_mask)

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
