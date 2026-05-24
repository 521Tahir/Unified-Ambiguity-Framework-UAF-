import torch

def ambiguity_weight_from_probs(probs: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    CHADO-MAD proxy: uncertainty near 0.5 is high.
    probs: [B, C]
    return: [B, 1] weights
    """
    u = 1.0 - 2.0 * torch.abs(probs - 0.5)
    u = torch.clamp(u, 0.0, 1.0)
    w = (u.mean(dim=1, keepdim=True) ** gamma)
    return w


def compute_mad_scores(probs: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Returns per-sample MAD ambiguity score.

    probs: [B, C]
    return: [B]
    """
    with torch.no_grad():
        return ambiguity_weight_from_probs(probs, gamma=gamma).squeeze(1)
