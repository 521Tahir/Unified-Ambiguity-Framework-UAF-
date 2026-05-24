import torch

def sinkhorn_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 0.1, iters: int = 20) -> torch.Tensor:
    """
    Differentiable Sinkhorn distance. Always runs in float32 for numerical stability.
    a, b: [B, D] embeddings
    returns: scalar OT distance
    """
    # Force float32 — critical: fp16 underflows exp(-C/eps) to 0
    a = a.float()
    b = b.float()

    C = torch.cdist(a, b, p=2) ** 2  # [B,B]
    # Normalize cost to prevent exp underflow
    C = C / (C.max().detach() + 1e-8)

    B = a.size(0)
    mu = torch.full((B,), 1.0 / B, device=a.device)
    nu = torch.full((B,), 1.0 / B, device=a.device)

    # Log-domain Sinkhorn for stability
    log_mu = torch.log(mu)
    log_nu = torch.log(nu)
    log_K = -C / eps

    log_u = torch.zeros(B, device=a.device)
    log_v = torch.zeros(B, device=a.device)

    for _ in range(iters):
        log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_nu - torch.logsumexp(log_K.t() + log_u.unsqueeze(0), dim=1)

    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    P = torch.exp(log_P)
    dist = torch.sum(P * C)

    if not torch.isfinite(dist):
        return torch.tensor(0.0, device=a.device, requires_grad=True)
    return dist


def ot_counterfactual_consistency(z: torch.Tensor, sigma: float = 0.05, eps: float = 0.1, iters: int = 20) -> torch.Tensor:
    """
    CHADO OT consistency: penalize distribution shift under small perturbation.
    """
    noise = sigma * torch.randn_like(z)
    z_cf = z + noise
    return sinkhorn_distance(z, z_cf, eps=eps, iters=iters)
