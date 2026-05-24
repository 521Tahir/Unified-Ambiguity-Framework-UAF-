"""
Ambiguity proxy weight functions for CHADO curriculum reweighting.

All public functions return a [B, 1] float tensor in [0, 1].
Drop-in replacements for the original MAD weight.

Proxies
-------
mad      : CHADO-MAD  mean(clamp(1-2|p-0.5|,0,1))^γ
entropy  : Normalised predictive entropy  H(p)/log(K)
margin   : 1 - max_c p_c
mc_var   : Mean per-class variance over K dropout passes (classifier head only)
bald     : H(E[p]) - E[H(p)]  (information gain / epistemic signal)
"""

import torch
import torch.nn as nn

EPS = 1e-8
MC_K_DEFAULT = 5          # cheap: runs only through the small MLP classifier head


# ── individual proxy functions ─────────────────────────────────────────────────

def mad_weight(probs: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    u = (1.0 - 2.0 * torch.abs(probs - 0.5)).clamp(0.0, 1.0)
    return u.mean(dim=1, keepdim=True) ** gamma


def entropy_weight(probs: torch.Tensor, is_binary: bool = False) -> torch.Tensor:
    if is_binary:
        p = probs.clamp(EPS, 1.0 - EPS)
        h = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean(dim=1, keepdim=True)
        return (h / torch.log(torch.tensor(2.0, device=probs.device))).clamp(0, 1)
    p = probs.clamp(EPS, 1.0)
    K = probs.shape[1]
    h = -(p * torch.log(p)).sum(dim=1, keepdim=True)
    return (h / torch.log(torch.tensor(float(K), device=probs.device))).clamp(0, 1)


def margin_weight(probs: torch.Tensor, is_binary: bool = False) -> torch.Tensor:
    if is_binary:
        return (1.0 - (2.0 * torch.abs(probs - 0.5)).max(dim=1, keepdim=True).values).clamp(0, 1)
    return (1.0 - probs.max(dim=1, keepdim=True).values).clamp(0, 1)


def mc_var_weight(cls_feat: torch.Tensor,
                  classifier: nn.Module,
                  K: int = MC_K_DEFAULT,
                  is_binary: bool = False) -> torch.Tensor:
    """Cheap MC-dropout variance — reruns only the small MLP head K times."""
    was_training = classifier.training
    classifier.train()          # activate dropout
    all_probs = []
    with torch.no_grad():
        for _ in range(K):
            lk = classifier(cls_feat)
            pk = torch.sigmoid(lk) if is_binary else torch.softmax(lk, dim=-1)
            all_probs.append(pk)
    if not was_training:
        classifier.eval()
    var = torch.stack(all_probs, 0).var(0).mean(dim=-1, keepdim=True)   # [B,1]
    return (var / (var.max() + EPS)).clamp(0, 1)


def bald_weight(cls_feat: torch.Tensor,
                classifier: nn.Module,
                K: int = MC_K_DEFAULT,
                is_binary: bool = False) -> torch.Tensor:
    """BALD = H(E[p]) - E[H(p)]  via K cheap MLP-head MC passes."""
    was_training = classifier.training
    classifier.train()
    all_probs = []
    with torch.no_grad():
        for _ in range(K):
            lk = classifier(cls_feat)
            pk = torch.sigmoid(lk) if is_binary else torch.softmax(lk, dim=-1)
            all_probs.append(pk)
    if not was_training:
        classifier.eval()

    stacked = torch.stack(all_probs, 0)          # [K, B, C]
    mean_p  = stacked.mean(0).clamp(EPS, 1.0)    # [B, C]

    if is_binary:
        q = (1.0 - mean_p).clamp(EPS, 1.0)
        H_mean = -(mean_p * torch.log(mean_p) + q * torch.log(q)).mean(dim=-1)
        per_k  = []
        for pk in all_probs:
            p_ = pk.clamp(EPS, 1.0 - EPS)
            per_k.append(-(p_ * torch.log(p_) + (1 - p_) * torch.log(1 - p_)).mean(dim=-1))
    else:
        H_mean = -(mean_p * torch.log(mean_p)).sum(dim=-1)
        per_k  = [-(pk.clamp(EPS, 1.0) * torch.log(pk.clamp(EPS, 1.0))).sum(dim=-1)
                  for pk in all_probs]

    b = (H_mean - torch.stack(per_k, 0).mean(0)).clamp(min=0)    # [B]
    return (b / (b.max() + EPS)).unsqueeze(1).clamp(0, 1)         # [B,1]


# ── unified selector ───────────────────────────────────────────────────────────

def get_ambiguity_weight(proxy_type: str,
                          probs: torch.Tensor,
                          cls_feat=None,
                          classifier=None,
                          is_binary: bool = False,
                          gamma: float = 1.0,
                          K_mc: int = MC_K_DEFAULT) -> torch.Tensor:
    """
    Returns [B, 1] curriculum weight in [0, 1].

    proxy_type : 'mad' | 'entropy' | 'margin' | 'mc_var' | 'bald'
    cls_feat   : [B, d] features fed into classifier (required for mc_var / bald)
    classifier : nn.Module with Dropout (required for mc_var / bald)
    """
    if proxy_type == 'mad':
        return mad_weight(probs, gamma=gamma)
    if proxy_type == 'entropy':
        return entropy_weight(probs, is_binary=is_binary)
    if proxy_type == 'margin':
        return margin_weight(probs, is_binary=is_binary)
    if proxy_type in ('mc_var', 'bald'):
        if cls_feat is None or classifier is None:
            # Graceful fallback to entropy when features not available
            return entropy_weight(probs, is_binary=is_binary)
        fn = mc_var_weight if proxy_type == 'mc_var' else bald_weight
        return fn(cls_feat, classifier, K=K_mc, is_binary=is_binary)
    raise ValueError(f"Unknown proxy_type '{proxy_type}'. "
                     "Choose: mad | entropy | margin | mc_var | bald")
