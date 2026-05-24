import torch
import torch.nn as nn
from typing import Optional

from src.chado.mad import ambiguity_weight_from_probs
from src.chado.ot import ot_counterfactual_consistency
from src.chado.hyperbolic import HyperbolicUncertainty
from src.chado.causal import CausalDisentangler

class ChadoObjective(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor,
        mad_enable: bool, mad_lambda: float, mad_gamma: float,
        hyp_enable: bool, hyp_lambda: float, hyp_c: float,
        ot_enable: bool, ot_lambda: float, ot_sigma: float, ot_eps: float, ot_iters: int,
        causal_enable: bool, causal_lambda: float
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.mad_enable = mad_enable
        self.mad_lambda = float(mad_lambda)
        self.mad_gamma = float(mad_gamma)

        self.hyp_enable = hyp_enable
        self.hyp_lambda = float(hyp_lambda)
        self.hyp = HyperbolicUncertainty(c=hyp_c) if hyp_enable else None

        self.ot_enable = ot_enable
        self.ot_lambda = float(ot_lambda)
        self.ot_sigma = float(ot_sigma)
        self.ot_eps = float(ot_eps)
        self.ot_iters = int(ot_iters)

        self.causal_enable = causal_enable
        self.causal_lambda = float(causal_lambda)

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        z_shared: Optional[torch.Tensor] = None,
        z_spec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        base = self.bce(logits, y)
        probs = torch.sigmoid(logits)

        total = base

        # MAD
        if self.mad_enable and self.mad_lambda > 0:
            w = ambiguity_weight_from_probs(probs, gamma=self.mad_gamma)  # [B,1]
            per_elem = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")  # [B,6]
            per_sample = per_elem.mean(dim=1, keepdim=True)
            mad_loss = (w * per_sample).mean()
            total = (1.0 - self.mad_lambda) * total + self.mad_lambda * mad_loss

        # Hyperbolic uncertainty
        if self.hyp_enable and self.hyp_lambda > 0:
            total = total + self.hyp_lambda * self.hyp(z, probs)

        # OT counterfactual consistency
        if self.ot_enable and self.ot_lambda > 0:
            total = total + self.ot_lambda * ot_counterfactual_consistency(
                z, sigma=self.ot_sigma, eps=self.ot_eps, iters=self.ot_iters
            )

        # Causal disentanglement orthogonality
        if self.causal_enable and self.causal_lambda > 0 and (z_shared is not None) and (z_spec is not None):
            total = total + self.causal_lambda * CausalDisentangler.orth_loss(z_shared, z_spec)

        return total
