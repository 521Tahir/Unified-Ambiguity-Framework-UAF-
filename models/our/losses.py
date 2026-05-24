import torch
import torch.nn as nn
from .mad import ambiguity_weight_from_probs
from .ot import ot_counterfactual_consistency
from .hyperbolic import HyperbolicUncertainty

class ChadoLoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor,
        mad_enable: bool, mad_lambda: float, mad_gamma: float,
        hyp_enable: bool, hyp_lambda: float, hyp_c: float,
        ot_enable: bool, ot_lambda: float, ot_sigma: float, ot_eps: float, ot_iters: int
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

    def forward(self, logits: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,6]
        y: [B,6]
        z: [B,D] fused embedding (for OT + hyperbolic regularizers)
        """
        base = self.bce(logits, y)

        probs = torch.sigmoid(logits)

        total = base

        # MAD ambiguity reweighting
        if self.mad_enable and self.mad_lambda > 0:
            w = ambiguity_weight_from_probs(probs, gamma=self.mad_gamma)  # [B,1]
            per_elem = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")  # [B,6]
            per_sample = per_elem.mean(dim=1, keepdim=True)  # [B,1]
            mad_loss = (w * per_sample).mean()
            total = (1.0 - self.mad_lambda) * total + self.mad_lambda * mad_loss

        # Hyperbolic uncertainty regularizer
        if self.hyp_enable and self.hyp_lambda > 0:
            total = total + self.hyp_lambda * self.hyp(z, probs)

        # OT counterfactual consistency
        if self.ot_enable and self.ot_lambda > 0:
            total = total + self.ot_lambda * ot_counterfactual_consistency(
                z, sigma=self.ot_sigma, eps=self.ot_eps, iters=self.ot_iters
            )

        return total
