import torch
import torch.nn as nn

class HyperbolicUncertainty(nn.Module):
    """
    Lightweight hyperbolic-inspired uncertainty regularizer:
    - pushes ambiguous samples to have larger 'radius' in embedding norm
    - stabilizes calibration and prevents overconfident FP explosions (e.g., sad)
    This is a CHADO-compatible component without heavy manifold ops.
    """
    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = float(c)

    def forward(self, z: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        z: [B, D] fused embedding
        probs: [B, C] sigmoid probs
        returns: scalar regularization loss
        """
        # ambiguity proxy: high near 0.5
        amb = 1.0 - 2.0 * torch.abs(probs - 0.5)
        amb = torch.clamp(amb, 0.0, 1.0).mean(dim=1)  # [B]

        # radius = ||z|| ; encourage radius proportional to ambiguity
        radius = torch.norm(z, dim=1)  # [B]
        # normalize
        radius = radius / (radius.mean().detach() + 1e-6)

        # MSE alignment between ambiguity and radius
        return ((radius - (1.0 + amb)) ** 2).mean()
