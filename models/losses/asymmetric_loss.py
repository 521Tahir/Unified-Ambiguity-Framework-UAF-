"""
Asymmetric Loss (ASL) for multi-label classification.
Ref: Ben-Baruch et al. "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
https://arxiv.org/abs/2009.14119

Key idea:
  - Positive samples: standard focal loss  -(1-p)^gamma_pos * log(p)
  - Negative samples: probability shifted focal loss with harder margin
    -(p_shifted)^gamma_neg * log(1-p_shifted), where p_shifted = max(p - margin, 0)

gamma_neg > gamma_pos discourages the model from predicting false positives
on the abundant negative samples (rare class problem).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    ASL for multi-label with class imbalance.

    gamma_neg: focusing parameter for negatives (recommended 4)
    gamma_pos: focusing parameter for positives (recommended 0 or 1)
    margin:    probability shift for negatives (clips overconfident negatives)
    eps:       label smoothing / numerical stability
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        margin: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.margin = margin
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C] raw logits (before sigmoid)
        targets: [B, C] binary {0, 1}
        """
        probs = torch.sigmoid(logits)

        # Shift negatives: clip probabilities below margin to 0
        probs_neg = (probs - self.margin).clamp(min=0)

        # Log probabilities
        log_pos = torch.log(probs.clamp(min=self.eps))
        log_neg = torch.log((1.0 - probs_neg).clamp(min=self.eps))

        # Focal weights
        pos_weight = (1.0 - probs) ** self.gamma_pos
        neg_weight = probs_neg ** self.gamma_neg

        # ASL per-element loss
        loss = targets * pos_weight * log_pos + (1.0 - targets) * neg_weight * log_neg
        loss = -loss  # [B, C]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # [B, C]


class AsymmetricLossOptimized(nn.Module):
    """
    Memory-efficient ASL (avoids storing intermediate tensors).
    Same math as AsymmetricLoss but uses in-place operations.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        margin: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.margin = margin
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        xs_pos = probs
        xs_neg = 1.0 - probs

        # Shift negatives
        if self.margin > 0:
            xs_neg = (xs_neg + self.margin).clamp(max=1)

        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))

        loss = targets * log_pos + (1.0 - targets) * log_neg

        # Focal scaling
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            prob_pos = targets * xs_pos
            prob_neg = (1.0 - targets) * (1.0 - xs_neg)
            focal_w = targets * (1.0 - prob_pos) ** self.gamma_pos + \
                      (1.0 - targets) * prob_neg ** self.gamma_neg
            loss *= focal_w

        return -loss.mean()
