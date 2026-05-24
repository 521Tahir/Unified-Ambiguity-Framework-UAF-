"""
CHADO 5-factor causal disentanglement module.

Factor taxonomy (matching §3.3 of the paper):
  z_c (d_c): conversational context — what the prior turns have established
  z_u (d_u): speaker individuality / cultural expression pattern
  z_t (d_t): temporality — position of the utterance within the conversation
  z_m (d_m): modality interaction — cross-modal coherence / conflict signal
  z_e (d_e): residual variation — everything not captured above

Constraints enforced:
  - Sum of factor dims == d_model (lossless linear decomposition)
  - Pairwise orthogonality on all C(5,2)=10 pairs (independence proxy)

Supervised auxiliary losses (when labels available in batch):
  z_u → speaker classification CE     (IEMOCAP: 10 speakers, MELD: 7 classes)
  z_t → turn-position regression MSE  (normalised turn index ∈ [0,1])
  z_m → cross-modal agreement MSE     (cos_sim(e_T, e_A) computed in model.py)
  z_c → context alignment loss        (1 − cos_sim(z_c, context_emb))

For CMU-MOSEI (monologue, no dialogue structure): z_u/z_t/z_c supervision is
skipped (signals passed as None); only z_m and independence loss apply.
"""
from __future__ import annotations

from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _factor_mlp(d_in: int, d_out: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_in),
        nn.LayerNorm(d_in),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_in, d_out),
    )


# ---------------------------------------------------------------------------
# main module
# ---------------------------------------------------------------------------

class FactorDisentangler(nn.Module):
    """
    Decomposes fused embedding h ∈ ℝ^d_model into 5 named factors.
    All factor dims must sum to d_model so the decomposition is complete.

    Default split for d_model=256: d_c=64, d_u=64, d_t=32, d_m=64, d_e=32.
    Classification head uses cat(z_c, z_u, z_m) ∈ ℝ^(d_c+d_u+d_m).
    """

    D_C = 64   # context
    D_U = 64   # individuality
    D_T = 32   # temporality
    D_M = 64   # modality
    D_E = 32   # residual

    def __init__(
        self,
        d_model: int = 256,
        n_speakers: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_c, d_u, d_t, d_m, d_e = self.D_C, self.D_U, self.D_T, self.D_M, self.D_E
        assert d_c + d_u + d_t + d_m + d_e == d_model, (
            f"Factor dims {d_c}+{d_u}+{d_t}+{d_m}+{d_e}={d_c+d_u+d_t+d_m+d_e} "
            f"!= d_model={d_model}"
        )
        self.d_c, self.d_u, self.d_t, self.d_m, self.d_e = d_c, d_u, d_t, d_m, d_e
        self.d_cls = d_c + d_u + d_m   # 192 — used by downstream classifier

        self.norm = nn.LayerNorm(d_model)

        # factor projection heads
        self.head_c = _factor_mlp(d_model, d_c, dropout)
        self.head_u = _factor_mlp(d_model, d_u, dropout)
        self.head_t = _factor_mlp(d_model, d_t, dropout)
        self.head_m = _factor_mlp(d_model, d_m, dropout)
        self.head_e = _factor_mlp(d_model, d_e, dropout)

        # auxiliary supervision heads
        self.speaker_head = nn.Linear(d_u, n_speakers)   # z_u → speaker class
        self.turn_head    = nn.Linear(d_t, 1)             # z_t → turn pos ∈ [0,1]
        self.modal_head   = nn.Linear(d_m, 1)             # z_m → cos_sim(T, A)

    # ------------------------------------------------------------------
    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, d_model] fused embedding (after _aux_proj in model.py)
        Returns:
            z_c, z_u, z_t, z_m, z_e — individual factor tensors
        """
        z = self.norm(z)
        return (
            self.head_c(z),
            self.head_u(z),
            self.head_t(z),
            self.head_m(z),
            self.head_e(z),
        )

    # ------------------------------------------------------------------
    def cls_features(
        self,
        z_c: torch.Tensor,
        z_u: torch.Tensor,
        z_m: torch.Tensor,
    ) -> torch.Tensor:
        """Return concatenated task-relevant factors → [B, d_cls]."""
        return torch.cat([z_c, z_u, z_m], dim=1)

    # ------------------------------------------------------------------
    def supervision_losses(
        self,
        z_c: torch.Tensor,
        z_u: torch.Tensor,
        z_t: torch.Tensor,
        z_m: torch.Tensor,
        *,
        speaker_id:  torch.Tensor | None = None,   # [B] long, -1 = unknown
        turn_norm:   torch.Tensor | None = None,   # [B] float [0,1], -1 = skip
        modal_sim:   torch.Tensor | None = None,   # [B] float cos_sim(T,A)
        context_emb: torch.Tensor | None = None,   # [B, d_c]
    ) -> dict[str, torch.Tensor]:
        """
        Compute named auxiliary supervision losses.
        Each loss is only computed when its signal is not None and has valid samples.
        """
        losses: dict[str, torch.Tensor] = {}

        # z_u: speaker identity classification
        if speaker_id is not None:
            mask = speaker_id >= 0
            if mask.any():
                sp_logits = self.speaker_head(z_u[mask])
                losses["speaker"] = F.cross_entropy(sp_logits, speaker_id[mask])

        # z_t: normalised turn-position regression
        if turn_norm is not None:
            mask = turn_norm >= 0
            if mask.any():
                pred = self.turn_head(z_t[mask]).squeeze(-1)
                losses["turn"] = F.mse_loss(pred, turn_norm[mask].float())

        # z_m: cross-modal agreement prediction
        if modal_sim is not None:
            pred = self.modal_head(z_m).squeeze(-1)
            losses["modal"] = F.mse_loss(pred, modal_sim.detach())

        # z_c: cosine alignment to context embedding
        if context_emb is not None:
            z_c_n = F.normalize(z_c, dim=1)
            ctx_n = F.normalize(context_emb.detach(), dim=1)
            losses["context"] = (1.0 - (z_c_n * ctx_n).sum(dim=1)).mean()

        return losses

    # ------------------------------------------------------------------
    @staticmethod
    def independence_loss(*factors: torch.Tensor) -> torch.Tensor:
        """
        Pairwise orthogonality over all C(n,2) factor pairs.
        For cross-dim pairs, only the min-dim prefix is compared.
        Returns mean squared cosine similarity (lower = more independent).
        """
        total = torch.tensor(0.0, device=factors[0].device)
        n_pairs = 0
        for a, b in combinations(factors, 2):
            min_d = min(a.size(1), b.size(1))
            a_n = F.normalize(a[:, :min_d], dim=1)
            b_n = F.normalize(b[:, :min_d], dim=1)
            total = total + ((a_n * b_n).sum(dim=1) ** 2).mean()
            n_pairs += 1
        return total / max(n_pairs, 1)
