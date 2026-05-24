"""
Unified CHADO model.

Two modes:
  - mode="trimodal"  : raw media (RoBERTa + wav2vec2 + ViT), single-label CE.
                       Used for IEMOCAP and MELD.
  - mode="feature"   : pre-extracted features (text tokens + audio/video numpy arrays),
                       multi-label BCE. Used for CMU-MOSEI.

CHADO auxiliary losses:
  - 5-factor disentanglement (FactorDisentangler) with supervised auxiliary losses
    where dataset signals are available (speaker_id, turn_norm, modal_sim, context_emb)
  - Hyperbolic norm-based ambiguity regularizer
  - Log-domain Sinkhorn OT consistency loss
  - MAD proxy reweighting
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal import FactorDisentangler
from .hyperbolic import HyperbolicUncertainty
from .ot import ot_counterfactual_consistency
from .mad import compute_mad_scores


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _modal_sim_from_raw_z(
    raw_z: torch.Tensor,
    modality_order: list[str],
    proj_dim: int = 256,
) -> torch.Tensor | None:
    """
    Compute cosine similarity between text and audio projections.
    raw_z = [B, M * proj_dim] with modalities in modality_order.
    Returns [B] float if both text and audio present, else None.
    """
    if "text" not in modality_order or "audio" not in modality_order:
        return None
    t_idx = modality_order.index("text")
    a_idx = modality_order.index("audio")
    e_T = raw_z[:, t_idx * proj_dim : (t_idx + 1) * proj_dim]
    e_A = raw_z[:, a_idx * proj_dim : (a_idx + 1) * proj_dim]
    return F.cosine_similarity(e_T, e_A, dim=1)   # [B]


# ---------------------------------------------------------------------------
# Trimodal CHADO  (IEMOCAP / MELD)
# ---------------------------------------------------------------------------

class CHADOTrimodal(nn.Module):
    """
    Backbone (CTNet or TriModalBaseline) + CHADO 5-factor disentanglement.

    Factor classifier replaces backbone classifier as the primary prediction head.
    Backbone logits are kept as an auxiliary consistency loss (w_backbone weight).
    """

    def __init__(
        self,
        text_model_name: str,
        audio_model_name: str,
        video_model_name: str,
        num_classes: int,
        proj_dim: int = 256,
        dropout: float = 0.2,
        use_text: bool = True,
        use_audio: bool = True,
        use_video: bool = True,
        use_gated_fusion: bool = True,
        backbone: str = "ctnet",
        n_heads: int = 4,
        n_speakers: int = 10,
        # CHADO toggles
        use_causal: bool = True,
        use_hyperbolic: bool = True,
        use_ot: bool = True,
        use_mad: bool = True,
        proxy_type: str = "mad",      # mad|entropy|margin|mc_var|bald
        # CHADO weights
        w_causal: float = 0.1,
        w_hyperbolic: float = 0.1,
        w_ot: float = 0.05,
        w_mad: float = 0.5,
        w_speaker: float = 0.1,
        w_turn: float = 0.05,
        w_modal: float = 0.05,
        w_context: float = 0.05,
        w_backbone: float = 0.2,   # auxiliary backbone CE loss weight
        use_residual_factor: bool = True,  # include z_e in independence loss
    ):
        super().__init__()
        self._backbone_type = backbone
        self._proj_dim = proj_dim
        n_mods = sum([use_text, use_audio, use_video])

        if backbone == "ctnet":
            from baselines.ctnet.ctnet_model import CTNetModel
            self.base = CTNetModel(
                text_model_name=text_model_name,
                audio_model_name=audio_model_name,
                video_model_name=video_model_name,
                num_classes=num_classes,
                proj_dim=proj_dim,
                n_heads=n_heads,
                dropout=dropout,
                use_text=use_text,
                use_audio=use_audio,
                use_video=use_video,
            )
            self._aux_proj = nn.Linear(n_mods * proj_dim, proj_dim) if n_mods > 1 else nn.Identity()
            self._modality_order = [m for m, u in [
                ("text", use_text), ("audio", use_audio), ("video", use_video)
            ] if u]
        else:
            from models.fusion.trimodal_fusion import TriModalBaseline
            self.base = TriModalBaseline(
                text_model_name=text_model_name,
                audio_model_name=audio_model_name,
                video_model_name=video_model_name,
                num_classes=num_classes,
                proj_dim=proj_dim,
                dropout=dropout,
                use_text=use_text,
                use_audio=use_audio,
                use_video=use_video,
                use_gated_fusion=use_gated_fusion,
            )
            self._aux_proj = nn.Identity()
            self._modality_order = [m for m, u in [
                ("text", use_text), ("audio", use_audio), ("video", use_video)
            ] if u]

        # CHADO component toggles and weights
        self.use_causal     = use_causal
        self.use_hyperbolic = use_hyperbolic
        self.use_ot         = use_ot
        self.use_mad        = use_mad
        self.proxy_type     = proxy_type
        self.w_causal       = w_causal
        self.w_hyperbolic   = w_hyperbolic
        self.w_ot           = w_ot
        self.w_mad          = w_mad
        self.w_speaker           = w_speaker
        self.w_turn              = w_turn
        self.w_modal             = w_modal
        self.w_context           = w_context
        self.w_backbone          = w_backbone
        self.use_residual_factor = use_residual_factor

        if use_causal:
            self.causal = FactorDisentangler(d_model=proj_dim, n_speakers=n_speakers)
            # Factor-based classifier: cat(z_c, z_u, z_m) → num_classes
            self.factor_classifier = nn.Sequential(
                nn.LayerNorm(self.causal.d_cls),
                nn.Linear(self.causal.d_cls, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, num_classes),
            )

        if use_hyperbolic:
            self.hyperbolic = HyperbolicUncertainty(c=1.0)

    # ------------------------------------------------------------------
    def forward(
        self,
        text_input=None,
        audio_wave=None,
        video_frames=None,
        modality_mask=None,
    ):
        backbone_logits, raw_z, gate = self.base(
            text_input=text_input,
            audio_wave=audio_wave,
            video_frames=video_frames,
            modality_mask=modality_mask,
        )
        fused = self._aux_proj(raw_z)     # [B, proj_dim]

        aux = {"backbone_logits": backbone_logits}

        # compute cross-modal similarity for z_m supervision
        modal_sim = _modal_sim_from_raw_z(
            raw_z, self._modality_order, self._proj_dim
        )
        if modal_sim is not None:
            aux["modal_sim"] = modal_sim

        if self.use_causal:
            z_c, z_u, z_t, z_m, z_e = self.causal(fused)
            aux["factors"] = (z_c, z_u, z_t, z_m, z_e)
            # primary logits from factor classifier
            cls_feat = self.causal.cls_features(z_c, z_u, z_m)
            factor_logits = self.factor_classifier(cls_feat)
            aux["factor_logits"] = factor_logits
            aux["cls_feat"] = cls_feat          # needed by mc_var / bald proxy
            # independence loss
            _indep_factors = (z_c, z_u, z_t, z_m, z_e) if self.use_residual_factor \
                             else (z_c, z_u, z_t, z_m)
            aux["indep_loss"] = FactorDisentangler.independence_loss(*_indep_factors)

        if self.use_hyperbolic:
            probs_for_hyp = torch.softmax(
                aux.get("factor_logits", backbone_logits), dim=-1
            )
            aux["hyp_loss"] = self.hyperbolic(fused, probs_for_hyp)

        if self.use_ot:
            aux["ot_loss"] = ot_counterfactual_consistency(fused)

        # primary logits: factor-based when causal is on, backbone otherwise
        logits = aux.get("factor_logits", backbone_logits)
        probs  = torch.softmax(logits.detach(), dim=-1)
        mad_scores = compute_mad_scores(probs).detach()

        return logits, mad_scores, aux

    # ------------------------------------------------------------------
    def loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aux: dict,
        label_smoothing: float = 0.0,
        class_weight: torch.Tensor | None = None,
        causal_signals: dict | None = None,
    ):
        """
        causal_signals (optional):
            speaker_id  : [B] long, -1 = unknown
            turn_norm   : [B] float [0,1], -1 = skip
            context_emb : [B, d_c] float, None = skip
        """
        from .mad import ambiguity_weight_from_probs

        # Primary classification loss (factor logits or backbone logits)
        ce = F.cross_entropy(
            logits, labels, weight=class_weight, label_smoothing=label_smoothing
        )
        total = ce

        # Auxiliary backbone consistency loss
        if self.w_backbone > 0 and "backbone_logits" in aux:
            bl = aux["backbone_logits"]
            if bl.shape == logits.shape:
                total = total + self.w_backbone * F.cross_entropy(
                    bl, labels, weight=class_weight
                )

        # Proxy reweighting (MAD or alternative)
        if self.use_mad and self.w_mad > 0:
            from .proxy import get_ambiguity_weight
            probs = torch.softmax(logits, dim=-1)
            w = get_ambiguity_weight(
                self.proxy_type, probs,
                cls_feat=aux.get("cls_feat"),
                classifier=getattr(self, "factor_classifier", None),
                is_binary=False,
            )
            per_sample = F.cross_entropy(
                logits, labels, weight=class_weight,
                label_smoothing=label_smoothing, reduction="none"
            )
            mad_loss = (w.squeeze(-1) * per_sample).mean()
            if torch.isfinite(mad_loss):
                total = (1.0 - self.w_mad) * total + self.w_mad * mad_loss

        # Causal losses
        if self.use_causal and "factors" in aux:
            z_c, z_u, z_t, z_m, z_e = aux["factors"]

            # independence across all 5 factors
            if torch.isfinite(aux["indep_loss"]):
                total = total + self.w_causal * aux["indep_loss"]

            # auxiliary supervision
            signals = causal_signals or {}
            sup = self.causal.supervision_losses(
                z_c, z_u, z_t, z_m,
                speaker_id  = signals.get("speaker_id"),
                turn_norm   = signals.get("turn_norm"),
                modal_sim   = aux.get("modal_sim"),
                context_emb = signals.get("context_emb"),
            )
            if "speaker" in sup and torch.isfinite(sup["speaker"]):
                total = total + self.w_speaker * sup["speaker"]
            if "turn" in sup and torch.isfinite(sup["turn"]):
                total = total + self.w_turn * sup["turn"]
            if "modal" in sup and torch.isfinite(sup["modal"]):
                total = total + self.w_modal * sup["modal"]
            if "context" in sup and torch.isfinite(sup["context"]):
                total = total + self.w_context * sup["context"]

        # Hyperbolic regularizer
        if self.use_hyperbolic and "hyp_loss" in aux:
            v = aux["hyp_loss"]
            if torch.isfinite(v):
                total = total + self.w_hyperbolic * v

        # OT consistency
        if self.use_ot and "ot_loss" in aux:
            v = aux["ot_loss"]
            if torch.isfinite(v):
                total = total + self.w_ot * v

        return total, ce


# ---------------------------------------------------------------------------
# Feature-based CHADO  (CMU-MOSEI)
# ---------------------------------------------------------------------------

class CHADOFeature(nn.Module):
    """
    CHADO on top of BaselineFusion (pre-extracted features).
    Multi-label classification with BCE.
    Speaker / turn supervision skipped (MOSEI = monologue-level, no dialogue structure).
    z_m supervision via modality agreement is still applied.
    """

    def __init__(
        self,
        num_classes: int = 6,
        d_model: int = 256,
        use_audio: bool = True,
        use_video: bool = True,
        text_model: str = "roberta-base",
        modality_dropout: float = 0.1,
        # CHADO toggles
        use_causal:     bool = True,
        use_hyperbolic: bool = True,
        use_ot:         bool = True,
        use_mad:        bool = True,
        proxy_type:     str  = "mad",   # mad|entropy|margin|mc_var|bald
        # CHADO weights
        w_causal:    float = 0.1,
        w_hyperbolic: float = 0.1,
        w_ot:        float = 0.05,
        w_mad:       float = 0.5,
        w_modal:     float = 0.05,
        mad_gamma:   float = 1.0,
        use_residual_factor: bool = True,
        # OT params
        ot_sigma: float = 0.05,
        ot_eps:   float = 0.1,
        ot_iters: int   = 20,
        # pos_weight for class imbalance
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        from models.fusion.mosei_fusion import BaselineFusion

        self.base = BaselineFusion(
            num_classes=num_classes,
            d_model=d_model,
            use_audio=use_audio,
            use_video=use_video,
            text_model=text_model,
            modality_dropout=modality_dropout,
        )

        self.use_causal     = use_causal
        self.use_hyperbolic = use_hyperbolic
        self.use_ot         = use_ot
        self.use_mad        = use_mad
        self.proxy_type     = proxy_type
        self.w_causal            = w_causal
        self.w_hyperbolic        = w_hyperbolic
        self.w_ot                = w_ot
        self.w_mad               = w_mad
        self.w_modal             = w_modal
        self.mad_gamma           = mad_gamma
        self.use_residual_factor = use_residual_factor
        self.ot_sigma            = ot_sigma
        self.ot_eps              = ot_eps
        self.ot_iters            = ot_iters

        if use_causal:
            # MOSEI has no speaker / turn info → n_speakers unused but must be set
            self.causal = FactorDisentangler(d_model=d_model, n_speakers=1)
            self.factor_classifier = nn.Sequential(
                nn.LayerNorm(self.causal.d_cls),
                nn.Linear(self.causal.d_cls, d_model),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(d_model, num_classes),
            )

        if use_hyperbolic:
            self.hyperbolic = HyperbolicUncertainty(c=1.0)

        self.register_buffer(
            "pos_weight",
            pos_weight if pos_weight is not None else torch.ones(num_classes),
        )

    def forward(self, batch):
        backbone_logits, z = self.base(batch)
        aux = {"backbone_logits": backbone_logits}
        probs = torch.sigmoid(backbone_logits)

        if self.use_causal:
            z_c, z_u, z_t, z_m, z_e = self.causal(z)
            aux["factors"] = (z_c, z_u, z_t, z_m, z_e)
            cls_feat = self.causal.cls_features(z_c, z_u, z_m)
            factor_logits = self.factor_classifier(cls_feat)
            aux["factor_logits"] = factor_logits
            aux["cls_feat"] = cls_feat          # needed by mc_var / bald proxy
            _indep_factors = (z_c, z_u, z_t, z_m, z_e) if self.use_residual_factor \
                             else (z_c, z_u, z_t, z_m)
            aux["indep_loss"] = FactorDisentangler.independence_loss(*_indep_factors)
            probs = torch.sigmoid(factor_logits)

        if self.use_hyperbolic:
            aux["hyp_loss"] = self.hyperbolic(z, probs)

        if self.use_ot:
            aux["ot_loss"] = ot_counterfactual_consistency(
                z, sigma=self.ot_sigma, eps=self.ot_eps, iters=self.ot_iters
            )

        logits = aux.get("factor_logits", backbone_logits)
        probs  = torch.sigmoid(logits.detach())
        mad_scores = compute_mad_scores(probs, gamma=self.mad_gamma).detach()

        return logits, z, mad_scores, aux

    def loss(self, logits, y, z, aux, modal_sim=None):
        from .mad import ambiguity_weight_from_probs

        base_bce = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=self.pos_weight
        )
        total = base_bce

        # auxiliary backbone loss
        if "backbone_logits" in aux:
            bl = aux["backbone_logits"]
            if bl.shape == logits.shape:
                total = total + 0.2 * F.binary_cross_entropy_with_logits(
                    bl, y, pos_weight=self.pos_weight
                )

        # Proxy reweighting (MAD or alternative)
        if self.use_mad and self.w_mad > 0:
            from .proxy import get_ambiguity_weight
            probs = torch.sigmoid(logits)
            w = get_ambiguity_weight(
                self.proxy_type, probs,
                cls_feat=aux.get("cls_feat"),
                classifier=getattr(self, "factor_classifier", None),
                is_binary=True,
                gamma=self.mad_gamma,
            )
            per_sample = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            ).mean(dim=1, keepdim=True)
            mad_loss = (w * per_sample).mean()
            if torch.isfinite(mad_loss):
                total = (1.0 - self.w_mad) * total + self.w_mad * mad_loss

        # Causal independence + z_m supervision
        if self.use_causal and "factors" in aux:
            z_c, z_u, z_t, z_m, z_e = aux["factors"]

            if torch.isfinite(aux["indep_loss"]):
                total = total + self.w_causal * aux["indep_loss"]

            if modal_sim is not None:
                sup = self.causal.supervision_losses(
                    z_c, z_u, z_t, z_m,
                    modal_sim=modal_sim,
                )
                if "modal" in sup and torch.isfinite(sup["modal"]):
                    total = total + self.w_modal * sup["modal"]

        if self.use_hyperbolic and "hyp_loss" in aux:
            if torch.isfinite(aux["hyp_loss"]):
                total = total + self.w_hyperbolic * aux["hyp_loss"]

        if self.use_ot and "ot_loss" in aux:
            if torch.isfinite(aux["ot_loss"]):
                total = total + self.w_ot * aux["ot_loss"]

        return total, base_bce
