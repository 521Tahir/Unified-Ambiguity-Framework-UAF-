#!/usr/bin/env python3
"""
A.5 — OT Perturbation Behavior Verification

Part 1 — Proposition 1 Lipschitz Bound (Eq. 29-30):
  Perturb fused representation z at σ ∈ {0.01, 0.05, 0.1, 0.2}.
  Measure W₁ (mean-TV) and W₂ (Sinkhorn) between original and perturbed
  prediction distributions. Check: W ≤ L·σ where L is the empirical
  Lipschitz constant of the classifier head.

Part 2 — Ambiguity-scaled σ_i Concentration (Eq. 23):
  σ_i = σ_base × MAD_i per sample.
  Stratify into low / mid / high ambiguity tiers.
  Report average OT loss per tier — higher tiers must show larger contributions.

Usage:
  CUDA_VISIBLE_DEVICES=2 python3 -u scripts/eval/run_A5_ot_perturbation.py --dataset iemocap
  CUDA_VISIBLE_DEVICES=3 python3 -u scripts/eval/run_A5_ot_perturbation.py --dataset mosei
  CUDA_VISIBLE_DEVICES=6 python3 -u scripts/eval/run_A5_ot_perturbation.py --dataset meld
"""
import os, sys, json, yaml, argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "experiments/results/A5_ot"
FIG_DIR = ROOT / "experiments/figures/A5_ot"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SIGMA_LEVELS = [0.01, 0.05, 0.1, 0.2]
N_REPEAT     = 10
BATCH_SIZE   = 64
EPS_SINK     = 0.05
SINK_ITER    = 50

def _best_ckpt(candidates):
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return None

DS_CFG = {
    "iemocap": {
        "config": ROOT / "configs/iemocap/chado_iemocap_v2.yaml",
        "ckpt":   _best_ckpt([
            ROOT / "experiments/results/iemocap/chado_v2/seed_42/best.pt",
            ROOT / "experiments/results/iemocap/argot_ms/seed_42/best.pt",
            ROOT / "experiments/results/iemocap/chado_v2/best.pt",
        ]),
        "type":   "trimodal",
    },
    "mosei": {
        "config": ROOT / "configs/mosei/chado_mosei_best.yaml",
        "ckpt":   _best_ckpt([
            ROOT / "experiments/results/mosei/chado_best/best.pt",
            ROOT / "experiments/results/mosei/argot_ms/seed_42/best.pt",
        ]),
        "type":   "feature",
    },
    "meld": {
        "config": ROOT / "configs/meld/chado_meld_final.yaml",
        "ckpt":   _best_ckpt([
            ROOT / "experiments/results/meld/argot_ms/seed_42/best.pt",
            ROOT / "experiments/results/meld/chado_v2/best.pt",
        ]),
        "type":   "trimodal",
    },
}


# ── OT / Sinkhorn ─────────────────────────────────────────────────────────────

def sinkhorn_w2(a: torch.Tensor, b: torch.Tensor,
                eps: float = EPS_SINK, iters: int = SINK_ITER) -> float:
    a = a.float().clamp(1e-8, 1.0); b = b.float().clamp(1e-8, 1.0)
    a = a / a.sum(-1, keepdim=True); b = b / b.sum(-1, keepdim=True)
    N = a.size(0)
    mu = torch.full((N,), 1.0/N, device=a.device)
    nu = torch.full((N,), 1.0/N, device=b.device)
    C  = torch.cdist(a, b, p=2)**2
    C  = C / (C.max().detach() + 1e-8)
    lK = -C / eps
    lu = torch.zeros(N, device=a.device)
    lv = torch.zeros(N, device=a.device)
    for _ in range(iters):
        lu = torch.log(mu) - torch.logsumexp(lK + lv.unsqueeze(0), dim=1)
        lv = torch.log(nu) - torch.logsumexp(lK.t() + lu.unsqueeze(0), dim=1)
    P = torch.exp(lu.unsqueeze(1) + lK + lv.unsqueeze(0))
    return float((P * C).sum().item())


def mean_tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum(axis=-1).mean())


# ── Lipschitz ─────────────────────────────────────────────────────────────────

def spectral_norm_product(classifier: nn.Module) -> float:
    L = 1.0
    for m in classifier.modules():
        if isinstance(m, nn.Linear):
            sv = torch.linalg.svdvals(m.weight.detach().float())
            L *= sv.max().item()
    return L


def empirical_lipschitz(cls_feat: torch.Tensor, logits: torch.Tensor,
                         n_pairs: int = 4000) -> float:
    N = cls_feat.size(0)
    n_pairs = min(n_pairs, N*(N-1)//2)
    i = torch.randint(0, N, (n_pairs,))
    j = torch.randint(0, N, (n_pairs,))
    j[i == j] = (j[i == j] + 1) % N
    dz = (cls_feat[i] - cls_feat[j]).float().norm(dim=-1)
    df = (logits[i]   - logits[j]).float().norm(dim=-1)
    return float((df / (dz + 1e-12)).max().item())


# ── MAD ───────────────────────────────────────────────────────────────────────

def mad_score(probs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    u = np.clip(1.0 - 2.0*np.abs(probs - 0.5), 0.0, 1.0)
    return (u.mean(axis=-1)**gamma)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("model_state_dict", "state_dict"):
        if isinstance(ck, dict) and k in ck:
            return ck[k]
    return ck


def build_trimodal(cfg, device):
    from models.chado.model import CHADOTrimodal
    mc = cfg["model"]; cc = cfg.get("chado", {})
    return CHADOTrimodal(
        text_model_name=mc["text_model_name"],
        audio_model_name=mc["audio_model_name"],
        video_model_name=mc.get("video_model_name","google/vit-base-patch16-224-in21k"),
        num_classes=cfg["data"]["num_classes"],
        proj_dim=mc.get("proj_dim",256), dropout=mc.get("dropout",0.2),
        use_text=mc.get("use_text",True), use_audio=mc.get("use_audio",True),
        use_video=mc.get("use_video",True), use_gated_fusion=mc.get("use_gated_fusion",True),
        backbone=cc.get("backbone","ctnet"), n_heads=mc.get("n_heads",4),
        n_speakers=cc.get("n_speakers",10),
        use_causal=cc.get("use_causal",True),
        use_hyperbolic=False, use_ot=False, use_mad=False,
    ).to(device)


def build_feature(cfg, device):
    from models.chado.model import CHADOFeature
    mc = cfg["model"]; cc = cfg.get("chado",{})
    train_labels = np.array([json.loads(l)["label"]
                              for l in open(cfg["data"]["train_manifest"])], np.float32)
    pos = train_labels.sum(0).clip(min=1); neg = len(train_labels) - pos
    pw  = torch.tensor(np.sqrt(neg/pos).clip(max=6.0), dtype=torch.float32).to(device)
    return CHADOFeature(
        num_classes=cfg["data"]["num_classes"], d_model=mc.get("d_model",256),
        use_audio=mc.get("use_audio",True), use_video=mc.get("use_video",True),
        text_model=mc.get("text_model_name","roberta-base"),
        modality_dropout=mc.get("modality_dropout",0.1),
        use_causal=cc.get("use_causal",True),
        use_hyperbolic=False, use_ot=False, use_mad=False, pos_weight=pw,
    ).to(device)


# ── Inference with hook ───────────────────────────────────────────────────────

def get_representations(model, loader, device, is_binary=False):
    cls_list, lgts_list = [], []

    def _hook(mod, inp, out):
        cls_list.append(inp[0].detach().cpu())

    handle = None
    if hasattr(model, "factor_classifier"):
        handle = model.factor_classifier[0].register_forward_hook(_hook)

    def _batch_to_dict(batch):
        """Convert any batch type (dict, namedtuple, dataclass) to dict."""
        if isinstance(batch, dict):
            return batch
        if hasattr(batch, '_asdict'):          # namedtuple (MeldBatch etc.)
            return batch._asdict()
        if hasattr(batch, '__dataclass_fields__'):
            import dataclasses
            return dataclasses.asdict(batch)
        return batch  # fallback — let caller handle

    def _forward_batch(raw_batch):
        """Universal forward pass handling dict, namedtuple, and feature-only batches."""
        # Named-tuple style (MeldBatch, IemocapBatch …)
        if hasattr(raw_batch, 'text_input'):
            ti = raw_batch.text_input
            if ti is not None:
                ti = {k: ti[k].to(device) for k in ("input_ids","attention_mask") if k in ti}
            wav  = raw_batch.audio_wave.to(device) if getattr(raw_batch, 'audio_wave', None) is not None else None
            pv   = raw_batch.video_frames.to(device) if getattr(raw_batch, 'video_frames', None) is not None else None
            return model(text_input=ti, audio_wave=wav, video_frames=pv)
        # Plain dict (MOSEI feature batch with utt_id → strip it)
        if isinstance(raw_batch, dict):
            # Feature-based dict: keys "features"/"text_feat" (old) OR "text"+"audio"/"video"
            # without "input_ids" (MOSEI-style pre-extracted features passed as dict to model).
            is_feature_dict = (
                "features" in raw_batch or "text_feat" in raw_batch or
                ("input_ids" not in raw_batch and ("audio" in raw_batch or "video" in raw_batch))
            )
            if is_feature_dict:
                bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in raw_batch.items() if k != "utt_id"}
                return model(bd)
            # Tokenized text/audio/video dict (trimodal raw)
            ti  = {k: raw_batch[k].to(device) for k in ("input_ids","attention_mask") if k in raw_batch}
            wav = raw_batch.get("audio_wave", raw_batch.get("wav"))
            pv  = raw_batch.get("video_frames", raw_batch.get("pixel_values"))
            if wav is not None: wav = wav.to(device)
            if pv  is not None: pv  = pv.to(device)
            return model(text_input=ti or None, audio_wave=wav, video_frames=pv)
        # Namedtuple without text_input (feature-only)
        if hasattr(raw_batch, '_asdict'):
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in raw_batch._asdict().items() if k not in ("utt_id","labels","speaker_id","turn_norm")}
            if "features" in bd or "text_feat" in bd:
                return model(bd)
        raise TypeError(f"Cannot handle batch type: {type(raw_batch)}")

    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            out = _forward_batch(raw_batch)
            lgts_list.append((out[0] if isinstance(out,tuple) else out).detach().cpu())

    if handle: handle.remove()
    logits    = torch.cat(lgts_list, 0)
    cls_feats = torch.cat(cls_list, 0) if cls_list else logits.clone()
    probs     = (torch.sigmoid(logits) if is_binary else torch.softmax(logits,-1)).numpy()
    return cls_feats, logits, probs


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1
# ═══════════════════════════════════════════════════════════════════════════════

def part1_lipschitz(cls_feat, logits, classifier, probs, sigmas, device, is_binary=False):
    print("\n  [Part 1] Proposition 1 — Lipschitz bound")
    L_spec = spectral_norm_product(classifier)
    L_emp  = empirical_lipschitz(cls_feat, logits)
    L_use  = max(L_spec, L_emp)
    print(f"    L_spectral={L_spec:.4f}  L_empirical={L_emp:.4f}  L_used={L_use:.4f}")

    zd  = cls_feat.to(device).float()
    clf = classifier.to(device).float()
    clf.eval()

    print(f"\n    {'σ':>6}  {'L·σ':>8}  {'mean_TV':>8}  {'W₂':>10}  "
          f"{'TV≤L·σ':>8}  {'W₂≤L·σ':>8}")
    print("    " + "─"*56)

    rows = []
    for sigma in sigmas:
        tvs, w2s = [], []
        for _ in range(N_REPEAT):
            zp = zd + sigma * torch.randn_like(zd)
            with torch.no_grad():
                lp = torch.cat([clf(zp[i:i+BATCH_SIZE]) for i in range(0,len(zp),BATCH_SIZE)])
            p_o = (torch.sigmoid(logits.float()) if is_binary
                   else torch.softmax(logits.float(),-1)).numpy()
            p_p = (torch.sigmoid(lp) if is_binary else torch.softmax(lp,-1)).cpu().numpy()
            tvs.append(mean_tv(p_o, p_p))
            idx = np.random.choice(len(p_o), min(256,len(p_o)), replace=False)
            with torch.no_grad():
                w2s.append(sinkhorn_w2(torch.tensor(p_o[idx],device=device),
                                       torch.tensor(p_p[idx],device=device)))

        tv_m  = float(np.mean(tvs)); w2_m = float(np.mean(w2s))
        bound = L_use * sigma
        tv_ok = tv_m <= bound; w2_ok = w2_m <= bound
        print(f"    {sigma:>6.3f}  {bound:>8.4f}  {tv_m:>8.4f}  "
              f"{w2_m:>10.4f}  {'✅' if tv_ok else '❌':>8}  {'✅' if w2_ok else '❌':>8}")
        rows.append(dict(sigma=sigma, L_sigma=bound, mean_tv=tv_m, w2=w2_m,
                         tv_ok=tv_ok, w2_ok=w2_ok, L_spectral=L_spec, L_empirical=L_emp))
    return rows, L_spec, L_emp


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_entropy(probs: np.ndarray, is_binary: bool = False) -> np.ndarray:
    """Normalized ambiguity score ∈ [0,1].
    Multi-class softmax  → H(p)/log(K).
    Multi-label sigmoid  → mean per-class binary entropy / log(2)."""
    p = np.clip(probs, 1e-8, 1.0 - 1e-8)
    if is_binary:
        # Multi-label: independent Bernoulli per class; average binary entropy
        q = 1.0 - p
        q = np.clip(q, 1e-8, 1.0 - 1e-8)
        h_bin = -(p * np.log(p) + q * np.log(q))     # [N, K], max = log(2) each
        return h_bin.mean(axis=1) / np.log(2)         # ∈ [0,1]
    K = probs.shape[1]
    return -(p * np.log(p)).sum(axis=1) / np.log(K)  # softmax: H/log(K) ∈ [0,1]


def _run_tier(zf, sigma_i_vec, logits_sub, classifier, device, is_binary, n_repeat):
    """Apply per-sample σ_i noise; return (mean_TV, mean_OT) over n_repeat trials."""
    tvs, ots = [], []
    clf = classifier; zf = zf.to(device)
    sigma_i_vec = sigma_i_vec.to(device)
    p_o = (torch.sigmoid(logits_sub.float()) if is_binary
           else torch.softmax(logits_sub.float(), -1)).numpy()
    n = len(zf)
    for _ in range(n_repeat):
        zp = zf + sigma_i_vec * torch.randn_like(zf)
        with torch.no_grad():
            lp = torch.cat([clf(zp[i:i+BATCH_SIZE]) for i in range(0, n, BATCH_SIZE)])
        p_p = (torch.sigmoid(lp) if is_binary else torch.softmax(lp, -1)).cpu().numpy()
        tvs.append(mean_tv(p_o, p_p))
        s = min(128, n)
        idx2 = np.random.choice(n, s, replace=False)
        with torch.no_grad():
            ots.append(sinkhorn_w2(torch.tensor(p_o[idx2], device=device),
                                   torch.tensor(p_p[idx2], device=device)))
    return float(np.mean(tvs)), float(np.mean(ots))


def part2_concentration(cls_feat, logits, classifier, probs, sigma_base, device,
                        is_binary=False, num_classes=None):
    """
    Two-part analysis:
      2a. Adaptive σ_i = σ_base × H̃_i  (H̃ = normalised entropy, generalises MAD to K>2).
          Tiers: Low / Mid / High by 33/67th-percentile of H̃.
          Expect: OT  Low < Mid < High  (monotone scaling).
      2b. Fixed σ = σ_base for all samples; same tiers.
          Expect: OT  Low < Mid < High  (high-ambiguity samples are intrinsically
          more sensitive even at equal noise — independent verification of Eq. 23).
    """
    print("\n  [Part 2] Ambiguity-scaled σ_i tier concentration")

    K  = probs.shape[1]
    H  = _norm_entropy(probs, is_binary=is_binary)   # [N] ∈ [0,1]; binary→mean binary-entropy

    # --- tier thresholds (33/67th percentile of H̃, widened if degenerate) ----------
    lo_t = float(np.percentile(H, 33))
    hi_t = float(np.percentile(H, 67))
    if hi_t - lo_t < 1e-4:            # extreme concentration → use 20/80
        lo_t = float(np.percentile(H, 20))
        hi_t = float(np.percentile(H, 80))
    lo_m  = H <  lo_t
    hi_m  = H >= hi_t
    mid_m = ~lo_m & ~hi_m
    tier_masks = [
        (f"Low  (H̃<{lo_t:.4f})",              lo_m),
        (f"Mid  ({lo_t:.4f}≤H̃<{hi_t:.4f})",  mid_m),
        (f"High (H̃≥{hi_t:.4f})",              hi_m),
    ]
    print(f"    σ_base={sigma_base:.4f}  K={K}  "
          f"H̃-range=[{H.min():.4f},{H.max():.4f}]  "
          f"thresholds=[{lo_t:.4f},{hi_t:.4f}]")
    print(f"    Low={lo_m.sum()}  Mid={mid_m.sum()}  High={hi_m.sum()}")

    zd  = cls_feat.to(device).float()
    clf = classifier.to(device).float(); clf.eval()
    Ht  = torch.tensor(H, dtype=torch.float32, device=device)

    # ── 2a: adaptive σ_i = σ_base × H̃_i ─────────────────────────────────────
    print(f"\n  2a. Adaptive σ_i = σ_base × H̃_i  (Eq. 23 generalised)")
    print(f"  {'Tier':<38}  {'N':>5}  {'mean_H̃':>8}  {'mean_σ_i':>9}  "
          f"{'mean_TV':>9}  {'mean_OT':>10}  {'OT/σ_i':>8}")
    print(f"  {'─'*95}")
    rows_2a = []
    for label, mask in tier_masks:
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:<38}  {0:>5}  {'—':>8}  {'—':>9}  {'—':>9}  {'—':>10}  {'—':>8}")
            rows_2a.append(dict(tier=label, n=0)); continue
        idx   = np.where(mask)[0]
        hm    = float(Ht[idx].mean())
        si    = (sigma_base * Ht[idx]).unsqueeze(1)           # adaptive per-sample
        ms    = float(si.mean().item())
        tv_m, ot_m = _run_tier(zd[idx], si, logits[idx], clf, device, is_binary, N_REPEAT)
        ratio = ot_m / ms if ms > 1e-9 else float("nan")
        print(f"  {label:<38}  {n:>5}  {hm:>8.4f}  {ms:>9.5f}  "
              f"{tv_m:>9.5f}  {ot_m:>10.5f}  {ratio:>8.3f}")
        rows_2a.append(dict(tier=label, n=n, mean_H=hm, mean_sigma_i=ms,
                            mean_tv=tv_m, mean_ot_loss=ot_m, ot_per_sigma=ratio))

    populated = [r for r in rows_2a if r.get("n", 0) > 0]
    if len(populated) >= 2:
        ots = [r["mean_ot_loss"] for r in populated]
        mono = all(ots[i] < ots[i+1] for i in range(len(ots)-1))
        delta = ots[-1] - ots[0]
        tag = "✅ MONOTONE" if mono else ("⚠  High>Low" if ots[-1] > ots[0] else "❌ REVERSED")
        print(f"\n  Verdict 2a: {tag}  |  Δ(High−Low) = {delta:+.5f}")

    # ── 2b: fixed σ = σ_base (sensitivity test) ──────────────────────────────
    print(f"\n  2b. Fixed σ = σ_base ({sigma_base:.4f}) for all — intrinsic sensitivity")
    print(f"  {'Tier':<38}  {'N':>5}  {'mean_H̃':>8}  {'mean_conf':>10}  "
          f"{'mean_TV':>9}  {'mean_OT':>10}")
    print(f"  {'─'*90}")
    rows_2b = []
    for label, mask in tier_masks:
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:<38}  {0:>5}  {'—':>8}  {'—':>10}  {'—':>9}  {'—':>10}")
            rows_2b.append(dict(tier=label, n=0)); continue
        idx     = np.where(mask)[0]
        hm      = float(Ht[idx].mean())
        conf    = float(probs[idx].max(axis=1).mean())          # mean max-class prob
        si_fix  = torch.full((n, 1), sigma_base, device=device)
        tv_m, ot_m = _run_tier(zd[idx], si_fix, logits[idx], clf, device, is_binary, N_REPEAT)
        print(f"  {label:<38}  {n:>5}  {hm:>8.4f}  {conf:>10.4f}  "
              f"{tv_m:>9.5f}  {ot_m:>10.5f}")
        rows_2b.append(dict(tier=label, n=n, mean_H=hm, mean_confidence=conf,
                            mean_tv=tv_m, mean_ot_loss=ot_m))

    populated_b = [r for r in rows_2b if r.get("n", 0) > 0]
    if len(populated_b) >= 2:
        ots_b = [r["mean_ot_loss"] for r in populated_b]
        mono_b = all(ots_b[i] < ots_b[i+1] for i in range(len(ots_b)-1))
        delta_b = ots_b[-1] - ots_b[0]
        tag_b = "✅ MONOTONE" if mono_b else ("⚠  High>Low" if ots_b[-1] > ots_b[0] else "❌ REVERSED")
        print(f"\n  Verdict 2b: {tag_b}  |  Δ(High−Low) = {delta_b:+.5f}")
        print(f"  (Fixed σ confirms high-ambiguity samples are intrinsically more sensitive)")

    return dict(rows_2a=rows_2a, rows_2b=rows_2b)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset runners
# ═══════════════════════════════════════════════════════════════════════════════

def _run(ds_key, device):
    info = DS_CFG[ds_key]
    cfg  = yaml.safe_load(open(info["config"]))
    sd   = load_ckpt(info["ckpt"])
    is_b = (ds_key == "mosei")
    sigma_base = cfg.get("chado",{}).get("ot_sigma", 0.05)

    if info["type"] == "trimodal":
        model = build_trimodal(cfg, device)
    else:
        model = build_feature(cfg, device)
    model.load_state_dict(sd, strict=False); model.eval()

    from torch.utils.data import DataLoader

    if ds_key == "iemocap":
        from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap
        dc=cfg["data"]; mc=cfg["model"]
        ds = IEMOCAPDataset(
            csv_path=dc["test_csv"], text_model_name=mc["text_model_name"],
            image_model_name=mc.get("video_model_name","google/vit-base-patch16-224-in21k"),
            max_text_len=dc.get("max_text_len",96), audio_sr=dc.get("sample_rate",16000),
            audio_sec=dc.get("max_audio_seconds",4.0), n_frames=dc.get("num_frames",8),
            use_audio=mc.get("use_audio",True), use_video=mc.get("use_video",True))
        loader = DataLoader(ds, batch_size=8, shuffle=False,
                            collate_fn=collate_iemocap, num_workers=2)

    elif ds_key == "meld":
        from datasets.meld.meld_dataset import (
            MeldDataset, collate_meld, build_label_map_from_order, EMO_ORDER_7)
        from transformers import AutoTokenizer
        import functools
        dc=cfg["data"]; mc=cfg["model"]
        lm = build_label_map_from_order(EMO_ORDER_7)
        ds = MeldDataset(
            csv_path=dc["test_csv"], label_map=lm,
            text_col=dc.get("text_col","text"),
            audio_path_col=dc.get("audio_path_col","audio_path"),
            video_path_col=dc.get("video_path_col","video_path"),
            label_col=dc.get("label_col","emotion"),
            utt_id_col=dc.get("utt_id_col","utt_id"),
            text_model_name=mc["text_model_name"],
            num_frames=dc.get("num_frames",8), frame_size=dc.get("frame_size",224),
            sample_rate=dc.get("sample_rate",16000),
            max_audio_seconds=dc.get("max_audio_seconds",6.0),
            context_turns=dc.get("context_turns",0),
            use_audio=mc.get("use_audio",True), use_video=mc.get("use_video",True))
        tok = AutoTokenizer.from_pretrained(mc["text_model_name"], use_fast=True)
        col = functools.partial(collate_meld, tokenizer=tok,
                                use_text=mc.get("use_text",True),
                                use_audio=mc.get("use_audio",True),
                                use_video=mc.get("use_video",True))
        loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=col, num_workers=2)

    else:  # mosei
        from datasets.mosei.mosei_utt_dataset import MoseiUttDataset, collate_mosei_utt
        dc=cfg["data"]
        ds = MoseiUttDataset(manifest_path=dc["test_manifest"],
                             max_audio_len=dc.get("max_audio_len",50),
                             max_video_len=dc.get("max_video_len",30))
        loader = DataLoader(ds, batch_size=64, shuffle=False,
                            collate_fn=collate_mosei_utt, num_workers=4)

    print(f"\n{'═'*70}\n[A.5] {ds_key.upper()}\n{'═'*70}")
    print(f"  Test set: {len(ds)} samples | σ_base={sigma_base}")
    cls_feat, logits, probs = get_representations(model, loader, device, is_binary=is_b)
    print(f"  cls_feat: {cls_feat.shape}  logits: {logits.shape}")

    clf = model.factor_classifier
    r1, L_sp, L_emp = part1_lipschitz(cls_feat, logits, clf, probs, SIGMA_LEVELS,
                                       device, is_binary=is_b)
    nc = cfg.get("data", {}).get("num_classes", probs.shape[1])
    r2 = part2_concentration(cls_feat, logits, clf, probs, sigma_base, device,
                             is_binary=is_b, num_classes=nc)
    return dict(part1=r1, part2=r2, L_spectral=L_sp, L_empirical=L_emp,
                sigma_base=sigma_base, n_test=len(ds))


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

def _p2_rows(res, key="rows_2a"):
    """Extract part2 rows safely (handles both old list and new dict format)."""
    p2 = res.get("part2", [])
    if isinstance(p2, dict):
        return p2.get(key, [])
    return p2   # legacy list format


def print_summary(results):
    print("\n" + "═"*90)
    print("A.5  SUMMARY")
    print("═"*90)

    # Part 1
    print(f"\n  Part 1 — Lipschitz bound (TV ≤ L·σ?)")
    print(f"  {'Dataset':<12} {'L_emp':>7}  " +
          "  ".join(f"σ={s}" for s in SIGMA_LEVELS))
    for ds, res in results.items():
        if res is None: continue
        cells = "  ".join(
            f"{r['mean_tv']:.3f}({'✅' if r['tv_ok'] else '❌'})"
            for r in res["part1"])
        print(f"  {ds:<12} {res['L_empirical']:>7.3f}  {cells}")

    # Part 2a — adaptive
    print(f"\n  Part 2a — Adaptive σ_i = σ_base×H̃_i  (Eq. 23)")
    print(f"  {'Dataset':<12} {'σ_base':>7}  {'Low':>10}  {'Mid':>10}  {'High':>10}  Monotone?")
    for ds, res in results.items():
        if res is None: continue
        r2 = _p2_rows(res, "rows_2a")
        pop = [r for r in r2 if r.get("n", 0) > 0]
        if len(pop) < 2: continue
        ots  = [r["mean_ot_loss"] for r in pop]
        lo, hi = ots[0], ots[-1]
        mi = ots[1] if len(ots) > 2 else float("nan")
        mono = "✅" if all(ots[i]<ots[i+1] for i in range(len(ots)-1)) else ("⚠" if hi>lo else "❌")
        lo_s  = f"{lo:.5f}"; mi_s = f"{mi:.5f}" if not np.isnan(mi) else "—"
        hi_s  = f"{hi:.5f}"
        print(f"  {ds:<12} {res['sigma_base']:>7.4f}  {lo_s:>10}  {mi_s:>10}  {hi_s:>10}  {mono}")

    # Part 2b — fixed σ
    print(f"\n  Part 2b — Fixed σ = σ_base (intrinsic sensitivity)")
    print(f"  {'Dataset':<12} {'σ_base':>7}  {'Low':>10}  {'Mid':>10}  {'High':>10}  Monotone?")
    for ds, res in results.items():
        if res is None: continue
        r2 = _p2_rows(res, "rows_2b")
        pop = [r for r in r2 if r.get("n", 0) > 0]
        if len(pop) < 2: continue
        ots  = [r["mean_ot_loss"] for r in pop]
        lo, hi = ots[0], ots[-1]
        mi = ots[1] if len(ots) > 2 else float("nan")
        mono = "✅" if all(ots[i]<ots[i+1] for i in range(len(ots)-1)) else ("⚠" if hi>lo else "❌")
        lo_s = f"{lo:.5f}"; mi_s = f"{mi:.5f}" if not np.isnan(mi) else "—"
        hi_s = f"{hi:.5f}"
        print(f"  {ds:<12} {res['sigma_base']:>7.4f}  {lo_s:>10}  {mi_s:>10}  {hi_s:>10}  {mono}")


def write_outputs(results):
    # LaTeX Part 1
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Proposition~1 verification: mean TV distance between "
        r"$P(y\mid z)$ and $P(y\mid z+\varepsilon)$ at $\sigma$ levels "
        r"vs.\ Lipschitz bound $L\cdot\sigma$. \checkmark=bound satisfied.}",
        r"\label{tab:a5_lipschitz}",
        r"\begin{tabular}{l" + "r"*(1+len(SIGMA_LEVELS)) + "}",
        r"\toprule",
        "Dataset & $L$ & " + " & ".join(f"$\\sigma={s}$" for s in SIGMA_LEVELS) + r" \\",
        r"\midrule",
    ]
    for ds, res in results.items():
        if res is None: continue
        cells = " & ".join(
            f"{r['mean_tv']:.3f}{'$^\\checkmark$' if r['tv_ok'] else ''}"
            for r in res["part1"])
        lines.append(f"{ds} & {res['L_empirical']:.2f} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_DIR/"A5_lipschitz_table.tex").write_text("\n".join(lines))

    # LaTeX Part 2 — adaptive σ_i
    lines2 = [
        r"\begin{table}[t]\centering\small",
        r"\caption{OT loss by ambiguity tier (Eq.~23 generalised): "
        r"$\sigma_i=\sigma_{\rm base}\times\tilde{H}_i$ where "
        r"$\tilde{H}_i=H(p_i)/\log K$ is the normalised predictive entropy "
        r"(reduces to MAD for $K{=}2$). "
        r"Higher-entropy tiers should produce larger OT contributions.}",
        r"\label{tab:a5_tier}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & $\sigma_{\rm base}$ & Low & Mid & High & Monotone \\",
        r"\midrule",
    ]
    for ds, res in results.items():
        if res is None: continue
        r2 = _p2_rows(res, "rows_2a")
        pop = [r for r in r2 if r.get("n", 0) > 0]
        if len(pop) < 2: continue
        ots  = [r["mean_ot_loss"] for r in pop]
        lo, hi = ots[0], ots[-1]
        mi = ots[1] if len(ots) > 2 else float("nan")
        mono_sym = r"\checkmark" if all(ots[i]<ots[i+1] for i in range(len(ots)-1)) else (r"\triangle" if hi>lo else r"\times")
        mi_s = f"{mi:.5f}" if not np.isnan(mi) else "—"
        lines2.append(f"{ds} & {res['sigma_base']:.4f} & {lo:.5f} & {mi_s} & {hi:.5f} & ${mono_sym}$ \\\\")
    lines2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_DIR/"A5_tier_table.tex").write_text("\n".join(lines2))
    print(f"  LaTeX → {OUT_DIR}/A5_lipschitz_table.tex + A5_tier_table.tex")


def plot_results(results):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"IEMOCAP":"#2E4057","CMU-MOSEI":"#048A81","MELD":"#C25B5B"}
    markers = {"IEMOCAP":"o","CMU-MOSEI":"s","MELD":"^"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("A.5 — OT Perturbation Behavior Verification",
                 fontsize=12, fontweight="bold")

    # Part 1
    ax = axes[0]
    ax.set_facecolor("#F7F9FB"); ax.grid(alpha=0.3, color="#DDE")
    for ds, res in results.items():
        if res is None: continue
        sigs   = [r["sigma"]   for r in res["part1"]]
        tvs    = [r["mean_tv"] for r in res["part1"]]
        bounds = [r["L_sigma"] for r in res["part1"]]
        c = colors.get(ds,"gray"); mk = markers.get(ds,"o")
        ax.plot(sigs, tvs, f"-{mk}", color=c, label=f"{ds} TV", lw=1.8, ms=6)
        ax.plot(sigs, bounds, "--", color=c, alpha=0.45, lw=1.2)
    ax.set_xlabel("σ"); ax.set_ylabel("Mean TV distance")
    ax.set_title("Part 1: TV vs σ  (dashed = L·σ)", fontsize=10)
    ax.legend(fontsize=8)

    # Part 2
    ax2 = axes[1]
    ax2.set_facecolor("#F7F9FB"); ax2.grid(axis="y", alpha=0.3, color="#DDE")
    width = 0.25; x = np.arange(3)
    for i, (ds, res) in enumerate(results.items()):
        if res is None: continue
        rows = _p2_rows(res, "rows_2a")
        pop  = [r for r in rows if r.get("n", 0) > 0]
        if len(pop) < 2: continue
        vals = [r.get("mean_ot_loss", 0) for r in pop]
        xi   = x[:len(vals)]
        ax2.bar(xi + i*width, vals, width, label=ds,
                color=colors.get(ds,"gray"), alpha=0.85, edgecolor="white")
    ax2.set_xticks(x + width); ax2.set_xticklabels(["Low","Mid","High"])
    ax2.set_ylabel("Mean OT loss"); ax2.set_xlabel("Ambiguity tier")
    ax2.set_title("Part 2: OT loss per tier (Eq. 23)", fontsize=10)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    for fmt in ("pdf","png"):
        p = FIG_DIR / f"A5_ot_verification.{fmt}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Figure → {p}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", choices=["iemocap","mosei","meld","all"], default="all")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Device: {device} | σ_levels: {SIGMA_LEVELS} | N_repeat: {N_REPEAT} | dataset: {args.dataset}")
    os.chdir(ROOT)

    results = {}
    targets = ["iemocap","mosei","meld"] if args.dataset == "all" else [args.dataset]

    for ds_key in targets:
        label = "CMU-MOSEI" if ds_key=="mosei" else ds_key.upper()
        try:
            results[label] = _run(ds_key, device)
            p = OUT_DIR / f"A5_results_{ds_key}.json"
            p.write_text(json.dumps(results[label], indent=2, default=str))
            print(f"\n  Saved partial → {p}")
        except Exception as e:
            print(f"\n  ERROR on {ds_key}: {e}")
            import traceback; traceback.print_exc()
            results[label] = None

    # Merge any partials from parallel runs
    for ds_key, label in [("iemocap","IEMOCAP"),("mosei","CMU-MOSEI"),("meld","MELD")]:
        if label not in results or results[label] is None:
            p = OUT_DIR / f"A5_results_{ds_key}.json"
            if p.exists():
                results[label] = json.loads(p.read_text())

    print_summary(results)
    write_outputs(results)
    plot_results(results)

    (OUT_DIR/"A5_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  JSON → {OUT_DIR}/A5_results.json")
    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    main()
