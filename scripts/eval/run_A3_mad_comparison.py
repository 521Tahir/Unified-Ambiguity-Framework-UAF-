#!/usr/bin/env python3
"""
A.3 — MAD vs Classical Uncertainty Baselines (Table 4 Extension)

Compares AR-GOT's MAD proxy against classical uncertainty proxies via
Pearson r and Spearman ρ correlation with ground-truth annotator ambiguity.

Proxies:
  MAD               CHADO: mean(clamp(1 - 2|p - 0.5|, 0, 1))^γ
  Entropy           Predictive entropy H(p) / log(C), normalized to [0,1]
  Margin            1 − max_p  (confidence gap)
  MC-Var            Mean class variance over K=10 MC-dropout passes
  BALD              H(E[p_k]) − E[H(p_k)]  (information-gain proxy)

Ground-truth annotator ambiguity (model-independent):
  IEMOCAP  — per-utterance normalized vote entropy (EmoEvaluation .txt)
  CMU-MOSEI — normalized entropy of label_raw intensity distribution
  MELD     — KNN soft-label entropy (pre-computed in A.1)

Usage:
  CUDA_VISIBLE_DEVICES=5 python3 -u scripts/eval/run_A3_mad_comparison.py
"""
import os, sys, json, re, glob, yaml, argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr, entropy as scipy_entropy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

BASE    = ROOT
CACHE   = f'{BASE}/experiments/results/reviewer_issues/cache'
OUT_DIR = f'{BASE}/experiments/results/A3_mad'
FIG_DIR = f'{BASE}/experiments/figures/A3_mad'
for d in (OUT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

EPS = 1e-8
MC_K = 10          # MC-dropout passes

# ── Proxy config per dataset ───────────────────────────────────────────────────
IEMOCAP_CACHE  = f'{CACHE}/A1_iemocap_chado.npz'
MOSEI_CACHE    = f'{CACHE}/A1v2_mosei_chado_best.npz'
MELD_CACHE     = f'{CACHE}/A1_meld_chado_v2.npz'

IEMOCAP_MC_CACHE  = f'{CACHE}/A3_iemocap_chado_mc.npz'
MOSEI_MC_CACHE    = f'{CACHE}/A3_mosei_chado_mc.npz'
MELD_MC_CACHE     = f'{CACHE}/A3_meld_chado_mc.npz'

IEMOCAP_CFG  = f'{BASE}/configs/iemocap/chado_iemocap_v2.yaml'
IEMOCAP_CKPT = f'{BASE}/experiments/results/iemocap/chado_v2/best.pt'
MOSEI_CFG    = f'{BASE}/configs/mosei/chado_mosei_best.yaml'
MOSEI_CKPT   = f'{BASE}/experiments/results/mosei/chado_best/best.pt'
MELD_CFG     = f'{BASE}/configs/meld/chado_meld_v2.yaml'
MELD_CKPT    = f'{BASE}/experiments/results/meld/chado_v2/best.pt'

IEMOCAP_ROOT = '/home/tahirahmad/IEMOCAP_Final/dataset/IEMOCAP_full_release'
MOSEI_MANIFEST = f'{BASE}/data/processed/mosei/mosei_utt_test.jsonl'
MELD_KNN_JSON  = f'{CACHE}/meld_knn_entropy.json'

IEMOCAP_CAT = {
    'neutral':0,'neu':0,'happy':1,'hap':1,'excitement':1,'excited':1,'exc':1,
    'anger':2,'angry':2,'ang':2,'frustration':2,'fru':2,'sad':3,'sadness':3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Uncertainty proxy computations (from cached probs, no model needed)
# ═══════════════════════════════════════════════════════════════════════════════

def mad(probs, gamma=1.0):
    """MAD per sample from CHADO model formula. Works for both softmax and sigmoid."""
    u = np.clip(1.0 - 2.0 * np.abs(probs - 0.5), 0.0, 1.0)   # [N, C]
    return (u.mean(axis=-1) ** gamma)                            # [N]


def pred_entropy(probs, is_binary=False):
    """Normalised predictive entropy H(p)/log(C) → [0,1]."""
    if is_binary:
        # MOSEI multi-label: mean binary entropy over 6 emotions, normalised by log(2)
        p = np.clip(probs, EPS, 1.0 - EPS)
        bin_h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))  # [N, 6]
        return bin_h.mean(axis=-1) / np.log(2.0)
    else:
        p = np.clip(probs, EPS, 1.0)
        h = -(p * np.log(p)).sum(axis=-1)                          # [N]
        n_cls = probs.shape[-1]
        return h / np.log(n_cls)


def margin(probs, is_binary=False):
    """1 - max_p uncertainty proxy.
    Multi-label (binary): 1 - max_c(2|sig_c - 0.5|)  (inverted max-certainty).
    """
    if is_binary:
        return 1.0 - (2.0 * np.abs(probs - 0.5)).max(axis=-1)
    return 1.0 - probs.max(axis=-1)


def mc_variance(mc_probs, is_binary=False):
    """Mean per-class variance over MC-dropout passes. mc_probs: [K, N, C]."""
    return mc_probs.var(axis=0).mean(axis=-1)                    # [N]


def bald(mc_probs, is_binary=False):
    """BALD = H(E[p_k]) - E[H(p_k)].  mc_probs: [K, N, C]."""
    n_cls = mc_probs.shape[-1]
    mean_p = mc_probs.mean(axis=0)                               # [N, C]

    if is_binary:
        mean_p_clipped = np.clip(mean_p, EPS, 1.0 - EPS)
        H_mean = -(mean_p_clipped * np.log(mean_p_clipped) +
                   (1 - mean_p_clipped) * np.log(1 - mean_p_clipped))  # [N, 6]
        H_mean_total = H_mean.mean(axis=-1)                       # [N]
        per_k = []
        for k in range(len(mc_probs)):
            pk = np.clip(mc_probs[k], EPS, 1.0 - EPS)
            h  = -(pk * np.log(pk) + (1 - pk) * np.log(1 - pk)) # [N, 6]
            per_k.append(h.mean(axis=-1))
        mean_H_total = np.mean(per_k, axis=0)                    # [N]
    else:
        mean_p_clipped = np.clip(mean_p, EPS, 1.0)
        H_mean_total = -(mean_p_clipped * np.log(mean_p_clipped)).sum(axis=-1)  # [N]
        per_k = []
        for k in range(len(mc_probs)):
            pk = np.clip(mc_probs[k], EPS, 1.0)
            per_k.append(-(pk * np.log(pk)).sum(axis=-1))
        mean_H_total = np.mean(per_k, axis=0)

    return np.clip(H_mean_total - mean_H_total, 0.0, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Ground-truth annotator ambiguity
# ═══════════════════════════════════════════════════════════════════════════════

def get_iemocap_annotator_entropy(eval_root):
    """Returns {uid: normalised_vote_entropy} for utterances with ≥2 annotators."""
    files = glob.glob(os.path.join(eval_root,'Session*/dialog/EmoEvaluation/*.txt'))
    hdr   = re.compile(r'\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+\S+\s+\[')
    rtr   = re.compile(r'C-\w+:\s*([^;]+)')
    out   = {}
    for fp in files:
        content = open(fp, errors='ignore').read()
        pos     = [m.start() for m in re.finditer(r'\[[\d.]+\s*-', content)]
        pos.append(len(content))
        for i in range(len(pos) - 1):
            blk = content[pos[i]:pos[i+1]]
            hm  = hdr.search(blk)
            if not hm: continue
            uid    = hm.group(3)
            labels = []
            for rm in rtr.finditer(blk):
                parts = [p.strip().rstrip(';').strip().lower()
                         for p in rm.group(1).split(';')]
                labels.extend([IEMOCAP_CAT[p] for p in parts if p in IEMOCAP_CAT])
            if len(labels) >= 2:
                cnt = np.bincount(labels, minlength=4).astype(float)
                p   = cnt / cnt.sum()
                pnz = p[p > EPS]
                h   = float(scipy_entropy(pnz)) / np.log(4) if len(pnz) >= 2 else 0.0
                out[uid] = h
    return out


def get_mosei_entropy(manifest_path):
    """Returns {uid: normalised_entropy_of_label_raw}."""
    out = {}
    for line in open(manifest_path):
        s   = json.loads(line)
        raw = np.array(s.get('label_raw', [0.0]*6), float).clip(min=0)
        tot = raw.sum()
        if tot < EPS:
            continue
        p  = raw / tot
        pnz = p[p > EPS]
        h   = float(scipy_entropy(pnz)) / np.log(6) if len(pnz) >= 2 else 0.0
        out[s['utt_id']] = h
    return out


def get_meld_knn_entropy(json_path):
    """Returns list of KNN entropy values aligned with MELD test indices."""
    data = json.load(open(json_path))
    # Keys are '0','1',...  aligned by position to the test CSV order
    n = len(data)
    ents = [float(data[str(i)]) for i in range(n)]
    return np.array(ents)


# ═══════════════════════════════════════════════════════════════════════════════
# MC-dropout model inference
# ═══════════════════════════════════════════════════════════════════════════════

def load_sd(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict'], ckpt
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict'], ckpt
    return ckpt, {}


def enable_mc_dropout(model):
    """Put model in eval mode but keep all Dropout layers in train mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


# ── IEMOCAP MC-dropout ────────────────────────────────────────────────────────

def mc_infer_iemocap(device, K=MC_K):
    if os.path.exists(IEMOCAP_MC_CACHE):
        print(f'  [cache] MC-IEMOCAP')
        d = np.load(IEMOCAP_MC_CACHE, allow_pickle=True)
        return d['mc_probs'], d['utt_ids'].tolist()

    cfg  = yaml.safe_load(open(IEMOCAP_CFG))
    sd, _= load_sd(IEMOCAP_CKPT)

    from models.chado.model import CHADOTrimodal
    mc = cfg['model']; cc = cfg.get('chado', {})
    model = CHADOTrimodal(
        text_model_name=mc['text_model_name'],
        audio_model_name=mc['audio_model_name'],
        video_model_name=mc.get('video_model_name','google/vit-base-patch16-224-in21k'),
        num_classes=cfg['data']['num_classes'],
        proj_dim=mc.get('proj_dim',256), dropout=mc.get('dropout',0.2),
        use_text=mc.get('use_text',True), use_audio=mc.get('use_audio',True),
        use_video=mc.get('use_video',True), use_gated_fusion=mc.get('use_gated_fusion',True),
        backbone=cc.get('backbone','ctnet'), n_heads=mc.get('n_heads',4),
        n_speakers=cc.get('n_speakers',10),
        use_causal=cc.get('use_causal',True), use_hyperbolic=cc.get('use_hyperbolic',True),
        use_ot=cc.get('use_ot',True), use_mad=False,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    enable_mc_dropout(model)

    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap
    dc = cfg['data']
    ds = IEMOCAPDataset(
        csv_path=dc['test_csv'],
        text_model_name=mc['text_model_name'],
        image_model_name=mc.get('video_model_name','google/vit-base-patch16-224-in21k'),
        max_text_len=dc.get('max_text_len',96),
        audio_sr=dc.get('sample_rate',16000),
        audio_sec=dc.get('max_audio_seconds',4.0),
        n_frames=dc.get('num_frames',8),
        use_audio=mc.get('use_audio',True),
        use_video=mc.get('use_video',True),
    )
    utt_ids = ds.df['utt_id'].astype(str).tolist()

    def _one_pass(bs):
        loader = DataLoader(ds, batch_size=bs, shuffle=False,
                            collate_fn=collate_iemocap, num_workers=0)
        probs_list = []
        for batch in loader:
            ti  = {k: batch[k].to(device) for k in ('input_ids','attention_mask')}
            wav = batch.get('wav', batch.get('audio_wave'))
            if wav is not None: wav = wav.to(device)
            pv  = batch.get('pixel_values', batch.get('video_frames'))
            if pv is not None: pv = pv.to(device)
            with torch.no_grad():
                out    = model(text_input=ti, audio_wave=wav, video_frames=pv)
                logits = out[0] if isinstance(out, tuple) else out
                probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(probs_list)

    mc_all = []
    for k in range(K):
        print(f'    IEMOCAP MC pass {k+1}/{K}', end='\r', flush=True)
        for bs in [4, 2, 1]:
            try:
                mc_all.append(_one_pass(bs)); break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
    print()
    mc_probs = np.array(mc_all)  # [K, N, C]
    np.savez(IEMOCAP_MC_CACHE, mc_probs=mc_probs.astype(np.float32),
             utt_ids=np.array(utt_ids))
    print(f'  [saved] {IEMOCAP_MC_CACHE}')
    return mc_probs, utt_ids


# ── CMU-MOSEI MC-dropout ──────────────────────────────────────────────────────

def mc_infer_mosei(device, K=MC_K):
    if os.path.exists(MOSEI_MC_CACHE):
        print(f'  [cache] MC-MOSEI')
        d = np.load(MOSEI_MC_CACHE, allow_pickle=True)
        return d['mc_probs'], d['utt_ids'].tolist()

    cfg  = yaml.safe_load(open(MOSEI_CFG))
    sd, _= load_sd(MOSEI_CKPT)

    from models.chado.model import CHADOFeature
    mc_cfg = cfg['model']; cc = cfg.get('chado', {})
    train_labels = np.array([json.loads(l)['label']
                              for l in open(cfg['data']['train_manifest'])],
                             dtype=np.float32)
    pos = train_labels.sum(0).clip(min=1)
    neg = len(train_labels) - pos
    pw  = torch.tensor(np.sqrt(neg/pos).clip(max=6.0), dtype=torch.float32).to(device)
    model = CHADOFeature(
        num_classes=cfg['data']['num_classes'],
        d_model=mc_cfg.get('d_model',256),
        use_audio=mc_cfg.get('use_audio',True),
        use_video=mc_cfg.get('use_video',True),
        text_model=mc_cfg.get('text_model_name','roberta-base'),
        modality_dropout=mc_cfg.get('modality_dropout',0.1),
        use_causal=cc.get('use_causal',True), use_hyperbolic=cc.get('use_hyperbolic',True),
        use_ot=cc.get('use_ot',True), use_mad=False,
        pos_weight=pw,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    enable_mc_dropout(model)

    from datasets.mosei.mosei_utt_dataset import MoseiUttDataset, collate_mosei_utt
    dc = cfg['data']
    ds = MoseiUttDataset(manifest_path=dc['test_manifest'],
                         max_audio_len=dc.get('max_audio_len',50),
                         max_video_len=dc.get('max_video_len',30))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=collate_mosei_utt)

    utt_ids_ref = []
    mc_all = []
    for k in range(K):
        print(f'    MOSEI MC pass {k+1}/{K}', end='\r', flush=True)
        sigs_k = []; uids_k = []
        for batch in loader:
            bd = {kk: v.to(device) if isinstance(v, torch.Tensor) else v
                  for kk, v in batch.items()}
            with torch.no_grad():
                out    = model(bd)
                logits = out[0] if isinstance(out, tuple) else out
                sigs_k.append(torch.sigmoid(logits).cpu().numpy())
            uids_k.extend(batch['utt_id'])
        mc_all.append(np.vstack(sigs_k))
        if k == 0:
            utt_ids_ref = uids_k
    print()
    mc_probs = np.array(mc_all)  # [K, N, 6]
    np.savez(MOSEI_MC_CACHE, mc_probs=mc_probs.astype(np.float32),
             utt_ids=np.array(utt_ids_ref))
    print(f'  [saved] {MOSEI_MC_CACHE}')
    return mc_probs, utt_ids_ref


# ── MELD MC-dropout ───────────────────────────────────────────────────────────

def mc_infer_meld(device, K=MC_K):
    if os.path.exists(MELD_MC_CACHE):
        print(f'  [cache] MC-MELD')
        d = np.load(MELD_MC_CACHE, allow_pickle=True)
        return d['mc_probs']

    cfg  = yaml.safe_load(open(MELD_CFG))
    sd, _= load_sd(MELD_CKPT)

    from models.chado.model import CHADOTrimodal
    mc_cfg = cfg['model']; cc = cfg.get('chado', {})
    model = CHADOTrimodal(
        text_model_name=mc_cfg['text_model_name'],
        audio_model_name=mc_cfg['audio_model_name'],
        video_model_name=mc_cfg.get('video_model_name','google/vit-base-patch16-224-in21k'),
        num_classes=cfg['data']['num_classes'],
        proj_dim=mc_cfg.get('proj_dim',256), dropout=mc_cfg.get('dropout',0.2),
        use_text=mc_cfg.get('use_text',True), use_audio=mc_cfg.get('use_audio',True),
        use_video=mc_cfg.get('use_video',True),
        use_gated_fusion=mc_cfg.get('use_gated_fusion',True),
        backbone=cc.get('backbone','ctnet'), n_heads=mc_cfg.get('n_heads',4),
        n_speakers=cc.get('n_speakers',7),
        use_causal=cc.get('use_causal',True), use_hyperbolic=cc.get('use_hyperbolic',True),
        use_ot=cc.get('use_ot',True), use_mad=False,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    enable_mc_dropout(model)

    from datasets.meld.meld_dataset import (
        MeldDataset, collate_meld, build_label_map_from_order, EMO_ORDER_7)
    from transformers import AutoTokenizer
    import functools
    dc = cfg['data']
    label_map = build_label_map_from_order(EMO_ORDER_7)
    ds = MeldDataset(
        csv_path=dc['test_csv'], label_map=label_map,
        text_col=dc.get('text_col','text'),
        audio_path_col=dc.get('audio_path_col','audio_path'),
        video_path_col=dc.get('video_path_col','video_path'),
        label_col=dc.get('label_col','emotion'),
        utt_id_col=dc.get('utt_id_col','utt_id'),
        text_model_name=mc_cfg['text_model_name'],
        num_frames=dc.get('num_frames',8), frame_size=dc.get('frame_size',224),
        sample_rate=dc.get('sample_rate',16000),
        max_audio_seconds=dc.get('max_audio_seconds',6.0),
        context_turns=dc.get('context_turns',0),
        use_audio=mc_cfg.get('use_audio',True),
        use_video=mc_cfg.get('use_video',True),
    )
    tok     = AutoTokenizer.from_pretrained(mc_cfg['text_model_name'], use_fast=True)
    collate = functools.partial(collate_meld, tokenizer=tok,
                                use_text=mc_cfg.get('use_text',True),
                                use_audio=mc_cfg.get('use_audio',True),
                                use_video=mc_cfg.get('use_video',True))

    def _one_pass(bs):
        loader = DataLoader(ds, batch_size=bs, shuffle=False,
                            collate_fn=collate, num_workers=0)
        pl = []
        for batch in loader:
            ti  = {k: batch.text_input[k].to(device) for k in ('input_ids','attention_mask')}
            wav = batch.audio_wave.to(device) if batch.audio_wave is not None else None
            pv  = batch.video_frames.to(device) if batch.video_frames is not None else None
            with torch.no_grad():
                out    = model(text_input=ti, audio_wave=wav, video_frames=pv)
                logits = out[0] if isinstance(out, tuple) else out
                pl.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(pl)

    mc_all = []
    for k in range(K):
        print(f'    MELD MC pass {k+1}/{K}', end='\r', flush=True)
        for bs in [8, 4, 2, 1]:
            try:
                mc_all.append(_one_pass(bs)); break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
    print()
    mc_probs = np.array(mc_all)  # [K, N, 7]
    np.savez(MELD_MC_CACHE, mc_probs=mc_probs.astype(np.float32))
    print(f'  [saved] {MELD_MC_CACHE}')
    return mc_probs


# ═══════════════════════════════════════════════════════════════════════════════
# Correlation computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_corr(proxy_vals, gt_vals):
    """Returns (Pearson_r, Spearman_rho) with p-values. Masks NaN."""
    mask = np.isfinite(proxy_vals) & np.isfinite(gt_vals)
    if mask.sum() < 5:
        return (float('nan'), float('nan')), (float('nan'), float('nan'))
    x = proxy_vals[mask]; y = gt_vals[mask]
    pr, pp  = pearsonr(x, y)
    sr, sp  = spearmanr(x, y)
    return (pr, pp), (sr, sp)


def fmt_corr(r, p):
    if np.isnan(r): return '  —   '
    star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '   '))
    return f'{r:+.3f}{star}'


# ═══════════════════════════════════════════════════════════════════════════════
# Per-dataset pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_iemocap(device):
    print('\n[A.3] IEMOCAP ...')
    if not os.path.exists(IEMOCAP_CACHE):
        print('  cache missing — run A.1 first'); return None
    if not os.path.isdir(IEMOCAP_ROOT):
        print(f'  annotator root missing: {IEMOCAP_ROOT}'); return None

    d     = np.load(IEMOCAP_CACHE, allow_pickle=True)
    uids  = d['utt_ids'].tolist()
    probs = d['probs']   # [N, 4]

    gt_dict = get_iemocap_annotator_entropy(IEMOCAP_ROOT)
    gt_vals = np.array([gt_dict.get(u, float('nan')) for u in uids])
    valid   = np.isfinite(gt_vals) & (gt_vals >= 0)
    print(f'  N={valid.sum()} samples aligned with annotator entropy')

    # Non-MC proxies
    proxies = {
        'MAD':     mad(probs),
        'Entropy': pred_entropy(probs),
        'Margin':  margin(probs),
    }

    # MC-dropout proxies
    print('  Computing MC-dropout ...')
    mc_p, mc_uids = mc_infer_iemocap(device)
    # Align MC probs with cached probs order
    uid2idx = {u: i for i, u in enumerate(mc_uids)}
    mc_order = [uid2idx.get(u, -1) for u in uids]
    good_mc  = [i for i in mc_order if i >= 0]
    mask_mc  = np.array([i >= 0 for i in mc_order])
    mc_p_aligned = np.zeros((MC_K, len(uids), 4), dtype=np.float32)
    mc_p_aligned[:, mask_mc, :] = mc_p[:, good_mc, :]

    proxies['MC-Var'] = mc_variance(mc_p_aligned)
    proxies['BALD']   = bald(mc_p_aligned)

    results = {}
    for name, vals in proxies.items():
        pr_tup, sr_tup = compute_corr(vals[valid], gt_vals[valid])
        results[name] = {'pearson': pr_tup, 'spearman': sr_tup, 'n': int(valid.sum())}
        print(f'  {name:<12} Pearson={fmt_corr(*pr_tup)}  Spearman={fmt_corr(*sr_tup)}')

    return results


def run_mosei(device):
    print('\n[A.3] CMU-MOSEI ...')
    if not os.path.exists(MOSEI_CACHE):
        print('  cache missing — run A.1 first'); return None
    if not os.path.exists(MOSEI_MANIFEST):
        print(f'  manifest missing: {MOSEI_MANIFEST}'); return None

    d     = np.load(MOSEI_CACHE, allow_pickle=True)
    uids  = d['utt_ids'].tolist()
    sigs  = d['sig_probs']   # [N, 6]

    gt_dict = get_mosei_entropy(MOSEI_MANIFEST)
    gt_vals = np.array([gt_dict.get(u, float('nan')) for u in uids])
    valid   = np.isfinite(gt_vals)
    print(f'  N={valid.sum()} samples with label_raw entropy')

    proxies = {
        'MAD':     mad(sigs),
        'Entropy': pred_entropy(sigs, is_binary=True),
        'Margin':  margin(sigs, is_binary=True),
    }

    print('  Computing MC-dropout ...')
    mc_s, mc_uids = mc_infer_mosei(device)
    uid2idx = {u: i for i, u in enumerate(mc_uids)}
    mc_order = [uid2idx.get(u, -1) for u in uids]
    good_mc  = [i for i in mc_order if i >= 0]
    mask_mc  = np.array([i >= 0 for i in mc_order])
    mc_s_aligned = np.zeros((MC_K, len(uids), 6), dtype=np.float32)
    mc_s_aligned[:, mask_mc, :] = mc_s[:, good_mc, :]

    proxies['MC-Var'] = mc_variance(mc_s_aligned, is_binary=True)
    proxies['BALD']   = bald(mc_s_aligned, is_binary=True)

    results = {}
    for name, vals in proxies.items():
        pr_tup, sr_tup = compute_corr(vals[valid], gt_vals[valid])
        results[name] = {'pearson': pr_tup, 'spearman': sr_tup, 'n': int(valid.sum())}
        print(f'  {name:<12} Pearson={fmt_corr(*pr_tup)}  Spearman={fmt_corr(*sr_tup)}')

    return results


def run_meld(device):
    print('\n[A.3] MELD ...')
    if not os.path.exists(MELD_CACHE):
        print('  cache missing — run A.1 first'); return None
    if not os.path.exists(MELD_KNN_JSON):
        print(f'  KNN entropy missing: {MELD_KNN_JSON}'); return None

    d     = np.load(MELD_CACHE, allow_pickle=True)
    probs = d['probs']   # [N, 7]
    gt_vals = get_meld_knn_entropy(MELD_KNN_JSON)

    N = min(len(probs), len(gt_vals))
    probs   = probs[:N]
    gt_vals = gt_vals[:N]
    valid   = np.isfinite(gt_vals)
    print(f'  N={valid.sum()} samples with KNN entropy')

    proxies = {
        'MAD':     mad(probs),
        'Entropy': pred_entropy(probs),
        'Margin':  margin(probs),
    }

    print('  Computing MC-dropout ...')
    mc_p = mc_infer_meld(device)
    mc_p = mc_p[:, :N, :]

    proxies['MC-Var'] = mc_variance(mc_p)
    proxies['BALD']   = bald(mc_p)

    results = {}
    for name, vals in proxies.items():
        pr_tup, sr_tup = compute_corr(vals[valid], gt_vals[valid])
        results[name] = {'pearson': pr_tup, 'spearman': sr_tup, 'n': int(valid.sum())}
        print(f'  {name:<12} Pearson={fmt_corr(*pr_tup)}  Spearman={fmt_corr(*sr_tup)}')

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Print full table + framing
# ═══════════════════════════════════════════════════════════════════════════════

PROXY_DISPLAY = [
    ('MAD',     'MAD (ours)'),
    ('Entropy', 'Pred. Entropy'),
    ('Margin',  '1 − max_p'),
    ('MC-Var',  'MC-Var'),
    ('BALD',    'BALD'),
]


def print_table(iemo, mosei, meld):
    print('\n' + '='*100)
    print('A.3  TABLE — Pearson / Spearman Correlation of Uncertainty Proxies with Annotator Ambiguity')
    print('='*100)
    hdr_ds = ['IEMOCAP', 'CMU-MOSEI', 'MELD']
    hdr_sub = 'annotator vote ent.    label_raw ent.    KNN ent.'
    print(f"  {'Proxy':<18}", end='')
    for ds in hdr_ds:
        print(f"  {'Pearson':>10}  {'Spearman':>10}", end='')
    print()
    print(f"  {'':18}", end='')
    for ds, sub in zip(hdr_ds, ['(vote ent.)', '(label_raw H)', '(KNN ent.)']):
        print(f"  {ds+' '+sub:>22}", end='')
    print()
    print('  ' + '─'*96)

    all_data = [('IEMOCAP', iemo), ('CMU-MOSEI', mosei), ('MELD', meld)]

    # Determine best per (dataset, metric)
    best_p = {}; best_s = {}
    for ds, res in all_data:
        if res is None: continue
        max_p = max((res[p]['pearson'][0] for p, _ in PROXY_DISPLAY
                     if p in res and not np.isnan(res[p]['pearson'][0])), default=0)
        max_s = max((res[p]['spearman'][0] for p, _ in PROXY_DISPLAY
                     if p in res and not np.isnan(res[p]['spearman'][0])), default=0)
        best_p[ds] = max_p
        best_s[ds] = max_s

    for proxy_key, proxy_label in PROXY_DISPLAY:
        row = f'  {proxy_label:<18}'
        for ds, res in all_data:
            if res is None or proxy_key not in res:
                row += f"  {'—':>10}  {'—':>10}"
                continue
            pr_r, pr_p  = res[proxy_key]['pearson']
            sr_r, sr_p  = res[proxy_key]['spearman']
            pf = fmt_corr(pr_r, pr_p) if not np.isnan(pr_r) else '  —   '
            sf = fmt_corr(sr_r, sr_p) if not np.isnan(sr_r) else '  —   '
            # Bold indicator for best in column
            pb = '◀' if (not np.isnan(pr_r) and abs(pr_r - best_p.get(ds,0)) < 1e-6) else ' '
            sb = '◀' if (not np.isnan(sr_r) and abs(sr_r - best_s.get(ds,0)) < 1e-6) else ' '
            row += f"  {pf}{pb:1s}  {sf}{sb:1s}"
        print(row)

    print()
    print('  *** p<0.001  ** p<0.01  * p<0.05    ◀ best in column')
    print('  MAD = CHADO ambiguity proxy: mean(clamp(1−2|p−0.5|,0,1))^γ')


def framing(iemo, mosei, meld):
    """Honest framing based on results."""
    print('\n' + '='*72)
    print('A.3  FRAMING — Honest Assessment')
    print('='*72)

    for ds, res, gt_desc in [
        ('IEMOCAP',   iemo,  'annotator vote entropy'),
        ('CMU-MOSEI', mosei, 'label_raw distribution entropy'),
        ('MELD',      meld,  'KNN soft-label entropy'),
    ]:
        if res is None:
            print(f'  {ds}: no results'); continue

        def get_r(proxy, metric='spearman'):
            if proxy not in res: return float('nan')
            return res[proxy][metric][0]

        mad_s  = get_r('MAD')
        ent_s  = get_r('Entropy')
        mar_s  = get_r('Margin')
        mc_s   = get_r('MC-Var')
        bald_s = get_r('BALD')

        non_mad = [x for x in [ent_s, mar_s, mc_s, bald_s] if not np.isnan(x)]
        best_baseline = max(non_mad) if non_mad else float('nan')

        if np.isnan(mad_s):
            verdict = '❓ MAD not evaluated'
        elif mad_s > best_baseline + 0.02:
            verdict = ('✅ MAD STRONGER than all standard proxies '
                       f'(Spearman ρ={mad_s:+.3f} vs best baseline ρ={best_baseline:+.3f}). '
                       'Claim: MAD provides a superior ambiguity signal.')
        elif abs(mad_s - max(non_mad + [mad_s])) < 0.02:
            verdict = (f'⚠  MAD SIMILAR to best baseline (ρ={mad_s:+.3f} vs ρ={best_baseline:+.3f}, '
                       f'Δ<0.02). Reframe as: MAD is a normalised ambiguity scalar—convenient '
                       f'and parameter-free, not a fundamentally novel uncertainty measure.')
        else:
            verdict = (f'ℹ  MAD BELOW best baseline (ρ={mad_s:+.3f} vs ρ={best_baseline:+.3f}). '
                       f'Reframe: MAD is a lightweight curriculum signal (no dropout overhead), '
                       f'trading absolute correlation for computational simplicity.')

        print(f'\n  {ds} (GT: {gt_desc})')
        print(f'    MAD ρ={mad_s:+.4f}  Entropy ρ={ent_s:+.4f}  '
              f'Margin ρ={mar_s:+.4f}  MC-Var ρ={mc_s:+.4f}  BALD ρ={bald_s:+.4f}')
        print(f'    Verdict: {verdict}')


def plot_bars(iemo, mosei, meld):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    datasets   = [('IEMOCAP', iemo), ('CMU-MOSEI', mosei), ('MELD', meld)]
    proxy_keys = [k for k, _ in PROXY_DISPLAY]
    proxy_labels = [l for _, l in PROXY_DISPLAY]
    colors = ['#2E4057', '#A0C4FF', '#C25B5B', '#048A81', '#E07B3F']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('A.3 — MAD vs Classical Uncertainty Proxies\n'
                 'Spearman ρ with Annotator Ambiguity', fontsize=12, fontweight='bold')

    for ax, (ds, res) in zip(axes, datasets):
        ax.set_facecolor('#F7F9FB')
        ax.grid(axis='y', color='#DDEAF7', linewidth=0.8, zorder=0)
        if res is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(ds); continue

        vals = [res[k]['spearman'][0] if k in res and not np.isnan(res[k]['spearman'][0])
                else 0.0 for k in proxy_keys]
        x = np.arange(len(proxy_keys))
        best_idx = int(np.argmax(vals))
        bars = ax.bar(x, vals, color=colors, zorder=3, alpha=0.9, width=0.6,
                      edgecolor='white', linewidth=0.5)
        for xi, v in enumerate(vals):
            fw = 'bold' if xi == best_idx else 'normal'
            ax.text(xi, v + 0.005, f'{v:+.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight=fw)
        ax.set_xticks(x)
        ax.set_xticklabels(proxy_labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Spearman ρ', fontsize=9)
        ax.set_title(ds, fontsize=11, fontweight='bold')
        ax.axhline(0, color='#999', linewidth=0.8, zorder=2)

    from matplotlib.patches import Patch
    handles = [Patch(color=colors[i], label=proxy_labels[i])
               for i in range(len(proxy_keys))]
    fig.legend(handles=handles, loc='lower center', ncol=len(proxy_keys),
               bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    for fmt in ('pdf', 'png'):
        p = f'{FIG_DIR}/A3_mad_comparison.{fmt}'
        plt.savefig(p, dpi=200 if fmt=='pdf' else 150, bbox_inches='tight')
        print(f'  Figure → {p}')
    plt.close()


def write_latex(iemo, mosei, meld):
    proxy_keys   = [k for k, _ in PROXY_DISPLAY]
    proxy_labels = [l for _, l in PROXY_DISPLAY]
    datasets     = [('IEMOCAP', iemo), ('CMU-MOSEI', mosei), ('MELD', meld)]

    def cell(res, proxy, metric):
        if res is None or proxy not in res: return '—'
        r, p = res[proxy][metric]
        if np.isnan(r): return '—'
        star = '^{***}' if p < 0.001 else ('^{**}' if p < 0.01 else ('^{*}' if p < 0.05 else ''))
        return f'${r:+.3f}{star}$'

    # Bold best per column
    best_r = {}
    for ds, res in datasets:
        if res is None: continue
        for metric in ('pearson','spearman'):
            vals = {k: res[k][metric][0] for k in proxy_keys
                    if k in res and not np.isnan(res[k][metric][0])}
            if vals:
                best_r[(ds, metric)] = max(vals, key=vals.get)

    rows = []
    for pk, pl in zip(proxy_keys, proxy_labels):
        row = f'  {pl}'
        for ds, res in datasets:
            for metric in ('pearson','spearman'):
                c = cell(res, pk, metric)
                if (ds, metric) in best_r and best_r[(ds, metric)] == pk:
                    c = c.replace('$','').replace('+','')
                    c = f'$\\mathbf{{{c[1:-1]}}}$' if c != '—' else c
                row += f' & {c}'
        row += ' \\\\'
        rows.append(row)

    hdr = (
        '\\textbf{Proxy} & '
        '\\multicolumn{2}{c}{\\textbf{IEMOCAP}} & '
        '\\multicolumn{2}{c}{\\textbf{CMU-MOSEI}} & '
        '\\multicolumn{2}{c}{\\textbf{MELD}} \\\\\n'
        '& $r_P$ & $\\rho_S$ & $r_P$ & $\\rho_S$ & $r_P$ & $\\rho_S$ \\\\'
    )
    table = (
        '\\begin{table}[t]\n'
        '\\centering\\small\n'
        '\\caption{Pearson $r_P$ and Spearman $\\rho_S$ correlation of uncertainty '
        'proxies with annotator ambiguity (A.3). GT ambiguity: IEMOCAP = normalised '
        'annotator vote entropy; CMU-MOSEI = normalised \\texttt{label\\_raw} intensity entropy; '
        'MELD = KNN soft-label entropy. Significance: $^{*}p{<}0.05$, $^{**}p{<}0.01$, '
        '$^{***}p{<}0.001$. Bold = best in column.}\n'
        '\\label{tab:mad_comparison}\n'
        '\\begin{tabular}{l|cc|cc|cc}\n'
        '\\toprule\n'
        f'{hdr}\n'
        '\\midrule\n'
        + '\n'.join(rows) + '\n'
        '\\bottomrule\n'
        '\\end{tabular}\n'
        '\\end{table}\n'
    )
    tex_path = f'{OUT_DIR}/A3_mad_table.tex'
    with open(tex_path, 'w') as f:
        f.write(table)
    print(f'\n  LaTeX → {tex_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-mc', action='store_true',
                        help='Skip MC-dropout inference (only non-MC proxies)')
    parser.add_argument('--dataset', choices=['iemocap','mosei','meld','all'], default='all',
                        help='Run only one dataset (for parallel execution)')
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f'Device: {device}  | MC-dropout K={MC_K}  | skip-mc={args.skip_mc}  | dataset={args.dataset}')

    os.makedirs(f'{BASE}/experiments/logs', exist_ok=True)

    if args.skip_mc:
        # Non-MC proxies only — fast path
        print('\n[skip-mc] Computing non-MC proxies from cached probs …')

        def non_mc(cache, is_binary=False, uids_key='utt_ids'):
            d = np.load(cache, allow_pickle=True)
            probs = d['probs'] if not is_binary else d['sig_probs']
            return {'probs': probs,
                    'uids': d[uids_key].tolist() if uids_key in d.files else None}

        def build_results_non_mc(probs, is_binary=False):
            return {
                'MAD':     mad(probs),
                'Entropy': pred_entropy(probs, is_binary=is_binary),
                'Margin':  margin(probs, is_binary=is_binary),
            }

        def correlate(proxies, gt_vals, valid):
            results = {}
            for name, vals in proxies.items():
                pr_tup, sr_tup = compute_corr(vals[valid], gt_vals[valid])
                results[name] = {'pearson': pr_tup, 'spearman': sr_tup,
                                 'n': int(valid.sum())}
            return results

        iemo_d    = non_mc(IEMOCAP_CACHE)
        ie_gt     = get_iemocap_annotator_entropy(IEMOCAP_ROOT)
        ie_gt_v   = np.array([ie_gt.get(u, float('nan'))
                               for u in iemo_d['uids']])
        ie_valid  = np.isfinite(ie_gt_v)
        iemo      = correlate(build_results_non_mc(iemo_d['probs']), ie_gt_v, ie_valid)

        mosei_d   = non_mc(MOSEI_CACHE, is_binary=True)
        mo_gt     = get_mosei_entropy(MOSEI_MANIFEST)
        mo_gt_v   = np.array([mo_gt.get(u, float('nan'))
                               for u in mosei_d['uids']])
        mo_valid  = np.isfinite(mo_gt_v)
        mosei     = correlate(build_results_non_mc(mosei_d['probs'], is_binary=True),
                              mo_gt_v, mo_valid)

        meld_d    = np.load(MELD_CACHE, allow_pickle=True)
        meld_probs = meld_d['probs']
        ml_gt_v   = get_meld_knn_entropy(MELD_KNN_JSON)
        N         = min(len(meld_probs), len(ml_gt_v))
        ml_valid  = np.isfinite(ml_gt_v[:N])
        meld      = correlate(build_results_non_mc(meld_probs[:N]),
                              ml_gt_v[:N], ml_valid)

        for name, res in [('IEMOCAP', iemo), ('CMU-MOSEI', mosei), ('MELD', meld)]:
            print(f'\n  {name}:')
            for k in ['MAD','Entropy','Margin']:
                if k in res:
                    pr, pp = res[k]['pearson']
                    sr, sp = res[k]['spearman']
                    print(f'    {k:<12} Pearson={fmt_corr(pr,pp)}  Spearman={fmt_corr(sr,sp)}')
    else:
        ds = args.dataset
        iemo  = run_iemocap(device) if ds in ('iemocap', 'all') else None
        mosei = run_mosei(device)   if ds in ('mosei',   'all') else None
        meld  = run_meld(device)    if ds in ('meld',    'all') else None

        # Save per-dataset partial result so parallel runs can be merged
        if ds != 'all':
            part = {'IEMOCAP': iemo, 'CMU-MOSEI': mosei, 'MELD': meld}
            part_path = f'{OUT_DIR}/A3_results_{ds}.json'
            with open(part_path, 'w') as f:
                json.dump(part, f, indent=2, default=str)
            print(f'\n  Partial JSON → {part_path}')

    # Merge any partial results that exist alongside current run results
    def _load_partial(dataset_key, file_key):
        p = f'{OUT_DIR}/A3_results_{file_key}.json'
        if os.path.exists(p):
            d = json.load(open(p))
            return d.get(dataset_key)
        return None

    if args.dataset == 'all':
        pass  # iemo/mosei/meld already set
    else:
        iemo  = iemo  or _load_partial('IEMOCAP',   'iemocap')
        mosei = mosei or _load_partial('CMU-MOSEI', 'mosei')
        meld  = meld  or _load_partial('MELD',      'meld')

    print_table(iemo, mosei, meld)
    framing(iemo, mosei, meld)

    if iemo or mosei or meld:
        plot_bars(iemo, mosei, meld)
        write_latex(iemo, mosei, meld)

    out = {'IEMOCAP': iemo, 'CMU-MOSEI': mosei, 'MELD': meld}
    with open(f'{OUT_DIR}/A3_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n  JSON → {OUT_DIR}/A3_results.json')
    print('Done.')


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
