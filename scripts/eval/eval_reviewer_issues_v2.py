#!/usr/bin/env python3
"""
Reviewer Issues v2 — Correct metrics aligned with paper (AR-GOT).

Fixes over v1:
  - IEMOCAP : chado_seed456 (79.06%/79.50%); direct CHADOTrimodal construction
  - MOSEI   : per-label binary thresholding (wacc=74.75%, wF1=55.08%); tuned thresholds
  - MELD    : chado_seed123 (66.59%/65.63%); optional (slow video decode)
  - Issue 1.1: MOSEI uses wacc + weighted_f1 per stratum
  - Issue 1.2: Z_cls norm extracted via aux["factors"]
  - Issue 1.3: reads test_results.json (no re-inference needed)
  - Issue 1.4: per-label KL/JSD/Wasserstein vs annotator distributions

Usage:
  CUDA_VISIBLE_DEVICES=5 python3 -u scripts/eval/eval_reviewer_issues_v2.py [--skip-meld]
"""
import os, sys, json, re, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, entropy as scipy_entropy, wasserstein_distance
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
import yaml
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

BASE    = ROOT
RES     = f'{BASE}/experiments/results'
OUT_DIR = f'{BASE}/experiments/results/reviewer_issues'
FIG_DIR = f'{BASE}/experiments/figures/reviewer_issues'
CACHE   = f'{OUT_DIR}/cache'
for d in (OUT_DIR, FIG_DIR, CACHE):
    os.makedirs(d, exist_ok=True)

# ── Colour palette (consistent with paper figures) ───────────────────────────
C = {
    'CHADO':   '#2E4057',
    'MulT':    '#048A81',
    'MM-DFN':  '#C25B5B',
    'CTNet':   '#E07B3F',
    'LF-LSTM': '#6B8CAE',
    'BP-MulT': '#8B5CF6',
    'bg':      '#F7F9FB',
    'grid':    '#DDEAF7',
}
matplotlib.rc('font', family='sans-serif', size=11)
matplotlib.rcParams['axes.spines.top']   = False
matplotlib.rcParams['axes.spines.right'] = False

MOSEI_EMOTIONS = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']

# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_sd(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        return ckpt['model'], ckpt.get('meta', {})
    for key in ('model_state_dict', 'state_dict'):
        if key in ckpt:
            return ckpt[key], {}
    return ckpt, {}


def ent_norm(p, n_cls):
    pnz = p[p > 1e-10]
    if len(pnz) < 2:
        return 0.0
    return float(scipy_entropy(pnz)) / np.log(n_cls)


def compute_ece(probs, labels, n_bins=10):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)
    bins        = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (confidences >= bins[b]) & (confidences < bins[b + 1])
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / max(len(labels), 1))


def jsd(p, q, eps=1e-10):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * scipy_entropy(p, m) + 0.5 * scipy_entropy(q, m))


# ─── Single-label stratification ─────────────────────────────────────────────
def stratified_metrics(probs, labels, amb_scores, n_strata=3):
    """Split into Low / Medium / High terciles; return acc, macro_f1, ECE."""
    order  = np.argsort(amb_scores)
    splits = np.array_split(order, n_strata)
    names  = ['Low', 'Medium', 'High']
    out    = {}
    for name, idx in zip(names, splits):
        p_s = probs[idx]; l_s = labels[idx]; a_s = amb_scores[idx]
        pred = p_s.argmax(axis=1)
        acc  = float(accuracy_score(l_s, pred))
        mf1  = float(f1_score(l_s, pred, average='macro', zero_division=0))
        ece  = compute_ece(p_s, l_s)
        out[name] = {'acc': round(acc,4), 'macro_f1': round(mf1,4),
                     'ece': round(ece,4), 'n': int(len(idx)),
                     'mean_amb': round(float(a_s.mean()),4)}
    return out


# ─── Multi-label stratification (CMU-MOSEI) ──────────────────────────────────
def stratified_metrics_multilabel(sig_probs, bin_labels, amb_scores, thresholds, n_strata=3):
    """
    Split into Low / Medium / High terciles; return wacc + weighted_f1.
    sig_probs : [N, 6] float in (0,1)
    bin_labels: [N, 6] binary int  (label_raw > 0.5)
    thresholds: [6] float
    """
    thr = np.asarray(thresholds)
    order  = np.argsort(amb_scores)
    splits = np.array_split(order, n_strata)
    names  = ['Low', 'Medium', 'High']
    out    = {}
    for name, idx in zip(names, splits):
        sp = sig_probs[idx]          # [m, 6]
        sl = bin_labels[idx]         # [m, 6]
        sa = amb_scores[idx]
        # apply thresholds
        pred = (sp >= thr).astype(int)
        # wacc: mean binary accuracy per class
        bin_accs = [(sl[:, c] == pred[:, c]).mean() for c in range(6)]
        wacc = float(np.mean(bin_accs))
        wf1  = float(f1_score(sl, pred, average='weighted', zero_division=0))
        mf1  = float(f1_score(sl, pred, average='macro', zero_division=0))
        out[name] = {'wacc': round(wacc,4), 'weighted_f1': round(wf1,4),
                     'macro_f1': round(mf1,4), 'n': int(len(idx)),
                     'mean_amb': round(float(sa.mean()),4)}
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Model-independent ambiguity
# ═══════════════════════════════════════════════════════════════════════════════

IEMOCAP_CAT = {
    'neutral': 0, 'neu': 0,
    'happy': 1, 'hap': 1, 'excitement': 1, 'excited': 1, 'exc': 1,
    'anger': 2, 'angry': 2, 'ang': 2, 'frustration': 2, 'fru': 2,
    'sad': 3, 'sadness': 3,
}


def parse_iemocap_annotator_entropy(eval_root):
    files = glob.glob(os.path.join(eval_root, 'Session*/dialog/EmoEvaluation/*.txt'))
    hdr   = re.compile(r'\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+\S+\s+\[')
    rtr   = re.compile(r'C-\w+:\s*([^;]+)')
    out   = {}
    for fp in files:
        content = open(fp, errors='ignore').read()
        pos = [m.start() for m in re.finditer(r'\[[\d.]+\s*-', content)]
        pos.append(len(content))
        for i in range(len(pos) - 1):
            blk = content[pos[i]:pos[i + 1]]
            hm  = hdr.search(blk)
            if not hm: continue
            uid    = hm.group(3)
            labels = []
            for rm in rtr.finditer(blk):
                parts = [p.strip().rstrip(';').strip().lower()
                         for p in rm.group(1).split(';')]
                labs = [IEMOCAP_CAT[p] for p in parts if p in IEMOCAP_CAT]
                if labs: labels.extend(labs)
            if len(labels) >= 2:
                cnt = np.bincount(labels, minlength=4).astype(float)
                out[uid] = ent_norm(cnt / cnt.sum(), 4)
    print(f'  IEMOCAP: {len(out)} utterances, '
          f'{sum(v > 0 for v in out.values())} with disagreement')
    return out


def compute_mosei_label_entropy(manifest_path):
    out = {}
    for line in open(manifest_path):
        s   = json.loads(line)
        raw = np.array(s.get('label_raw', [0.0]*6), float)
        tot = raw.sum()
        out[s['utt_id']] = 0.0 if tot < 1e-6 else ent_norm(raw / tot, 6)
    print(f'  CMU-MOSEI: {len(out)} samples, '
          f'{sum(v > 0 for v in out.values())} non-zero entropy')
    return out


def compute_meld_knn_entropy(test_csv, train_csv, device, k=15):
    from transformers import AutoTokenizer, AutoModel
    from sklearn.neighbors import NearestNeighbors
    EMO_MAP = {'neutral':0,'joy':1,'surprise':2,'anger':3,'sadness':4,'disgust':5,'fear':6}
    tok  = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
    bert = AutoModel.from_pretrained('roberta-base').to(device)
    bert.eval()
    def embed(texts, bs=32):
        embs = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = tok(texts[i:i+bs], padding=True, truncation=True,
                          max_length=128, return_tensors='pt').to(device)
                out = bert(**enc)
                embs.append(out.last_hidden_state[:,0,:].cpu().numpy())
        return np.vstack(embs) if embs else np.zeros((0,768))
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)
    tr_texts  = df_tr['text'].astype(str).tolist()
    tr_labels = np.array([EMO_MAP.get(str(l).lower(),0) for l in df_tr['emotion']])
    te_texts  = df_te['text'].astype(str).tolist()
    print(f'  MELD KNN: embedding {len(tr_texts)} train + {len(te_texts)} test …')
    tr_embs = embed(tr_texts)
    te_embs = embed(te_texts)
    del bert; torch.cuda.empty_cache()
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nn.fit(tr_embs)
    _, indices = nn.kneighbors(te_embs)
    out = {}
    for pos, idx in enumerate(indices):
        cnt  = np.bincount(tr_labels[idx], minlength=7).astype(float)
        out[str(pos)] = ent_norm(cnt/cnt.sum(), 7)
    print(f'  MELD KNN: {len(out)} samples, mean={np.mean(list(out.values())):.4f}')
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Inference — IEMOCAP (CHADOTrimodal + baselines)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_iemocap_chado(cfg, device):
    from models.chado.model import CHADOTrimodal
    mc = cfg['model']; cc = cfg.get('chado', {})
    model = CHADOTrimodal(
        text_model_name=mc['text_model_name'],
        audio_model_name=mc['audio_model_name'],
        video_model_name=mc.get('video_model_name','google/vit-base-patch16-224-in21k'),
        num_classes=cfg['data']['num_classes'],
        proj_dim=mc.get('proj_dim', 256),
        dropout=mc.get('dropout', 0.2),
        use_text=mc.get('use_text', True),
        use_audio=mc.get('use_audio', True),
        use_video=mc.get('use_video', True),
        use_gated_fusion=mc.get('use_gated_fusion', True),
        backbone=cc.get('backbone', 'ctnet'),
        n_heads=mc.get('n_heads', 4),
        n_speakers=cc.get('n_speakers', 10),
        use_causal=cc.get('use_causal', True),
        use_hyperbolic=cc.get('use_hyperbolic', True),
        use_ot=cc.get('use_ot', True),
        use_mad=cc.get('use_mad', False),
        w_causal=cc.get('w_causal', 0.05),
        w_hyperbolic=cc.get('w_hyperbolic', 0.05),
        w_ot=cc.get('w_ot', 0.02),
        w_mad=cc.get('w_mad', 0.5),
        w_speaker=cc.get('w_speaker', 0.1),
        w_turn=cc.get('w_turn', 0.05),
        w_modal=cc.get('w_modal', 0.05),
        w_context=cc.get('w_context', 0.05),
    ).to(device)
    return model


def _build_iemocap_baseline(cfg, device):
    sys.path.insert(0, f'{BASE}/scripts/train')
    from train_baseline import build_model
    return build_model(cfg).to(device)


def _iemocap_dataset(cfg):
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap
    mc = cfg['model']; dc = cfg['data']
    ds = IEMOCAPDataset(
        csv_path=dc['test_csv'],
        text_model_name=mc['text_model_name'],
        image_model_name=mc.get('video_model_name','google/vit-base-patch16-224-in21k'),
        max_text_len=dc.get('max_text_len', 96),
        audio_sr=dc.get('sample_rate', 16000),
        audio_sec=dc.get('max_audio_seconds', 4.0),
        n_frames=dc.get('num_frames', 8),
        use_audio=mc.get('use_audio', True),
        use_video=mc.get('use_video', True),
    )
    return ds, collate_iemocap


@torch.no_grad()
def infer_iemocap(ckpt_path, cfg_path, device, model_key='chado', bs=8):
    """
    Returns dict with keys: utt_ids, probs [N,4], labels [N], z_cls_norms [N] (CHADO only).
    Caches to CACHE/{tag}.npz.
    """
    tag      = f'iemocap_{os.path.basename(os.path.dirname(ckpt_path))}'
    cache_path = f'{CACHE}/{tag}.npz'
    if os.path.exists(cache_path):
        print(f'    [cache hit] {tag}')
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    cfg = yaml.safe_load(open(cfg_path))
    sd, meta = load_sd(ckpt_path)

    is_chado = model_key.startswith('chado') or any('causal' in k for k in sd)
    if is_chado:
        model = _build_iemocap_chado(cfg, device)
    else:
        model = _build_iemocap_baseline(cfg, device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds, collate_fn = _iemocap_dataset(cfg)
    # utt_ids come from dataframe (not in batch — collate_iemocap omits them)
    all_utt_ids = ds.df['utt_id'].astype(str).tolist()
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    all_probs, all_labels, all_norms = [], [], []
    for batch in loader:
        ti = {'input_ids':      batch['input_ids'].to(device),
              'attention_mask': batch['attention_mask'].to(device)}
        wav = batch.get('wav')
        if wav is None: wav = batch.get('audio_wave')
        if wav is not None: wav = wav.to(device)
        pv = batch.get('pixel_values')
        if pv is None: pv = batch.get('video_frames')
        if pv is not None: pv = pv.to(device)

        out = model(text_input=ti, audio_wave=wav, video_frames=pv)
        logits, _, aux = out if isinstance(out, tuple) and len(out) == 3 else (out[0], None, {})
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(batch['label'].tolist())

        if is_chado and 'factors' in aux:
            z_c, z_u, z_t, z_m, z_e = aux['factors']
            z_cls = torch.cat([z_c, z_u, z_m], dim=1)
            norms = z_cls.norm(dim=1).cpu().numpy()
        else:
            norms = np.full(probs.shape[0], float('nan'))
        all_norms.append(norms)

    result = {
        'utt_ids':     np.array(all_utt_ids),
        'probs':       np.vstack(all_probs).astype(np.float32),
        'labels':      np.array(all_labels, dtype=np.int64),
        'z_cls_norms': np.concatenate(all_norms).astype(np.float32),
    }
    np.savez(cache_path, **result)
    print(f'    [saved] {cache_path}')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Inference — CMU-MOSEI (CHADOFeature + baselines)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_mosei_chado(cfg, pos_weight, device):
    from models.chado.model import CHADOFeature
    mc = cfg['model']; cc = cfg.get('chado', {})
    model = CHADOFeature(
        num_classes=cfg['data']['num_classes'],
        d_model=mc.get('d_model', 256),
        use_audio=mc.get('use_audio', True),
        use_video=mc.get('use_video', True),
        text_model=mc.get('text_model_name', 'roberta-base'),
        modality_dropout=mc.get('modality_dropout', 0.1),
        use_causal=cc.get('use_causal', True),
        use_hyperbolic=cc.get('use_hyperbolic', True),
        use_ot=cc.get('use_ot', True),
        use_mad=cc.get('use_mad', True),
        w_causal=cc.get('w_causal', 0.005),
        w_hyperbolic=cc.get('w_hyperbolic', 0.005),
        w_ot=cc.get('w_ot', 0.002),
        w_mad=cc.get('w_mad', 0.02),
        mad_gamma=cc.get('mad_gamma', 0.5),
        ot_sigma=cc.get('ot_sigma', 0.05),
        ot_eps=cc.get('ot_eps', 0.1),
        ot_iters=cc.get('ot_iters', 20),
        pos_weight=pos_weight,
    ).to(device)
    return model


def _build_mosei_baseline(cfg, device):
    sys.path.insert(0, f'{BASE}/scripts/train')
    from train_baseline_mosei import build_model as _bm
    return _bm(cfg).to(device)


def _mosei_dataset(cfg):
    from datasets.mosei.mosei_utt_dataset import MoseiUttDataset, collate_mosei_utt
    dc = cfg['data']
    ds = MoseiUttDataset(
        manifest_path=dc['test_manifest'],
        max_audio_len=dc.get('max_audio_len', 50),
        max_video_len=dc.get('max_video_len', 30),
    )
    return ds, collate_mosei_utt


def _mosei_pos_weight(cfg):
    dc = cfg['data']
    train_labels = np.array([
        json.loads(l)['label']
        for l in open(dc['train_manifest'])
    ], dtype=np.float32)
    pos = train_labels.sum(0).clip(min=1)
    neg = len(train_labels) - pos
    raw_pw = neg / pos
    loss_type = cfg.get('train', {}).get('loss', 'bce_sqrt')
    if loss_type == 'bce_full':
        pw = raw_pw.clip(max=12.0)
    else:
        pw = np.sqrt(raw_pw).clip(max=6.0)
    return torch.tensor(pw, dtype=torch.float32)


@torch.no_grad()
def infer_mosei(ckpt_path, cfg_path, device, model_key='chado', bs=32):
    """
    Returns dict: utt_ids, sig_probs [N,6], bin_labels [N,6], thresholds [6], z_cls_norms [N].
    """
    tag        = f'mosei_{os.path.basename(os.path.dirname(ckpt_path))}'
    cache_path = f'{CACHE}/{tag}.npz'
    if os.path.exists(cache_path):
        print(f'    [cache hit] {tag}')
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    cfg = yaml.safe_load(open(cfg_path))
    sd, meta = load_sd(ckpt_path)
    thresholds = np.array(meta.get('thresholds', [0.5]*6), dtype=np.float32)

    is_chado = model_key.startswith('chado') or any('causal' in k for k in sd)
    try:
        pos_weight = _mosei_pos_weight(cfg).to(device) if is_chado else None
    except Exception:
        pos_weight = None

    if is_chado:
        model = _build_mosei_chado(cfg, pos_weight, device)
    else:
        try:
            model = _build_mosei_baseline(cfg, device)
        except Exception:
            model = _build_mosei_chado(cfg, pos_weight, device)  # fallback
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds, collate_fn = _mosei_dataset(cfg)
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    all_sigs, all_bins, all_uids, all_norms = [], [], [], []
    for batch in loader:
        bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in batch.items()}
        out = model(bd)
        # CHADOFeature returns (logits, z, mad_scores, aux) — 4-tuple
        # baselines return (logits, ...) of varying length
        if isinstance(out, tuple):
            logits = out[0]
            aux    = out[-1] if isinstance(out[-1], dict) else {}
        else:
            logits = out; aux = {}
        sig_p = torch.sigmoid(logits).cpu().numpy()
        all_sigs.append(sig_p)
        all_bins.append((batch['label'].numpy() > 0.5).astype(np.int8))
        all_uids.extend(batch['utt_id'])

        if is_chado and 'factors' in aux:
            z_c, z_u, z_t, z_m, z_e = aux['factors']
            z_cls = torch.cat([z_c, z_u, z_m], dim=1)
            norms = z_cls.norm(dim=1).cpu().numpy()
        else:
            norms = np.full(sig_p.shape[0], float('nan'))
        all_norms.append(norms)

    result = {
        'utt_ids':     np.array(all_uids),
        'sig_probs':   np.vstack(all_sigs).astype(np.float32),
        'bin_labels':  np.vstack(all_bins).astype(np.int8),
        'thresholds':  thresholds,
        'z_cls_norms': np.concatenate(all_norms).astype(np.float32),
    }
    np.savez(cache_path, **result)
    print(f'    [saved] {cache_path}')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Inference — MELD (CHADOTrimodal + baselines)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_meld_chado(cfg, device):
    from models.chado.model import CHADOTrimodal
    mc = cfg['model']; cc = cfg.get('chado', {})
    model = CHADOTrimodal(
        text_model_name=mc['text_model_name'],
        audio_model_name=mc['audio_model_name'],
        video_model_name=mc.get('video_model_name','google/vit-base-patch16-224-in21k'),
        num_classes=cfg['data']['num_classes'],
        proj_dim=mc.get('proj_dim', 256),
        dropout=mc.get('dropout', 0.2),
        use_text=mc.get('use_text', True),
        use_audio=mc.get('use_audio', True),
        use_video=mc.get('use_video', True),
        use_gated_fusion=mc.get('use_gated_fusion', True),
        backbone=cc.get('backbone', 'ctnet'),
        n_heads=mc.get('n_heads', 4),
        n_speakers=cc.get('n_speakers', 7),
        use_causal=cc.get('use_causal', True),
        use_hyperbolic=cc.get('use_hyperbolic', True),
        use_ot=cc.get('use_ot', True),
        use_mad=cc.get('use_mad', True),
        w_causal=cc.get('w_causal', 0.01),
        w_hyperbolic=cc.get('w_hyperbolic', 0.01),
        w_ot=cc.get('w_ot', 0.005),
        w_mad=cc.get('w_mad', 0.5),
        w_speaker=cc.get('w_speaker', 0.1),
        w_turn=cc.get('w_turn', 0.05),
        w_modal=cc.get('w_modal', 0.05),
        w_context=cc.get('w_context', 0.05),
    ).to(device)
    return model


def _build_meld_baseline(cfg, device):
    sys.path.insert(0, f'{BASE}/scripts/train')
    from train_baseline import build_model
    return build_model(cfg).to(device)


def _meld_dataset(cfg):
    from datasets.meld.meld_dataset import (
        MeldDataset, collate_meld, build_label_map_from_order, EMO_ORDER_7
    )
    mc = cfg['model']; dc = cfg['data']
    label_map = build_label_map_from_order(EMO_ORDER_7)
    ds = MeldDataset(
        csv_path=dc['test_csv'], label_map=label_map,
        text_col=dc.get('text_col','text'),
        audio_path_col=dc.get('audio_path_col','audio_path'),
        video_path_col=dc.get('video_path_col','video_path'),
        label_col=dc.get('label_col','emotion'),
        utt_id_col=dc.get('utt_id_col','utt_id'),
        text_model_name=mc['text_model_name'],
        num_frames=dc.get('num_frames', 8),
        frame_size=dc.get('frame_size', 224),
        sample_rate=dc.get('sample_rate', 16000),
        max_audio_seconds=dc.get('max_audio_seconds', 6.0),
    )
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mc['text_model_name'], use_fast=True)
    import functools
    collate = functools.partial(collate_meld, tokenizer=tok,
                                use_text=mc.get('use_text',True),
                                use_audio=mc.get('use_audio',True),
                                use_video=mc.get('use_video',True))
    return ds, collate


@torch.no_grad()
def infer_meld(ckpt_path, cfg_path, device, model_key='chado', bs=8):
    tag        = f'meld_{os.path.basename(os.path.dirname(ckpt_path))}'
    cache_path = f'{CACHE}/{tag}.npz'
    if os.path.exists(cache_path):
        print(f'    [cache hit] {tag}')
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    cfg = yaml.safe_load(open(cfg_path))
    sd, meta = load_sd(ckpt_path)

    is_chado = model_key.startswith('chado') or any('causal' in k for k in sd)
    if is_chado:
        model = _build_meld_chado(cfg, device)
    else:
        model = _build_meld_baseline(cfg, device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds, collate_fn = _meld_dataset(cfg)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4,
                        collate_fn=collate_fn, pin_memory=True)

    all_probs, all_labels, all_norms = [], [], []
    for batch in loader:
        # MeldBatch namedtuple
        ti  = {k: batch.text_input[k].to(device) for k in ('input_ids','attention_mask')}
        wav = batch.audio_wave.to(device) if batch.audio_wave is not None else None
        pv  = batch.video_frames.to(device) if batch.video_frames is not None else None
        lbl = batch.labels  # [B] long

        out = model(text_input=ti, audio_wave=wav, video_frames=pv)
        logits, _, aux = out if isinstance(out, tuple) and len(out) == 3 else (out[0], None, {})
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(lbl.cpu().numpy())

        if is_chado and 'factors' in aux:
            z_c, z_u, z_t, z_m, z_e = aux['factors']
            z_cls = torch.cat([z_c, z_u, z_m], dim=1)
            norms = z_cls.norm(dim=1).cpu().numpy()
        else:
            norms = np.full(probs.shape[0], float('nan'))
        all_norms.append(norms)

    result = {
        'probs':       np.vstack(all_probs).astype(np.float32),
        'labels':      np.concatenate(all_labels).astype(np.int64),
        'z_cls_norms': np.concatenate(all_norms).astype(np.float32),
    }
    np.savez(cache_path, **result)
    print(f'    [saved] {cache_path}')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Model registries
# ═══════════════════════════════════════════════════════════════════════════════

IEMOCAP_MODELS = [
    ('CHADO',   'chado',         f'{BASE}/configs/iemocap/chado_iemocap.yaml',            'chado'),
    ('MulT',    'mult',          f'{BASE}/configs/iemocap/mult_iemocap.yaml',             'mult'),
    ('MM-DFN',  'mmdfn',         f'{BASE}/configs/iemocap/mmdfn_iemocap.yaml',            'mmdfn'),
    ('CTNet',   'ctnet',         f'{BASE}/configs/iemocap/ctnet_iemocap.yaml',            'ctnet'),
    ('LF-LSTM', 'lflstm',        f'{BASE}/configs/iemocap/lflstm_iemocap.yaml',           'lflstm'),
    ('BP-MulT', 'bpmult',        f'{BASE}/configs/iemocap/bpmult_iemocap.yaml',           'bpmult'),
]

MOSEI_MODELS = [
    ('CHADO',   'chado',   f'{BASE}/configs/mosei/chado_mosei.yaml',   'chado'),
    ('MulT',    'mult',    f'{BASE}/configs/mosei/mult_mosei.yaml',     'mult'),
    ('MM-DFN',  'mmdfn',   f'{BASE}/configs/mosei/mmdfn_mosei.yaml',   'mmdfn'),
    ('CTNet',   'ctnet',   f'{BASE}/configs/mosei/ctnet_mosei.yaml',   'ctnet'),
    ('LF-LSTM', 'lflstm',  f'{BASE}/configs/mosei/lflstm_mosei.yaml',  'lflstm'),
    ('BP-MulT', 'bpmult',  f'{BASE}/configs/mosei/bpmult_mosei.yaml',  'bpmult'),
]

MELD_MODELS = [
    ('CHADO',   'chado_seed123', f'{BASE}/configs/meld/chado_meld_seed123.yaml',  'chado'),
    ('MulT',    'mult',          f'{BASE}/configs/meld/mult_meld.yaml',            'mult'),
    ('MM-DFN',  'mmdfn',         f'{BASE}/configs/meld/mmdfn_meld.yaml',           'mmdfn'),
    ('CTNet',   'ctnet',         f'{BASE}/configs/meld/ctnet_meld.yaml',           'ctnet'),
    ('LF-LSTM', 'lflstm',        f'{BASE}/configs/meld/lflstm_meld.yaml',          'lflstm'),
    ('BP-MulT', 'bpmult',        f'{BASE}/configs/meld/bpmult_meld.yaml',          'bpmult'),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 1.1 — Stratified analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_issue11(device, iemocap_amb, mosei_amb, meld_amb, skip_meld=False):
    print('\n' + '='*70)
    print('ISSUE 1.1 — Stratified Analysis (Model-Independent Ambiguity)')
    print('='*70)
    all_results = {}

    # ── IEMOCAP ──────────────────────────────────────────────────────────────
    if iemocap_amb:
        print(f'\n--- IEMOCAP ({len(iemocap_amb)} amb scores) ---')
        iemo_res = {}
        for display, ckpt_dir, cfg_path, mkey in IEMOCAP_MODELS:
            ckpt = f'{RES}/iemocap/{ckpt_dir}/best.pt'
            if not os.path.exists(ckpt):
                print(f'  [{display}] checkpoint missing → skip')
                continue
            if not os.path.exists(cfg_path):
                print(f'  [{display}] config missing: {cfg_path} → skip')
                continue
            print(f'  [{display}] …', end='', flush=True)
            try:
                res = infer_iemocap(ckpt, cfg_path, device, mkey)
                print(f' {len(res["labels"])} samples', end='')
            except Exception as e:
                print(f' ERROR: {e}'); continue

            # align
            uids = res['utt_ids'].tolist()
            amb_a, probs_a, labels_a = [], [], []
            for i, uid in enumerate(uids):
                if uid in iemocap_amb:
                    amb_a.append(iemocap_amb[uid])
                    probs_a.append(res['probs'][i])
                    labels_a.append(res['labels'][i])

            if len(amb_a) < 10:
                print(f' [WARN] only {len(amb_a)} aligned → skip'); continue

            amb_a    = np.array(amb_a)
            probs_a  = np.array(probs_a)
            labels_a = np.array(labels_a)
            strata = stratified_metrics(probs_a, labels_a, amb_a)
            iemo_res[display] = strata
            print()
            for sname, m in strata.items():
                print(f'    {sname:7s}(N={m["n"]:4d}, amb={m["mean_amb"]:.3f}): '
                      f'acc={m["acc"]:.3f}  f1={m["macro_f1"]:.3f}  '
                      f'ECE={m["ece"]:.3f}')
        all_results['IEMOCAP'] = iemo_res

    # ── CMU-MOSEI ─────────────────────────────────────────────────────────────
    if mosei_amb:
        print(f'\n--- CMU-MOSEI ({len(mosei_amb)} amb scores) ---')
        mosei_res = {}
        for display, ckpt_dir, cfg_path, mkey in MOSEI_MODELS:
            ckpt = f'{RES}/mosei/{ckpt_dir}/best.pt'
            if not os.path.exists(ckpt) or not os.path.exists(cfg_path):
                print(f'  [{display}] missing → skip'); continue
            print(f'  [{display}] …', end='', flush=True)
            try:
                res = infer_mosei(ckpt, cfg_path, device, mkey)
                print(f' {len(res["bin_labels"])} samples', end='')
            except Exception as e:
                print(f' ERROR: {e}'); continue

            uids = res['utt_ids'].tolist()
            amb_a, sigs_a, bins_a = [], [], []
            for i, uid in enumerate(uids):
                if uid in mosei_amb:
                    amb_a.append(mosei_amb[uid])
                    sigs_a.append(res['sig_probs'][i])
                    bins_a.append(res['bin_labels'][i])

            if len(amb_a) < 10:
                print(f' [WARN] only {len(amb_a)} aligned → skip'); continue

            strata = stratified_metrics_multilabel(
                np.array(sigs_a), np.array(bins_a),
                np.array(amb_a), res['thresholds']
            )
            mosei_res[display] = strata
            print()
            for sname, m in strata.items():
                print(f'    {sname:7s}(N={m["n"]:4d}, amb={m["mean_amb"]:.3f}): '
                      f'wacc={m["wacc"]:.3f}  wF1={m["weighted_f1"]:.3f}')
        all_results['CMU-MOSEI'] = mosei_res

    # ── MELD ─────────────────────────────────────────────────────────────────
    if meld_amb and not skip_meld:
        print(f'\n--- MELD ({len(meld_amb)} amb scores) ---')
        meld_res = {}
        for display, ckpt_dir, cfg_path, mkey in MELD_MODELS:
            ckpt = f'{RES}/meld/{ckpt_dir}/best.pt'
            if not os.path.exists(ckpt) or not os.path.exists(cfg_path):
                print(f'  [{display}] missing → skip'); continue
            print(f'  [{display}] …', end='', flush=True)
            try:
                res = infer_meld(ckpt, cfg_path, device, mkey)
                print(f' {len(res["labels"])} samples', end='')
            except Exception as e:
                print(f' ERROR: {e}'); continue

            # Positional index alignment
            probs = res['probs']
            labels = res['labels']
            amb_a, probs_a, labels_a = [], [], []
            for pos in range(len(labels)):
                uid = str(pos)
                if uid in meld_amb:
                    amb_a.append(meld_amb[uid])
                    probs_a.append(probs[pos])
                    labels_a.append(labels[pos])

            if len(amb_a) < 10:
                print(f' [WARN] only {len(amb_a)} aligned → skip'); continue

            strata = stratified_metrics(np.array(probs_a), np.array(labels_a), np.array(amb_a))
            meld_res[display] = strata
            print()
            for sname, m in strata.items():
                print(f'    {sname:7s}(N={m["n"]:4d}, amb={m["mean_amb"]:.3f}): '
                      f'acc={m["acc"]:.3f}  f1={m["macro_f1"]:.3f}')
        all_results['MELD'] = meld_res

    out_path = f'{OUT_DIR}/issue11_stratified_v2.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n  Saved → {out_path}')
    _plot_stratified(all_results)
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 1.2 — Radius-ambiguity Spearman correlation
# ═══════════════════════════════════════════════════════════════════════════════

def run_issue12(device, iemocap_amb, mosei_amb):
    print('\n' + '='*70)
    print('ISSUE 1.2 — Radius–Ambiguity Spearman Correlation')
    print('='*70)

    PAIRS = [
        ('IEMOCAP', 'chado_seed456',     f'{BASE}/configs/iemocap/chado_iemocap_seed456.yaml',
         'chado_wo_hyperbolic', f'{BASE}/configs/iemocap/chado_iemocap_wo_hyperbolic.yaml',
         iemocap_amb, 'iemocap'),
        ('CMU-MOSEI', 'chado',           f'{BASE}/configs/mosei/chado_mosei.yaml',
         'chado_wo_hyperbolic', f'{BASE}/configs/mosei/chado_mosei_wo_hyperbolic.yaml',
         mosei_amb, 'mosei'),
    ]

    results = {}
    for ds_name, ckpt_full, cfg_full, ckpt_abl, cfg_abl, amb_dict, ds_key in PAIRS:
        print(f'\n  {ds_name}:')
        res_pairs = {}

        for label, ckpt_dir, cfg_path, mkey in [
            ('CHADO (full)', ckpt_full, cfg_full, 'chado'),
            ('CHADO w/o hyp', ckpt_abl, cfg_abl, 'chado_wo_hyperbolic'),
        ]:
            ckpt = f'{RES}/{ds_key}/{ckpt_dir}/best.pt'
            if not os.path.exists(ckpt) or not os.path.exists(cfg_path):
                print(f'    [{label}] missing → skip'); continue

            try:
                if ds_key == 'iemocap':
                    res = infer_iemocap(ckpt, cfg_path, device, mkey)
                    uids = res['utt_ids'].tolist()
                    norms = res['z_cls_norms']
                    amb_a = np.array([amb_dict.get(u, float('nan')) for u in uids])
                else:
                    res = infer_mosei(ckpt, cfg_path, device, mkey)
                    uids = res['utt_ids'].tolist()
                    norms = res['z_cls_norms']
                    amb_a = np.array([amb_dict.get(u, float('nan')) for u in uids])

                mask = np.isfinite(norms) & np.isfinite(amb_a)
                norms_m = norms[mask]; amb_m = amb_a[mask]
                if len(norms_m) < 20:
                    print(f'    [{label}] insufficient aligned samples'); continue

                rho, pval = spearmanr(norms_m, amb_m)
                print(f'    {label}: Spearman ρ={rho:+.4f}  p={pval:.4f}  N={mask.sum()}')
                res_pairs[label] = {'spearman_r': float(rho), 'spearman_p': float(pval),
                                    'n': int(mask.sum()), 'mean_norm': float(norms_m.mean())}
            except Exception as e:
                print(f'    [{label}] ERROR: {e}')

        results[ds_name] = res_pairs

    out_path = f'{OUT_DIR}/issue12_radius_correlation_v2.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved → {out_path}')
    _plot_radius_corr(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 1.3 — Component ablation (from test_results.json)
# ═══════════════════════════════════════════════════════════════════════════════

def run_issue13():
    print('\n' + '='*70)
    print('ISSUE 1.3 — Component Ablation (IEMOCAP + CMU-MOSEI)')
    print('='*70)

    ABLATIONS_IEMOCAP = [
        ('AR-GOT (Full)',      'chado_seed456',   'acc', 'macro_f1'),
        ('w/o Causal',         'chado_wo_causal', 'acc', 'macro_f1'),
        ('w/o Hyperbolic',     'chado_wo_hyperbolic','acc','macro_f1'),
        ('w/o MAD',            'chado_wo_mad',    'acc', 'macro_f1'),
        ('w/o OT',             'chado_wo_ot',     'acc', 'macro_f1'),
        ('Text-only',          'chado_T',         'acc', 'macro_f1'),
        ('Text+Audio',         'chado_TA',        'acc', 'macro_f1'),
        ('Text+Video',         'chado_TV',        'acc', 'macro_f1'),
    ]

    ABLATIONS_MOSEI = [
        ('AR-GOT (Full)',      'chado',            'wacc', 'weighted_f1'),
        ('w/o Causal',         'chado_wo_causal',  'wacc', 'weighted_f1'),
        ('w/o Hyperbolic',     'chado_wo_hyperbolic','wacc','weighted_f1'),
        ('w/o MAD',            'chado_wo_mad',     'wacc', 'weighted_f1'),
        ('w/o OT',             'chado_wo_ot',      'wacc', 'weighted_f1'),
        ('Text-only',          'chado_T',          'wacc', 'weighted_f1'),
        ('Text+Audio',         'chado_TA',         'wacc', 'weighted_f1'),
        ('Text+Video',         'chado_TV',         'wacc', 'weighted_f1'),
    ]

    results = {'IEMOCAP': {}, 'CMU-MOSEI': {}}

    print('\n  IEMOCAP ablations:')
    print(f'  {"Variant":<24} {"Acc%":>7} {"MacF1%":>8} {"WtF1%":>8}')
    print('  ' + '-'*52)
    for label, ckpt_dir, m1, m2 in ABLATIONS_IEMOCAP:
        path = f'{RES}/iemocap/{ckpt_dir}/test_results.json'
        if not os.path.exists(path):
            print(f'  {label:<24} {"—":>7} {"—":>8}'); continue
        d = json.load(open(path))
        acc = d.get('accuracy', d.get('acc', 0))
        mf1 = d.get('macro_f1', 0)
        wf1 = d.get('weighted_f1', 0)
        print(f'  {label:<24} {acc*100:>6.2f}%  {mf1*100:>7.2f}%  {wf1*100:>7.2f}%')
        results['IEMOCAP'][label] = {'acc': acc, 'macro_f1': mf1, 'weighted_f1': wf1}

    print('\n  CMU-MOSEI ablations:')
    print(f'  {"Variant":<24} {"WAcc%":>7} {"WtF1%":>8} {"MacF1%":>8}')
    print('  ' + '-'*52)
    for label, ckpt_dir, m1, m2 in ABLATIONS_MOSEI:
        path = f'{RES}/mosei/{ckpt_dir}/test_results.json'
        if not os.path.exists(path):
            print(f'  {label:<24} {"—":>7} {"—":>8}'); continue
        d = json.load(open(path))
        wacc = d.get('wacc', 0)
        wf1  = d.get('weighted_f1', 0)
        mf1  = d.get('macro_f1', 0)
        print(f'  {label:<24} {wacc*100:>6.2f}%  {wf1*100:>7.2f}%  {mf1*100:>7.2f}%')
        results['CMU-MOSEI'][label] = {'wacc': wacc, 'weighted_f1': wf1, 'macro_f1': mf1}

    out_path = f'{OUT_DIR}/issue13_ablation_v2.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved → {out_path}')
    _plot_ablation(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 1.4 — KL / JSD / Wasserstein vs annotator distribution
# ═══════════════════════════════════════════════════════════════════════════════

def _divergence_row(pred_dist, anno_dist, eps=1e-8):
    """KL(anno||pred), JSD(anno,pred), Wasserstein."""
    pred  = np.asarray(pred_dist, float).clip(min=eps)
    anno  = np.asarray(anno_dist, float).clip(min=eps)
    pred /= pred.sum(); anno /= anno.sum()
    kl   = float(scipy_entropy(anno, pred))
    jsd_ = jsd(anno, pred)
    wass = float(wasserstein_distance(np.arange(len(pred)), np.arange(len(pred)), anno, pred))
    return kl, jsd_, wass


def run_issue14(device, iemocap_amb, mosei_amb, iemocap_eval_root=None, mosei_manifest=None):
    print('\n' + '='*70)
    print('ISSUE 1.4 — Predicted vs Annotator Distribution Divergence')
    print('='*70)
    results = {}

    # ── IEMOCAP ──────────────────────────────────────────────────────────────
    if iemocap_eval_root and iemocap_amb:
        print('\n  IEMOCAP — KL/JSD/Wass (predicted softmax vs rater-vote distribution):')
        # Load raw annotator vote distributions {uid: np.array [4]}
        anno_dists = _load_iemocap_annotator_dists(iemocap_eval_root)

        ckpt     = f'{RES}/iemocap/chado_seed456/best.pt'
        cfg_path = f'{BASE}/configs/iemocap/chado_iemocap_seed456.yaml'
        if os.path.exists(ckpt):
            try:
                res  = infer_iemocap(ckpt, cfg_path, device, 'chado')
                uids = res['utt_ids'].tolist()
                kls, jsds, wass = [], [], []
                for i, uid in enumerate(uids):
                    if uid not in anno_dists: continue
                    kl, js, wa = _divergence_row(res['probs'][i], anno_dists[uid])
                    kls.append(kl); jsds.append(js); wass.append(wa)
                n = len(kls)
                results['IEMOCAP'] = {
                    'kl_div_mean': round(float(np.mean(kls)), 4),
                    'jsd_mean':    round(float(np.mean(jsds)), 4),
                    'wasserstein_mean': round(float(np.mean(wass)), 4),
                    'n': n,
                }
                print(f'  CHADO  KL={np.mean(kls):.4f}  JSD={np.mean(jsds):.4f}'
                      f'  Wass={np.mean(wass):.4f}  (N={n})')
            except Exception as e:
                print(f'  ERROR: {e}')

    # ── CMU-MOSEI ─────────────────────────────────────────────────────────────
    if mosei_manifest and mosei_amb:
        print('\n  CMU-MOSEI — per-label KL/JSD/Wass (sigmoid prob vs label_raw norm):')
        # Load label_raw distributions {uid: np.array [6]}
        anno_dists_m = {}
        for line in open(mosei_manifest):
            s   = json.loads(line)
            raw = np.array(s.get('label_raw', [0.0]*6), float)
            tot = raw.sum()
            if tot > 1e-6:
                anno_dists_m[s['utt_id']] = raw / tot

        ckpt     = f'{RES}/mosei/chado/best.pt'
        cfg_path = f'{BASE}/configs/mosei/chado_mosei.yaml'
        if os.path.exists(ckpt):
            try:
                res  = infer_mosei(ckpt, cfg_path, device, 'chado')
                uids = res['utt_ids'].tolist()
                kls, jsds, wass = [], [], []
                for i, uid in enumerate(uids):
                    if uid not in anno_dists_m: continue
                    kl, js, wa = _divergence_row(res['sig_probs'][i], anno_dists_m[uid])
                    kls.append(kl); jsds.append(js); wass.append(wa)
                n = len(kls)
                results['CMU-MOSEI'] = {
                    'kl_div_mean': round(float(np.mean(kls)), 4),
                    'jsd_mean':    round(float(np.mean(jsds)), 4),
                    'wasserstein_mean': round(float(np.mean(wass)), 4),
                    'n': n,
                }
                print(f'  CHADO  KL={np.mean(kls):.4f}  JSD={np.mean(jsds):.4f}'
                      f'  Wass={np.mean(wass):.4f}  (N={n})')
            except Exception as e:
                print(f'  ERROR: {e}')

    out_path = f'{OUT_DIR}/issue14_divergence_v2.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved → {out_path}')
    _plot_divergence(results)
    return results


def _load_iemocap_annotator_dists(eval_root):
    """Return {uid: np.array [4]} normalized vote distribution."""
    files = glob.glob(os.path.join(eval_root, 'Session*/dialog/EmoEvaluation/*.txt'))
    hdr   = re.compile(r'\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+\S+\s+\[')
    rtr   = re.compile(r'C-\w+:\s*([^;]+)')
    out   = {}
    for fp in files:
        content = open(fp, errors='ignore').read()
        pos = [m.start() for m in re.finditer(r'\[[\d.]+\s*-', content)]
        pos.append(len(content))
        for i in range(len(pos)-1):
            blk = content[pos[i]:pos[i+1]]
            hm  = hdr.search(blk)
            if not hm: continue
            uid    = hm.group(3)
            labels = []
            for rm in rtr.finditer(blk):
                parts = [p.strip().rstrip(';').strip().lower()
                         for p in rm.group(1).split(';')]
                labs = [IEMOCAP_CAT[p] for p in parts if p in IEMOCAP_CAT]
                if labs: labels.extend(labs)
            if len(labels) >= 2:
                cnt = np.bincount(labels, minlength=4).astype(float)
                out[uid] = cnt / cnt.sum()
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_stratified(all_results):
    strata = ['Low', 'Medium', 'High']
    for ds, ds_results in all_results.items():
        if not ds_results: continue
        methods = list(ds_results.keys())
        x     = np.arange(len(strata))
        width = 0.8 / max(len(methods), 1)

        # Determine primary metrics
        first_val = next(iter(ds_results.values()))
        first_s   = next(iter(first_val.values()))
        if 'wacc' in first_s:
            metrics = [('wacc', 'WAcc (Avg Binary Acc)'), ('weighted_f1', 'Weighted-F1')]
        else:
            metrics = [('acc', 'Accuracy'), ('macro_f1', 'Macro-F1'), ('ece', 'ECE ↓')]

        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
        if len(metrics) == 1: axes = [axes]
        fig.patch.set_facecolor(C['bg'])
        fig.suptitle(f'{ds} — Stratified by Model-Independent Ambiguity',
                     fontweight='bold', fontsize=13, y=1.02)

        for ax_i, (metric, title) in enumerate(metrics):
            ax = axes[ax_i]
            ax.set_facecolor(C['bg'])
            ax.grid(axis='y', color=C['grid'], lw=0.8, alpha=0.7)
            for i, method in enumerate(methods):
                vals  = [ds_results[method].get(s, {}).get(metric, np.nan) for s in strata]
                color = C.get(method, '#9CA3AF')
                ax.bar(x + i*width - 0.4 + width/2, vals, width*0.9,
                       label=method, color=color, alpha=0.85,
                       edgecolor='#1a2740' if method == 'CHADO' else 'white',
                       linewidth=2.0 if method == 'CHADO' else 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(['Low\nAmb.', 'Med.\nAmb.', 'High\nAmb.'], fontsize=10)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title, fontweight='bold')
            if ax_i == 0:
                ax.legend(fontsize=9, loc='upper right')

        plt.tight_layout()
        slug = ds.replace(' ', '_').replace('-', '_').lower()
        for ext in ('pdf', 'png'):
            fig.savefig(f'{FIG_DIR}/issue11_{slug}_v2.{ext}',
                        dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Drop plot: Low→High delta
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fig2.patch.set_facecolor(C['bg'])
        ax2.set_facecolor(C['bg'])
        ax2.grid(axis='y', color=C['grid'], lw=0.8, alpha=0.7)
        metric0 = metrics[0][0]
        drops = []
        for method in methods:
            lo  = ds_results[method].get('Low',{}).get(metric0, np.nan)
            hi  = ds_results[method].get('High',{}).get(metric0, np.nan)
            drops.append(lo - hi if not (np.isnan(lo) or np.isnan(hi)) else np.nan)

        x2 = np.arange(len(methods))
        bars = ax2.bar(x2, drops, color=[C.get(m,'#9CA3AF') for m in methods],
                       alpha=0.85, edgecolor=['#1a2740' if m=='CHADO' else 'white' for m in methods],
                       linewidth=[2.0 if m=='CHADO' else 1.0 for m in methods])
        ax2.axhline(0, color='#555', lw=0.8)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(methods, rotation=20, ha='right')
        ax2.set_ylabel(f'Drop in {metrics[0][1]} (Low→High)', fontsize=11)
        ax2.set_title(f'{ds} — Performance Drop vs Ambiguity', fontweight='bold')
        plt.tight_layout()
        for ext in ('pdf', 'png'):
            fig2.savefig(f'{FIG_DIR}/issue11_{slug}_drop_v2.{ext}',
                         dpi=200, bbox_inches='tight')
        plt.close(fig2)

    print(f'  Stratified figures → {FIG_DIR}/')


def _plot_radius_corr(results):
    fig, axes = plt.subplots(1, len(results), figsize=(7*len(results), 5))
    fig.patch.set_facecolor(C['bg'])
    if len(results) == 1: axes = [axes]
    for ax, (ds, res_pairs) in zip(axes, results.items()):
        ax.set_facecolor(C['bg'])
        ax.grid(color=C['grid'], lw=0.8, alpha=0.7)
        labels = list(res_pairs.keys())
        rhos   = [res_pairs[l].get('spearman_r', 0) for l in labels]
        colors = ['#2E4057' if 'full' in l.lower() or 'Full' in l else '#C25B5B' for l in labels]
        ax.bar(labels, rhos, color=colors, alpha=0.85)
        ax.axhline(0, color='#555', lw=0.8)
        ax.set_title(f'{ds} — Spearman ρ(‖Z_cls‖, Ambiguity)', fontweight='bold')
        ax.set_ylabel('Spearman ρ')
        for i, (l, r) in enumerate(zip(labels, rhos)):
            ax.text(i, r + 0.005 * np.sign(r), f'{r:+.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    for ext in ('pdf','png'):
        fig.savefig(f'{FIG_DIR}/issue12_radius_corr_v2.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def _plot_ablation(results):
    for ds, res in results.items():
        if not res: continue
        labels = list(res.keys())
        metric_key = 'wacc' if 'wacc' in next(iter(res.values())) else 'acc'
        metric2    = 'weighted_f1' if metric_key == 'wacc' else 'macro_f1'
        vals1 = [res[l].get(metric_key, 0)*100 for l in labels]
        vals2 = [res[l].get(metric2, 0)*100 for l in labels]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.4), 5))
        fig.patch.set_facecolor(C['bg'])
        ax.set_facecolor(C['bg'])
        ax.grid(axis='y', color=C['grid'], lw=0.8, alpha=0.7)
        ax.bar(x - width/2, vals1, width, label='WAcc%' if metric_key=='wacc' else 'Acc%',
               color='#2E4057', alpha=0.85)
        ax.bar(x + width/2, vals2, width,
               label='WtF1%' if metric2=='weighted_f1' else 'MacF1%',
               color='#048A81', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Score (%)')
        ax.set_title(f'{ds} — Component Ablation', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        slug = ds.replace(' ', '_').replace('-','_').lower()
        for ext in ('pdf','png'):
            fig.savefig(f'{FIG_DIR}/issue13_{slug}_ablation_v2.{ext}', dpi=200, bbox_inches='tight')
        plt.close(fig)
    print(f'  Ablation figures → {FIG_DIR}/')


def _plot_divergence(results):
    if not results: return
    ds_names = list(results.keys())
    metrics  = ['kl_div_mean', 'jsd_mean', 'wasserstein_mean']
    labels   = ['KL Divergence', 'Jensen-Shannon Div.', 'Wasserstein Dist.']

    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    fig.patch.set_facecolor(C['bg'])
    for ax, metric, mlabel in zip(axes, metrics, labels):
        ax.set_facecolor(C['bg'])
        ax.grid(axis='y', color=C['grid'], lw=0.8, alpha=0.7)
        vals = [results.get(ds, {}).get(metric, 0) for ds in ds_names]
        ax.bar(ds_names, vals, color=['#2E4057','#048A81'][:len(ds_names)], alpha=0.85)
        ax.set_title(mlabel, fontweight='bold')
        ax.set_ylabel(mlabel)
    plt.tight_layout()
    for ext in ('pdf','png'):
        fig.savefig(f'{FIG_DIR}/issue14_divergence_v2.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Divergence figure → {FIG_DIR}/')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-meld', action='store_true',
                   help='Skip slow MELD video-decode inference')
    p.add_argument('--skip-issue11', action='store_true')
    p.add_argument('--skip-issue12', action='store_true')
    p.add_argument('--skip-issue14', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Output : {OUT_DIR}')
    print(f'Figures: {FIG_DIR}')
    print(f'Cache  : {CACHE}')

    # ── Ambiguity scores ──────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('Computing Model-Independent Ambiguity Scores')
    print('='*70)

    IEMOCAP_EVAL_ROOT = '/home/tahirahmad/IEMOCAP_Final/dataset/IEMOCAP_full_release'
    MOSEI_MANIFEST    = f'{BASE}/data/processed/mosei/mosei_utt_test.jsonl'
    MELD_TEST_CSV     = f'{BASE}/data/processed/meld/meld_test.csv'
    MELD_TRAIN_CSV    = f'{BASE}/data/processed/meld/meld_train.csv'

    iemocap_amb = {}
    if os.path.isdir(IEMOCAP_EVAL_ROOT):
        print('\n  IEMOCAP — annotator vote entropy …')
        iemocap_amb = parse_iemocap_annotator_entropy(IEMOCAP_EVAL_ROOT)

    mosei_amb = {}
    if os.path.exists(MOSEI_MANIFEST):
        print('\n  CMU-MOSEI — label_raw entropy …')
        mosei_amb = compute_mosei_label_entropy(MOSEI_MANIFEST)

    meld_amb = {}
    if not args.skip_meld and os.path.exists(MELD_TEST_CSV) and os.path.exists(MELD_TRAIN_CSV):
        meld_cache = f'{CACHE}/meld_knn_entropy.json'
        if os.path.exists(meld_cache):
            meld_amb = json.load(open(meld_cache))
            print(f'\n  MELD — loaded KNN entropy from cache ({len(meld_amb)} samples)')
        else:
            print('\n  MELD — computing KNN soft-label entropy …')
            meld_amb = compute_meld_knn_entropy(MELD_TEST_CSV, MELD_TRAIN_CSV, device)
            json.dump(meld_amb, open(meld_cache, 'w'))

    # ── Run issues ────────────────────────────────────────────────────────────
    all_output = {}

    if not args.skip_issue11:
        all_output['issue_1.1'] = run_issue11(
            device, iemocap_amb, mosei_amb, meld_amb,
            skip_meld=args.skip_meld
        )

    if not args.skip_issue12:
        all_output['issue_1.2'] = run_issue12(device, iemocap_amb, mosei_amb)

    all_output['issue_1.3'] = run_issue13()

    if not args.skip_issue14:
        all_output['issue_1.4'] = run_issue14(
            device, iemocap_amb, mosei_amb,
            iemocap_eval_root=IEMOCAP_EVAL_ROOT if os.path.isdir(IEMOCAP_EVAL_ROOT) else None,
            mosei_manifest=MOSEI_MANIFEST if os.path.exists(MOSEI_MANIFEST) else None,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)

    print('\n━━━ ISSUE 1.1: Stratified Analysis ━━━')
    for ds, ds_res in all_output.get('issue_1.1', {}).items():
        print(f'\n  {ds}:')
        for model, strata in ds_res.items():
            metric = 'wacc' if 'wacc' in next(iter(strata.values())) else 'macro_f1'
            lo = strata.get('Low',{}).get(metric, 0)
            hi = strata.get('High',{}).get(metric, 0)
            mark = ' ←AR-GOT' if model == 'CHADO' else ''
            print(f'    {model:<12} Low={lo:.3f} → High={hi:.3f}  drop={lo-hi:.3f}{mark}')

    print('\n━━━ ISSUE 1.2: Radius Correlation ━━━')
    for ds, pairs in all_output.get('issue_1.2', {}).items():
        for label, r in pairs.items():
            if 'spearman_r' in r:
                print(f'  {ds}/{label}: ρ={r["spearman_r"]:+.4f}  p={r["spearman_p"]:.4f}')

    print('\n━━━ ISSUE 1.3: Ablation ━━━')
    for ds, res in all_output.get('issue_1.3', {}).items():
        print(f'  {ds}:')
        for variant, m in res.items():
            key1 = 'wacc' if 'wacc' in m else 'acc'
            key2 = 'weighted_f1' if 'weighted_f1' in m else 'macro_f1'
            print(f'    {variant:<24} {m.get(key1,0)*100:.2f}%  {m.get(key2,0)*100:.2f}%')

    print('\n━━━ ISSUE 1.4: Distribution Divergence ━━━')
    for ds, r in all_output.get('issue_1.4', {}).items():
        if 'kl_div_mean' in r:
            print(f'  {ds}: KL={r["kl_div_mean"]:.4f}  JSD={r["jsd_mean"]:.4f}'
                  f'  Wass={r["wasserstein_mean"]:.4f}  (N={r["n"]})')

    out_path = f'{OUT_DIR}/all_reviewer_issues_v2.json'
    with open(out_path, 'w') as f:
        json.dump(all_output, f, indent=2, default=str)
    print(f'\n  Full results → {out_path}')
    print(f'  Figures      → {FIG_DIR}/')


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
    main()
