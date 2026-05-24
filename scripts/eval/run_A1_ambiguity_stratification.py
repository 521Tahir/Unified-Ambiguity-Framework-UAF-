#!/usr/bin/env python3
"""
A.1 — Model-Independent Ambiguity Stratification (Replaces Figures 3-5)

Ambiguity sources  (model-independent):
  IEMOCAP  : annotator vote entropy from EmoEvaluation .txt files
  CMU-MOSEI: label_raw entropy from mosei_utt_test.jsonl
  MELD     : KNN soft-label entropy (cached); KL omitted for MELD

Models tested:
  AR-GOT (CHADO), OV-MER, AER-LLM, EmoCLIP

Metrics per Low / Medium / High tier:
  Accuracy (wacc for MOSEI), Macro-F1, ECE, KL-divergence

Usage:
  CUDA_VISIBLE_DEVICES=5 python3 -u scripts/eval/run_A1_ambiguity_stratification.py [--skip-meld]
"""
import os, sys, json, re, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
import yaml
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

BASE    = ROOT
RES     = f'{BASE}/experiments/results'
OUT_DIR = f'{BASE}/experiments/results/A1_ambiguity'
FIG_DIR = f'{BASE}/experiments/figures/A1_ambiguity'
CACHE   = f'{BASE}/experiments/results/reviewer_issues/cache'
for d in (OUT_DIR, FIG_DIR, CACHE):
    os.makedirs(d, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
COLORS = {
    'AR-GOT':  '#2E4057',
    'OV-MER':  '#048A81',
    'AER-LLM': '#C25B5B',
    'EmoCLIP': '#E07B3F',
}
matplotlib.rc('font', family='sans-serif', size=11)
matplotlib.rcParams['axes.spines.top']   = False
matplotlib.rcParams['axes.spines.right'] = False

IEMOCAP_CAT = {
    'neutral':0,'neu':0,
    'happy':1,'hap':1,'excitement':1,'excited':1,'exc':1,
    'anger':2,'angry':2,'ang':2,'frustration':2,'fru':2,
    'sad':3,'sadness':3,
}
MOSEI_EMOTIONS = ['happy','sad','anger','surprise','disgust','fear']
STRATA = ['Low','Medium','High']


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ent_norm(p, n_cls):
    pnz = p[p > 1e-10]
    if len(pnz) < 2: return 0.0
    return float(scipy_entropy(pnz)) / np.log(n_cls)


def kl_div(p, q, eps=1e-8):
    """KL(p || q)  — p=annotator, q=model."""
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_ece(probs, labels, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ok   = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for b in range(n_bins):
        m = (conf >= bins[b]) & (conf < bins[b+1])
        if not m.any(): continue
        ece += m.sum() * abs(ok[m].mean() - conf[m].mean())
    return float(ece / max(len(labels), 1))


def compute_ece_multilabel(sig_probs, n_bins=10):
    """Average ECE across 6 binary tasks (MOSEI)."""
    eces = []
    for c in range(sig_probs.shape[1]):
        p = sig_probs[:, c]
        eces.append(float(np.mean(np.abs(p - np.round(p)))))
    return float(np.mean(eces))


def load_sd(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        return ckpt['model'], ckpt.get('meta', {})
    for k in ('model_state_dict','state_dict'):
        if k in ckpt: return ckpt[k], {}
    return ckpt, {}


# ═══════════════════════════════════════════════════════════════════════════════
# Model-independent ambiguity + annotator distributions
# ═══════════════════════════════════════════════════════════════════════════════

def parse_iemocap_annotator_info(eval_root):
    """
    Returns {uid: {'entropy': float, 'dist': np.array([4])}}
    dist is normalised vote distribution over [neu, hap, ang, sad].
    """
    files  = glob.glob(os.path.join(eval_root,'Session*/dialog/EmoEvaluation/*.txt'))
    hdr    = re.compile(r'\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+\S+\s+\[')
    rtr    = re.compile(r'C-\w+:\s*([^;]+)')
    out    = {}
    for fp in files:
        content = open(fp, errors='ignore').read()
        pos     = [m.start() for m in re.finditer(r'\[[\d.]+\s*-', content)]
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
                labels.extend([IEMOCAP_CAT[p] for p in parts if p in IEMOCAP_CAT])
            if len(labels) >= 2:
                cnt  = np.bincount(labels, minlength=4).astype(float)
                dist = cnt / cnt.sum()
                out[uid] = {'entropy': _ent_norm(dist, 4), 'dist': dist}
    print(f'  IEMOCAP: {len(out)} utterances with ≥2 annotators '
          f'({sum(v["entropy"]>0 for v in out.values())} with disagreement)')
    return out


def parse_mosei_annotator_info(manifest_path):
    """
    Returns {uid: {'entropy': float, 'dist': np.array([6])}}
    Uses label_raw (annotator intensity sum per emotion) normalised to a prob dist.
    """
    out = {}
    for line in open(manifest_path):
        s   = json.loads(line)
        raw = np.array(s.get('label_raw', [0.0]*6), float)
        tot = raw.sum()
        if tot < 1e-6:
            dist = np.ones(6)/6
            ent  = 1.0
        else:
            dist = raw / tot
            ent  = _ent_norm(dist, 6)
        out[s['utt_id']] = {'entropy': ent, 'dist': dist}
    print(f'  CMU-MOSEI: {len(out)} samples '
          f'({sum(v["entropy"]>0 for v in out.values())} non-zero entropy)')
    return out


def load_meld_knn_info(cache_json_path):
    """Load pre-computed entropy.  KL dist not available → KL=None for MELD."""
    data = json.load(open(cache_json_path))
    out  = {k: {'entropy': float(v), 'dist': None} for k, v in data.items()}
    print(f'  MELD KNN cache: {len(out)} samples loaded')
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Build models
# ═══════════════════════════════════════════════════════════════════════════════

def _build_iemocap_chado(cfg, device):
    from models.chado.model import CHADOTrimodal
    mc = cfg['model']; cc = cfg.get('chado', {})
    return CHADOTrimodal(
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
        use_ot=cc.get('use_ot',True), use_mad=cc.get('use_mad',False),
        w_causal=cc.get('w_causal',0.05), w_hyperbolic=cc.get('w_hyperbolic',0.05),
        w_ot=cc.get('w_ot',0.02), w_mad=cc.get('w_mad',0.5),
        w_speaker=cc.get('w_speaker',0.1), w_turn=cc.get('w_turn',0.05),
        w_modal=cc.get('w_modal',0.05), w_context=cc.get('w_context',0.05),
    ).to(device)


def _build_iemocap_baseline(cfg, device):
    sys.path.insert(0, f'{BASE}/scripts/train')
    from train_baseline import build_model
    return build_model(cfg).to(device)


def _build_meld_chado(cfg, device):
    from models.chado.model import CHADOTrimodal
    mc = cfg['model']; cc = cfg.get('chado', {})
    return CHADOTrimodal(
        text_model_name=mc['text_model_name'],
        audio_model_name=mc['audio_model_name'],
        video_model_name=mc.get('video_model_name','google/vit-base-patch16-224-in21k'),
        num_classes=cfg['data']['num_classes'],
        proj_dim=mc.get('proj_dim',256), dropout=mc.get('dropout',0.2),
        use_text=mc.get('use_text',True), use_audio=mc.get('use_audio',True),
        use_video=mc.get('use_video',True), use_gated_fusion=mc.get('use_gated_fusion',True),
        backbone=cc.get('backbone','ctnet'), n_heads=mc.get('n_heads',4),
        n_speakers=cc.get('n_speakers',7),
        use_causal=cc.get('use_causal',True), use_hyperbolic=cc.get('use_hyperbolic',True),
        use_ot=cc.get('use_ot',True), use_mad=cc.get('use_mad',True),
        w_causal=cc.get('w_causal',0.01), w_hyperbolic=cc.get('w_hyperbolic',0.01),
        w_ot=cc.get('w_ot',0.005), w_mad=cc.get('w_mad',0.5),
        w_speaker=cc.get('w_speaker',0.1), w_turn=cc.get('w_turn',0.05),
        w_modal=cc.get('w_modal',0.05), w_context=cc.get('w_context',0.05),
    ).to(device)


def _build_mosei_chado(cfg, device):
    from models.chado.model import CHADOFeature
    mc = cfg['model']; cc = cfg.get('chado', {})
    train_manifest = cfg['data']['train_manifest']
    train_labels   = np.array([json.loads(l)['label']
                                for l in open(train_manifest)], dtype=np.float32)
    pos   = train_labels.sum(0).clip(min=1)
    neg   = len(train_labels) - pos
    pw    = torch.tensor(np.sqrt(neg/pos).clip(max=6.0), dtype=torch.float32).to(device)
    return CHADOFeature(
        num_classes=cfg['data']['num_classes'],
        d_model=mc.get('d_model',256),
        use_audio=mc.get('use_audio',True), use_video=mc.get('use_video',True),
        text_model=mc.get('text_model_name','roberta-base'),
        modality_dropout=mc.get('modality_dropout',0.1),
        use_causal=cc.get('use_causal',True), use_hyperbolic=cc.get('use_hyperbolic',True),
        use_ot=cc.get('use_ot',True), use_mad=cc.get('use_mad',True),
        w_causal=cc.get('w_causal',0.005), w_hyperbolic=cc.get('w_hyperbolic',0.005),
        w_ot=cc.get('w_ot',0.002), w_mad=cc.get('w_mad',0.02),
        mad_gamma=cc.get('mad_gamma',0.5), ot_sigma=cc.get('ot_sigma',0.05),
        ot_eps=cc.get('ot_eps',0.1), ot_iters=cc.get('ot_iters',20),
        pos_weight=pw,
    ).to(device)


def _build_mosei_baseline(cfg, device):
    sys.path.insert(0, f'{BASE}/scripts/train')
    from train_baseline_mosei import build_model as _bm
    return _bm(cfg).to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer_iemocap(ckpt_path, cfg_path, device, model_key, bs=8):
    tag        = f'A1_iemocap_{os.path.basename(os.path.dirname(ckpt_path))}'
    cache_path = f'{CACHE}/{tag}.npz'
    if os.path.exists(cache_path):
        print(f'    [cache] {tag}')
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    cfg   = yaml.safe_load(open(cfg_path))
    sd, _ = load_sd(ckpt_path)
    is_chado = model_key == 'chado'

    model = _build_iemocap_chado(cfg, device) if is_chado \
            else _build_iemocap_baseline(cfg, device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap
    dc = cfg['data']; mc = cfg['model']
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
    all_utt_ids = ds.df['utt_id'].astype(str).tolist()

    def _run_loader(batch_size):
        nw = min(4, batch_size)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=nw, collate_fn=collate_iemocap,
                            pin_memory=True,
                            persistent_workers=(nw > 0),
                            prefetch_factor=2 if nw > 0 else None)
        _probs, _labels = [], []
        for batch in loader:
            ti  = {k: batch[k].to(device) for k in ('input_ids','attention_mask')}
            wav = batch.get('wav', batch.get('audio_wave'))
            if wav is not None: wav = wav.to(device)
            pv  = batch.get('pixel_values', batch.get('video_frames'))
            if pv is not None: pv = pv.to(device)
            out    = model(text_input=ti, audio_wave=wav, video_frames=pv)
            logits = out[0] if isinstance(out, tuple) else out
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            _probs.append(probs)
            _labels.extend(batch['label'].tolist())
        return _probs, _labels

    all_probs, all_labels = None, None
    for bs_try in [bs, max(1, bs // 2), max(1, bs // 4), 1]:
        try:
            print(f' [bs={bs_try}]', end='', flush=True)
            all_probs, all_labels = _run_loader(bs_try)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f' [OOM@bs={bs_try}]', end='', flush=True)
    if all_probs is None:
        raise RuntimeError('CUDA OOM at all batch sizes — cannot infer')

    result = {'utt_ids': np.array(all_utt_ids),
              'probs':   np.vstack(all_probs).astype(np.float32),
              'labels':  np.array(all_labels, dtype=np.int64)}
    np.savez(cache_path, **result)
    print(f'    [saved] {cache_path}')
    return result


def _tune_thresholds_wacc(val_sigs, val_bins, grid=np.arange(0.05, 0.96, 0.01)):
    """Per-class threshold tuned to maximise binary accuracy on validation set."""
    thr = np.zeros(val_sigs.shape[1], dtype=np.float32)
    for c in range(val_sigs.shape[1]):
        best_acc, best_t = 0.0, 0.5
        for t in grid:
            acc = (val_bins[:,c] == (val_sigs[:,c] >= t).astype(int)).mean()
            if acc > best_acc:
                best_acc, best_t = acc, t
        thr[c] = best_t
    return thr


@torch.no_grad()
def _collect_mosei_sigs(model, manifest_path, cfg, device, bs=32):
    from datasets.mosei.mosei_utt_dataset import MoseiUttDataset, collate_mosei_utt
    dc = cfg['data']
    ds = MoseiUttDataset(manifest_path=manifest_path,
                         max_audio_len=dc.get('max_audio_len',50),
                         max_video_len=dc.get('max_video_len',30))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0,
                        collate_fn=collate_mosei_utt)
    sigs, bins, uids = [], [], []
    for batch in loader:
        bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in batch.items()}
        out    = model(bd)
        logits = out[0] if isinstance(out, tuple) else out
        sigs.append(torch.sigmoid(logits).cpu().numpy())
        bins.append((batch['label'].numpy() > 0.5).astype(np.int8))
        uids.extend(batch['utt_id'])
    return np.vstack(sigs), np.vstack(bins), uids


@torch.no_grad()
def infer_mosei(ckpt_path, cfg_path, device, model_key, bs=32):
    tag        = f'A1v2_mosei_{os.path.basename(os.path.dirname(ckpt_path))}'
    cache_path = f'{CACHE}/{tag}.npz'
    if os.path.exists(cache_path):
        print(f'    [cache] {tag}')
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    cfg      = yaml.safe_load(open(cfg_path))
    sd, _    = load_sd(ckpt_path)
    is_chado = model_key == 'chado'

    model = _build_mosei_chado(cfg, device) if is_chado \
            else _build_mosei_baseline(cfg, device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # ── Tune thresholds on validation (maximise per-class binary accuracy) ──
    print(' [tuning thresholds on val…]', end='', flush=True)
    val_sigs, val_bins, _ = _collect_mosei_sigs(model, cfg['data']['val_manifest'],
                                                 cfg, device, bs)
    thresholds = _tune_thresholds_wacc(val_sigs, val_bins)
    print(f' thr={thresholds.round(2)}', end='', flush=True)

    # ── Collect test predictions ──────────────────────────────────────────────
    test_sigs, test_bins, test_uids = _collect_mosei_sigs(
        model, cfg['data']['test_manifest'], cfg, device, bs)

    # Sanity check
    pred  = (test_sigs >= thresholds).astype(int)
    baccs = [(test_bins[:,c]==pred[:,c]).mean() for c in range(6)]
    print(f' wacc={np.mean(baccs):.4f}', end='', flush=True)

    result = {'utt_ids':   np.array(test_uids),
              'sig_probs': test_sigs.astype(np.float32),
              'bin_labels':test_bins.astype(np.int8),
              'thresholds':thresholds}
    np.savez(cache_path, **result)
    print(f'    [saved] {cache_path}')
    return result


@torch.no_grad()
def infer_meld(ckpt_path, cfg_path, device, model_key, bs=8):
    tag        = f'A1_meld_{os.path.basename(os.path.dirname(ckpt_path))}'
    cache_path = f'{CACHE}/{tag}.npz'
    if os.path.exists(cache_path):
        print(f'    [cache] {tag}')
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    cfg   = yaml.safe_load(open(cfg_path))
    sd, _ = load_sd(ckpt_path)
    is_chado = model_key == 'chado'

    model = _build_meld_chado(cfg, device) if is_chado \
            else _build_iemocap_baseline(cfg, device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    from datasets.meld.meld_dataset import (
        MeldDataset, collate_meld, build_label_map_from_order, EMO_ORDER_7)
    from transformers import AutoTokenizer
    import functools
    dc = cfg['data']; mc = cfg['model']
    label_map = build_label_map_from_order(EMO_ORDER_7)
    ds = MeldDataset(
        csv_path=dc['test_csv'], label_map=label_map,
        text_col=dc.get('text_col','text'),
        audio_path_col=dc.get('audio_path_col','audio_path'),
        video_path_col=dc.get('video_path_col','video_path'),
        label_col=dc.get('label_col','emotion'),
        utt_id_col=dc.get('utt_id_col','utt_id'),
        text_model_name=mc['text_model_name'],
        num_frames=dc.get('num_frames',8), frame_size=dc.get('frame_size',224),
        sample_rate=dc.get('sample_rate',16000),
        max_audio_seconds=dc.get('max_audio_seconds',6.0),
    )
    tok     = AutoTokenizer.from_pretrained(mc['text_model_name'], use_fast=True)
    collate = functools.partial(collate_meld, tokenizer=tok,
                                use_text=mc.get('use_text',True),
                                use_audio=mc.get('use_audio',True),
                                use_video=mc.get('use_video',True))
    def _run_meld_loader(batch_size):
        nw = min(4, batch_size)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=nw, collate_fn=collate, pin_memory=True,
                            persistent_workers=(nw > 0),
                            prefetch_factor=2 if nw > 0 else None)
        _probs, _labels = [], []
        for batch in loader:
            ti  = {k: batch.text_input[k].to(device) for k in ('input_ids','attention_mask')}
            wav = batch.audio_wave.to(device) if batch.audio_wave is not None else None
            pv  = batch.video_frames.to(device) if batch.video_frames is not None else None
            out    = model(text_input=ti, audio_wave=wav, video_frames=pv)
            logits = out[0] if isinstance(out, tuple) else out
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            _probs.append(probs)
            _labels.append(batch.labels.cpu().numpy())
        return _probs, _labels

    all_probs, all_labels = None, None
    for bs_try in [bs, max(1, bs // 2), max(1, bs // 4), 1]:
        try:
            print(f' [bs={bs_try}]', end='', flush=True)
            all_probs, all_labels = _run_meld_loader(bs_try)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f' [OOM@bs={bs_try}]', end='', flush=True)
    if all_probs is None:
        raise RuntimeError('CUDA OOM at all batch sizes — cannot infer')

    result = {'probs':  np.vstack(all_probs).astype(np.float32),
              'labels': np.concatenate(all_labels).astype(np.int64)}
    np.savez(cache_path, **result)
    print(f'    [saved] {cache_path}')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Stratified metrics
# ═══════════════════════════════════════════════════════════════════════════════

def stratify_iemocap(res, ann_info):
    """Align predictions with annotator info, return per-stratum metrics."""
    uids = res['utt_ids'].tolist()
    probs = res['probs']; labels = res['labels']

    # Align
    ambs, probs_a, labels_a, dists_a = [], [], [], []
    for i, uid in enumerate(uids):
        if uid in ann_info:
            info = ann_info[uid]
            ambs.append(info['entropy'])
            probs_a.append(probs[i])
            labels_a.append(labels[i])
            dists_a.append(info['dist'])

    if len(ambs) < 10:
        return None

    ambs = np.array(ambs); probs_a = np.array(probs_a)
    labels_a = np.array(labels_a); dists_a = np.array(dists_a)

    order  = np.argsort(ambs)
    splits = np.array_split(order, 3)
    out    = {}
    for name, idx in zip(STRATA, splits):
        p = probs_a[idx]; l = labels_a[idx]
        d = dists_a[idx]; a = ambs[idx]
        pred = p.argmax(axis=1)
        acc  = float(accuracy_score(l, pred))
        mf1  = float(f1_score(l, pred, average='macro', zero_division=0))
        ece  = compute_ece(p, l)
        # KL: mean KL(annotator_dist || model_probs) per sample
        kl   = float(np.mean([kl_div(d[j], p[j]) for j in range(len(idx))]))
        out[name] = {'acc': round(acc,4), 'macro_f1': round(mf1,4),
                     'ece': round(ece,4), 'kl': round(kl,4),
                     'n': int(len(idx)), 'mean_amb': round(float(a.mean()),4)}
    return out


def stratify_mosei(res, ann_info):
    uids = res['utt_ids'].tolist()
    sigs = res['sig_probs']; bins = res['bin_labels']
    thr  = np.asarray(res['thresholds'])

    ambs, sigs_a, bins_a, dists_a = [], [], [], []
    for i, uid in enumerate(uids):
        if uid in ann_info:
            info = ann_info[uid]
            ambs.append(info['entropy'])
            sigs_a.append(sigs[i])
            bins_a.append(bins[i])
            dists_a.append(info['dist'])

    if len(ambs) < 10:
        return None

    ambs = np.array(ambs); sigs_a = np.array(sigs_a)
    bins_a = np.array(bins_a); dists_a = np.array(dists_a)

    order  = np.argsort(ambs)
    splits = np.array_split(order, 3)
    out    = {}
    for name, idx in zip(STRATA, splits):
        sp = sigs_a[idx]; sl = bins_a[idx]
        sd = dists_a[idx]; sa = ambs[idx]
        pred = (sp >= thr).astype(int)
        bin_accs = [(sl[:,c] == pred[:,c]).mean() for c in range(6)]
        wacc = float(np.mean(bin_accs))
        wf1  = float(f1_score(sl, pred, average='weighted', zero_division=0))
        mf1  = float(f1_score(sl, pred, average='macro', zero_division=0))
        ece  = compute_ece_multilabel(sp)
        # Per-label binary KL: KL([a_c, 1-a_c] || [p_c, 1-p_c])
        kl_vals = []
        for j in range(len(idx)):
            label_kls = []
            for c in range(6):
                ac = sd[j][c]; pc = sp[j][c]
                p_bin = np.array([ac, 1-ac])
                q_bin = np.array([pc, 1-pc])
                label_kls.append(kl_div(p_bin, q_bin))
            kl_vals.append(np.mean(label_kls))
        kl = float(np.mean(kl_vals))
        out[name] = {'wacc': round(wacc,4), 'weighted_f1': round(wf1,4),
                     'macro_f1': round(mf1,4),
                     'ece': round(ece,4), 'kl': round(kl,4),
                     'n': int(len(idx)), 'mean_amb': round(float(sa.mean()),4)}
    return out


def stratify_meld(res, knn_info):
    probs  = res['probs']; labels = res['labels']
    N = len(labels)

    ambs, probs_a, labels_a = [], [], []
    for pos in range(N):
        uid = str(pos)
        if uid in knn_info:
            ambs.append(knn_info[uid]['entropy'])
            probs_a.append(probs[pos])
            labels_a.append(labels[pos])

    if len(ambs) < 10:
        return None

    ambs = np.array(ambs); probs_a = np.array(probs_a)
    labels_a = np.array(labels_a)

    order  = np.argsort(ambs)
    splits = np.array_split(order, 3)
    out    = {}
    for name, idx in zip(STRATA, splits):
        p = probs_a[idx]; l = labels_a[idx]; a = ambs[idx]
        pred = p.argmax(axis=1)
        acc  = float(accuracy_score(l, pred))
        mf1  = float(f1_score(l, pred, average='macro', zero_division=0))
        ece  = compute_ece(p, l)
        out[name] = {'acc': round(acc,4), 'macro_f1': round(mf1,4),
                     'ece': round(ece,4), 'kl': None,   # KL not available for MELD
                     'n': int(len(idx)), 'mean_amb': round(float(a.mean()),4)}
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Model registries
# ═══════════════════════════════════════════════════════════════════════════════

# Best compatible (new-arch) checkpoints per dataset
# IEMOCAP: chado (acc=77.1%, mf1=77.5%) — beats all baselines
# MOSEI  : chado_best (wacc=80.9%) — highest wAcc; thresholds re-tuned on val for wacc
# MELD   : chado_v2 (acc=63.0%, mf1=46.3%) — best new-arch checkpoint available
IEMOCAP_REGISTRY = [
    ('AR-GOT',  'chado',   f'{BASE}/configs/iemocap/chado_iemocap.yaml',        'chado'),
    ('OV-MER',  'ovmer',   f'{BASE}/configs/iemocap/ovmer_iemocap.yaml',         'ovmer'),
    ('AER-LLM', 'aerllm',  f'{BASE}/configs/iemocap/aerllm_iemocap.yaml',        'aerllm'),
    ('EmoCLIP', 'emoclip', f'{BASE}/configs/iemocap/emoclip_iemocap.yaml',       'emoclip'),
]

MOSEI_REGISTRY = [
    ('AR-GOT',  'chado_best', f'{BASE}/configs/mosei/chado_mosei_best.yaml',  'chado'),
    ('OV-MER',  'ovmer',      f'{BASE}/configs/mosei/ovmer_mosei.yaml',        'ovmer'),
    ('AER-LLM', 'aerllm',     f'{BASE}/configs/mosei/aerllm_mosei.yaml',       'aerllm'),
    ('EmoCLIP', 'emoclip',    f'{BASE}/configs/mosei/emoclip_mosei.yaml',      'emoclip'),
]

MELD_REGISTRY = [
    ('AR-GOT',  'chado_v2', f'{BASE}/configs/meld/chado_meld_v2.yaml',  'chado'),
    ('OV-MER',  'ovmer',    f'{BASE}/configs/meld/ovmer_meld.yaml',      'ovmer'),
    ('AER-LLM', 'aerllm',   f'{BASE}/configs/meld/aerllm_meld.yaml',     'aerllm'),
    ('EmoCLIP', 'emoclip',  f'{BASE}/configs/meld/emoclip_meld.yaml',    'emoclip'),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Print table
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(dataset, results, is_mosei=False):
    kl_avail = (dataset != 'MELD')
    w = 10

    header1 = f"{'Dataset':<10}{'Tier':<9}{'Model':<10}"
    if is_mosei:
        header1 += f"{'wAcc':>{w}}{'wF1':>{w}}{'Macro-F1':>{w}}{'ECE':>{w}}"
    else:
        header1 += f"{'Acc':>{w}}{'Macro-F1':>{w}}{'ECE':>{w}}"
    if kl_avail:
        header1 += f"{'KL':>{w}}"
    header1 += f"{'N':>{w}}"
    print('\n' + header1)
    print('-' * len(header1))

    registry = MOSEI_REGISTRY if is_mosei else \
               IEMOCAP_REGISTRY if dataset == 'IEMOCAP' else MELD_REGISTRY

    for tier in STRATA:
        for i, (name, _, _, _) in enumerate(registry):
            m = results.get(name, {}).get(tier)
            if m is None:
                continue
            ds_col   = dataset if i == 0 else ''
            tier_col = tier    if i == 0 else ''
            row = f"{ds_col:<10}{tier_col:<9}{name:<10}"
            if is_mosei:
                row += f"{m['wacc']:>{w}.4f}{m['weighted_f1']:>{w}.4f}{m['macro_f1']:>{w}.4f}{m['ece']:>{w}.4f}"
            else:
                row += f"{m['acc']:>{w}.4f}{m['macro_f1']:>{w}.4f}{m['ece']:>{w}.4f}"
            if kl_avail:
                row += f"{m['kl']:>{w}.4f}" if m['kl'] is not None else f"{'—':>{w}}"
            row += f"{m['n']:>{w}}"
            print(row)
        print()

    # ── Robustness summary: Low→High degradation per model ───────────────────
    print(f"  {'Robustness (Low→High degradation — lower drop = more robust)'}")
    acc_key = 'wacc' if is_mosei else 'acc'
    print(f"  {'Model':<10}  {'Low':>8}  {'High':>8}  {'Drop':>8}  {'Rank'}")
    drops = []
    for name, _, _, _ in registry:
        lo = results.get(name, {}).get('Low',  {}).get(acc_key, None)
        hi = results.get(name, {}).get('High', {}).get(acc_key, None)
        if lo is not None and hi is not None:
            drops.append((name, lo, hi, lo - hi))
    drops.sort(key=lambda x: x[3])   # ascending: smallest drop = most robust
    for rank, (name, lo, hi, drop) in enumerate(drops, 1):
        marker = ' ← BEST' if rank == 1 else ''
        print(f"  {name:<10}  {lo:>8.4f}  {hi:>8.4f}  {drop:>+8.4f}{marker}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_dataset(ds_name, results, metric_key, metric_label, is_mosei=False, ax=None):
    registry = MOSEI_REGISTRY if is_mosei else \
               IEMOCAP_REGISTRY if ds_name == 'IEMOCAP' else MELD_REGISTRY
    x   = np.arange(3)
    w   = 0.18
    ax  = ax or plt.gca()
    ax.set_facecolor('#F7F9FB')
    ax.grid(axis='y', color='#DDEAF7', linewidth=0.8, zorder=0)

    for j, (name, _, _, _) in enumerate(registry):
        if name not in results: continue
        vals = [results[name].get(t, {}).get(metric_key, 0) or 0 for t in STRATA]
        bars = ax.bar(x + j*w - 1.5*w, vals, width=w,
                      color=COLORS[name], label=name, zorder=3, alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(['Low', 'Medium', 'High'])
    ax.set_xlabel('Ambiguity Tier')
    ax.set_ylabel(metric_label)
    ax.set_title(f'{ds_name}')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] + 0.12))


def plot_all(all_results):
    metrics = [
        ('acc',      'Accuracy',   False, False),
        ('macro_f1', 'Macro-F1',   False, False),
        ('ece',      'ECE',        False, False),
        ('kl',       'KL-Div',     False, True),   # skip MELD
        ('wacc',     'wAcc (MOSEI)' , True, False),
    ]
    ds_list = ['IEMOCAP', 'CMU-MOSEI', 'MELD']
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle('A.1 — Ambiguity-Stratified Performance\n'
                 '(model-independent ambiguity tiers)', fontsize=13, fontweight='bold')

    metric_cols = [
        ('macro_f1', 'Macro-F1'),
        ('acc',      'Accuracy / wAcc'),
        ('ece',      'ECE'),
        ('kl',       'KL-Divergence'),
    ]

    for row_i, ds in enumerate(ds_list):
        if ds not in all_results: continue
        is_mosei = (ds == 'CMU-MOSEI')
        ds_res   = all_results[ds]

        for col_i, (mkey, mlabel) in enumerate(metric_cols):
            ax = axes[row_i][col_i]
            if mkey == 'kl' and ds == 'MELD':
                ax.text(0.5, 0.5, 'KL not available\n(annotator dist\nnot recorded)',
                        ha='center', va='center', transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_title(f'{ds} — {mlabel}')
                ax.axis('off')
                continue
            if mkey == 'acc' and is_mosei:
                actual_key = 'wacc'
            else:
                actual_key = mkey
            _plot_dataset(ds, ds_res, actual_key, mlabel, is_mosei=is_mosei, ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = f'{FIG_DIR}/A1_ambiguity_stratification.pdf'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    fig_path2 = fig_path.replace('.pdf', '.png')
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Figure saved → {fig_path}')
    print(f'  Figure saved → {fig_path2}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-meld',   action='store_true')
    p.add_argument('--skip-mosei',  action='store_true')
    p.add_argument('--skip-iemocap',action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Output : {OUT_DIR}')

    # ── Annotator / ambiguity info ────────────────────────────────────────────
    iemocap_info, mosei_info, meld_info = {}, {}, {}

    iemocap_root = '/home/tahirahmad/IEMOCAP_Final/dataset/IEMOCAP_full_release'
    if not args.skip_iemocap and os.path.isdir(iemocap_root):
        print('\n[Ambiguity] IEMOCAP annotator vote distributions …')
        iemocap_info = parse_iemocap_annotator_info(iemocap_root)

    mosei_manifest = f'{BASE}/data/processed/mosei/mosei_utt_test.jsonl'
    if not args.skip_mosei and os.path.exists(mosei_manifest):
        print('\n[Ambiguity] CMU-MOSEI label_raw distributions …')
        mosei_info = parse_mosei_annotator_info(mosei_manifest)

    meld_knn_cache = f'{CACHE}/meld_knn_entropy.json'
    if not args.skip_meld and os.path.exists(meld_knn_cache):
        print('\n[Ambiguity] MELD KNN entropy (cached) …')
        meld_info = load_meld_knn_info(meld_knn_cache)

    all_results = {}

    # ══════════════════════════════════════════════════════════════════════════
    # IEMOCAP
    # ══════════════════════════════════════════════════════════════════════════
    if iemocap_info and not args.skip_iemocap:
        print('\n' + '='*70)
        print('IEMOCAP — Model-Independent Ambiguity Stratification')
        print('='*70)
        iemo_res = {}
        for display, ckpt_dir, cfg_path, mkey in IEMOCAP_REGISTRY:
            ckpt = f'{RES}/iemocap/{ckpt_dir}/best.pt'
            if not os.path.exists(ckpt) or not os.path.exists(cfg_path):
                print(f'  [{display}] missing → skip'); continue
            print(f'  [{display}] inferring …', end='', flush=True)
            try:
                res = infer_iemocap(ckpt, cfg_path, device, mkey)
                print(f' {len(res["labels"])} samples', end='', flush=True)
            except Exception as e:
                print(f' ERROR: {e}'); continue
            strata = stratify_iemocap(res, iemocap_info)
            if strata is None:
                print(' [WARN] alignment failed'); continue
            iemo_res[display] = strata
            print()
            for tier, m in strata.items():
                kl_str = f'  KL={m["kl"]:.4f}' if m["kl"] is not None else ''
                print(f'    {tier:7s} (N={m["n"]:4d}, amb={m["mean_amb"]:.3f}): '
                      f'acc={m["acc"]:.4f}  mF1={m["macro_f1"]:.4f}  '
                      f'ECE={m["ece"]:.4f}{kl_str}')
        all_results['IEMOCAP'] = iemo_res

    # ══════════════════════════════════════════════════════════════════════════
    # CMU-MOSEI
    # ══════════════════════════════════════════════════════════════════════════
    if mosei_info and not args.skip_mosei:
        print('\n' + '='*70)
        print('CMU-MOSEI — Model-Independent Ambiguity Stratification')
        print('='*70)
        mosei_res = {}
        for display, ckpt_dir, cfg_path, mkey in MOSEI_REGISTRY:
            ckpt = f'{RES}/mosei/{ckpt_dir}/best.pt'
            if not os.path.exists(ckpt) or not os.path.exists(cfg_path):
                print(f'  [{display}] missing → skip'); continue
            print(f'  [{display}] inferring …', end='', flush=True)
            try:
                res = infer_mosei(ckpt, cfg_path, device, mkey)
                print(f' {len(res["bin_labels"])} samples', end='', flush=True)
            except Exception as e:
                print(f' ERROR: {e}'); continue
            strata = stratify_mosei(res, mosei_info)
            if strata is None:
                print(' [WARN] alignment failed'); continue
            mosei_res[display] = strata
            print()
            for tier, m in strata.items():
                print(f'    {tier:7s} (N={m["n"]:4d}, amb={m["mean_amb"]:.3f}): '
                      f'wacc={m["wacc"]:.4f}  wF1={m["weighted_f1"]:.4f}  '
                      f'mF1={m["macro_f1"]:.4f}  ECE={m["ece"]:.4f}  KL={m["kl"]:.4f}')
        all_results['CMU-MOSEI'] = mosei_res

    # ══════════════════════════════════════════════════════════════════════════
    # MELD
    # ══════════════════════════════════════════════════════════════════════════
    if meld_info and not args.skip_meld:
        print('\n' + '='*70)
        print('MELD — KNN Soft-Label Ambiguity Stratification')
        print('='*70)
        meld_res = {}
        for display, ckpt_dir, cfg_path, mkey in MELD_REGISTRY:
            ckpt = f'{RES}/meld/{ckpt_dir}/best.pt'
            if not os.path.exists(ckpt) or not os.path.exists(cfg_path):
                print(f'  [{display}] missing → skip'); continue
            print(f'  [{display}] inferring …', end='', flush=True)
            try:
                res = infer_meld(ckpt, cfg_path, device, mkey)
                print(f' {len(res["labels"])} samples', end='', flush=True)
            except Exception as e:
                print(f' ERROR: {e}'); continue
            strata = stratify_meld(res, meld_info)
            if strata is None:
                print(' [WARN] alignment failed'); continue
            meld_res[display] = strata
            print()
            for tier, m in strata.items():
                print(f'    {tier:7s} (N={m["n"]:4d}, amb={m["mean_amb"]:.3f}): '
                      f'acc={m["acc"]:.4f}  mF1={m["macro_f1"]:.4f}  '
                      f'ECE={m["ece"]:.4f}  KL=—')
        all_results['MELD'] = meld_res

    # ══════════════════════════════════════════════════════════════════════════
    # Summary tables
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('SUMMARY TABLES — A.1 Model-Independent Ambiguity Stratification')
    print('='*70)

    if 'IEMOCAP' in all_results:
        print_table('IEMOCAP', all_results['IEMOCAP'], is_mosei=False)
    if 'CMU-MOSEI' in all_results:
        print_table('CMU-MOSEI', all_results['CMU-MOSEI'], is_mosei=True)
    if 'MELD' in all_results:
        print_table('MELD', all_results['MELD'], is_mosei=False)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = f'{OUT_DIR}/A1_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n  Results saved → {out_path}')

    # ── Figures ───────────────────────────────────────────────────────────────
    if all_results:
        plot_all(all_results)

    print('\nDone.')


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
    main()
