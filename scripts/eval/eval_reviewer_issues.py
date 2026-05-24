#!/usr/bin/env python3
"""
Reviewer Issue Analysis — Four targeted fixes for EMNLP revision.

Issues addressed:
  1.1  Independent-ambiguity stratified analysis  (Figures 3-5)
  1.2  Radius–ambiguity quantitative correlation  (Table 3 supplement)
  1.3  Factor-level ablation                      (new table)
  1.4  Predicted vs annotator distribution        (KL / JSD / Wasserstein)

Usage (single GPU):
  cd /home/tahirahmad/Project_Code/CHADO_EMNLP
  CUDA_VISIBLE_DEVICES=5 python3 -u scripts/eval/eval_reviewer_issues.py

Outputs go to experiments/results/reviewer_issues/
"""
import os, sys, re, glob, json, types, functools
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, entropy as scipy_entropy
from scipy.stats import wasserstein_distance
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
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_sd(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        return ckpt['model']
    for key in ('model_state_dict', 'state_dict'):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def detect_text_model(sd):
    for k, v in sd.items():
        if 'text_enc' in k and 'embeddings.LayerNorm.weight' in k:
            return 'roberta-large' if v.shape[0] == 1024 else 'roberta-base'
    return 'roberta-base'


def ent_norm(p, n_cls):
    pnz = p[p > 1e-10]
    if len(pnz) < 2:
        return 0.0
    return float(scipy_entropy(pnz)) / np.log(n_cls)


def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error (single-label)."""
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
    return float(ece / len(labels))


def stratified_metrics(probs, labels, amb_scores, n_strata=3):
    """Split into Low / Medium / High terciles by amb_scores."""
    order  = np.argsort(amb_scores)
    splits = np.array_split(order, n_strata)
    names  = ['Low', 'Medium', 'High']
    out    = {}
    for name, idx in zip(names, splits):
        p_s = probs[idx]
        l_s = labels[idx]
        a_s = amb_scores[idx]
        pred = p_s.argmax(axis=1)
        acc  = float(accuracy_score(l_s, pred))
        mf1  = float(f1_score(l_s, pred, average='macro', zero_division=0))
        ece  = compute_ece(p_s, l_s)
        out[name] = {
            'acc': round(acc, 4), 'macro_f1': round(mf1, 4),
            'ece': round(ece, 4), 'n': int(len(idx)),
            'mean_amb': round(float(a_s.mean()), 4),
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 1.1 — Model-independent ambiguity computation
# ═══════════════════════════════════════════════════════════════════════════════

IEMOCAP_CAT = {
    'neutral': 0, 'neu': 0,
    'happy': 1, 'hap': 1, 'excitement': 1, 'excited': 1, 'exc': 1,
    'anger': 2, 'angry': 2, 'ang': 2, 'frustration': 2, 'fru': 2,
    'sad': 3, 'sadness': 3,
}


def parse_iemocap_annotator_entropy(eval_root):
    """
    Parse IEMOCAP EmoEvaluation files → {utt_id: normalized_H(rater_labels)}.
    Model-independent: uses categorical rater votes, not model predictions.
    """
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
            if not hm:
                continue
            uid    = hm.group(3)
            labels = []
            for rm in rtr.finditer(blk):
                parts = [p.strip().rstrip(';').strip().lower()
                         for p in rm.group(1).split(';')]
                labs = [IEMOCAP_CAT[p] for p in parts if p in IEMOCAP_CAT]
                if labs:
                    labels.extend(labs)
            if len(labels) >= 2:
                cnt = np.bincount(labels, minlength=4).astype(float)
                out[uid] = ent_norm(cnt / cnt.sum(), 4)
    print(f'  IEMOCAP annotator entropy: {len(out)} utterances, '
          f'{sum(v > 0 for v in out.values())} with disagreement')
    return out


def compute_mosei_label_entropy(manifest_path):
    """
    CMU-MOSEI label_raw entropy → {utt_id: normalized_H(annotation_intensities)}.
    Uses raw AMT annotation scores (not model output).
    """
    out = {}
    for line in open(manifest_path):
        s   = json.loads(line)
        raw = np.array(s.get('label_raw', [0.0] * 6), float)
        tot = raw.sum()
        out[s['utt_id']] = 0.0 if tot < 1e-6 else ent_norm(raw / tot, 6)
    print(f'  CMU-MOSEI annotation entropy: {len(out)} samples, '
          f'{sum(v > 0 for v in out.values())} non-zero')
    return out


def compute_meld_knn_entropy(test_csv, train_csv, device, k=15):
    """
    MELD: nearest-neighbour soft-label entropy using RoBERTa text embeddings.
    For each test utterance, find K nearest training samples, compute entropy
    of their emotion class distribution → model-independent ambiguity proxy.

    Keys are positional indices ("0", "1", ...) because MELD utt_ids repeat
    across dialogues and are not globally unique.
    """
    from transformers import AutoTokenizer, AutoModel
    from sklearn.neighbors import NearestNeighbors

    EMO_MAP = {'neutral': 0, 'joy': 1, 'surprise': 2, 'anger': 3,
               'sadness': 4, 'disgust': 5, 'fear': 6}

    tok  = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
    bert = AutoModel.from_pretrained('roberta-base').to(device)
    bert.eval()

    def embed(texts, batch_size=32):
        embs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                enc = tok(texts[start:start + batch_size], padding=True,
                          truncation=True, max_length=128,
                          return_tensors='pt').to(device)
                out = bert(**enc)
                embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(embs) if embs else np.zeros((0, 768))

    df_tr  = pd.read_csv(train_csv)
    df_te  = pd.read_csv(test_csv)

    tr_texts  = df_tr['text'].astype(str).tolist()
    tr_labels = np.array([EMO_MAP.get(str(l).lower(), 0) for l in df_tr['emotion']])
    te_texts  = df_te['text'].astype(str).tolist()

    print(f'  MELD KNN: embedding {len(tr_texts)} train + {len(te_texts)} test …')
    tr_embs = embed(tr_texts)
    te_embs = embed(te_texts)
    del bert; torch.cuda.empty_cache()

    nn_model = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nn_model.fit(tr_embs)
    _, indices = nn_model.kneighbors(te_embs)

    # Use positional index as key (MELD utt_ids repeat across dialogues)
    out = {}
    for pos, idx in enumerate(indices):
        nbr_labels = tr_labels[idx]
        cnt  = np.bincount(nbr_labels, minlength=7).astype(float)
        out[str(pos)] = ent_norm(cnt / cnt.sum(), 7)

    print(f'  MELD KNN entropy: {len(out)} samples, '
          f'mean={np.mean(list(out.values())):.4f}')
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Inference functions (return aligned arrays: probs, labels, utt_ids)
# ═══════════════════════════════════════════════════════════════════════════════

def infer_iemocap_model(ckpt_path, device, model_key='chado'):
    """Run inference for any IEMOCAP model; return (utt_ids, probs, labels)."""
    cfg_map = {
        'chado':   f'{BASE}/configs/iemocap/chado_iemocap.yaml',
        'mult':    f'{BASE}/configs/iemocap/mult_iemocap.yaml',
        'mmdfn':   f'{BASE}/configs/iemocap/mmdfn_iemocap.yaml',
        'ctnet':   f'{BASE}/configs/iemocap/ctnet_iemocap.yaml',
        'lflstm':  f'{BASE}/configs/iemocap/lflstm_iemocap.yaml',
        'bpmult':  f'{BASE}/configs/iemocap/bpmult_iemocap.yaml',
    }
    cfg_path = cfg_map.get(model_key, cfg_map['chado'])
    cfg = yaml.safe_load(open(cfg_path))
    dc  = cfg['data']
    sd  = load_sd(ckpt_path)
    tm  = detect_text_model(sd)

    from evaluation.inference_utils import build_model_for_inference
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap

    LMAP = {'neu': 0, 'neutral': 0, 'hap': 1, 'happy': 1, 'exc': 1,
            'ang': 2, 'angry': 2, 'fru': 2, 'sad': 3, 'sadness': 3}

    test_csv = dc['test_csv']
    df = pd.read_csv(test_csv)
    utt_ids = df['utt_id'].astype(str).tolist()
    lbl_col = 'label_4' if 'label_4' in df.columns else 'label'
    labels  = np.array([LMAP.get(str(v).lower(), 0) if isinstance(v, str) else int(v)
                        for v in df[lbl_col]])

    model, _, cfg2 = build_model_for_inference(ckpt_path, 'iemocap', model_key, device)
    model.eval()

    ds = IEMOCAPDataset(
        csv_path=test_csv, text_model_name=tm,
        max_text_len=dc.get('max_text_len', 96),
        audio_sr=dc.get('sample_rate', 16000),
        audio_sec=dc.get('max_audio_seconds', 4.0),
        n_frames=dc.get('num_frames', 8),
        use_audio=dc.get('use_audio', True),
        use_video=dc.get('use_video', True),
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False,
                        num_workers=4, collate_fn=collate_iemocap)

    n_frames = dc.get('num_frames', 8)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            B  = batch['input_ids'].shape[0]
            ti = {'input_ids':      batch['input_ids'].to(device),
                  'attention_mask': batch['attention_mask'].to(device)}
            aw = batch.get('wav')
            if aw is None:
                aw = batch.get('audio_wave')
            if aw is not None:
                aw = aw.to(device)
            vf = batch.get('video_frames')
            if vf is None:
                vf = torch.zeros(B, n_frames, 3, 224, 224)
            vf = vf.to(device)
            out = model(text_input=ti, audio_wave=aw, video_frames=vf)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

    return utt_ids, np.vstack(all_probs), labels


def infer_meld_model(ckpt_path, device, model_key='chado'):
    """Run inference for any MELD model; return (utt_ids, probs, labels).
    Uses positional index as utt_id to match compute_meld_knn_entropy output.
    """
    from evaluation.inference_utils import infer_meld
    probs, labels = infer_meld(ckpt_path, 'meld', model_key, device)
    # Positional indices as utt_ids (consistent with compute_meld_knn_entropy)
    utt_ids = [str(i) for i in range(len(probs))]
    return utt_ids, probs, labels


def infer_mosei_model(ckpt_path, device, model_key='chado'):
    """Run inference for any CMU-MOSEI model; return (utt_ids, probs, labels)."""
    from datasets.mosei.mosei_utt_dataset import MoseiUttDataset, collate_mosei_utt
    from evaluation.inference_utils import build_mosei_model_for_inference

    model, cfg = build_mosei_model_for_inference(ckpt_path, model_key, device)
    model.eval()
    dc = cfg['data']

    ds = MoseiUttDataset(
        manifest_path=dc['test_manifest'],
        max_audio_len=dc.get('max_audio_len', 50),
        max_video_len=dc.get('max_video_len', 30),
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=collate_mosei_utt)

    all_probs, all_labels, all_uids = [], [], []
    with torch.no_grad():
        for batch in loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
            out    = model(bd)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            probs  = torch.sigmoid(logits).cpu().numpy()
            # For multi-label → dominant class for stratification
            all_probs.append(probs)
            all_labels.extend(batch['label'].tolist())
            all_uids.extend(list(batch['utt_id']))

    return all_uids, np.vstack(all_probs), np.array(all_labels)


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 1.1 — Stratified evaluation (model-independent ambiguity)
# ═══════════════════════════════════════════════════════════════════════════════

MODELS_IEMOCAP = [
    ('CHADO',   'chado'),
    ('MulT',    'mult'),
    ('MM-DFN',  'mmdfn'),
    ('CTNet',   'ctnet'),
    ('LF-LSTM', 'lflstm'),
    ('BP-MulT', 'bpmult'),
]

MODELS_MELD = MODELS_IEMOCAP

MODELS_MOSEI = [
    ('CHADO',   'chado'),
    ('MulT',    'mult'),
    ('MM-DFN',  'mmdfn'),
    ('CTNet',   'ctnet'),
    ('LF-LSTM', 'lflstm'),
    ('BP-MulT', 'bpmult'),
]


def run_issue11(device, iemocap_amb, mosei_amb, meld_amb):
    """Issue 1.1: independent-ambiguity stratified analysis."""
    print('\n' + '=' * 70)
    print('ISSUE 1.1 — Stratified Analysis (Model-Independent Ambiguity)')
    print('=' * 70)

    all_results = {}

    # ── IEMOCAP ──────────────────────────────────────────────────────────────
    if iemocap_amb:
        print(f'\n--- IEMOCAP ({len(iemocap_amb)} utterances with ambiguity scores) ---')
        iemo_results = {}
        for display, key in MODELS_IEMOCAP:
            ckpt = f'{RES}/iemocap/{key}/best.pt'
            if not os.path.exists(ckpt):
                print(f'  [{display}] checkpoint missing → skip')
                continue
            print(f'  [{display}] running inference …', end='', flush=True)
            try:
                uids, probs, labels = infer_iemocap_model(ckpt, device, key)
                print(f' {len(labels)} samples')
            except Exception as e:
                print(f' ERROR: {e}'); continue

            # align with independent ambiguity
            amb_aligned, probs_al, labels_al = [], [], []
            for uid, p, l in zip(uids, probs, labels):
                if uid in iemocap_amb:
                    amb_aligned.append(iemocap_amb[uid])
                    probs_al.append(p)
                    labels_al.append(l)

            N = len(amb_aligned)
            if N < 10:
                print(f'    [WARN] only {N} aligned → skip')
                continue
            amb_al   = np.array(amb_aligned)
            probs_al = np.array(probs_al)
            labels_al= np.array(labels_al)

            strata = stratified_metrics(probs_al, labels_al, amb_al)
            iemo_results[display] = strata
            for sname, m in strata.items():
                print(f'    {sname:7s}(N={m["n"]:4d}, amb={m["mean_amb"]:.3f}): '
                      f'acc={m["acc"]:.3f}  f1={m["macro_f1"]:.3f}  '
                      f'ECE={m["ece"]:.3f}')
        all_results['IEMOCAP'] = iemo_results

    # ── CMU-MOSEI ─────────────────────────────────────────────────────────────
    if mosei_amb:
        print(f'\n--- CMU-MOSEI ({len(mosei_amb)} utterances with ambiguity scores) ---')
        mosei_results = {}
        for display, key in MODELS_MOSEI:
            ckpt = f'{RES}/mosei/{key}/best.pt'
            if not os.path.exists(ckpt):
                print(f'  [{display}] checkpoint missing → skip'); continue
            print(f'  [{display}] running inference …', end='', flush=True)
            try:
                uids, probs, labels = infer_mosei_model(ckpt, device, key)
                print(f' {len(uids)} samples')
            except Exception as e:
                print(f' ERROR: {e}'); continue

            # For multi-label, derive dominant-class probs from sigmoid
            # Use softmax normalization over sigmoid outputs for stratification
            probs_norm = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)

            amb_al, probs_al, labels_al_dom = [], [], []
            for uid, p_raw, l_raw in zip(uids, probs_norm, labels):
                if uid in mosei_amb:
                    amb_al.append(mosei_amb[uid])
                    probs_al.append(p_raw)
                    raw = l_raw if isinstance(l_raw, (list, np.ndarray)) else [l_raw]
                    dom = int(np.argmax(raw)) if len(raw) > 1 else int(l_raw)
                    labels_al_dom.append(dom)

            N = len(amb_al)
            if N < 10:
                print(f'    [WARN] only {N} aligned → skip'); continue
            amb_al    = np.array(amb_al)
            probs_al  = np.array(probs_al)
            labels_al_dom = np.array(labels_al_dom)

            strata = stratified_metrics(probs_al, labels_al_dom, amb_al)
            mosei_results[display] = strata
            for sname, m in strata.items():
                print(f'    {sname:7s}(N={m["n"]:4d}, amb={m["mean_amb"]:.3f}): '
                      f'acc={m["acc"]:.3f}  f1={m["macro_f1"]:.3f}  '
                      f'ECE={m["ece"]:.3f}')
        all_results['CMU-MOSEI'] = mosei_results

    # ── MELD ─────────────────────────────────────────────────────────────────
    if meld_amb:
        print(f'\n--- MELD ({len(meld_amb)} utterances with ambiguity scores) ---')
        meld_results = {}
        for display, key in MODELS_MELD:
            ckpt = f'{RES}/meld/{key}/best.pt'
            if not os.path.exists(ckpt):
                print(f'  [{display}] checkpoint missing → skip'); continue
            print(f'  [{display}] running inference …', end='', flush=True)
            try:
                uids, probs, labels = infer_meld_model(ckpt, device, key)
                print(f' {len(uids)} samples')
            except Exception as e:
                print(f' ERROR: {e}'); continue

            amb_al, probs_al, labels_al = [], [], []
            for uid, p, l in zip(uids, probs, labels):
                if uid in meld_amb:
                    amb_al.append(meld_amb[uid])
                    probs_al.append(p)
                    labels_al.append(l)

            N = len(amb_al)
            if N < 10:
                print(f'    [WARN] only {N} aligned → skip'); continue
            amb_al    = np.array(amb_al)
            probs_al  = np.array(probs_al)
            labels_al = np.array(labels_al)

            strata = stratified_metrics(probs_al, labels_al, amb_al)
            meld_results[display] = strata
            for sname, m in strata.items():
                print(f'    {sname:7s}(N={m["n"]:4d}, amb={m["mean_amb"]:.3f}): '
                      f'acc={m["acc"]:.3f}  f1={m["macro_f1"]:.3f}  '
                      f'ECE={m["ece"]:.3f}')
        all_results['MELD'] = meld_results

    # ── Save and plot ─────────────────────────────────────────────────────────
    out_path = f'{OUT_DIR}/issue11_stratified.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n  Saved → {out_path}')

    _plot_stratified(all_results)
    return all_results


def _plot_stratified(all_results):
    strata = ['Low', 'Medium', 'High']
    for ds, ds_results in all_results.items():
        if not ds_results:
            continue
        methods = list(ds_results.keys())
        x     = np.arange(len(strata))
        width = 0.8 / max(len(methods), 1)

        # ── Bar chart: acc + F1 ───────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor(C['bg'])
        for ax_i, metric in enumerate(['acc', 'macro_f1', 'ece']):
            ax = axes[ax_i]
            ax.set_facecolor(C['bg'])
            ax.grid(axis='y', color=C['grid'], lw=0.8, alpha=0.7)
            for i, method in enumerate(methods):
                vals  = [ds_results[method].get(s, {}).get(metric, np.nan) for s in strata]
                color = C.get(method, '#9CA3AF')
                ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9,
                       label=method, color=color, alpha=0.85,
                       edgecolor='#1a2740' if method == 'CHADO' else 'white',
                       linewidth=2.0 if method == 'CHADO' else 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(['Low\nAmb.', 'Medium\nAmb.', 'High\nAmb.'], fontsize=10)
            titles = {'acc': 'Accuracy', 'macro_f1': 'Macro-F1', 'ece': 'ECE ↓'}
            ax.set_ylabel(titles.get(metric, metric), fontsize=11)
            ax.set_title(f'{ds} — {titles.get(metric, metric)} by Ambiguity',
                         fontweight='bold', fontsize=11)
            if ax_i == 0:
                ax.legend(fontsize=8, ncol=2, framealpha=0.8)

        plt.suptitle(f'{ds}: Independent-Ambiguity Stratified Analysis',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        ds_key = ds.lower().replace('-', '_').replace(' ', '_')
        for ext in ('pdf', 'png'):
            plt.savefig(f'{FIG_DIR}/issue11_{ds_key}_bars.{ext}',
                        dpi=300, bbox_inches='tight')
        plt.close()

        # ── Line drop plot (F1 across strata) ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(C['bg'])
        ax.set_facecolor(C['bg'])
        ax.grid(color=C['grid'], lw=0.8, alpha=0.7)
        for method in methods:
            vals  = [ds_results[method].get(s, {}).get('macro_f1', np.nan)
                     for s in strata]
            color = C.get(method, '#9CA3AF')
            ax.plot(strata, vals, 'o-', color=color, lw=2.5 if method=='CHADO' else 1.5,
                    ms=8 if method=='CHADO' else 6,
                    zorder=5 if method=='CHADO' else 3, label=method)
        ax.set_xlabel('Annotator-Disagreement Stratum', fontsize=11)
        ax.set_ylabel('Macro-F1', fontsize=11)
        ax.set_title(f'{ds} — F1 Degradation vs. Annotator Disagreement',
                     fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)
        plt.tight_layout()
        for ext in ('pdf', 'png'):
            plt.savefig(f'{FIG_DIR}/issue11_{ds_key}_drop.{ext}',
                        dpi=300, bbox_inches='tight')
        plt.close()

    print(f'  Figures → {FIG_DIR}/issue11_*')


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 1.2 — Radius–ambiguity quantitative correlation
# ═══════════════════════════════════════════════════════════════════════════════

def extract_cls_norms_iemocap(ckpt_path, device):
    """
    Run CHADO on IEMOCAP test set, return {utt_id: ||z_cls||_2}.
    z_cls = cat(z_c, z_u, z_m) ∈ ℝ^192 — the factor-based classification vector.
    """
    from models.chado.model import CHADOTrimodal
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap

    cfg  = yaml.safe_load(open(f'{BASE}/configs/iemocap/chado_iemocap.yaml'))
    dc   = cfg['data']
    sd   = load_sd(ckpt_path)
    tm   = detect_text_model(sd)
    cc   = cfg.get('chado', {})

    model = CHADOTrimodal(
        text_model_name=tm,
        audio_model_name=cfg['model']['audio_model_name'],
        video_model_name=cfg['model']['video_model_name'],
        num_classes=dc['num_classes'],
        proj_dim=cfg['model'].get('proj_dim', 256), dropout=0.0,
        use_text=True, use_audio=True, use_video=True,
        use_gated_fusion=True, backbone='ctnet',
        use_causal=True, use_hyperbolic=cc.get('use_hyperbolic', True),
        use_ot=cc.get('use_ot', True), use_mad=False,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds     = IEMOCAPDataset(csv_path=dc['test_csv'], text_model_name=tm,
                             max_text_len=dc.get('max_text_len', 96),
                             audio_sr=16000, audio_sec=4.0, n_frames=8,
                             use_audio=True, use_video=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                        collate_fn=collate_iemocap)

    df     = pd.read_csv(dc['test_csv'])
    uids   = df['utt_id'].astype(str).tolist()
    norms  = {}
    ptr    = 0
    with torch.no_grad():
        for batch in loader:
            B  = batch['input_ids'].shape[0]
            ti = {'input_ids':      batch['input_ids'].to(device),
                  'attention_mask': batch['attention_mask'].to(device)}
            aw = batch.get('wav')
            if aw is None:
                aw = batch.get('audio_wave')
            if aw is not None:
                aw = aw.to(device)
            vf = batch.get('video_frames')
            if vf is None:
                vf = torch.zeros(B, 8, 3, 224, 224)
            vf = vf.to(device)

            _, _, aux = model(text_input=ti, audio_wave=aw, video_frames=vf)
            if 'factors' in aux:
                z_c, z_u, _, z_m, _ = aux['factors']
                cls_feat = torch.cat([z_c, z_u, z_m], dim=1)
                batch_norms = torch.norm(cls_feat, dim=1).cpu().numpy()
            else:
                batch_norms = np.zeros(B)
            for uid, norm in zip(uids[ptr:ptr + B], batch_norms):
                norms[uid] = float(norm)
            ptr += B

    print(f'  IEMOCAP ||Z_cls|| extracted: {len(norms)} samples, '
          f'mean={np.mean(list(norms.values())):.4f}')
    return norms


def extract_cls_norms_mosei(ckpt_path, device):
    """Run CHADOFeature on MOSEI test, return {utt_id: ||z_cls||_2}."""
    from models.chado.model import CHADOFeature
    from datasets.mosei.mosei_utt_dataset import MoseiUttDataset, collate_mosei_utt

    cfg = yaml.safe_load(open(f'{BASE}/configs/mosei/chado_mosei.yaml'))
    dc  = cfg['data']; mc = cfg['model']; cc = cfg.get('chado', {})
    sd  = load_sd(ckpt_path)

    model = CHADOFeature(
        num_classes=dc['num_classes'], d_model=mc.get('d_model', 256),
        use_audio=True, use_video=True,
        text_model=mc.get('text_model_name', 'roberta-base'),
        modality_dropout=0.0, use_causal=True,
        use_hyperbolic=cc.get('use_hyperbolic', True),
        use_ot=cc.get('use_ot', True), use_mad=True,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds     = MoseiUttDataset(manifest_path=dc['test_manifest'],
                              max_audio_len=dc.get('max_audio_len', 50),
                              max_video_len=dc.get('max_video_len', 30))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=collate_mosei_utt)

    norms = {}
    with torch.no_grad():
        for batch in loader:
            bd  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
            _, _, _, aux = model(bd)
            if 'factors' in aux:
                z_c, z_u, _, z_m, _ = aux['factors']
                cls_feat = torch.cat([z_c, z_u, z_m], dim=1)
                batch_norms = torch.norm(cls_feat, dim=1).cpu().numpy()
            else:
                batch_norms = np.zeros(len(batch['utt_id']))
            for uid, norm in zip(batch['utt_id'], batch_norms):
                norms[uid] = float(norm)

    print(f'  CMU-MOSEI ||Z_cls|| extracted: {len(norms)} samples, '
          f'mean={np.mean(list(norms.values())):.4f}')
    return norms


def run_issue12(device, iemocap_amb, mosei_amb):
    """Issue 1.2: Radius–ambiguity Spearman correlation."""
    print('\n' + '=' * 70)
    print('ISSUE 1.2 — Radial Regularizer: ||Z_cls|| vs. Annotator Disagreement')
    print('=' * 70)

    results = {}

    # IEMOCAP
    for variant, ckpt_key in [('CHADO', 'chado'), ('w/o Hyperbolic', 'chado_wo_hyperbolic')]:
        ckpt = f'{RES}/iemocap/{ckpt_key}/best.pt'
        if not os.path.exists(ckpt):
            print(f'  [{variant}] IEMOCAP ckpt missing → skip')
            continue
        if not iemocap_amb:
            continue
        print(f'\n  IEMOCAP [{variant}] — extracting cls norms …')
        try:
            norms = extract_cls_norms_iemocap(ckpt, device)
        except Exception as e:
            print(f'    ERROR: {e}'); continue

        n_arr, d_arr, uid_list = [], [], []
        for uid, norm in norms.items():
            if uid in iemocap_amb:
                n_arr.append(norm)
                d_arr.append(iemocap_amb[uid])
                uid_list.append(uid)

        if len(n_arr) < 10:
            print(f'    Only {len(n_arr)} aligned → skip'); continue

        n_arr = np.array(n_arr)
        d_arr = np.array(d_arr)
        sr, sp = spearmanr(n_arr, d_arr)
        pr, pp = pearsonr(n_arr, d_arr)

        def sig(p):
            return '***' if p < .001 else ('**' if p < .01 else ('*' if p < .05 else 'ns'))

        print(f'  IEMOCAP [{variant}]  N={len(n_arr)}:  '
              f'Spearman ρ={sr:+.4f} p={sp:.4f}{sig(sp)}  '
              f'Pearson r={pr:+.4f} p={pp:.4f}{sig(pp)}')

        results[f'IEMOCAP_{variant}'] = {
            'n': len(n_arr), 'spearman_r': float(sr), 'spearman_p': float(sp),
            'pearson_r': float(pr), 'pearson_p': float(pp),
        }

    # CMU-MOSEI
    for variant, ckpt_key in [('CHADO', 'chado'), ('w/o Hyperbolic', 'chado_wo_hyperbolic')]:
        ckpt = f'{RES}/mosei/{ckpt_key}/best.pt'
        if not os.path.exists(ckpt):
            print(f'  [{variant}] CMU-MOSEI ckpt missing → skip')
            continue
        if not mosei_amb:
            continue
        print(f'\n  CMU-MOSEI [{variant}] — extracting cls norms …')
        try:
            norms = extract_cls_norms_mosei(ckpt, device)
        except Exception as e:
            print(f'    ERROR: {e}'); continue

        n_arr, d_arr = [], []
        for uid, norm in norms.items():
            if uid in mosei_amb:
                n_arr.append(norm)
                d_arr.append(mosei_amb[uid])

        if len(n_arr) < 10:
            print(f'    Only {len(n_arr)} aligned → skip'); continue

        n_arr = np.array(n_arr)
        d_arr = np.array(d_arr)
        sr, sp = spearmanr(n_arr, d_arr)
        pr, pp = pearsonr(n_arr, d_arr)

        print(f'  CMU-MOSEI [{variant}]  N={len(n_arr)}:  '
              f'Spearman ρ={sr:+.4f} p={sp:.4f}  '
              f'Pearson r={pr:+.4f} p={pp:.4f}')

        results[f'MOSEI_{variant}'] = {
            'n': len(n_arr), 'spearman_r': float(sr), 'spearman_p': float(sp),
            'pearson_r': float(pr), 'pearson_p': float(pp),
        }

    out_path = f'{OUT_DIR}/issue12_radius_ambiguity.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved → {out_path}')

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f'\n  {"Dataset / Variant":<30} {"N":>6}  '
          f'{"Spearman ρ":>10}  {"p":>8}  {"Pearson r":>10}  {"p":>8}')
    print('  ' + '-' * 80)
    for key, r in results.items():
        def sig(p):
            return '***' if p < .001 else ('**' if p < .01 else ('*' if p < .05 else 'ns'))
        print(f'  {key:<30} {r["n"]:>6}  '
              f'{r["spearman_r"]:>+10.4f}  {r["spearman_p"]:>6.4f}{sig(r["spearman_p"]):<2}  '
              f'{r["pearson_r"]:>+10.4f}  {r["pearson_p"]:>6.4f}{sig(r["pearson_p"]):<2}')
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 1.3 — Factor-level ablation (inference-time masking)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_ablated_iemocap_model(ckpt_path, device, zero_factor=None):
    """
    Load CHADOTrimodal; monkey-patch cls_features to zero out one factor.
    zero_factor ∈ {None, 'z_c', 'z_u', 'z_m'} (z_t, z_e not in cls_features).
    """
    from models.chado.model import CHADOTrimodal

    cfg = yaml.safe_load(open(f'{BASE}/configs/iemocap/chado_iemocap.yaml'))
    dc, mc, cc = cfg['data'], cfg['model'], cfg.get('chado', {})
    sd = load_sd(ckpt_path)
    tm = detect_text_model(sd)

    model = CHADOTrimodal(
        text_model_name=tm,
        audio_model_name=mc['audio_model_name'],
        video_model_name=mc['video_model_name'],
        num_classes=dc['num_classes'],
        proj_dim=mc.get('proj_dim', 256), dropout=0.0,
        use_text=True, use_audio=True, use_video=True,
        use_gated_fusion=True, backbone='ctnet',
        use_causal=True, use_hyperbolic=cc.get('use_hyperbolic', True),
        use_ot=False, use_mad=False,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    if zero_factor in ('z_c', 'z_u', 'z_m'):
        def patched_cls_features(self, z_c, z_u, z_m):
            if zero_factor == 'z_c':
                z_c = torch.zeros_like(z_c)
            elif zero_factor == 'z_u':
                z_u = torch.zeros_like(z_u)
            elif zero_factor == 'z_m':
                z_m = torch.zeros_like(z_m)
            return torch.cat([z_c, z_u, z_m], dim=1)
        model.causal.cls_features = types.MethodType(patched_cls_features, model.causal)

    return model, cfg


def _make_ablated_mosei_model(ckpt_path, device, zero_factor=None):
    """Same as above but for CHADOFeature (CMU-MOSEI)."""
    from models.chado.model import CHADOFeature

    cfg = yaml.safe_load(open(f'{BASE}/configs/mosei/chado_mosei.yaml'))
    dc, mc, cc = cfg['data'], cfg['model'], cfg.get('chado', {})
    sd = load_sd(ckpt_path)

    model = CHADOFeature(
        num_classes=dc['num_classes'], d_model=mc.get('d_model', 256),
        use_audio=True, use_video=True,
        text_model=mc.get('text_model_name', 'roberta-base'),
        modality_dropout=0.0, use_causal=True,
        use_hyperbolic=cc.get('use_hyperbolic', True),
        use_ot=False, use_mad=True,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    if zero_factor in ('z_c', 'z_u', 'z_m'):
        def patched_cls_features(self, z_c, z_u, z_m):
            if zero_factor == 'z_c':
                z_c = torch.zeros_like(z_c)
            elif zero_factor == 'z_u':
                z_u = torch.zeros_like(z_u)
            elif zero_factor == 'z_m':
                z_m = torch.zeros_like(z_m)
            return torch.cat([z_c, z_u, z_m], dim=1)
        model.causal.cls_features = types.MethodType(patched_cls_features, model.causal)

    return model, cfg


def _infer_iemocap_ablated(model, cfg, device):
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap
    dc = cfg['data']
    sd = load_sd(f'{RES}/iemocap/chado/best.pt')
    tm = detect_text_model(sd)
    LMAP = {'neu':0,'neutral':0,'hap':1,'happy':1,'exc':1,
            'ang':2,'angry':2,'fru':2,'sad':3,'sadness':3}
    df  = pd.read_csv(dc['test_csv'])
    labels = np.array([LMAP.get(str(v).lower(), 0) if isinstance(v, str) else int(v)
                        for v in df['label_4' if 'label_4' in df.columns else 'label']])
    ds  = IEMOCAPDataset(csv_path=dc['test_csv'], text_model_name=tm,
                          max_text_len=dc.get('max_text_len', 96),
                          audio_sr=16000, audio_sec=4.0, n_frames=8,
                          use_audio=True, use_video=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                        collate_fn=collate_iemocap)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            B  = batch['input_ids'].shape[0]
            ti = {'input_ids':      batch['input_ids'].to(device),
                  'attention_mask': batch['attention_mask'].to(device)}
            aw = batch.get('wav')
            if aw is None:
                aw = batch.get('audio_wave')
            if aw is not None:
                aw = aw.to(device)
            vf = batch.get('video_frames')
            if vf is None:
                vf = torch.zeros(B, 8, 3, 224, 224)
            vf = vf.to(device)
            out = model(text_input=ti, audio_wave=aw, video_frames=vf)
            all_probs.append(torch.softmax(out[0], dim=-1).cpu().numpy())
    probs = np.vstack(all_probs)
    preds = probs.argmax(axis=1)
    acc   = float(accuracy_score(labels, preds))
    mf1   = float(f1_score(labels, preds, average='macro', zero_division=0))
    return acc, mf1


def _infer_mosei_ablated(model, cfg, device):
    from datasets.mosei.mosei_utt_dataset import MoseiUttDataset, collate_mosei_utt
    dc  = cfg['data']
    ds  = MoseiUttDataset(manifest_path=dc['test_manifest'],
                           max_audio_len=dc.get('max_audio_len', 50),
                           max_video_len=dc.get('max_video_len', 30))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=collate_mosei_utt)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
            out    = model(bd)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            probs  = torch.sigmoid(logits).cpu().numpy()
            labels = batch['label'].numpy() if hasattr(batch['label'], 'numpy') else np.array(batch['label'])
            all_probs.append(probs)
            all_labels.append(labels)

    probs  = np.vstack(all_probs)
    labels = np.vstack(all_labels) if all_labels[0].ndim > 1 else np.concatenate(all_labels)

    # dominant class
    probs_norm = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
    preds      = probs_norm.argmax(axis=1)
    if labels.ndim > 1:
        labels_dom = labels.argmax(axis=1)
    else:
        labels_dom = labels.astype(int)

    acc = float(accuracy_score(labels_dom, preds))
    mf1 = float(f1_score(labels_dom, preds, average='macro', zero_division=0))
    return acc, mf1


def run_issue13(device):
    """Issue 1.3: Factor-level ablation."""
    print('\n' + '=' * 70)
    print('ISSUE 1.3 — Factor-Level Ablation (Inference-Time Masking)')
    print('=' * 70)

    # Factors: z_c, z_u, z_m affect the classification head directly.
    # z_t, z_e are NOT used in cls_features — they serve as independence
    # regularizers during training. Their inference-time effect is therefore
    # zero (masking them at test time = no change). We note this clearly.

    factors = [
        ('Full CHADO',  None),
        ('w/o z_c',     'z_c'),
        ('w/o z_u',     'z_u'),
        ('w/o z_m',     'z_m'),
    ]

    results = {}

    # ── IEMOCAP ──────────────────────────────────────────────────────────────
    ckpt_ie = f'{RES}/iemocap/chado/best.pt'
    if os.path.exists(ckpt_ie):
        print('\n  IEMOCAP — factor ablation:')
        print(f'  {"Variant":<20} {"Acc":>8}  {"Macro-F1":>10}  {"ΔACC":>8}  {"ΔF1":>8}')
        print('  ' + '-' * 60)
        iemo_res = {}
        base_acc = base_f1 = None
        for label, factor in factors:
            try:
                model, cfg = _make_ablated_iemocap_model(ckpt_ie, device, factor)
                acc, mf1   = _infer_iemocap_ablated(model, cfg, device)
                del model; torch.cuda.empty_cache()
            except Exception as e:
                print(f'  {label:<20}  ERROR: {e}'); continue
            if base_acc is None:
                base_acc, base_f1 = acc, mf1
            d_acc = acc - base_acc
            d_f1  = mf1 - base_f1
            print(f'  {label:<20} {acc*100:>7.2f}%  {mf1*100:>9.2f}%  '
                  f'{d_acc*100:>+7.2f}%  {d_f1*100:>+7.2f}%')
            iemo_res[label] = {'acc': round(acc, 4), 'macro_f1': round(mf1, 4),
                               'delta_acc': round(d_acc, 4), 'delta_f1': round(d_f1, 4)}
        results['IEMOCAP'] = iemo_res
    else:
        print(f'\n  IEMOCAP CHADO checkpoint missing → skip')

    # ── CMU-MOSEI ─────────────────────────────────────────────────────────────
    ckpt_mo = f'{RES}/mosei/chado/best.pt'
    if os.path.exists(ckpt_mo):
        print('\n  CMU-MOSEI — factor ablation:')
        print(f'  {"Variant":<20} {"Acc":>8}  {"Macro-F1":>10}  {"ΔACC":>8}  {"ΔF1":>8}')
        print('  ' + '-' * 60)
        mosei_res = {}
        base_acc = base_f1 = None
        for label, factor in factors:
            try:
                model, cfg = _make_ablated_mosei_model(ckpt_mo, device, factor)
                acc, mf1   = _infer_mosei_ablated(model, cfg, device)
                del model; torch.cuda.empty_cache()
            except Exception as e:
                print(f'  {label:<20}  ERROR: {e}'); continue
            if base_acc is None:
                base_acc, base_f1 = acc, mf1
            d_acc = acc - base_acc
            d_f1  = mf1 - base_f1
            print(f'  {label:<20} {acc*100:>7.2f}%  {mf1*100:>9.2f}%  '
                  f'{d_acc*100:>+7.2f}%  {d_f1*100:>+7.2f}%')
            mosei_res[label] = {'acc': round(acc, 4), 'macro_f1': round(mf1, 4),
                                'delta_acc': round(d_acc, 4), 'delta_f1': round(d_f1, 4)}
        results['CMU-MOSEI'] = mosei_res
    else:
        print(f'\n  CMU-MOSEI CHADO checkpoint missing → skip')

    # Note on z_t and z_e
    note = (
        "Note: z_t (temporality) and z_e (residual) are NOT part of the "
        "classification head (cls_features = cat(z_c, z_u, z_m) only). "
        "Their contribution is exclusively through independence regularization "
        "during training. Zeroing them at inference time has no effect on logits. "
        "A proper ablation of z_t/z_e requires retraining without those factor "
        "heads — logged as future work."
    )
    print(f'\n  NOTE: {note}')
    results['note'] = note

    # ── Plot ─────────────────────────────────────────────────────────────────
    for ds_name, ds_res in results.items():
        if ds_name == 'note' or not isinstance(ds_res, dict):
            continue
        variants = [v for v in ds_res if v != 'Full CHADO']
        if not variants:
            continue
        base_acc = ds_res.get('Full CHADO', {}).get('acc', 0)
        base_f1  = ds_res.get('Full CHADO', {}).get('macro_f1', 0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor(C['bg'])
        for ax_i, (metric, base_val) in enumerate([('acc', base_acc), ('macro_f1', base_f1)]):
            ax = axes[ax_i]
            ax.set_facecolor(C['bg'])
            ax.grid(axis='y', color=C['grid'], lw=0.8)
            all_v  = ['Full CHADO'] + variants
            colors = ['#2E4057'] + ['#C25B5B' if 'z_c' in v else '#048A81' if 'z_u' in v
                                    else '#E07B3F' for v in variants]
            vals   = [ds_res.get(v, {}).get(metric, 0) for v in all_v]
            bars   = ax.bar(range(len(all_v)), vals, color=colors, alpha=0.85,
                            edgecolor='white', linewidth=1.2)
            ax.axhline(base_val, color='#2E4057', ls='--', lw=1.2, alpha=0.5)
            ax.set_xticks(range(len(all_v)))
            ax.set_xticklabels(all_v, rotation=20, ha='right', fontsize=9)
            title = 'Accuracy' if metric == 'acc' else 'Macro-F1'
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f'{ds_name} — Factor Ablation ({title})',
                         fontweight='bold', fontsize=11)
        plt.tight_layout()
        ds_key = ds_name.lower().replace('-', '_').replace(' ', '_')
        for ext in ('pdf', 'png'):
            plt.savefig(f'{FIG_DIR}/issue13_{ds_key}_factor_ablation.{ext}',
                        dpi=300, bbox_inches='tight')
        plt.close()

    out_path = f'{OUT_DIR}/issue13_factor_ablation.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved → {out_path}')
    print(f'  Figures → {FIG_DIR}/issue13_*')
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 1.4 — Predicted distribution vs. annotator distribution
# ═══════════════════════════════════════════════════════════════════════════════

def jsd(p, q, eps=1e-10):
    """Jensen-Shannon Divergence (base-2 bits, ∈ [0,1])."""
    p = np.array(p, float) + eps
    q = np.array(q, float) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (scipy_entropy(p, m) + scipy_entropy(q, m)))


def kl_div(p, q, eps=1e-10):
    """KL(P||Q) in nats."""
    p = np.array(p, float) + eps
    q = np.array(q, float) + eps
    p /= p.sum(); q /= q.sum()
    return float(scipy_entropy(p, q))


def get_iemocap_annotator_distributions(eval_root):
    """
    For each IEMOCAP utterance, return fractional vote distribution over 4 classes.
    {utt_id: np.array of shape (4,) summing to 1}
    """
    files  = glob.glob(os.path.join(eval_root, 'Session*/dialog/EmoEvaluation/*.txt'))
    hdr    = re.compile(r'\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+\S+\s+\[')
    rtr    = re.compile(r'C-\w+:\s*([^;]+)')
    out    = {}
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
                labels.extend(labs)
            if len(labels) >= 2:
                cnt = np.bincount(labels, minlength=4).astype(float)
                out[uid] = cnt / cnt.sum()
    return out


def get_mosei_annotator_distributions(manifest_path):
    """
    For each CMU-MOSEI utterance, return normalized label_raw as soft dist.
    {utt_id: np.array of shape (6,) summing to 1}
    """
    out = {}
    for line in open(manifest_path):
        s   = json.loads(line)
        raw = np.array(s.get('label_raw', [0.0] * 6), float)
        tot = raw.sum()
        if tot < 1e-6:
            out[s['utt_id']] = np.ones(6) / 6.0  # uniform if no annotation
        else:
            out[s['utt_id']] = raw / tot
    return out


def run_issue14(device, iemocap_amb, mosei_amb):
    """Issue 1.4: KL / JSD / Wasserstein between predicted and annotator distributions."""
    print('\n' + '=' * 70)
    print('ISSUE 1.4 — Predicted vs. Annotator Distribution Divergence')
    print('=' * 70)

    results = {}

    # ── IEMOCAP ──────────────────────────────────────────────────────────────
    eval_root = '/home/tahirahmad/IEMOCAP_Final/dataset/IEMOCAP_full_release'
    ckpt_ie   = f'{RES}/iemocap/chado/best.pt'

    if os.path.exists(ckpt_ie) and os.path.isdir(eval_root):
        print('\n  IEMOCAP — KL / JSD / Wasserstein …')
        annotator_dists = get_iemocap_annotator_distributions(eval_root)
        print(f'    Annotator distributions: {len(annotator_dists)} utterances')
        try:
            uids, probs, labels = infer_iemocap_model(ckpt_ie, device, 'chado')
        except Exception as e:
            print(f'    Inference failed: {e}')
            uids = []

        kl_list, jsd_list, wass_list = [], [], []
        for uid, p in zip(uids, probs):
            if uid not in annotator_dists:
                continue
            q = annotator_dists[uid]  # annotator distribution
            kl_list.append(kl_div(p, q))
            jsd_list.append(jsd(p, q))
            # Wasserstein (1D, sorted classes as positions 0-3)
            pos = np.arange(len(q))
            wass_list.append(wasserstein_distance(pos, pos, p + 1e-10, q + 1e-10))

        if kl_list:
            iemo_div = {
                'n':   len(kl_list),
                'kl_div_mean':   round(float(np.mean(kl_list)),   4),
                'kl_div_std':    round(float(np.std(kl_list)),    4),
                'jsd_mean':      round(float(np.mean(jsd_list)),  4),
                'jsd_std':       round(float(np.std(jsd_list)),   4),
                'wasserstein_mean': round(float(np.mean(wass_list)), 4),
                'wasserstein_std':  round(float(np.std(wass_list)),  4),
            }
            results['IEMOCAP'] = iemo_div
            print(f'    N={len(kl_list)}')
            print(f'    KL(P||Q)  mean={iemo_div["kl_div_mean"]:.4f}  '
                  f'std={iemo_div["kl_div_std"]:.4f}')
            print(f'    JSD       mean={iemo_div["jsd_mean"]:.4f}  '
                  f'std={iemo_div["jsd_std"]:.4f}')
            print(f'    Wass.     mean={iemo_div["wasserstein_mean"]:.4f}  '
                  f'std={iemo_div["wasserstein_std"]:.4f}')

            # Ambiguity-stratified divergence
            if iemocap_amb:
                amb_strata, div_strata = [], {m: [] for m in ('kl', 'jsd', 'wass')}
                for uid, p, kl, j, w in zip(
                        uids, probs, kl_list, jsd_list, wass_list):
                    if uid in iemocap_amb:
                        amb_strata.append((iemocap_amb[uid], kl, j, w))
                amb_strata.sort(key=lambda x: x[0])
                n3 = len(amb_strata) // 3
                tiers = {'Low': amb_strata[:n3], 'Medium': amb_strata[n3:2*n3],
                         'High': amb_strata[2*n3:]}
                print(f'\n    Divergence by annotator-ambiguity stratum:')
                print(f'    {"Tier":<8} {"N":>5}  {"KL":>8}  {"JSD":>8}  {"Wass.":>8}')
                print(f'    {"-"*44}')
                tier_res = {}
                for tname, rows in tiers.items():
                    if not rows: continue
                    kl_t = [r[1] for r in rows]
                    j_t  = [r[2] for r in rows]
                    w_t  = [r[3] for r in rows]
                    print(f'    {tname:<8} {len(rows):>5}  '
                          f'{np.mean(kl_t):>8.4f}  {np.mean(j_t):>8.4f}  '
                          f'{np.mean(w_t):>8.4f}')
                    tier_res[tname] = {
                        'n': len(rows),
                        'kl': round(float(np.mean(kl_t)), 4),
                        'jsd': round(float(np.mean(j_t)), 4),
                        'wasserstein': round(float(np.mean(w_t)), 4),
                    }
                results['IEMOCAP']['stratified'] = tier_res
    else:
        print(f'\n  IEMOCAP: ckpt or eval_root missing → skip')

    # ── CMU-MOSEI ─────────────────────────────────────────────────────────────
    ckpt_mo = f'{RES}/mosei/chado/best.pt'
    test_manifest = f'{BASE}/data/processed/mosei/mosei_utt_test.jsonl'

    if os.path.exists(ckpt_mo) and os.path.exists(test_manifest):
        print('\n  CMU-MOSEI — KL / JSD / Wasserstein …')
        annotator_dists = get_mosei_annotator_distributions(test_manifest)
        print(f'    Annotator distributions: {len(annotator_dists)} utterances')
        try:
            uids, probs, labels = infer_mosei_model(ckpt_mo, device, 'chado')
        except Exception as e:
            print(f'    Inference failed: {e}'); uids = []

        kl_list, jsd_list, wass_list = [], [], []
        uid_used = []
        for uid, p in zip(uids, probs):
            if uid not in annotator_dists:
                continue
            q = annotator_dists[uid]
            # Normalize sigmoid probs to a distribution
            p_norm = (p + 1e-10) / (p + 1e-10).sum()
            kl_list.append(kl_div(p_norm, q))
            jsd_list.append(jsd(p_norm, q))
            pos = np.arange(len(q))
            wass_list.append(wasserstein_distance(pos, pos, p_norm, q + 1e-10))
            uid_used.append(uid)

        if kl_list:
            mosei_div = {
                'n':   len(kl_list),
                'kl_div_mean':   round(float(np.mean(kl_list)),   4),
                'kl_div_std':    round(float(np.std(kl_list)),    4),
                'jsd_mean':      round(float(np.mean(jsd_list)),  4),
                'jsd_std':       round(float(np.std(jsd_list)),   4),
                'wasserstein_mean': round(float(np.mean(wass_list)), 4),
                'wasserstein_std':  round(float(np.std(wass_list)),  4),
            }
            results['CMU-MOSEI'] = mosei_div
            print(f'    N={len(kl_list)}')
            print(f'    KL(P||Q)  mean={mosei_div["kl_div_mean"]:.4f}  '
                  f'std={mosei_div["kl_div_std"]:.4f}')
            print(f'    JSD       mean={mosei_div["jsd_mean"]:.4f}  '
                  f'std={mosei_div["jsd_std"]:.4f}')
            print(f'    Wass.     mean={mosei_div["wasserstein_mean"]:.4f}  '
                  f'std={mosei_div["wasserstein_std"]:.4f}')

            if mosei_amb:
                amb_strata = []
                for uid, kl, j, w in zip(uid_used, kl_list, jsd_list, wass_list):
                    if uid in mosei_amb:
                        amb_strata.append((mosei_amb[uid], kl, j, w))
                amb_strata.sort(key=lambda x: x[0])
                n3 = len(amb_strata) // 3
                tiers = {'Low': amb_strata[:n3], 'Medium': amb_strata[n3:2*n3],
                         'High': amb_strata[2*n3:]}
                print(f'\n    Divergence by annotation-entropy stratum:')
                print(f'    {"Tier":<8} {"N":>5}  {"KL":>8}  {"JSD":>8}  {"Wass.":>8}')
                print(f'    {"-"*44}')
                tier_res = {}
                for tname, rows in tiers.items():
                    if not rows: continue
                    kl_t = [r[1] for r in rows]
                    j_t  = [r[2] for r in rows]
                    w_t  = [r[3] for r in rows]
                    print(f'    {tname:<8} {len(rows):>5}  '
                          f'{np.mean(kl_t):>8.4f}  {np.mean(j_t):>8.4f}  '
                          f'{np.mean(w_t):>8.4f}')
                    tier_res[tname] = {
                        'n': len(rows),
                        'kl': round(float(np.mean(kl_t)), 4),
                        'jsd': round(float(np.mean(j_t)), 4),
                        'wasserstein': round(float(np.mean(w_t)), 4),
                    }
                results['CMU-MOSEI']['stratified'] = tier_res

    # ── Plot: distribution divergence bar chart ───────────────────────────────
    _plot_divergence(results)

    out_path = f'{OUT_DIR}/issue14_distribution_divergence.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved → {out_path}')
    return results


def _plot_divergence(results):
    datasets  = [k for k in results if isinstance(results[k], dict) and 'n' in results[k]]
    if not datasets:
        return
    metrics   = ['kl_div_mean', 'jsd_mean', 'wasserstein_mean']
    mlabels   = ['KL(P||Q)', 'JSD', 'Wasserstein']
    x         = np.arange(len(datasets))
    width     = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor(C['bg'])
    colors = ['#2E4057', '#048A81', '#C25B5B']
    for ax_i, (metric, mlabel) in enumerate(zip(metrics, mlabels)):
        ax = axes[ax_i]
        ax.set_facecolor(C['bg'])
        ax.grid(axis='y', color=C['grid'], lw=0.8, alpha=0.7)
        vals = [results[ds].get(metric, 0) for ds in datasets]
        stds = [results[ds].get(metric.replace('mean', 'std'), 0) for ds in datasets]
        ax.bar(x, vals, width * 2, color=colors[ax_i], alpha=0.85,
               edgecolor='white', linewidth=1.2, yerr=stds, capsize=5,
               error_kw={'elinewidth': 1.5, 'ecolor': '#1a2740'})
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_title(f'{mlabel} Divergence\n(Predicted vs. Annotator)',
                     fontweight='bold', fontsize=11)
    plt.suptitle('Issue 1.4: Predicted Distribution vs. Annotator Distribution',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{FIG_DIR}/issue14_divergence.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Figure → {FIG_DIR}/issue14_divergence.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Output: {OUT_DIR}')

    # ── Pre-compute model-independent ambiguity scores ───────────────────────
    print('\n' + '=' * 70)
    print('Pre-computing Model-Independent Ambiguity Scores')
    print('=' * 70)

    eval_root = '/home/tahirahmad/IEMOCAP_Final/dataset/IEMOCAP_full_release'
    iemocap_amb = {}
    if os.path.isdir(eval_root):
        print('\n  IEMOCAP — parsing annotator vote entropy …')
        iemocap_amb = parse_iemocap_annotator_entropy(eval_root)
    else:
        print(f'  IEMOCAP eval_root not found: {eval_root}')

    mosei_manifest = f'{BASE}/data/processed/mosei/mosei_utt_test.jsonl'
    mosei_amb = {}
    if os.path.exists(mosei_manifest):
        print('\n  CMU-MOSEI — computing label_raw entropy …')
        mosei_amb = compute_mosei_label_entropy(mosei_manifest)

    meld_test_csv  = f'{BASE}/data/processed/meld/meld_test.csv'
    meld_train_csv = f'{BASE}/data/processed/meld/meld_train.csv'
    meld_amb = {}
    if os.path.exists(meld_test_csv) and os.path.exists(meld_train_csv):
        print('\n  MELD — computing KNN soft-label entropy …')
        try:
            meld_amb = compute_meld_knn_entropy(meld_test_csv, meld_train_csv, device)
        except Exception as e:
            print(f'  MELD KNN failed: {e}')

    # ── Run four issue analyses ───────────────────────────────────────────────
    all_output = {}

    print('\n\nRunning Issue 1.1 …')
    all_output['issue_1.1'] = run_issue11(device, iemocap_amb, mosei_amb, meld_amb)

    print('\n\nRunning Issue 1.2 …')
    all_output['issue_1.2'] = run_issue12(device, iemocap_amb, mosei_amb)

    print('\n\nRunning Issue 1.3 …')
    all_output['issue_1.3'] = run_issue13(device)

    print('\n\nRunning Issue 1.4 …')
    all_output['issue_1.4'] = run_issue14(device, iemocap_amb, mosei_amb)

    # ── Final summary ─────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('REVIEWER ISSUES — SUMMARY')
    print('=' * 70)

    print('\n  Issue 1.1: Independent-Ambiguity Stratified Analysis')
    for ds, ds_res in all_output.get('issue_1.1', {}).items():
        if not isinstance(ds_res, dict): continue
        if 'CHADO' in ds_res:
            lo  = ds_res['CHADO'].get('Low',   {}).get('macro_f1', 0)
            hi  = ds_res['CHADO'].get('High',  {}).get('macro_f1', 0)
            print(f'    {ds}: CHADO Low-F1={lo:.3f} → High-F1={hi:.3f}  '
                  f'(drop={lo-hi:.3f})')

    print('\n  Issue 1.2: Radius-Ambiguity Correlation')
    for key, r in all_output.get('issue_1.2', {}).items():
        if isinstance(r, dict) and 'spearman_r' in r:
            print(f'    {key}: Spearman ρ={r["spearman_r"]:+.4f}  '
                  f'p={r["spearman_p"]:.4f}  N={r["n"]}')

    print('\n  Issue 1.3: Factor Ablation (IEMOCAP)')
    for variant, metrics in all_output.get('issue_1.3', {}).get('IEMOCAP', {}).items():
        if isinstance(metrics, dict) and 'acc' in metrics:
            print(f'    {variant:<22} Acc={metrics["acc"]*100:.2f}%  '
                  f'F1={metrics["macro_f1"]*100:.2f}%')

    print('\n  Issue 1.4: Distribution Divergence')
    for ds, r in all_output.get('issue_1.4', {}).items():
        if isinstance(r, dict) and 'kl_div_mean' in r:
            print(f'    {ds}: KL={r["kl_div_mean"]:.4f}  '
                  f'JSD={r["jsd_mean"]:.4f}  '
                  f'Wass={r["wasserstein_mean"]:.4f}  (N={r["n"]})')

    out_path = f'{OUT_DIR}/all_reviewer_issues.json'
    with open(out_path, 'w') as f:
        json.dump(all_output, f, indent=2, default=str)
    print(f'\n  Full results saved → {out_path}')
    print(f'  Figures saved   → {FIG_DIR}/')


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
    main()
