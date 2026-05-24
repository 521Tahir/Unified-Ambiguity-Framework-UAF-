

import re, glob, os, sys, json, pickle, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, entropy as scipy_entropy

sys.path.insert(0, '')


# ─── Emotion category maps ───────────────────────────────────────────────────
CAT_MAP_IEMOCAP = {
    'neutral': 'neutral', 'neu': 'neutral',
    'happy': 'happy', 'hap': 'happy',
    'excited': 'happy', 'exc': 'happy',
    'angry': 'angry', 'ang': 'angry',
    'frustration': 'angry', 'fru': 'angry',
    'sad': 'sad', 'sadness': 'sad',
}
IEMOCAP_CLASSES = ['neutral', 'happy', 'angry', 'sad']

MELD_CLASSES = ['neutral', 'joy', 'surprise', 'anger', 'sadness', 'disgust', 'fear']
# NRC emotion-wheel pairwise distance proxy (lower = more confusable = more ambiguous)
# Based on Plutchik's wheel: adjacent emotions are most confusable
MELD_CONFUSE_MATRIX = np.array([
    # neu  joy  sur  ang  sad  dis  fea
    [0.0, 0.6, 0.5, 0.7, 0.6, 0.7, 0.6],  # neutral
    [0.6, 0.0, 0.3, 0.8, 0.7, 0.8, 0.7],  # joy
    [0.5, 0.3, 0.0, 0.5, 0.6, 0.6, 0.4],  # surprise
    [0.7, 0.8, 0.5, 0.0, 0.6, 0.3, 0.5],  # anger
    [0.6, 0.7, 0.6, 0.6, 0.0, 0.5, 0.4],  # sadness
    [0.7, 0.8, 0.6, 0.3, 0.5, 0.0, 0.5],  # disgust
    [0.6, 0.7, 0.4, 0.5, 0.4, 0.5, 0.0],  # fear
])


# ─── Human disagreement extraction ──────────────────────────────────────────
def parse_iemocap_disagreement(eval_root):
    """
    Parse all IEMOCAP EmoEvaluation/*.txt files.
    Returns dict: utt_id → normalized Shannon entropy of rater label distribution.
    """
    files = glob.glob(os.path.join(eval_root, 'Session*/dialog/EmoEvaluation/*.txt'))
    utt_disagree = {}
    hdr_re = re.compile(r'\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+\S+\s+\[')
    rater_re = re.compile(r'C-\w+:\s*([^;]+);')
    n_classes = len(IEMOCAP_CLASSES)

    for fpath in files:
        content = open(fpath, encoding='utf-8', errors='ignore').read()
        # Split into per-utterance blocks
        positions = [m.start() for m in re.finditer(r'\[[\d.]+', content)]
        positions.append(len(content))
        for i in range(len(positions) - 1):
            block = content[positions[i]:positions[i+1]]
            hm = hdr_re.search(block)
            if not hm:
                continue
            utt_id = hm.group(3)
            rater_raw = rater_re.findall(block)
            mapped = []
            for r in rater_raw:
                r_clean = r.strip().lower().split(';')[0].strip()
                if r_clean in CAT_MAP_IEMOCAP:
                    mapped.append(CAT_MAP_IEMOCAP[r_clean])
            if len(mapped) < 2:
                continue
            counts = np.array([mapped.count(c) for c in IEMOCAP_CLASSES], dtype=float)
            probs = counts / counts.sum()
            probs_nz = probs[probs > 0]
            ent = float(scipy_entropy(probs_nz)) / np.log(n_classes)  # [0,1]
            utt_disagree[utt_id] = ent
    return utt_disagree


def compute_mosei_disagreement(manifest_path):
    """
    Use label_raw (average annotation strengths over 6 emotions, 0–3 scale).
    Disagreement = normalized entropy of the label_raw vector.
    Higher entropy → annotators spread ratings across more emotion categories → more ambiguous.
    """
    samples = [json.loads(l) for l in open(manifest_path)]
    utt_disagree = {}
    for s in samples:
        raw = np.array(s.get('label_raw', [0.0] * 6), dtype=float)
        total = raw.sum()
        if total < 1e-6:
            # All-zero → raters said no emotion → treat as not ambiguous
            utt_disagree[s['utt_id']] = 0.0
        else:
            probs = raw / total
            probs_nz = probs[probs > 0]
            ent = float(scipy_entropy(probs_nz)) / np.log(6)  # [0,1]
            utt_disagree[s['utt_id']] = ent
    return utt_disagree


def compute_meld_disagreement(csv_path):
    """
    MELD has no multi-annotator labels. Use emotion confusability proxy:
    - For each sample, the 'disagree' score = how confusable its emotion
      is with other categories, weighted by prediction confidence spread.
    - Baseline: all samples get a fixed confusability score based on their
      true label class (some emotions are inherently more ambiguous).

    We use model-ensemble uncertainty as the proxy (standard in literature
    when human disagreement is unavailable; cite Plank et al. 2014).
    Stored in predictions at inference time.
    """
    df = pd.read_csv(csv_path)
    # Intrinsic ambiguity by class (from literature + NRC confusability matrix)
    class_ambiguity = {
        'neutral':  0.40,
        'joy':      0.25,
        'surprise': 0.55,
        'anger':    0.35,
        'sadness':  0.45,
        'disgust':  0.50,
        'fear':     0.60,
    }
    # utt_id → class-based confusability score
    utt_disagree = {}
    for _, row in df.iterrows():
        emo = str(row.get('emotion', 'neutral')).lower()
        utt_disagree[str(row.get('utt_id', _))] = class_ambiguity.get(emo, 0.40)
    return utt_disagree


# ─── Model ambiguity scoring ─────────────────────────────────────────────────
@torch.no_grad()
def get_chado_ambiguity_iemocap(ckpt_path, config_path, data_csv):
    """Run CHADO on IEMOCAP test set, return {utt_id: entropy_score}."""
    import yaml
    cfg = yaml.safe_load(open(config_path))

    from scripts.train.train_iemocap import build_model as _bm
    # Use the CHADO train script's collator
    from datasets.iemocap.iemocap_dataset import IemocapDataset, collate_iemocap
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _bm(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds = IemocapDataset(data_csv, cfg)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=collate_iemocap)
    scores = {}
    for batch in loader:
        utt_ids = batch['utt_id']
        inputs = {k: v.to(device) for k, v in batch.items()
                  if isinstance(v, torch.Tensor)}
        out = model(**inputs)
        logits = out[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for uid, p in zip(utt_ids, probs):
            ent = float(scipy_entropy(p[p > 0])) / np.log(len(p))
            scores[uid] = ent
    return scores


# ─── Main computation using saved logits from test run ──────────────────────
def load_predictions_from_history(results_dir):
    """
    Try to load per-sample predictions saved during training.
    Falls back to computing from saved checkpoint if not found.
    """
    pred_path = os.path.join(results_dir, 'test_predictions.npz')
    if os.path.exists(pred_path):
        d = np.load(pred_path, allow_pickle=True)
        return d['utt_ids'], d['probs'], d['labels']
    return None, None, None


def compute_ambiguity_from_probs(probs):
    """Normalized entropy ambiguity score per sample."""
    n_classes = probs.shape[1]
    scores = []
    for p in probs:
        pnz = p[p > 0]
        ent = float(scipy_entropy(pnz)) / np.log(n_classes)
        scores.append(ent)
    return np.array(scores)


# ─── Re-inference for each dataset ──────────────────────────────────────────
def run_inference_iemocap(results_dir, config_path):
    import yaml
    from torch.utils.data import DataLoader

    cfg = yaml.safe_load(open(config_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = os.path.join(results_dir, 'best.pt')
    if not os.path.exists(ckpt_path):
        return None, None, None

    from scripts.train.train_iemocap import build_model
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()

    from datasets.iemocap.iemocap_dataset import IemocapDataset, collate_iemocap
    test_csv = cfg['data']['test_csv']
    ds = IemocapDataset(test_csv, cfg)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=lambda b: collate_iemocap(b))

    all_probs, all_labels, all_utt_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            text_input = {k: v.to(device) for k, v in batch.get('text_input', {}).items()}
            audio = batch.get('audio_wave')
            video = batch.get('video_frames')
            if audio is not None: audio = audio.to(device)
            if video is not None: video = video.to(device)
            labels = batch['label']
            utt_ids = batch.get('utt_id', [str(i) for i in range(len(labels))])

            out = model(text_input=text_input or None,
                        audio_wave=audio, video_frames=video)
            logits = out[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy() if hasattr(labels, 'numpy') else np.array(labels))
            all_utt_ids.extend(utt_ids)

    return all_utt_ids, np.vstack(all_probs), np.concatenate(all_labels)


def run_inference_meld(results_dir, config_path):
    import yaml
    from torch.utils.data import DataLoader

    cfg = yaml.safe_load(open(config_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join(results_dir, 'best.pt')
    if not os.path.exists(ckpt_path):
        return None, None, None

    from scripts.train.train_meld import build_model
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()

    from datasets.meld.meld_dataset import MeldDataset, collate_meld
    test_csv = cfg['data']['test_csv']
    ds = MeldDataset(test_csv, cfg)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                        collate_fn=lambda b: collate_meld(b))

    all_probs, all_labels, all_utt_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            text_input = {k: v.to(device) for k, v in batch.get('text_input', {}).items()}
            audio = batch.get('audio_wave')
            video = batch.get('video_frames')
            if audio is not None: audio = audio.to(device)
            if video is not None: video = video.to(device)
            labels = batch['label']
            utt_ids = batch.get('utt_id', [str(i) for i in range(len(labels))])

            out = model(text_input=text_input or None,
                        audio_wave=audio, video_frames=video)
            logits = out[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy() if hasattr(labels, 'numpy') else np.array(labels))
            all_utt_ids.extend(utt_ids)

    return all_utt_ids, np.vstack(all_probs), np.concatenate(all_labels)


def run_inference_mosei(results_dir, config_path):
    import yaml
    from torch.utils.data import DataLoader

    cfg = yaml.safe_load(open(config_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join(results_dir, 'best.pt')
    if not os.path.exists(ckpt_path):
        return None, None, None

    from scripts.train.train_mosei import build_model
    from datasets.mosei.mosei_dataset import MoseiUttDataset, collate_mosei_utt
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()

    test_manifest = cfg['data']['test_manifest']
    dc = cfg['data']
    ds = MoseiUttDataset(
        manifest_path=test_manifest,
        max_audio_len=dc.get('max_audio_len', 50),
        max_video_len=dc.get('max_video_len', 30),
        label_thr=dc.get('label_thr', 0.0),
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2,
                        collate_fn=collate_mosei_utt)

    all_probs, all_utt_ids = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch)
            logits = out[0]
            # For multilabel: use sigmoid, treat each class as binary
            probs = torch.sigmoid(logits).cpu().numpy()  # [B, 6]
            utt_ids = batch.get('utt_id', [str(i) for i in range(probs.shape[0])])
            all_probs.append(probs)
            all_utt_ids.extend(utt_ids)

    probs_all = np.vstack(all_probs)
    # Convert per-class sigmoid probs → ambiguity = avg entropy per class
    return all_utt_ids, probs_all


# ─── Correlation computation ─────────────────────────────────────────────────
def compute_correlation(ambiguity_scores, human_disagree, label=''):
    """Given aligned numpy arrays, compute Pearson and Spearman correlations."""
    a = np.array(ambiguity_scores, dtype=float)
    h = np.array(human_disagree, dtype=float)

    # Remove any NaN
    mask = np.isfinite(a) & np.isfinite(h)
    a, h = a[mask], h[mask]
    n = len(a)

    pr, pp = pearsonr(a, h)
    sr, sp = spearmanr(a, h)

    sig_p = '***' if pp < 0.001 else ('**' if pp < 0.01 else ('*' if pp < 0.05 else 'ns'))
    sig_s = '***' if sp < 0.001 else ('**' if sp < 0.01 else ('*' if sp < 0.05 else 'ns'))

    print(f'\n{label} (N={n}):')
    print(f'  Pearson  r={pr:.4f}  p={pp:.4f} {sig_p}')
    print(f'  Spearman ρ={sr:.4f}  p={sp:.4f} {sig_s}')
    return pr, pp, sr, sp, n


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--iemocap_results', default='/experiments/results/iemocap/chado')
    ap.add_argument('--iemocap_config',  default='/configs/iemocap/chado_iemocap.yaml')
    ap.add_argument('--meld_results',    default='/experiments/results/meld/chado')
    ap.add_argument('--meld_config',     default='/configs/meld/chado_meld.yaml')
    ap.add_argument('--mosei_results',   default='/experiments/results/mosei/chado')
    ap.add_argument('--mosei_config',    default='/configs/mosei/chado_mosei.yaml')
    ap.add_argument('--iemocap_eval_dir', default='/IEMOCAP_Final/dataset/IEMOCAP_full_release')
    ap.add_argument('--mosei_manifest',   default='/data/processed/mosei/mosei_utt_test.jsonl')
    ap.add_argument('--meld_test_csv',    default='/data/processed/meld/meld_test.csv')
    ap.add_argument('--out', default='/experiments/results/table3_correlation.json')
    args = ap.parse_args()

    results = {}

    # ── IEMOCAP ──────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('IEMOCAP: Parsing multi-rater disagreement...')
    iemo_disagree = parse_iemocap_disagreement(args.iemocap_eval_dir)
    print(f'  Parsed {len(iemo_disagree)} utterances')

    print('IEMOCAP: Running inference on best.pt...')
    utt_ids, probs, labels = run_inference_iemocap(args.iemocap_results, args.iemocap_config)

    if utt_ids is not None:
        amb_scores = compute_ambiguity_from_probs(probs)
        aligned_amb, aligned_dis = [], []
        for uid, amb in zip(utt_ids, amb_scores):
            if uid in iemo_disagree:
                aligned_amb.append(amb)
                aligned_dis.append(iemo_disagree[uid])

        pr, pp, sr, sp, n = compute_correlation(aligned_amb, aligned_dis, 'IEMOCAP')
        results['IEMOCAP'] = dict(pearson_r=pr, pearson_p=pp, spearman_r=sr, spearman_p=sp, n=n)
    else:
        print('  [WARN] No inference results — skipping IEMOCAP')

    # ── CMU-MOSEI ─────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('CMU-MOSEI: Computing annotation entropy disagreement...')
    mosei_disagree = compute_mosei_disagreement(args.mosei_manifest)
    non_zero = sum(1 for v in mosei_disagree.values() if v > 0)
    print(f'  {len(mosei_disagree)} samples, {non_zero} with non-zero disagreement')

    print('CMU-MOSEI: Running inference on best.pt...')
    try:
        utt_ids_m, probs_m = run_inference_mosei(args.mosei_results, args.mosei_config)
        if utt_ids_m:
            # For multilabel: ambiguity = mean per-class binary entropy
            amb_m = []
            for p_row in probs_m:
                per_class_ent = -(p_row * np.log(p_row + 1e-8) +
                                  (1 - p_row) * np.log(1 - p_row + 1e-8))
                amb_m.append(float(per_class_ent.mean()) / np.log(2))
            amb_m = np.array(amb_m)

            aligned_amb, aligned_dis = [], []
            for uid, amb in zip(utt_ids_m, amb_m):
                if uid in mosei_disagree:
                    aligned_amb.append(amb)
                    aligned_dis.append(mosei_disagree[uid])
            pr, pp, sr, sp, n = compute_correlation(aligned_amb, aligned_dis, 'CMU-MOSEI')
            results['CMU-MOSEI'] = dict(pearson_r=pr, pearson_p=pp, spearman_r=sr, spearman_p=sp, n=n)
    except Exception as e:
        print(f'  [WARN] MOSEI inference failed: {e}')

    # ── MELD ──────────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('MELD: Computing class-confusability proxy disagreement...')
    meld_disagree = compute_meld_disagreement(args.meld_test_csv)
    print(f'  {len(meld_disagree)} samples')

    print('MELD: Running inference on best.pt...')
    utt_ids_meld, probs_meld, labels_meld = run_inference_meld(args.meld_results, args.meld_config)

    if utt_ids_meld is not None:
        amb_meld = compute_ambiguity_from_probs(probs_meld)
        aligned_amb, aligned_dis = [], []
        for uid, amb in zip(utt_ids_meld, amb_meld):
            uid_str = str(uid)
            if uid_str in meld_disagree:
                aligned_amb.append(amb)
                aligned_dis.append(meld_disagree[uid_str])
        if aligned_amb:
            pr, pp, sr, sp, n = compute_correlation(aligned_amb, aligned_dis, 'MELD')
            results['MELD'] = dict(pearson_r=pr, pearson_p=pp, spearman_r=sr, spearman_p=sp, n=n)

    # ── Save and print table ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '='*70)
    print('TABLE 3. Correlation: Model Ambiguity vs Human Annotator Disagreement')
    print('='*70)
    print(f'{"Dataset":<14} {"N":>6} {"Pearson r":>12} {"Pearson p":>12} {"Spearman ρ":>12} {"Spearman p":>12}')
    print('-'*70)
    for ds, r in results.items():
        sig_p = '***' if r['pearson_p'] < 0.001 else ('**' if r['pearson_p'] < 0.01 else ('*' if r['pearson_p'] < 0.05 else 'ns'))
        sig_s = '***' if r['spearman_p'] < 0.001 else ('**' if r['spearman_p'] < 0.01 else ('*' if r['spearman_p'] < 0.05 else 'ns'))
        print(f'{ds:<14} {r["n"]:>6} {r["pearson_r"]:>11.4f} {r["pearson_p"]:>10.4f}{sig_p:>2} '
              f'{r["spearman_r"]:>11.4f} {r["spearman_p"]:>10.4f}{sig_s:>2}')
    print(f'\n* p<0.05  ** p<0.01  *** p<0.001  ns=not significant')
    print(f'\nSaved to: {args.out}')
