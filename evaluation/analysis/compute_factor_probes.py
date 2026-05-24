

import os, sys, re, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '/')
BASE = ''

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_ckpt(path):
    ck = torch.load(path, map_location='cpu')
    if isinstance(ck, dict):
        for k in ('model', 'model_state_dict', 'state_dict'):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
    return ck


def speaker_from_uid(uid):
    """Ses01M → 0, Ses01F → 1, ..., Ses05F → 9"""
    m = re.match(r'Ses0?(\d)([MF])', str(uid))
    if m:
        s = int(m.group(1)) - 1
        g = 0 if m.group(2) == 'M' else 1
        return s * 2 + g
    return -1


def dialogue_from_uid(uid):
    """Extract dialogue name: Ses01M_impro01 → Ses01M_impro01"""
    m = re.match(r'(Ses\d+[MF]_\w+?)_[MF]\d+', str(uid))
    return m.group(1) if m else str(uid)


def turn_index_from_uid(uid):
    """Extract turn number from utt_id: Ses01M_impro01_M005 → 5"""
    m = re.search(r'_[MF](\d+)$', str(uid))
    return int(m.group(1)) if m else 0


def compute_turn_norm(uids):
    """Normalize turn index within each dialogue to [0,1]."""
    dialogues = [dialogue_from_uid(u) for u in uids]
    turns     = [turn_index_from_uid(u) for u in uids]
    df = pd.DataFrame({'dlg': dialogues, 'turn': turns})
    df['max_turn'] = df.groupby('dlg')['turn'].transform('max').clip(lower=1)
    return (df['turn'] / df['max_turn']).values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Factor extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_factors(device, cfg_path, ckpt_path, csv_paths):
    """
    Run the CHADO model on the dataset and collect per-sample:
      factors: (z_c, z_u, z_t, z_m, z_e) — all as numpy arrays
      modal_sim: cosine sim between e_T and e_A from CTNet backbone
      utt_ids, labels
    Returns a dict of all collected arrays.
    """
    import yaml
    from models.chado.model import CHADOTrimodal
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap

    cfg = yaml.safe_load(open(cfg_path))
    dc, mc = cfg['data'], cfg['model']

    sd = load_ckpt(ckpt_path)
    # detect text model size
    text_model = 'roberta-large'
    for k, v in sd.items():
        if 'text_enc' in k and 'embeddings.LayerNorm.weight' in k:
            text_model = 'roberta-large' if v.shape[0] == 1024 else 'roberta-base'
            break

    model = CHADOTrimodal(
        text_model_name   = text_model,
        audio_model_name  = mc['audio_model_name'],
        video_model_name  = mc['video_model_name'],
        num_classes       = dc['num_classes'],
        proj_dim          = mc.get('proj_dim', 256),
        dropout           = 0.0,
        use_text=True, use_audio=True, use_video=True,
        use_gated_fusion  = mc.get('use_gated_fusion', True),
        backbone          = 'ctnet',
        n_heads           = mc.get('n_heads', 4),
        n_speakers        = cfg.get('chado', {}).get('n_speakers', 10),
        use_causal=True, use_hyperbolic=False, use_ot=False, use_mad=False,
    ).to(device).eval()
    model.load_state_dict(sd, strict=False)

    # ── Hooks ────────────────────────────────────────────────────────────────
    factor_buf = {}
    modal_sim_buf = []

    def _factor_hook(module, inp, out):
        zc, zu, zt, zm, ze = out
        factor_buf.setdefault('zc', []).append(zc.detach().cpu())
        factor_buf.setdefault('zu', []).append(zu.detach().cpu())
        factor_buf.setdefault('zt', []).append(zt.detach().cpu())
        factor_buf.setdefault('zm', []).append(zm.detach().cpu())
        factor_buf.setdefault('ze', []).append(ze.detach().cpu())

    model.causal.register_forward_hook(_factor_hook)

    # Hook on CTNet backbone to capture encoder outputs for modal_sim
    def _enc_hook(module, inp, out):
        # out = (backbone_logits, raw_z, gate)
        # raw_z from CTNet is [B, 3*256]; split first 256 = text, next = audio
        raw_z = out[1]
        if raw_z.shape[1] >= 512:
            et = F.normalize(raw_z[:, :256], dim=1)
            ea = F.normalize(raw_z[:, 256:512], dim=1)
            sim = (et * ea).sum(dim=1)
            modal_sim_buf.append(sim.detach().cpu())

    model.base.register_forward_hook(_enc_hook)

    # ── Data ─────────────────────────────────────────────────────────────────
    all_uids, all_labels = [], []
    n_frames = dc.get('num_frames', 8)
    fsize    = dc.get('frame_size', 224)

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f'  [skip] {csv_path}')
            continue
        df = pd.read_csv(csv_path)
        ds = IEMOCAPDataset(
            csv_path       = csv_path,
            text_model_name= text_model,
            max_text_len   = dc.get('max_text_len', 96),
            audio_sr       = dc.get('sample_rate', 16000),
            audio_sec      = dc.get('max_audio_seconds', 4.0),
            n_frames       = n_frames,
            use_audio      = True,
            use_video      = False,   # skip video for speed
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False,
                            num_workers=2, collate_fn=collate_iemocap)

        LABEL2ID = {'neu': 0, 'hap': 1, 'ang': 2, 'sad': 3}
        uid_list = df['utt_id'].tolist()
        lbl_list = [LABEL2ID.get(str(l).strip().lower(), 0)
                    for l in df['label_4'].tolist()]

        with torch.no_grad():
            uid_idx = 0
            for batch in loader:
                B = batch['input_ids'].shape[0]
                ti = {
                    'input_ids'     : batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                }
                aw = batch.get('wav', batch.get('audio_wave'))
                aw = aw.to(device) if aw is not None else None
                vf = torch.zeros(B, n_frames, 3, fsize, fsize, device=device)
                model(text_input=ti, audio_wave=aw, video_frames=vf)
                all_uids.extend(uid_list[uid_idx: uid_idx + B])
                all_labels.extend(lbl_list[uid_idx: uid_idx + B])
                uid_idx += B

    # ── Stack ─────────────────────────────────────────────────────────────────
    result = {
        'utt_ids'   : all_uids,
        'labels'    : np.array(all_labels),
        'zc'        : torch.cat(factor_buf['zc'], 0).numpy(),
        'zu'        : torch.cat(factor_buf['zu'], 0).numpy(),
        'zt'        : torch.cat(factor_buf['zt'], 0).numpy(),
        'zm'        : torch.cat(factor_buf['zm'], 0).numpy(),
        'ze'        : torch.cat(factor_buf['ze'], 0).numpy(),
    }
    if modal_sim_buf:
        result['modal_sim'] = torch.cat(modal_sim_buf, 0).numpy()
    result['speaker_id'] = np.array([speaker_from_uid(u) for u in all_uids])
    result['turn_norm']  = compute_turn_norm(all_uids)

    print(f'  Extracted N={len(all_uids)} samples')
    print(f'  Factor shapes: zc={result["zc"].shape}, zu={result["zu"].shape}, '
          f'zt={result["zt"].shape}, zm={result["zm"].shape}, ze={result["ze"].shape}')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe training
# ─────────────────────────────────────────────────────────────────────────────

def probe_classification(z, labels, label_name='', n_splits=5):
    """Train logistic regression probe, return mean accuracy (%)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    scaler = StandardScaler()
    z_s = scaler.fit_transform(z)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for tr, te in cv.split(z_s, labels):
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                                  solver='lbfgs')
        clf.fit(z_s[tr], labels[tr])
        accs.append(accuracy_score(labels[te], clf.predict(z_s[te])))
    val = float(np.mean(accs)) * 100
    print(f'    [{label_name}] Probe acc = {val:.1f}%')
    return val


def probe_regression_spearman(z, targets, label_name='', n_splits=5):
    """Train ridge regression probe, return mean Spearman ρ."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from scipy.stats import spearmanr

    scaler = StandardScaler()
    z_s = scaler.fit_transform(z)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rhos = []
    for tr, te in cv.split(z_s):
        reg = Ridge(alpha=1.0)
        reg.fit(z_s[tr], targets[tr])
        pred = reg.predict(z_s[te])
        rho, _ = spearmanr(pred, targets[te])
        rhos.append(float(rho))
    val = float(np.mean(rhos))
    print(f'    [{label_name}] Spearman ρ = {val:.3f}')
    return val


def probe_regression_r2(z, targets, label_name='', n_splits=5):
    """Train ridge regression probe, return mean R²."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    scaler = StandardScaler()
    z_s = scaler.fit_transform(z)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2s = []
    for tr, te in cv.split(z_s):
        reg = Ridge(alpha=1.0)
        reg.fit(z_s[tr], targets[tr])
        pred = reg.predict(z_s[te])
        r2s.append(float(r2_score(targets[te], pred)))
    val = float(np.mean(r2s))
    print(f'    [{label_name}] R² = {val:.3f}')
    return val


def context_alignment_score(zc, utt_ids, K=5):
    """
    Mean cosine similarity of each z_c to the mean z_c of its K previous turns.
    Higher = z_c better tracks conversational context.
    """
    dlgs = [dialogue_from_uid(u) for u in utt_ids]
    turns = [turn_index_from_uid(u) for u in utt_ids]
    uid2idx = {u: i for i, u in enumerate(utt_ids)}

    scores = []
    for i, uid in enumerate(utt_ids):
        dlg = dlgs[i]
        t   = turns[i]
        # find previous K turns in same dialogue
        prev_idxs = [
            j for j, (d, tn) in enumerate(zip(dlgs, turns))
            if d == dlg and 0 < (t - tn) <= K
        ]
        if not prev_idxs:
            continue
        ctx = zc[prev_idxs].mean(axis=0, keepdims=True)  # [1, d_c]
        zi  = zc[[i]]                                      # [1, d_c]
        # cosine similarity
        cos = (zi * ctx).sum() / (np.linalg.norm(zi) * np.linalg.norm(ctx) + 1e-8)
        scores.append(float(cos))

    val = float(np.mean(scores)) if scores else 0.0
    print(f'    [z_c context align] mean cos_sim = {val:.3f}')
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Read ablation results
# ─────────────────────────────────────────────────────────────────────────────

def read_test_result(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_ablation_results(dataset):
    """Read existing ablation test_results.json for all ablation variants."""
    base = f'{BASE}/experiments/results/{dataset}'
    variants = {
        'Full CHADO'      : f'{base}/chado/test_results.json',
        'w/o Causal'      : f'{base}/chado_wo_causal/test_results.json',
        'w/o Hyperbolic'  : f'{base}/chado_wo_hyperbolic/test_results.json',
        'w/o OT'          : f'{base}/chado_wo_ot/test_results.json',
        'w/o MAD'         : f'{base}/chado_wo_mad/test_results.json',
    }
    results = {}
    for name, path in variants.items():
        r = read_test_result(path)
        if r:
            results[name] = {
                'acc'   : round(r['accuracy'] * 100, 2),
                'mf1'   : round(r['macro_f1'] * 100, 2),
                'per_class': r.get('per_class', {}),
            }
    return results


def identify_failure_mode(full_result, ablation_result, dataset):
    """
    Compare per-class F1 drops to identify the dominant failure mode
    when a factor is removed.
    Returns the emotion class with the largest F1 drop.
    """
    if not full_result or not ablation_result:
        return 'N/A'
    full_pc = full_result.get('per_class', {})
    abl_pc  = ablation_result.get('per_class', {})
    drops = {}
    for cls in full_pc:
        if cls in abl_pc:
            drops[cls] = full_pc[cls]['f1'] - abl_pc[cls]['f1']
    if not drops:
        return 'N/A'
    worst = max(drops, key=drops.get)
    drop_val = drops[worst]
    return f'{worst} (Δ={drop_val*100:+.1f}pp)'


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table generation
# ─────────────────────────────────────────────────────────────────────────────

def write_factor_probe_table(probe_results, out_path):
    rows = [
        ('$z_u$', 'Speaker ID',          'Accuracy (\\%)',   probe_results['zu_spk'],   'Higher = better speaker disentanglement'),
        ('$z_t$', 'Temporal Order',      'Spearman $\\rho$', probe_results['zt_spear'],  'Higher = better turn-order encoding'),
        ('$z_c$', 'Dialogue Context',    'Cosine Sim.',      probe_results['zc_ctx'],    'Higher = better context tracking'),
        ('$z_m$', 'Modality Consistency','$R^2$',            probe_results['zm_r2'],     'Higher = better cross-modal agreement'),
        ('$z_e$', 'Emotion (residual)',  'Accuracy (\\%)',   probe_results['ze_emo'],    'Lower = residual does not encode emotion'),
    ]

    lines = [
        '% TABLE: Factor Probe Results',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Linear probe accuracy for each CHADO factor on its designated',
        'supervision task (IEMOCAP test set). Higher is better for $z_u$--$z_m$;',
        'lower is better for $z_e$ (residual should not encode task-relevant structure).}',
        '\\label{tab:factor_probes}',
        '\\setlength{\\tabcolsep}{5pt}',
        '\\begin{tabular}{llcc}',
        '\\toprule',
        '\\textbf{Factor} & \\textbf{Probe Task} & \\textbf{Metric} & \\textbf{Score} \\\\',
        '\\midrule',
    ]
    for factor, task, metric, score, note in rows:
        if isinstance(score, float):
            score_str = f'{score:.2f}' if metric != 'Accuracy (\\%)' else f'{score:.1f}'
        else:
            score_str = str(score)
        lines.append(f'{factor} & {task} & {metric} & {score_str} \\\\ % {note}')
    lines += [
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n  Saved factor probe table → {out_path}')


def write_ablation_failure_table(ie_results, meld_results, out_path):
    """
    Table with columns: Removed Component | IEMOCAP Acc | IEMOCAP Macro-F1 |
                        MELD Acc | MELD Macro-F1 | Main Failure Mode
    """
    ie_full   = ie_results.get('Full CHADO', {})
    meld_full = meld_results.get('Full CHADO', {})

    rows = []
    for variant in ['Full CHADO', 'w/o Causal', 'w/o Hyperbolic', 'w/o OT', 'w/o MAD']:
        ie_r   = ie_results.get(variant, {})
        meld_r = meld_results.get(variant, {})

        # failure mode from largest per-class F1 drop vs full CHADO
        if variant == 'Full CHADO':
            failure = '---'
        else:
            ie_fail   = identify_failure_mode(ie_full,   ie_r,   'iemocap')
            meld_fail = identify_failure_mode(meld_full, meld_r, 'meld')
            failure = f'IE: {ie_fail} / MELD: {meld_fail}'

        ie_acc  = f'{ie_r.get("acc",  "---")}'
        ie_mf1  = f'{ie_r.get("mf1",  "---")}'
        meld_acc = f'{meld_r.get("acc",  "---")}'
        meld_mf1 = f'{meld_r.get("mf1",  "---")}'

        rows.append((variant, ie_acc, ie_mf1, meld_acc, meld_mf1, failure))

    lines = [
        '% TABLE: Ablation with Failure Modes',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Ablation study with per-component failure analysis.',
        'Numbers are Accuracy / Macro-F1 (\\%).',
        'Failure mode shows the emotion class with the largest F1 drop',
        'when that component is removed.}',
        '\\label{tab:ablation_failure}',
        '\\setlength{\\tabcolsep}{4pt}',
        '\\begin{tabular}{l|cc|cc|l}',
        '\\toprule',
        ' & \\multicolumn{2}{c|}{\\textbf{IEMOCAP}} & \\multicolumn{2}{c|}{\\textbf{MELD}} & \\\\',
        '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
        '\\textbf{Variant} & Acc & Mac-F1 & Acc & Mac-F1 & \\textbf{Main Failure} \\\\',
        '\\midrule',
    ]

    for variant, ie_acc, ie_mf1, meld_acc, meld_mf1, failure in rows:
        bold = '\\textbf' if variant == 'Full CHADO' else ''
        if bold:
            lines.append(f'\\textbf{{{variant}}} & \\textbf{{{ie_acc}}} & \\textbf{{{ie_mf1}}} & {meld_acc} & {meld_mf1} & {failure} \\\\')
        else:
            lines.append(f'{variant} & {ie_acc} & {ie_mf1} & {meld_acc} & {meld_mf1} & {failure} \\\\')

    lines += [
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved ablation failure table → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    cfg_path  = f'{BASE}/configs/iemocap/chado_iemocap.yaml'
    ckpt_path = f'{BASE}/experiments/results/iemocap/chado/best.pt'

    import yaml
    cfg = yaml.safe_load(open(cfg_path))
    dc  = cfg['data']

    csv_paths = [dc['val_csv'], dc['test_csv']]

    # ── Step 1: Extract factors ───────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 1: Extracting 5 factors from CHADO (IEMOCAP)')
    print('='*60)
    data = extract_factors(device, cfg_path, ckpt_path, csv_paths)

    N = len(data['utt_ids'])
    print(f'\n  Total samples: N={N}')

    # ── Step 2: Run linear probes ─────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 2: Linear probe experiments')
    print('='*60)

    probe_results = {}

    # z_u → speaker ID (10 classes)
    print('\n  z_u → Speaker ID classification')
    spk = data['speaker_id']
    valid = spk >= 0
    probe_results['zu_spk'] = probe_classification(
        data['zu'][valid], spk[valid], 'z_u → speaker')

    # z_t → turn-position regression (Spearman ρ)
    print('\n  z_t → Turn position (Spearman ρ)')
    tn = data['turn_norm']
    probe_results['zt_spear'] = probe_regression_spearman(
        data['zt'], tn, 'z_t → turn_norm')

    # z_c → context alignment (cosine sim to previous turns)
    print('\n  z_c → Dialogue context alignment')
    probe_results['zc_ctx'] = context_alignment_score(
        data['zc'], data['utt_ids'], K=5)

    # z_m → cross-modal cosine similarity (R²)
    print('\n  z_m → Modality consistency (R²)')
    if 'modal_sim' in data:
        probe_results['zm_r2'] = probe_regression_r2(
            data['zm'], data['modal_sim'], 'z_m → modal_sim')
    else:
        # fallback: use dummy uniform signal
        probe_results['zm_r2'] = 0.0
        print('    [warn] modal_sim not available; R²=0.00')

    # z_e → emotion label (should be LOW — residual should not encode emotion)
    print('\n  z_e → Emotion label (should be LOW)')
    probe_results['ze_emo'] = probe_classification(
        data['ze'], data['labels'], 'z_e → emotion')

    # Bonus: z_c+z_u+z_m joint → emotion (should be HIGH — confirms classifier subspace)
    print('\n  [z_c||z_u||z_m] → Emotion (classifier subspace, should be HIGH)')
    z_cls = np.concatenate([data['zc'], data['zu'], data['zm']], axis=1)
    probe_cls_acc = probe_classification(z_cls, data['labels'], 'cls_subspace → emotion')
    probe_results['cls_subspace'] = probe_cls_acc

    # ── Step 3: Ablation failure modes ───────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 3: Reading ablation results')
    print('='*60)
    ie_ablation   = get_ablation_results('iemocap')
    meld_ablation = get_ablation_results('meld')

    print('\n  IEMOCAP ablation:')
    for k, v in ie_ablation.items():
        print(f'    {k:<20} acc={v["acc"]} mf1={v["mf1"]}')
    print('\n  MELD ablation:')
    for k, v in meld_ablation.items():
        print(f'    {k:<20} acc={v["acc"]} mf1={v["mf1"]}')

    # ── Step 4: Print summary ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print('FACTOR PROBE SUMMARY')
    print('='*60)
    print(f'  z_u → Speaker ID accuracy  : {probe_results["zu_spk"]:.1f}%')
    print(f'  z_t → Temporal Spearman ρ  : {probe_results["zt_spear"]:.3f}')
    print(f'  z_c → Context cos-sim       : {probe_results["zc_ctx"]:.3f}')
    print(f'  z_m → Modal consistency R²  : {probe_results["zm_r2"]:.3f}')
    print(f'  z_e → Emotion acc (residual): {probe_results["ze_emo"]:.1f}%  ← should be LOW')
    print(f'  cls → Emotion acc (z_c+u+m) : {probe_results["cls_subspace"]:.1f}%  ← should be HIGH')

    # ── Step 5: Write tables ──────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 5: Writing LaTeX tables')
    print('='*60)
    os.makedirs(f'{BASE}/experiments/tables', exist_ok=True)

    write_factor_probe_table(
        probe_results,
        f'{BASE}/experiments/tables/factor_probes.tex'
    )
    write_ablation_failure_table(
        ie_ablation, meld_ablation,
        f'{BASE}/experiments/tables/ablation_failure.tex'
    )

    # Save raw numbers
    out_json = f'{BASE}/experiments/results/factor_probe_results.json'
    with open(out_json, 'w') as f:
        json.dump({
            'probe_results' : probe_results,
            'iemocap_ablation': {k: {'acc': v['acc'], 'mf1': v['mf1']}
                                  for k, v in ie_ablation.items()},
            'meld_ablation'  : {k: {'acc': v['acc'], 'mf1': v['mf1']}
                                  for k, v in meld_ablation.items()},
        }, f, indent=2)
    print(f'\n  Raw results saved → {out_json}')
    print('\nDone.')
