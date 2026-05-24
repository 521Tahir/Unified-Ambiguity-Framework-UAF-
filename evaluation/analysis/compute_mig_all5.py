

import os, sys, re, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '')
BASE = ''

# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def speaker_from_uid(uid):
    m = re.match(r'Ses0?(\d)([MF])', str(uid))
    if m:
        return (int(m.group(1)) - 1) * 2 + (0 if m.group(2) == 'M' else 1)
    return -1

def session_from_uid(uid):
    m = re.match(r'Ses0?(\d)', str(uid))
    return int(m.group(1)) - 1 if m else -1

def dialogue_from_uid(uid):
    m = re.match(r'(Ses\d+[MF]_\w+?)_[MF]\d+', str(uid))
    return m.group(1) if m else str(uid)

def turn_index_from_uid(uid):
    m = re.search(r'_[MF](\d+)$', str(uid))
    return int(m.group(1)) if m else 0

def compute_turn_norm(uids):
    dlgs  = [dialogue_from_uid(u) for u in uids]
    turns = [turn_index_from_uid(u) for u in uids]
    df = pd.DataFrame({'dlg': dlgs, 'turn': turns})
    df['max_turn'] = df.groupby('dlg')['turn'].transform('max').clip(lower=1)
    return (df['turn'] / df['max_turn']).values.astype(np.float32)

def dialogue_id_from_uid(uids):
    """Map each uid to an integer dialogue ID."""
    dlgs = [dialogue_from_uid(u) for u in uids]
    uniq = {d: i for i, d in enumerate(sorted(set(dlgs)))}
    return np.array([uniq[d] for d in dlgs])

def load_ckpt(path):
    ck = torch.load(path, map_location='cpu')
    if isinstance(ck, dict):
        for k in ('model', 'model_state_dict', 'state_dict'):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
    return ck


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_factors(device, cfg_path, ckpt_path, csv_paths):
    import yaml
    from models.chado.model import CHADOTrimodal
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap

    cfg = yaml.safe_load(open(cfg_path))
    dc, mc = cfg['data'], cfg['model']
    sd = load_ckpt(ckpt_path)

    text_model = 'roberta-base'
    for k, v in sd.items():
        if 'text_enc' in k and 'embeddings.LayerNorm.weight' in k:
            text_model = 'roberta-large' if v.shape[0] == 1024 else 'roberta-base'
            break

    model = CHADOTrimodal(
        text_model_name  = text_model,
        audio_model_name = mc['audio_model_name'],
        video_model_name = mc['video_model_name'],
        num_classes      = dc['num_classes'],
        proj_dim         = mc.get('proj_dim', 256),
        dropout          = 0.0,
        use_text=True, use_audio=True, use_video=True,
        use_gated_fusion = mc.get('use_gated_fusion', True),
        backbone='ctnet', n_heads=mc.get('n_heads', 4),
        n_speakers       = cfg.get('chado', {}).get('n_speakers', 10),
        use_causal=True, use_hyperbolic=False, use_ot=False, use_mad=False,
    ).to(device).eval()
    model.load_state_dict(sd, strict=False)

    factor_buf    = {}
    modal_sim_buf = []

    def _factor_hook(module, inp, out):
        zc, zu, zt, zm, ze = out
        for key, z in zip(('zc','zu','zt','zm','ze'), out):
            factor_buf.setdefault(key, []).append(z.detach().cpu())

    model.causal.register_forward_hook(_factor_hook)

    def _enc_hook(module, inp, out):
        raw_z = out[1]
        if raw_z.shape[1] >= 512:
            et = F.normalize(raw_z[:, :256],    dim=1)
            ea = F.normalize(raw_z[:, 256:512], dim=1)
            modal_sim_buf.append((et * ea).sum(dim=1).detach().cpu())

    model.base.register_forward_hook(_enc_hook)

    LABEL2ID = {'neu': 0, 'hap': 1, 'ang': 2, 'sad': 3}
    all_uids, all_labels = [], []
    n_frames = dc.get('num_frames', 8)
    fsize    = dc.get('frame_size', 224)

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        ds = IEMOCAPDataset(
            csv_path        = csv_path,
            text_model_name = text_model,
            max_text_len    = dc.get('max_text_len', 96),
            audio_sr        = dc.get('sample_rate', 16000),
            audio_sec       = dc.get('max_audio_seconds', 4.0),
            n_frames        = n_frames,
            use_audio       = True,
            use_video       = False,
        )
        loader   = DataLoader(ds, batch_size=16, shuffle=False,
                              num_workers=2, collate_fn=collate_iemocap)
        uid_list = df['utt_id'].tolist()
        lbl_list = [LABEL2ID.get(str(l).strip().lower(), 0) for l in df['label_4']]

        with torch.no_grad():
            uid_idx = 0
            for batch in loader:
                B = batch['input_ids'].shape[0]
                ti = {'input_ids'     : batch['input_ids'].to(device),
                      'attention_mask': batch['attention_mask'].to(device)}
                aw = batch.get('wav', batch.get('audio_wave'))
                aw = aw.to(device) if aw is not None else None
                vf = torch.zeros(B, n_frames, 3, fsize, fsize, device=device)
                model(text_input=ti, audio_wave=aw, video_frames=vf)
                all_uids.extend(uid_list[uid_idx: uid_idx + B])
                all_labels.extend(lbl_list[uid_idx: uid_idx + B])
                uid_idx += B

    factors = {k: torch.cat(v, 0).numpy() for k, v in factor_buf.items()}
    modal_sim = torch.cat(modal_sim_buf, 0).numpy() if modal_sim_buf else None

    print(f'  N={len(all_uids)}')
    for k, v in factors.items():
        print(f'  {k}: {v.shape}')

    return factors, all_uids, np.array(all_labels), modal_sim


# ─────────────────────────────────────────────────────────────────────────────
# MIG computation
# ─────────────────────────────────────────────────────────────────────────────

def factor_mi(z, v_discrete, n_neighbors=10):
    """
    MI between factor z (continuous, [N, d]) and discrete signal v ([N]).
    Uses sklearn mutual_info_classif: treats each dim of z as a feature
    and computes the total MI as sum over dims, then averages.
    Returns a single scalar MI estimate.
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import StandardScaler

    z_s = StandardScaler().fit_transform(z)
    mi  = mutual_info_classif(z_s, v_discrete,
                               discrete_features=False,
                               n_neighbors=n_neighbors,
                               random_state=42)
    return float(mi.mean())   # mean MI per dimension


def entropy(v):
    """Entropy of discrete variable v."""
    counts = np.bincount(v.astype(int),
                         minlength=int(v.max()) + 1).astype(float)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def compute_mig_table(factors, signals):
    """
    Args:
        factors : dict {name: np.array [N, d]}
        signals : dict {name: np.array [N] discrete labels}
    Returns:
        mi_table  : dict {signal_name: {factor_name: mi_value}}
        mig_scores: dict {signal_name: mig_value}
    """
    factor_names = list(factors.keys())
    signal_names = list(signals.keys())

    mi_table   = {}
    mig_scores = {}

    for sname, v in signals.items():
        H_v = entropy(v)
        if H_v < 1e-8:
            print(f'  [skip] {sname}: zero entropy')
            continue

        mi_row = {}
        for fname in factor_names:
            z = factors[fname]
            mi = factor_mi(z, v)
            mi_row[fname] = mi
            print(f'    MI({fname}, {sname}) = {mi:.4f}')

        mi_table[sname] = mi_row

        # MIG: top1 - top2 gap normalized by H(v)
        mi_vals = sorted(mi_row.values(), reverse=True)
        if len(mi_vals) >= 2:
            mig = (mi_vals[0] - mi_vals[1]) / H_v
        else:
            mig = 0.0
        mig_scores[sname] = float(np.clip(mig, 0.0, 1.0))
        best_factor = max(mi_row, key=mi_row.get)
        print(f'  [{sname}] H={H_v:.3f}  MIG={mig:.3f}  best_factor={best_factor}\n')

    return mi_table, mig_scores


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def write_mig_table(mi_table, mig_scores, out_path):
    signal_display = {
        'speaker'  : 'Speaker Identity ($z_u$)',
        'temporal' : 'Temporal Order ($z_t$)',
        'context'  : 'Dialogue Context ($z_c$)',
        'modal'    : 'Modality Consistency ($z_m$)',
        'emotion'  : 'Emotion Label ($z_e$)',
    }
    factor_display = {
        'zc': '$z_c$', 'zu': '$z_u$', 'zt': '$z_t$',
        'zm': '$z_m$', 'ze': '$z_e$',
    }
    factor_order  = ['zc', 'zu', 'zt', 'zm', 'ze']
    signal_order  = ['speaker', 'temporal', 'context', 'modal', 'emotion']

    lines = [
        '% TABLE: MIG cross-MI matrix — all 5 CHADO factors',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Mutual Information Gap (MIG) analysis for all five CHADO factors',
        'on IEMOCAP ($N{=}1107$, val+test).',
        'Each cell shows mean MI between the factor (column) and supervision signal (row).',
        'Bold = highest MI per row (designating which factor best captures that signal).',
        'MIG = normalized gap between top-1 and top-2 MI per signal.',
        'Higher MIG indicates cleaner disentanglement for that signal.}',
        '\\label{tab:mig_all5}',
        '\\setlength{\\tabcolsep}{5pt}',
        '\\begin{tabular}{l|ccccc|c}',
        '\\toprule',
        '\\textbf{Supervision Signal} & ' +
        ' & '.join(factor_display[f] for f in factor_order) +
        ' & \\textbf{MIG} \\\\',
        '\\midrule',
    ]

    for sname in signal_order:
        if sname not in mi_table:
            continue
        row    = mi_table[sname]
        mig    = mig_scores.get(sname, 0.0)
        label  = signal_display.get(sname, sname)
        best_f = max(row, key=row.get)

        cells = []
        for f in factor_order:
            val = row.get(f, 0.0)
            s   = f'{val:.3f}'
            if f == best_f:
                s = f'\\textbf{{{s}}}'
            cells.append(s)

        mig_s = f'{mig:.3f}'
        lines.append(f'{label} & {" & ".join(cells)} & {mig_s} \\\\')

    # Overall MIG
    all_mig = [v for v in mig_scores.values()]
    overall = float(np.mean(all_mig)) if all_mig else 0.0

    lines += [
        '\\midrule',
        f'\\textbf{{Overall MIG}} & & & & & & \\textbf{{{overall:.3f}}} \\\\',
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n  Saved MIG table → {out_path}')
    return overall


def write_summary_table(mig_scores, overall, out_path):
    """Compact 2-column summary table matching the existing paper format."""
    signal_display = {
        'speaker'  : 'Speaker Identity',
        'temporal' : 'Temporal Order',
        'context'  : 'Dialogue Context',
        'modal'    : 'Modality Consistency',
        'emotion'  : 'Emotion Label (residual)',
    }
    signal_order = ['speaker', 'temporal', 'context', 'modal', 'emotion']

    lines = [
        '% TABLE: MIG Summary — 5 factors, IEMOCAP',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{MIG scores for all five CHADO supervision signals on IEMOCAP.',
        'Higher MIG indicates the designated factor captures its supervision signal',
        'more exclusively than other factors.}',
        '\\label{tab:mig_summary}',
        '\\begin{tabular}{lc}',
        '\\toprule',
        '\\textbf{Factor / Signal} & \\textbf{MIG (IEMOCAP)} \\\\',
        '\\midrule',
    ]
    for s in signal_order:
        label = signal_display.get(s, s)
        val   = mig_scores.get(s, 0.0)
        lines.append(f'{label} & {val:.3f} \\\\')

    lines += [
        '\\midrule',
        f'\\textbf{{Overall}} & \\textbf{{{overall:.3f}}} \\\\',
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved MIG summary table → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg_path  = f'{BASE}/configs/iemocap/chado_iemocap.yaml'
    ckpt_path = f'{BASE}/experiments/results/iemocap/chado/best.pt'

    import yaml
    cfg  = yaml.safe_load(open(cfg_path))
    dc   = cfg['data']
    csvs = [dc['val_csv'], dc['test_csv']]

    # ── Step 1: Extract all 5 factors ────────────────────────────────────────
    print('\n' + '='*60)
    print('Extracting 5 factors from CHADO (IEMOCAP)')
    print('='*60)
    factors, uids, labels, modal_sim = extract_all_factors(
        device, cfg_path, ckpt_path, csvs)

    # ── Step 2: Build supervision signals ────────────────────────────────────
    print('\n' + '='*60)
    print('Building supervision signals')
    print('='*60)

    # Speaker: 10 classes (already discrete)
    speaker = np.array([speaker_from_uid(u) for u in uids])
    speaker = np.where(speaker < 0, 0, speaker)

    # Temporal: discretize turn_norm into 5 bins
    turn_norm = compute_turn_norm(uids)
    temporal  = np.digitize(turn_norm,
                             bins=np.linspace(0, 1, 6)[1:-1]) # 5 bins: 0-4
    print(f'  temporal bins: {np.unique(temporal, return_counts=True)}')

    # Context: dialogue ID (utterances in same dialogue should cluster in z_c)
    context = dialogue_id_from_uid(uids)
    print(f'  unique dialogues: {len(np.unique(context))}')
    # Reduce to session-level (5 sessions) for tractable MIG
    context_session = np.array([session_from_uid(u) for u in uids])
    context_session = np.where(context_session < 0, 0, context_session)

    # Modal: discretize cross-modal cosine sim into 3 bins
    if modal_sim is not None:
        pcts = np.percentile(modal_sim, [33, 67])
        modal_disc = np.digitize(modal_sim, bins=pcts)  # 0=low, 1=mid, 2=high
        print(f'  modal bins: {np.unique(modal_disc, return_counts=True)}')
    else:
        modal_disc = None

    # Emotion: 4 classes
    emotion = labels

    signals = {
        'speaker' : speaker,
        'temporal': temporal,
        'context' : context_session,
        'emotion' : emotion,
    }
    if modal_disc is not None:
        signals['modal'] = modal_disc

    for k, v in signals.items():
        print(f'  {k}: classes={np.unique(v)}, N={len(v)}')

    # ── Step 3: Compute MIG table ─────────────────────────────────────────────
    print('\n' + '='*60)
    print('Computing MIG cross-MI table (5 factors × 5 signals)')
    print('='*60)
    mi_table, mig_scores = compute_mig_table(factors, signals)

    # ── Step 4: Print results ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print('MIG SUMMARY')
    print('='*60)
    signal_display = {
        'speaker' : 'Speaker Identity   (z_u)',
        'temporal': 'Temporal Order     (z_t)',
        'context' : 'Dialogue Context   (z_c)',
        'modal'   : 'Modality Consist.  (z_m)',
        'emotion' : 'Emotion Label      (z_e)',
    }
    factor_order = ['zc', 'zu', 'zt', 'zm', 'ze']
    print(f'\n  {"Signal":<30} {"zc":>6} {"zu":>6} {"zt":>6} {"zm":>6} {"ze":>6} {"MIG":>6}')
    print('  ' + '-'*68)
    all_mig = []
    for s, label in signal_display.items():
        if s not in mi_table:
            continue
        row  = mi_table[s]
        mig  = mig_scores.get(s, 0.0)
        all_mig.append(mig)
        vals = ' '.join(f'{row.get(f, 0.0):>6.3f}' for f in factor_order)
        print(f'  {label:<30} {vals} {mig:>6.3f}')
    overall = float(np.mean(all_mig))
    print(f'  {"Overall MIG":<30} {"":>36} {overall:>6.3f}')

    # ── Step 5: Compare with existing 3-factor values ────────────────────────
    print('\n' + '='*60)
    print('Comparison with existing paper MIG values')
    print('='*60)
    existing = {'speaker': 0.312, 'modal': 0.428, 'context': 0.271}
    for s, old_val in existing.items():
        new_val = mig_scores.get(s, 0.0)
        print(f'  {s:<12}: existing={old_val:.3f}  computed={new_val:.3f}')

    # ── Step 6: Write tables ──────────────────────────────────────────────────
    print('\n' + '='*60)
    print('Writing LaTeX tables')
    print('='*60)
    os.makedirs(f'{BASE}/experiments/tables', exist_ok=True)

    overall = write_mig_table(
        mi_table, mig_scores,
        f'{BASE}/experiments/tables/mig_all5.tex'
    )
    write_summary_table(
        mig_scores, overall,
        f'{BASE}/experiments/tables/mig_summary.tex'
    )

    # Save JSON
    out_json = f'{BASE}/experiments/results/mig_all5.json'
    json.dump({
        'mi_table'  : {s: {f: float(v) for f, v in row.items()}
                       for s, row in mi_table.items()},
        'mig_scores': {k: float(v) for k, v in mig_scores.items()},
        'overall_mig': overall,
    }, open(out_json, 'w'), indent=2)
    print(f'\n  JSON saved → {out_json}')
    print('\nDone.')
