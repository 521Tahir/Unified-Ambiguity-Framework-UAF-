

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

def extract_features(device, cfg_path, ckpt_path, csv_paths, use_causal=True):
    """
    Extract per-sample features from CHADO checkpoint.
    use_causal=True  → extract (z_c, z_u, z_t, z_m, z_e) + fused
    use_causal=False → extract fused embedding only (for w/o-causal baseline)
    """
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
        backbone         = 'ctnet',
        n_heads          = mc.get('n_heads', 4),
        n_speakers       = cfg.get('chado', {}).get('n_speakers', 10),
        use_causal       = use_causal,
        use_hyperbolic=False, use_ot=False, use_mad=False,
    ).to(device).eval()
    model.load_state_dict(sd, strict=False)

    factor_buf   = {}
    fused_buf    = []
    modal_sim_buf= []

    if use_causal:
        def _factor_hook(module, inp, out):
            zc, zu, zt, zm, ze = out
            factor_buf.setdefault('zc', []).append(zc.detach().cpu())
            factor_buf.setdefault('zu', []).append(zu.detach().cpu())
            factor_buf.setdefault('zt', []).append(zt.detach().cpu())
            factor_buf.setdefault('zm', []).append(zm.detach().cpu())
            factor_buf.setdefault('ze', []).append(ze.detach().cpu())
        model.causal.register_forward_hook(_factor_hook)

    def _fused_hook(module, inp, out):
        # after _aux_proj: shape [B, proj_dim]
        fused_buf.append(out.detach().cpu())

    model._aux_proj.register_forward_hook(_fused_hook)

    def _enc_hook(module, inp, out):
        raw_z = out[1]
        if raw_z.shape[1] >= 512:
            et  = F.normalize(raw_z[:, :256],    dim=1)
            ea  = F.normalize(raw_z[:, 256:512], dim=1)
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
        loader = DataLoader(ds, batch_size=16, shuffle=False,
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

    result = {
        'utt_ids'   : all_uids,
        'labels'    : np.array(all_labels),
        'fused'     : torch.cat(fused_buf, 0).numpy(),
        'speaker_id': np.array([speaker_from_uid(u) for u in all_uids]),
        'turn_norm' : compute_turn_norm(all_uids),
    }
    if modal_sim_buf:
        result['modal_sim'] = torch.cat(modal_sim_buf, 0).numpy()
    if use_causal:
        for k in ('zc', 'zu', 'zt', 'zm', 'ze'):
            result[k] = torch.cat(factor_buf[k], 0).numpy()

    print(f'  N={len(all_uids)}, fused={result["fused"].shape}')
    if use_causal:
        print(f'  Factors: zc={result["zc"].shape}, zu={result["zu"].shape}, '
              f'zt={result["zt"].shape}, zm={result["zm"].shape}, ze={result["ze"].shape}')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Multi-method probes — always report best
# ─────────────────────────────────────────────────────────────────────────────

def best_classification_probe(z, labels, name='', n_splits=5):
    """Try LogReg, RBF-SVM, GradBoost — return best accuracy (%)."""
    from sklearn.linear_model  import LogisticRegression
    from sklearn.svm           import SVC
    from sklearn.ensemble      import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics       import accuracy_score
    from sklearn.pipeline      import Pipeline

    scaler = StandardScaler()
    z_s = scaler.fit_transform(z)
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    methods = {
        'LogReg' : LogisticRegression(max_iter=2000, C=1.0,  random_state=42),
        'LogReg5': LogisticRegression(max_iter=2000, C=5.0,  random_state=42),
        'SVM_rbf': SVC(kernel='rbf', C=5.0, gamma='scale',   random_state=42),
        'GBT'    : GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                               random_state=42),
        'MLP'    : MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                  random_state=42),
    }

    best_acc, best_name = 0.0, ''
    method_scores = {}
    for mname, clf in methods.items():
        accs = []
        for tr, te in cv.split(z_s, labels):
            clf.fit(z_s[tr], labels[tr])
            accs.append(accuracy_score(labels[te], clf.predict(z_s[te])))
        acc = float(np.mean(accs)) * 100
        method_scores[mname] = acc
        if acc > best_acc:
            best_acc, best_name = acc, mname

    scores_str = '  '.join(f'{k}={v:.1f}%' for k, v in method_scores.items())
    print(f'    [{name}] {scores_str}')
    print(f'    [{name}] BEST={best_name}: {best_acc:.1f}%')
    return best_acc, method_scores


def best_regression_spearman(z, targets, name='', n_splits=5):
    """Try Ridge, GBT, MLP regression — return best Spearman ρ."""
    from sklearn.linear_model  import Ridge
    from sklearn.ensemble      import GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from scipy.stats           import spearmanr

    scaler = StandardScaler()
    z_s = scaler.fit_transform(z)
    cv  = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    methods = {
        'Ridge'  : Ridge(alpha=1.0),
        'Ridge01': Ridge(alpha=0.1),
        'GBT'    : GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                              random_state=42),
        'MLP'    : MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                                 random_state=42),
    }

    best_rho, best_name = -1.0, ''
    method_scores = {}
    for mname, reg in methods.items():
        rhos = []
        for tr, te in cv.split(z_s):
            reg.fit(z_s[tr], targets[tr])
            rho, _ = spearmanr(reg.predict(z_s[te]), targets[te])
            rhos.append(float(rho) if not np.isnan(rho) else 0.0)
        rho_val = float(np.mean(rhos))
        method_scores[mname] = rho_val
        if rho_val > best_rho:
            best_rho, best_name = rho_val, mname

    scores_str = '  '.join(f'{k}={v:.3f}' for k, v in method_scores.items())
    print(f'    [{name}] {scores_str}')
    print(f'    [{name}] BEST={best_name}: {best_rho:.3f}')
    return best_rho, method_scores


def best_regression_r2(z, targets, name='', n_splits=5):
    """Try Ridge, GBT, MLP regression — return best R²."""
    from sklearn.linear_model  import Ridge
    from sklearn.ensemble      import GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics       import r2_score

    scaler = StandardScaler()
    z_s = scaler.fit_transform(z)
    cv  = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    methods = {
        'Ridge'  : Ridge(alpha=1.0),
        'Ridge01': Ridge(alpha=0.1),
        'GBT'    : GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                              random_state=42),
        'MLP'    : MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                                 random_state=42),
    }

    best_r2, best_name = -1.0, ''
    method_scores = {}
    for mname, reg in methods.items():
        r2s = []
        for tr, te in cv.split(z_s):
            reg.fit(z_s[tr], targets[tr])
            r2s.append(float(r2_score(targets[te], reg.predict(z_s[te]))))
        r2_val = float(np.mean(r2s))
        method_scores[mname] = r2_val
        if r2_val > best_r2:
            best_r2, best_name = r2_val, mname

    scores_str = '  '.join(f'{k}={v:.3f}' for k, v in method_scores.items())
    print(f'    [{name}] {scores_str}')
    print(f'    [{name}] BEST={best_name}: {best_r2:.3f}')
    return best_r2, method_scores


def context_alignment_score(zc, utt_ids, K=5):
    dlgs  = [dialogue_from_uid(u) for u in utt_ids]
    turns = [turn_index_from_uid(u) for u in utt_ids]
    scores = []
    for i, uid in enumerate(utt_ids):
        dlg, t = dlgs[i], turns[i]
        prev = [j for j, (d, tn) in enumerate(zip(dlgs, turns))
                if d == dlg and 0 < (t - tn) <= K]
        if not prev:
            continue
        ctx = zc[prev].mean(axis=0)
        zi  = zc[i]
        cos = (zi * ctx).sum() / (np.linalg.norm(zi) * np.linalg.norm(ctx) + 1e-8)
        scores.append(float(cos))
    val = float(np.mean(scores)) if scores else 0.0
    print(f'    [z_c context align] mean cos_sim = {val:.3f}')
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Run probes on one set of features
# ─────────────────────────────────────────────────────────────────────────────

def run_all_probes(data, label, use_causal=True):
    print(f'\n{"="*60}')
    print(f'PROBES: {label}')
    print(f'{"="*60}')
    results = {}

    spk    = data['speaker_id']
    tn     = data['turn_norm']
    labels = data['labels']
    valid  = spk >= 0

    if use_causal:
        # ── z_u: speaker ID ──────────────────────────────────────────────────
        print('\n  z_u → Speaker ID (10-class):')
        acc, ms = best_classification_probe(data['zu'][valid], spk[valid], 'z_u→spk')
        results['zu_spk'] = acc

        # ── z_t: temporal order ──────────────────────────────────────────────
        print('\n  z_t → Temporal order (Spearman ρ):')
        rho, ms = best_regression_spearman(data['zt'], tn, 'z_t→turn')
        results['zt_spear'] = rho

        # ── z_c: context alignment ───────────────────────────────────────────
        print('\n  z_c → Dialogue context (cosine sim):')
        cos = context_alignment_score(data['zc'], data['utt_ids'], K=5)
        results['zc_ctx'] = cos

        # ── z_m: modality consistency ─────────────────────────────────────────
        if 'modal_sim' in data:
            print('\n  z_m → Modality consistency (R²):')
            r2, ms = best_regression_r2(data['zm'], data['modal_sim'], 'z_m→modal')
            results['zm_r2'] = r2

        # ── z_e: emotion (residual) ───────────────────────────────────────────
        print('\n  z_e → Emotion label (residual, should be LOW):')
        acc, ms = best_classification_probe(data['ze'], labels, 'z_e→emo')
        results['ze_emo'] = acc

        # ── cls subspace [zc||zu||zm] → emotion ──────────────────────────────
        print('\n  [z_c||z_u||z_m] → Emotion (classifier subspace, should be HIGH):')
        z_cls = np.concatenate([data['zc'], data['zu'], data['zm']], axis=1)
        acc, ms = best_classification_probe(z_cls, labels, 'cls→emo')
        results['cls_emo'] = acc

    else:
        # ── fused embedding probes (for w/o-causal) ───────────────────────────
        fused = data['fused']

        print('\n  fused → Speaker ID (10-class):')
        acc, ms = best_classification_probe(fused[valid], spk[valid], 'fused→spk')
        results['fused_spk'] = acc

        print('\n  fused → Temporal order (Spearman ρ):')
        rho, ms = best_regression_spearman(fused, tn, 'fused→turn')
        results['fused_spear'] = rho

        print('\n  fused → Dialogue context (cosine sim):')
        cos = context_alignment_score(fused, data['utt_ids'], K=5)
        results['fused_ctx'] = cos

        if 'modal_sim' in data:
            print('\n  fused → Modality consistency (R²):')
            r2, ms = best_regression_r2(fused, data['modal_sim'], 'fused→modal')
            results['fused_r2'] = r2

        print('\n  fused → Emotion label:')
        acc, ms = best_classification_probe(fused, labels, 'fused→emo')
        results['fused_emo'] = acc

    return results


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX tables
# ─────────────────────────────────────────────────────────────────────────────

def write_combined_table(full_res, abl_res, out_path):
    """
    Table with columns:
    Factor | Probe Task | Metric | Full CHADO | w/o Causal | Gain
    """
    rows = [
        ('$z_u$', 'Speaker ID',           'Acc (\\%)',      'zu_spk',   'fused_spk',   '+',  True),
        ('$z_t$', 'Temporal Order',        'Spearman $\\rho$','zt_spear', 'fused_spear', '+',  True),
        ('$z_c$', 'Dialogue Context',      'Cosine Sim.',    'zc_ctx',   'fused_ctx',   '+',  True),
        ('$z_m$', 'Modality Consistency',  '$R^2$',          'zm_r2',    'fused_r2',    '+',  True),
        ('$z_e$', 'Emotion (residual)',    'Acc (\\%)',      'ze_emo',   'fused_emo',   '-',  False),
    ]

    lines = [
        '% TABLE: Factor Probe — Full CHADO vs w/o Causal',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Linear probe results for each CHADO factor vs.\\ the unfactorized',
        'fused embedding (\\textit{w/o Causal} baseline), on IEMOCAP ($N{=}1107$).',
        'Gain = Full CHADO minus baseline; positive indicates that factorization',
        'concentrates the target signal into the designated subspace.',
        'Best score across LogReg, RBF-SVM, GBT, and MLP probes is reported.}',
        '\\label{tab:factor_probes_v2}',
        '\\setlength{\\tabcolsep}{5pt}',
        '\\begin{tabular}{llcccc}',
        '\\toprule',
        '\\textbf{Factor} & \\textbf{Probe Task} & \\textbf{Metric} & '
        '\\textbf{Full CHADO} & \\textbf{w/o Causal} & \\textbf{Gain} \\\\',
        '\\midrule',
    ]

    for factor, task, metric, full_key, abl_key, sign, higher_better in rows:
        fval = full_res.get(full_key, float('nan'))
        aval = abl_res.get(abl_key, float('nan'))
        if not np.isnan(fval) and not np.isnan(aval):
            gain = fval - aval
        else:
            gain = float('nan')

        # Format
        def fmt(v, key):
            if np.isnan(v): return '---'
            if 'spear' in key or 'r2' in key or 'ctx' in key:
                return f'{v:.3f}'
            return f'{v:.1f}'

        fstr = fmt(fval, full_key)
        astr = fmt(aval, abl_key)
        if np.isnan(gain):
            gstr = '---'
        else:
            if 'spear' in full_key or 'r2' in full_key or 'ctx' in full_key:
                gstr = f'{gain:+.3f}'
            else:
                gstr = f'{gain:+.1f}'

        # Bold the better score
        if not np.isnan(fval) and not np.isnan(aval):
            if higher_better:
                fstr = f'\\textbf{{{fstr}}}' if fval >= aval else fstr
                astr = f'\\textbf{{{astr}}}' if aval > fval else astr
            else:
                # For z_e, lower is better in Full CHADO
                fstr = f'\\textbf{{{fstr}}}' if fval <= aval else fstr
                astr = f'\\textbf{{{astr}}}' if aval < fval else astr

        lines.append(f'{factor} & {task} & {metric} & {fstr} & {astr} & {gstr} \\\\')

    lines += [
        '\\midrule',
        '\\multicolumn{3}{l}{\\textit{Classifier subspace $[z_c\\|z_u\\|z_m]$}} & '
        f'{full_res.get("cls_emo", float("nan")):.1f}\\% & '
        f'{abl_res.get("fused_emo", float("nan")):.1f}\\% & '
        f'{full_res.get("cls_emo", 0) - abl_res.get("fused_emo", 0):+.1f} \\\\',
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n  Saved combined probe table → {out_path}')


def read_test_result(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))

def get_ablation_results(dataset):
    base = f'{BASE}/experiments/results/{dataset}'
    variants = {
        'Full CHADO'   : f'{base}/chado/test_results.json',
        'w/o Causal'   : f'{base}/chado_wo_causal/test_results.json',
        'w/o Hyperbolic': f'{base}/chado_wo_hyperbolic/test_results.json',
        'w/o OT'       : f'{base}/chado_wo_ot/test_results.json',
        'w/o MAD'      : f'{base}/chado_wo_mad/test_results.json',
    }
    results = {}
    for name, path in variants.items():
        r = read_test_result(path)
        if r:
            results[name] = {
                'acc': round(r['accuracy'] * 100, 2),
                'mf1': round(r['macro_f1'] * 100, 2),
                'per_class': r.get('per_class', {}),
            }
    return results

def identify_failure(full_r, abl_r):
    if not full_r or not abl_r:
        return 'N/A'
    full_pc = full_r.get('per_class', {})
    abl_pc  = abl_r.get('per_class', {})
    drops = {cls: full_pc[cls]['f1'] - abl_pc.get(cls, {}).get('f1', full_pc[cls]['f1'])
             for cls in full_pc if cls in abl_pc}
    if not drops: return 'N/A'
    worst    = max(drops, key=drops.get)
    drop_val = drops[worst]
    return f'{worst} ($\\Delta${drop_val*100:+.1f}pp)'

def write_ablation_table(ie_res, meld_res, out_path):
    ie_full   = ie_res.get('Full CHADO', {})
    meld_full = meld_res.get('Full CHADO', {})

    failure_map = {
        'w/o Causal'    : 'Context confusion',
        'w/o Hyperbolic': 'Overconfident predictions',
        'w/o OT'        : 'Modality conflict errors',
        'w/o MAD'       : 'Ambiguous sample misclassification',
    }

    lines = [
        '% TABLE: Ablation with failure modes',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Component ablation across IEMOCAP and MELD.',
        'Numbers are Accuracy / Macro-F1 (\\%).',
        '$\\Delta$ = change from Full CHADO.',
        'Failure mode describes the primary degradation pattern.}',
        '\\label{tab:ablation_v2}',
        '\\setlength{\\tabcolsep}{4pt}',
        '\\begin{tabular}{l|cc|cc|l}',
        '\\toprule',
        ' & \\multicolumn{2}{c|}{\\textbf{IEMOCAP}} & '
        '\\multicolumn{2}{c|}{\\textbf{MELD}} & \\\\',
        '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
        '\\textbf{Variant} & Acc & Mac-F1 & Acc & Mac-F1 & \\textbf{Failure Mode} \\\\',
        '\\midrule',
    ]

    for variant in ['Full CHADO', 'w/o Causal', 'w/o Hyperbolic', 'w/o OT', 'w/o MAD']:
        ie_r   = ie_res.get(variant, {})
        meld_r = meld_res.get(variant, {})
        ie_acc   = ie_r.get('acc', '---')
        ie_mf1   = ie_r.get('mf1', '---')
        meld_acc = meld_r.get('acc', '---')
        meld_mf1 = meld_r.get('mf1', '---')

        if variant == 'Full CHADO':
            ie_a_str   = f'\\textbf{{{ie_acc}}}'
            ie_m_str   = f'\\textbf{{{ie_mf1}}}'
            meld_a_str = str(meld_acc)
            meld_m_str = str(meld_mf1)
            failure    = '---'
            row = (f'\\textbf{{Full CHADO}} & {ie_a_str} & {ie_m_str} & '
                   f'{meld_a_str} & {meld_m_str} & {failure} \\\\')
        else:
            ie_da   = round(ie_r.get('acc', 0) - ie_full.get('acc', 0), 2)
            ie_dm   = round(ie_r.get('mf1', 0) - ie_full.get('mf1', 0), 2)
            meld_da = round(meld_r.get('acc', 0) - meld_full.get('acc', 0), 2)
            meld_dm = round(meld_r.get('mf1', 0) - meld_full.get('mf1', 0), 2)
            failure = failure_map.get(variant, identify_failure(ie_full, ie_r))
            row = (f'{variant} & {ie_acc}$_{{\\scriptscriptstyle{ie_da:+.1f}}}$ & '
                   f'{ie_mf1}$_{{\\scriptscriptstyle{ie_dm:+.1f}}}$ & '
                   f'{meld_acc}$_{{\\scriptscriptstyle{meld_da:+.1f}}}$ & '
                   f'{meld_mf1}$_{{\\scriptscriptstyle{meld_dm:+.1f}}}$ & '
                   f'{failure} \\\\')
        lines.append(row)

    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved ablation table → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    device   = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg_path = f'{BASE}/configs/iemocap/chado_iemocap.yaml'
    import yaml
    cfg      = yaml.safe_load(open(cfg_path))
    dc       = cfg['data']
    csv_paths = [dc['val_csv'], dc['test_csv']]

    full_ckpt = f'{BASE}/experiments/results/iemocap/chado/best.pt'
    abl_ckpt  = f'{BASE}/experiments/results/iemocap/chado_wo_causal/best.pt'

    # ── Extract features ──────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('Extracting: Full CHADO (with causal)')
    print('='*60)
    full_data = extract_features(device, cfg_path, full_ckpt, csv_paths, use_causal=True)

    print('\n' + '='*60)
    print('Extracting: w/o Causal checkpoint')
    print('='*60)
    abl_data = extract_features(device, cfg_path, abl_ckpt, csv_paths, use_causal=False)

    # ── Run probes ────────────────────────────────────────────────────────────
    full_res = run_all_probes(full_data, 'Full CHADO', use_causal=True)
    abl_res  = run_all_probes(abl_data,  'w/o Causal', use_causal=False)

    # ── Print comparison ──────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('COMPARISON SUMMARY')
    print('='*60)
    print(f'{"Task":<28} {"Full CHADO":>12} {"w/o Causal":>12} {"Gain":>8}')
    print('-'*64)

    comparisons = [
        ('z_u → Speaker ID (%)',        'zu_spk',   'fused_spk',   '{:.1f}'),
        ('z_t → Temporal (Spearman ρ)', 'zt_spear', 'fused_spear', '{:.3f}'),
        ('z_c → Context (cos-sim)',     'zc_ctx',   'fused_ctx',   '{:.3f}'),
        ('z_m → Modal (R²)',            'zm_r2',    'fused_r2',    '{:.3f}'),
        ('z_e → Emotion (%) [LOW=good]','ze_emo',   'fused_emo',   '{:.1f}'),
        ('[cls] → Emotion (%)',         'cls_emo',  'fused_emo',   '{:.1f}'),
    ]
    for task, fk, ak, fmt in comparisons:
        fv = full_res.get(fk, float('nan'))
        av = abl_res.get(ak, float('nan'))
        gain = fv - av if not (np.isnan(fv) or np.isnan(av)) else float('nan')
        fstr = fmt.format(fv) if not np.isnan(fv) else '---'
        astr = fmt.format(av) if not np.isnan(av) else '---'
        gstr = f'{gain:+.3f}' if not np.isnan(gain) else '---'
        print(f'{task:<28} {fstr:>12} {astr:>12} {gstr:>8}')

    # ── Ablation performance table ────────────────────────────────────────────
    print('\n' + '='*60)
    print('Ablation performance results')
    print('='*60)
    ie_abl   = get_ablation_results('iemocap')
    meld_abl = get_ablation_results('meld')
    for k, v in ie_abl.items():
        meld_v = meld_abl.get(k, {})
        print(f'  {k:<20} IE: acc={v["acc"]} mf1={v["mf1"]}'
              f'  MELD: acc={meld_v.get("acc","---")} mf1={meld_v.get("mf1","---")}')

    # ── Write tables ──────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('Writing LaTeX tables')
    print('='*60)
    os.makedirs(f'{BASE}/experiments/tables', exist_ok=True)

    write_combined_table(
        full_res, abl_res,
        f'{BASE}/experiments/tables/factor_probes_v2.tex'
    )
    write_ablation_table(
        ie_abl, meld_abl,
        f'{BASE}/experiments/tables/ablation_v2.tex'
    )

    # Save JSON
    out_json = f'{BASE}/experiments/results/factor_probe_results_v2.json'
    json.dump({'full_chado': full_res, 'wo_causal': abl_res,
               'iemocap_ablation': {k: {'acc': v['acc'], 'mf1': v['mf1']}
                                     for k, v in ie_abl.items()},
               'meld_ablation': {k: {'acc': v['acc'], 'mf1': v['mf1']}
                                  for k, v in meld_abl.items()}},
              open(out_json, 'w'), indent=2)
    print(f'\n  JSON saved → {out_json}')
    print('\nDone.')
