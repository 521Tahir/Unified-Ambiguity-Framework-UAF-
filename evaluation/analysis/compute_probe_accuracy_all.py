
import os, sys, re, json, functools, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

sys.path.insert(0, '')
BASE = ''

FACTOR_NAMES  = ['zc', 'zu', 'zt', 'zm', 'ze']
FACTOR_DIMS   = {'zc':(0,64),'zu':(64,128),'zt':(128,160),'zm':(160,224),'ze':(224,256)}
FACTOR_LABELS = {
    'zc': 'z_c (context,  64d)',
    'zu': 'z_u (individ., 64d)',
    'zt': 'z_t (temporal, 32d)',
    'zm': 'z_m (modality, 64d)',
    'ze': 'z_e (residual, 32d)',
}
TARGET_FACTOR = {
    'speaker':'zu','temporal':'zt','context':'zc','modal':'zm','emotion':'ze'
}

# ─────────────────────────────────────────────────────────────────────────────
# Probe helpers
# ─────────────────────────────────────────────────────────────────────────────

def best_clf_probe(z, y):
    """Best Acc (%) across LogReg, SVM-RBF, GBT."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    z_s = StandardScaler().fit_transform(z.astype(np.float32))
    best = 0.0
    for clf in [
        LogisticRegression(max_iter=500, C=1.0),
        SVC(kernel='rbf', C=1.0, gamma='scale'),
        GradientBoostingClassifier(n_estimators=50, max_depth=3),
    ]:
        try:
            sc = cross_val_score(clf, z_s, y.astype(int), cv=5,
                                 scoring='accuracy', n_jobs=1)
            best = max(best, float(sc.mean()) * 100)
        except Exception:
            pass
    return best


def best_spearman_probe(z, y_cont):
    """Best Spearman ρ across Ridge + GBT regression."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    z_s = StandardScaler().fit_transform(z.astype(np.float32))
    best = 0.0
    for reg in [Ridge(alpha=1.0), GradientBoostingRegressor(n_estimators=50)]:
        try:
            preds = cross_val_predict(reg, z_s, y_cont.astype(np.float32), cv=5)
            rho   = float(spearmanr(preds, y_cont).statistic)
            best  = max(best, rho)
        except Exception:
            pass
    return best


def best_r2_probe(z, y_cont):
    """Best R² across Ridge + GBT regression."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    z_s = StandardScaler().fit_transform(z.astype(np.float32))
    best = -1.0
    for reg in [Ridge(alpha=0.1), GradientBoostingRegressor(n_estimators=50)]:
        try:
            sc   = cross_val_score(reg, z_s, y_cont.astype(np.float32), cv=5, scoring='r2')
            best = max(best, float(sc.mean()))
        except Exception:
            pass
    return best


def cosine_align(z, ctx_ref):
    """Mean cosine similarity between z and reference embedding ctx_ref.
    If dimensions differ, ctx_ref is sliced/projected to match z."""
    d = z.shape[1]
    if ctx_ref.shape[1] != d:
        ctx_ref = ctx_ref[:, :d]   # take first d dims of reference
    z_n = z      / (np.linalg.norm(z,       axis=1, keepdims=True) + 1e-8)
    r_n = ctx_ref/ (np.linalg.norm(ctx_ref, axis=1, keepdims=True) + 1e-8)
    return float((z_n * r_n).sum(axis=1).mean())


# Module-level hook helpers (must be at top level to be reusable across extraction fns)
def _keep2d(buf):
    """Hook that only accumulates 2-D tensors (CLS projections, not sequence)."""
    def hook(m, i, o):
        if o.dim() == 2:
            buf.append(F.normalize(o.detach().cpu(), dim=1))
    return hook


def _make_factor_hook(factor_buf, factor_names):
    """Hook for causal module that stores outputs WITHOUT replacing them (returns None)."""
    def hook(m, i, o):
        for k, z in zip(factor_names, o):
            factor_buf.setdefault(k, []).append(z.detach().cpu())
        # return None — do NOT replace module output
    return hook


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load checkpoint state dict
# ─────────────────────────────────────────────────────────────────────────────

def load_sd(path):
    ck = torch.load(path, map_location='cpu')
    if isinstance(ck, dict):
        for k in ('model', 'model_state_dict', 'state_dict'):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
    return ck


def has_5factor(sd):
    return any('head_c' in k for k in sd.keys())


# ─────────────────────────────────────────────────────────────────────────────
# IEMOCAP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ie_speaker(u):
    m = re.match(r'Ses0?(\d)([MF])', str(u))
    return (int(m.group(1))-1)*2+(0 if m.group(2)=='M' else 1) if m else 0

def _ie_session(u):
    m = re.match(r'Ses0?(\d)', str(u))
    return int(m.group(1))-1 if m else 0

def _ie_dlg(u):
    m = re.match(r'(Ses\d+[MF]_\w+?)_[MF]\d+', str(u))
    return m.group(1) if m else str(u)

def _ie_turn(u):
    m = re.search(r'_[MF](\d+)$', str(u))
    return int(m.group(1)) if m else 0

def ie_turn_norm(uids):
    dlgs  = [_ie_dlg(u)  for u in uids]
    turns = [_ie_turn(u) for u in uids]
    df = pd.DataFrame({'d':dlgs,'t':turns})
    df['mx'] = df.groupby('d')['t'].transform('max').clip(lower=1)
    return (df['t']/df['mx']).values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# IEMOCAP factor extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_iemocap(device, ckpt_path, use_causal=True):
    import yaml
    from models.chado.model import CHADOTrimodal
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap

    cfg = yaml.safe_load(open(f'{BASE}/configs/iemocap/chado_iemocap.yaml'))
    dc, mc = cfg['data'], cfg['model']
    sd = load_sd(ckpt_path)
    five_fac = has_5factor(sd)

    model = CHADOTrimodal(
        text_model_name='roberta-base',
        audio_model_name=mc['audio_model_name'],
        video_model_name=mc['video_model_name'],
        num_classes=dc['num_classes'],
        proj_dim=256, dropout=0.0,
        use_text=True, use_audio=True, use_video=True,
        use_gated_fusion=True, backbone='ctnet', n_heads=4,
        n_speakers=10,
        use_causal=use_causal, use_hyperbolic=False,
        use_ot=False, use_mad=False,
    ).to(device).eval()
    model.load_state_dict(sd, strict=False)

    factor_buf = {}
    fused_buf  = []
    audio_buf  = []
    text_buf   = []

    if use_causal and five_fac:
        model.causal.register_forward_hook(_make_factor_hook(factor_buf, FACTOR_NAMES))

    # always capture fused (for w/o Causal baseline or context alignment)
    model._aux_proj.register_forward_hook(
        lambda m,i,o: fused_buf.append(o.detach().cpu()))

    # per-modality proj encodings for modal cosine similarity (2D only)
    model.base.encoders.text_proj.register_forward_hook(_keep2d(text_buf))
    model.base.encoders.audio_proj.register_forward_hook(_keep2d(audio_buf))

    LABEL2ID = {'neu':0,'hap':1,'ang':2,'sad':3}
    csvs = [dc['val_csv'], dc['test_csv']]
    all_uids, all_labels = [], []

    for csv_path in csvs:
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        ds = IEMOCAPDataset(csv_path=csv_path, text_model_name='roberta-base',
                            max_text_len=96, audio_sr=16000, audio_sec=4.0,
                            n_frames=8, use_audio=True, use_video=False)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                            collate_fn=collate_iemocap)
        uid_list = df['utt_id'].tolist()
        lbl_list = [LABEL2ID.get(str(l).strip().lower(),0) for l in df['label_4']]
        with torch.no_grad():
            idx = 0
            for batch in loader:
                B = batch['input_ids'].shape[0]
                ti = {'input_ids':batch['input_ids'].to(device),
                      'attention_mask':batch['attention_mask'].to(device)}
                aw = batch.get('wav')
                if aw is None:
                    aw = batch.get('audio_wave')
                aw = aw.to(device) if aw is not None else None
                vf = torch.zeros(B,8,3,224,224,device=device)
                model(text_input=ti, audio_wave=aw, video_frames=vf)
                all_uids.extend(uid_list[idx:idx+B])
                all_labels.extend(lbl_list[idx:idx+B])
                idx += B

    fused = torch.cat(fused_buf, 0).numpy()
    text_emb  = torch.cat(text_buf,  0).numpy() if text_buf  else None
    audio_emb = torch.cat(audio_buf, 0).numpy() if audio_buf else None

    if use_causal and five_fac and factor_buf:
        factors = {k: torch.cat(v,0).numpy() for k,v in factor_buf.items()}
    else:
        # partition fused into positional pseudo-factors
        factors = {k: fused[:,lo:hi] for k,(lo,hi) in FACTOR_DIMS.items()}

    N = len(all_uids)
    speaker   = np.array([_ie_speaker(u) for u in all_uids])
    session   = np.array([_ie_session(u) for u in all_uids])
    turn_norm = ie_turn_norm(all_uids)
    labels    = np.array(all_labels)

    # context reference: mean fused over same dialogue
    dlg_ids = np.array([_ie_dlg(u) for u in all_uids])
    uniq_d  = {d:i for i,d in enumerate(sorted(set(dlg_ids)))}
    dlg_int = np.array([uniq_d[d] for d in dlg_ids])
    ctx_ref = np.zeros_like(fused)
    for d in np.unique(dlg_int):
        mask = dlg_int == d
        ctx_ref[mask] = fused[mask].mean(axis=0)

    # modal sim = cos(text, audio)
    if text_emb is not None and audio_emb is not None:
        dim = min(text_emb.shape[1], audio_emb.shape[1])
        modal_sim = (text_emb[:N, :dim] * audio_emb[:N, :dim]).sum(axis=1)
    else:
        modal_sim = np.zeros(N, dtype=np.float32)

    signals = {
        'speaker' : speaker,
        'temporal': turn_norm,
        'context' : ctx_ref,
        'modal'   : modal_sim,
        'emotion' : labels,
    }
    return factors, fused, signals, N


# ─────────────────────────────────────────────────────────────────────────────
# MELD factor extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_meld(device, ckpt_path):
    import yaml, functools
    from models.chado.model import CHADOTrimodal
    from datasets.meld.meld_dataset import MeldDataset, collate_meld, EMO_ORDER_7

    cfg = yaml.safe_load(open(f'{BASE}/configs/meld/chado_meld.yaml'))
    dc, mc = cfg['data'], cfg['model']
    sd = load_sd(ckpt_path)

    text_model = 'roberta-large'
    for k,v in sd.items():
        if 'text_enc' in k and 'embeddings.LayerNorm.weight' in k:
            text_model = 'roberta-large' if v.shape[0]==1024 else 'roberta-base'
            break

    model = CHADOTrimodal(
        text_model_name=text_model,
        audio_model_name=mc['audio_model_name'],
        video_model_name=mc['video_model_name'],
        num_classes=dc['num_classes'],
        proj_dim=256, dropout=0.0,
        use_text=True, use_audio=True, use_video=True,
        use_gated_fusion=True, backbone='ctnet', n_heads=4, n_speakers=7,
        use_causal=True, use_hyperbolic=False, use_ot=False, use_mad=False,
    ).to(device).eval()
    model.load_state_dict(sd, strict=False)

    fused_buf = []
    text_buf  = []
    audio_buf = []
    model._aux_proj.register_forward_hook(
        lambda m,i,o: fused_buf.append(o.detach().cpu()))
    model.base.encoders.text_proj.register_forward_hook(_keep2d(text_buf))
    model.base.encoders.audio_proj.register_forward_hook(_keep2d(audio_buf))

    MAIN_SPK = {'joey':0,'monica':1,'chandler':2,'ross':3,'rachel':4,'phoebe':5}
    raw_spk_map = {}
    for rp in [
        '/home/tahirahmad/CHADO_CLAUDECODE/data/raw/meld/MELD.Raw/dev_sent_emo.csv',
        '/home/tahirahmad/CHADO_CLAUDECODE/data/raw/meld/MELD.Raw/test_sent_emo.csv',
    ]:
        if os.path.exists(rp):
            df_r = pd.read_csv(rp)
            df_r.columns = [c.lower().replace(' ','_') for c in df_r.columns]
            for _,r in df_r.iterrows():
                key = (int(r['dialogue_id']), int(r['utterance_id']))
                raw_spk_map[key] = MAIN_SPK.get(str(r['speaker']).strip().lower(),6)

    all_labels, all_tnorm, all_speaker, all_diaid = [], [], [], []

    for split in ('val','test'):
        csv_path = dc['val_csv'] if split=='val' else dc['test_csv']
        if not os.path.exists(csv_path): continue
        proc_df   = pd.read_csv(csv_path)
        label_map = {e:i for i,e in enumerate(EMO_ORDER_7)}

        ds = MeldDataset(
            csv_path=csv_path, text_model_name=text_model,
            label_map=label_map, text_col='text', label_col='emotion',
            audio_path_col='audio_path', video_path_col='video_path',
            utt_id_col='utt_id', num_frames=8, frame_size=224,
            sample_rate=16000, max_audio_seconds=6.0,
            use_text=True, use_audio=True, use_video=False,
        )
        cfn = functools.partial(collate_meld, tokenizer=ds.tokenizer,
                                use_text=True, use_audio=True, use_video=False)
        loader = DataLoader(ds, batch_size=12, shuffle=False,
                            num_workers=0, collate_fn=cfn)

        spk_arr, tnrm_arr, dia_arr = [], [], []
        for _, row in proc_df.iterrows():
            did  = int(row['dialogue_id']); uid = int(row['utt_id'])
            spk_arr.append(raw_spk_map.get((did,uid),6))
            dia_arr.append(did)
            mx = proc_df[proc_df['dialogue_id']==did]['utt_id'].max()
            tnrm_arr.append(uid/max(1,mx))

        split_idx = 0
        with torch.no_grad():
            for batch in loader:
                B  = batch.text_input['input_ids'].shape[0]
                ti = {k:v.to(device) for k,v in batch.text_input.items()}
                aw = batch.audio_wave.to(device) if batch.audio_wave is not None else None
                vf = torch.zeros(B,8,3,224,224,device=device)
                model(text_input=ti, audio_wave=aw, video_frames=vf)
                for i in range(B):
                    idx = split_idx+i
                    all_speaker.append(spk_arr[idx] if idx<len(spk_arr) else 6)
                    all_tnorm.append(tnrm_arr[idx]  if idx<len(tnrm_arr) else 0.0)
                    all_diaid.append(dia_arr[idx]    if idx<len(dia_arr)  else 0)
                    all_labels.append(int(batch.labels[i].item()))
                split_idx += B

    fused     = torch.cat(fused_buf, 0).numpy()
    text_emb  = torch.cat(text_buf,  0).numpy() if text_buf  else None
    audio_emb = torch.cat(audio_buf, 0).numpy() if audio_buf else None
    N = fused.shape[0]

    factors = {k: fused[:,lo:hi] for k,(lo,hi) in FACTOR_DIMS.items()}

    # context ref: mean fused per dialogue
    dia_int = np.array(all_diaid)
    ctx_ref = np.zeros_like(fused)
    for d in np.unique(dia_int):
        mask = dia_int == d
        ctx_ref[mask] = fused[mask].mean(axis=0)

    if text_emb is not None and audio_emb is not None:
        dim = min(text_emb.shape[1], audio_emb.shape[1])
        modal_sim = (text_emb[:N,:dim]*audio_emb[:N,:dim]).sum(axis=1)
    else:
        modal_sim = np.zeros(N, dtype=np.float32)

    signals = {
        'speaker' : np.array(all_speaker),
        'temporal': np.array(all_tnorm, dtype=np.float32),
        'context' : ctx_ref,
        'modal'   : modal_sim,
        'emotion' : np.array(all_labels),
    }
    return factors, fused, signals, N


# ─────────────────────────────────────────────────────────────────────────────
# MOSEI factor extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_mosei(device, ckpt_path):
    import yaml
    from models.chado.model import CHADOFeature
    from datasets.mosei.mosei_utt_dataset import MoseiUttDataset, collate_mosei_utt

    cfg = yaml.safe_load(open(f'{BASE}/configs/mosei/chado_mosei.yaml'))
    dc = cfg['data']
    sd = load_sd(ckpt_path)

    d_model = 256
    model = CHADOFeature(text_model='roberta-base', num_classes=6, d_model=d_model,
                         use_audio=True, use_video=True,
                         use_causal=True, use_hyperbolic=False,
                         use_ot=False, use_mad=False).to(device).eval()
    model.load_state_dict(sd, strict=False)

    fused_buf = []
    model.base.fuse.register_forward_hook(
        lambda m,i,o: fused_buf.append(o.detach().cpu()))

    for manifest_key in ('test_manifest','val_manifest'):
        manifest = dc.get(manifest_key)
        if manifest and os.path.exists(manifest):
            break
    else:
        print('  [SKIP] MOSEI manifest not found')
        return None, None, None, 0

    ds = MoseiUttDataset(manifest_path=manifest, text_model_name='roberta-base',
                         max_text_len=96, max_audio_len=50, max_video_len=30,
                         label_thr=0.5)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0,
                        collate_fn=collate_mosei_utt)

    all_labels, all_video_id, all_utt_idx = [], [], []

    with torch.no_grad():
        for batch in loader:
            B = batch['audio'].shape[0]
            dev_batch = {'text':batch['text'],
                         'audio':batch['audio'].to(device),
                         'video':batch['video'].to(device),
                         'label':batch['label'].to(device)}
            model(dev_batch)
            lbl = batch['label'].numpy()
            dominant = np.argmax(lbl, axis=1)
            all_labels.extend(dominant.tolist())
            for uid in batch['utt_id']:
                parts = str(uid).rsplit('__',1)
                all_video_id.append(parts[0])
                all_utt_idx.append(int(parts[1]) if len(parts)>1 and parts[1].isdigit() else 0)

    fused = torch.cat(fused_buf, 0).numpy()
    N = min(fused.shape[0], len(all_labels))
    fused = fused[:N]; all_labels=all_labels[:N]
    all_video_id=all_video_id[:N]; all_utt_idx=all_utt_idx[:N]

    factors = {k: fused[:,lo:hi] for k,(lo,hi) in FACTOR_DIMS.items()}

    vid_uniq = {v:i for i,v in enumerate(sorted(set(all_video_id)))}
    vid_int  = np.array([vid_uniq[v] for v in all_video_id])

    # context ref: mean fused per video (speaker proxy)
    ctx_ref = np.zeros_like(fused)
    for v in np.unique(vid_int):
        mask = vid_int==v
        ctx_ref[mask] = fused[mask].mean(axis=0)

    utt_arr  = np.array(all_utt_idx, dtype=np.float32)
    max_utt  = max(1.0, utt_arr.max())
    turn_norm= utt_arr / max_utt

    # modal sim proxy: zm norm percentile (no raw audio/video features accessible)
    zm_norm  = np.linalg.norm(factors['zm'], axis=1)
    modal_sim= (zm_norm - zm_norm.min()) / (zm_norm.max()-zm_norm.min()+1e-8)

    signals = {
        'speaker' : np.digitize(vid_int, bins=np.percentile(vid_int,[20,40,60,80])),
        'temporal': turn_norm,
        'context' : ctx_ref,
        'modal'   : modal_sim,
        'emotion' : np.array(all_labels),
    }
    return factors, fused, signals, N


# ─────────────────────────────────────────────────────────────────────────────
# Run probes for one dataset
# ─────────────────────────────────────────────────────────────────────────────

def run_probes(factors, signals, fused=None, dataset_name='', arch_note=''):
    """
    factors: dict {name: [N,d]}   — 5 factor groups
    signals: dict {name: array}   — supervision signals
    fused  : [N,256] or None      — for w/o Causal baseline column
    """
    PROBE_CFG = {
        'speaker' : ('clf',   'Acc (%)',     'zu'),
        'temporal': ('spear', 'Spearman ρ',  'zt'),
        'context' : ('cos',   'Cos Sim',     'zc'),
        'modal'   : ('r2',    'R²',          'zm'),
        'emotion' : ('clf',   'Acc (%)',     'ze'),
    }

    results = {}  # signal → {factor → score}

    for sname, (ptype, metric, tgt) in PROBE_CFG.items():
        if sname not in signals:
            continue
        sig = signals[sname]
        row = {}

        for fname in FACTOR_NAMES:
            z = factors[fname]
            if ptype == 'clf':
                sc = best_clf_probe(z, sig)
            elif ptype == 'spear':
                sc = best_spearman_probe(z, sig)
            elif ptype == 'r2':
                sc = best_r2_probe(z, sig)
            elif ptype == 'cos':
                sc = cosine_align(z, sig)
            else:
                sc = 0.0
            row[fname] = sc

        # Fused baseline
        if fused is not None:
            z_f = fused
            if ptype == 'clf':
                row['fused'] = best_clf_probe(z_f, sig)
            elif ptype == 'spear':
                row['fused'] = best_spearman_probe(z_f, sig)
            elif ptype == 'r2':
                row['fused'] = best_r2_probe(z_f, sig)
            elif ptype == 'cos':
                row['fused'] = cosine_align(z_f, sig)

        results[sname] = row

    # ── Print ─────────────────────────────────────────────────────────────
    print(f'\n{"="*80}')
    print(f'  PROBE ACCURACY — {dataset_name}  {arch_note}')
    print(f'{"="*80}')

    has_fused = fused is not None
    cols = FACTOR_NAMES + (['fused'] if has_fused else [])
    hdr  = f'  {"Signal":<22} {"Metric":<12}'
    for c in cols:
        hdr += f'  {c:>8}'
    if has_fused:
        hdr += f'  {"Gain":>8}'
    print(hdr)
    print('  ' + '-'*80)

    for sname, (ptype, metric, tgt) in PROBE_CFG.items():
        if sname not in results:
            continue
        row = results[sname]
        fmt = '.2f' if ptype in ('clf',) else '.4f'
        line = f'  {sname:<22} {metric:<12}'
        for c in FACTOR_NAMES:
            val  = row.get(c, float('nan'))
            bold = '*' if c == tgt else ' '
            line += f'  {val:{fmt}}{bold}'
        if has_fused:
            fv   = row.get('fused', float('nan'))
            line += f'  {fv:{fmt}} '
            # Gain = tgt_factor - fused
            tgt_v = row.get(tgt, float('nan'))
            gain  = tgt_v - fv if not (np.isnan(tgt_v) or np.isnan(fv)) else float('nan')
            sign  = '+' if gain >= 0 else ''
            line += f'  {sign}{gain:{fmt}}'
        print(line)

    print(f'\n  * = target factor for that signal (expected to score highest)')

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    all_results = {}

    # ── IEMOCAP ───────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('  IEMOCAP  — extracting 5-factor checkpoint + w/o Causal baseline')
    print('='*80)
    try:
        ie_ckpt    = f'{BASE}/experiments/results/iemocap/chado/best.pt'
        ie_woc_ck  = f'{BASE}/experiments/results/iemocap/chado_wo_causal/best.pt'

        ie_fac, ie_fused, ie_sig, ie_N = extract_iemocap(device, ie_ckpt, use_causal=True)
        print(f'  IEMOCAP N={ie_N}')
        for k,v in ie_fac.items():
            print(f'    {k}: {v.shape}')

        # w/o Causal: extract fused embedding (no factorization)
        _, ie_fused_woc, _, _ = extract_iemocap(device, ie_woc_ck, use_causal=False)

        ie_res = run_probes(ie_fac, ie_sig, fused=ie_fused_woc,
                            dataset_name='IEMOCAP',
                            arch_note='[5-factor | fused=w/o Causal]')
        all_results['IEMOCAP'] = ie_res
    except Exception as e:
        print(f'  IEMOCAP failed: {e}')
        import traceback; traceback.print_exc()

    # ── MELD ──────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('  MELD  — extracting fused embedding (old checkpoint) + w/o Causal')
    print('='*80)
    try:
        meld_ckpt   = f'{BASE}/experiments/results/meld/chado/best.pt'
        meld_woc_ck = f'{BASE}/experiments/results/meld/chado_wo_causal/best.pt'

        meld_fac, meld_fused, meld_sig, meld_N = extract_meld(device, meld_ckpt)
        print(f'  MELD N={meld_N}')
        torch.cuda.empty_cache()

        meld_fused_woc = None
        try:
            _, meld_fused_woc, _, _ = extract_meld(device, meld_woc_ck)
            torch.cuda.empty_cache()
        except Exception as woc_e:
            print(f'  [WARN] MELD w/o Causal extraction failed: {woc_e}  — running without baseline')

        meld_res = run_probes(meld_fac, meld_sig, fused=meld_fused_woc,
                              dataset_name='MELD',
                              arch_note='[positional pseudo-factors]')
        all_results['MELD'] = meld_res
    except Exception as e:
        print(f'  MELD failed: {e}')
        import traceback; traceback.print_exc()

    # ── CMU-MOSEI ─────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('  CMU-MOSEI  — extracting fused embedding (old checkpoint) + w/o Causal')
    print('='*80)
    try:
        mosei_ckpt   = f'{BASE}/experiments/results/mosei/chado/best.pt'
        mosei_woc_ck = f'{BASE}/experiments/results/mosei/chado_wo_causal/best.pt'

        mosei_fac, mosei_fused, mosei_sig, mosei_N = extract_mosei(device, mosei_ckpt)
        print(f'  MOSEI N={mosei_N}')

        _, mosei_fused_woc, _, _ = extract_mosei(device, mosei_woc_ck)

        mosei_res = run_probes(mosei_fac, mosei_sig, fused=mosei_fused_woc,
                               dataset_name='CMU-MOSEI',
                               arch_note='[positional pseudo-factors | fused=w/o Causal]')
        all_results['CMU-MOSEI'] = mosei_res
    except Exception as e:
        print(f'  MOSEI failed: {e}')
        import traceback; traceback.print_exc()

    # ── Cross-dataset summary ─────────────────────────────────────────────
    print('\n' + '='*80)
    print('  CROSS-DATASET SUMMARY — Target Factor Score vs Fused Baseline')
    print('='*80)
    PROBE_CFG = {
        'speaker' : ('Acc (%)', '.2f', 'zu'),
        'temporal': ('Spear ρ', '.4f', 'zt'),
        'context' : ('Cos Sim', '.4f', 'zc'),
        'modal'   : ('R²',      '.4f', 'zm'),
        'emotion' : ('Acc (%)', '.2f', 'ze'),
    }
    print(f'\n  {"Signal":<20} {"Metric":<10}'
          f'{"IE-Tgt":>10}{"IE-Fuse":>10}{"IE-Δ":>8}'
          f'{"ML-Tgt":>10}{"ML-Fuse":>10}{"ML-Δ":>8}'
          f'{"MO-Tgt":>10}{"MO-Fuse":>10}{"MO-Δ":>8}')
    print('  '+'-'*100)

    for sname,(metric,fmt,tgt) in PROBE_CFG.items():
        def get(ds, key):
            r = all_results.get(ds,{}).get(sname,{})
            v = r.get(key, float('nan'))
            return v
        ie_t = get('IEMOCAP',   tgt);   ie_f = get('IEMOCAP',   'fused')
        ml_t = get('MELD',      tgt);   ml_f = get('MELD',      'fused')
        mo_t = get('CMU-MOSEI', tgt);   mo_f = get('CMU-MOSEI', 'fused')
        def d(a,b): return a-b if not (np.isnan(a) or np.isnan(b)) else float('nan')
        def s(x):
            if np.isnan(x): return '  N/A  '
            return f'{x:{fmt}}'
        def sd(x):
            if np.isnan(x): return '  N/A '
            return f'{x:+{fmt}}'
        print(f'  {sname:<20} {metric:<10}'
              f'{s(ie_t):>10}{s(ie_f):>10}{sd(d(ie_t,ie_f)):>8}'
              f'{s(ml_t):>10}{s(ml_f):>10}{sd(d(ml_t,ml_f)):>8}'
              f'{s(mo_t):>10}{s(mo_f):>10}{sd(d(mo_t,mo_f)):>8}')

    print('\n  IE=IEMOCAP  ML=MELD  MO=CMU-MOSEI')
    print('  Tgt=target factor probe  Fuse=unfactorized fused embedding probe')
    print('  Δ = Tgt - Fuse  (positive = factorization helps)')

    # Save JSON
    def convert(obj):
        if isinstance(obj,(np.float32,np.float64,float)): return float(obj)
        if isinstance(obj,(np.int32,np.int64,int)): return int(obj)
        if isinstance(obj,dict): return {k:convert(v) for k,v in obj.items()}
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    out = f'{BASE}/experiments/results/probe_accuracy_all.json'
    json.dump(convert(all_results), open(out,'w'), indent=2)
    print(f'\n  Saved → {out}')
    print('Done.')
