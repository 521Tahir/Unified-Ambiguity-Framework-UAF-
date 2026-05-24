
import os, sys, json, yaml, functools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '')
BASE    = ''
RES     = f'{BASE}/experiments/results'
OUT_FIG = f'{BASE}/experiments/figures/cross_dataset'
OUT_JSON= f'{BASE}/experiments/results/cross_dataset/cross_dataset_results.json'
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
C = {'bg': '#F7F9FB', 'grid': '#DDEAF7',
     'in_domain': '#2E4057', 'cross': '#C25B5B', 'accent': '#E07B3F',
     'methods': {'CHADO': '#2E4057', 'MulT': '#048A81', 'CTNet': '#E07B3F',
                 'LF-LSTM': '#6B8CAE', 'Baseline': '#9CA3AF'}}
matplotlib.rc('font', family='sans-serif', size=11)
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

# ── Shared emotion mapping IEMOCAP ↔ MELD ────────────────────────────────────
# IEMOCAP labels: neu=0, hap=1, ang=2, sad=3
# MELD labels:    neutral=0, surprise=1, fear=2, sadness=3, joy=4, disgust=5, anger=6
# But our MELD uses EMO_ORDER_7; check actual mapping
# We align: neutral, happy/joy, anger/angry, sad/sadness (4 shared classes)

# IEMOCAP id -> canonical name
IEMOCAP_ID2NAME = {0: 'neutral', 1: 'happy', 2: 'anger', 3: 'sadness'}
# MELD EMO_ORDER_7 mapping (from datasets/meld/meld_dataset.py)
# EMO_ORDER_7 = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
MELD_NAME2ID = {'neutral': 0, 'surprise': 1, 'fear': 2,
                'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}

# Cross-dataset mapping: IEMOCAP class → MELD class (canonical)
IEMOCAP_TO_MELD = {
    0: 0,  # neutral → neutral
    1: 4,  # happy   → joy
    2: 6,  # anger   → anger
    3: 3,  # sadness → sadness
}
# MELD class → IEMOCAP class (for MELD→IEMOCAP direction; only shared classes)
MELD_SHARED = {0, 3, 4, 6}  # neutral, sadness, joy, anger
MELD_TO_IEMOCAP = {0: 0, 4: 1, 6: 2, 3: 3}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_sd(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        return ckpt['model']
    for key in ('model_state_dict', 'state_dict'):
        if key in ckpt: return ckpt[key]
    return ckpt

def detect_text_model(sd):
    for k, v in sd.items():
        if 'text_enc' in k and 'embeddings.LayerNorm.weight' in k:
            return 'roberta-large' if v.shape[0] == 1024 else 'roberta-base'
    return 'roberta-base'


# ── Model loader: uses inference_utils for correct architecture ───────────────
def load_iemocap_model(ckpt_path, device, model_key='chado'):
    from evaluation.inference_utils import build_model_for_inference
    model, text_model, cfg = build_model_for_inference(ckpt_path, 'iemocap', model_key, device)
    return model, text_model


def load_meld_model(ckpt_path, device, model_key='chado'):
    from evaluation.inference_utils import build_model_for_inference
    model, text_model, cfg = build_model_for_inference(ckpt_path, 'meld', model_key, device)
    return model, text_model


# ── IEMOCAP inference (returns probs over 4 classes) ─────────────────────────
def infer_iemocap_model(model, text_model, device, csv_path=None):
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap
    cfg = yaml.safe_load(open(f'{BASE}/configs/iemocap/chado_iemocap.yaml'))
    dc = cfg['data']
    csv_path = csv_path or dc['test_csv']
    LABEL2ID = {"neu": 0, "hap": 1, "ang": 2, "sad": 3}
    df = pd.read_csv(csv_path)
    lbl_col = 'label' if 'label' in df.columns else 'label_4'
    labels = np.array([LABEL2ID.get(str(v).strip(), 0) for v in df[lbl_col].values])

    ds = IEMOCAPDataset(
        csv_path=csv_path, text_model_name=text_model,
        max_text_len=dc.get('max_text_len', 96),
        audio_sr=dc.get('sample_rate', 16000),
        audio_sec=dc.get('max_audio_seconds', 4.0),
        n_frames=dc.get('num_frames', 8),
        use_audio=True, use_video=True,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                        collate_fn=collate_iemocap)
    B_nf, B_fs = dc.get('num_frames', 8), dc.get('frame_size', 224)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            B = batch['input_ids'].shape[0]
            ti = {'input_ids': batch['input_ids'].to(device),
                  'attention_mask': batch['attention_mask'].to(device)}
            aw = batch.get('wav')
            if aw is None: aw = batch.get('audio_wave')
            if aw is not None: aw = aw.to(device)
            # Use real video frames from batch (pixel_values key from IEMOCAPDataset)
            vf = None
            for key in ('video_frames', 'pixel_values', 'video'):
                val = batch.get(key)
                if val is not None:
                    vf = val.to(device)
                    break
            if vf is None:
                vf = torch.zeros(B, B_nf, 3, B_fs, B_fs, device=device)
            out = model(text_input=ti, audio_wave=aw, video_frames=vf)
            all_probs.append(torch.softmax(out[0], dim=-1).cpu().numpy())
    return np.vstack(all_probs), labels


def infer_meld_model(model, text_model, device):
    from datasets.meld.meld_dataset import MeldDataset, collate_meld, \
        build_label_map_from_order, EMO_ORDER_7
    from transformers import AutoTokenizer
    cfg = yaml.safe_load(open(f'{BASE}/configs/meld/chado_meld.yaml'))
    dc = cfg['data']
    tok = AutoTokenizer.from_pretrained(text_model, use_fast=True)
    label_map = build_label_map_from_order(EMO_ORDER_7)
    ds = MeldDataset(
        csv_path=dc['test_csv'], text_model_name=text_model,
        label_map=label_map, text_col=dc['text_col'], label_col=dc['label_col'],
        audio_path_col=dc['audio_path_col'], video_path_col=dc['video_path_col'],
        utt_id_col=dc.get('utt_id_col', 'utt_id'),
        num_frames=dc.get('num_frames', 8), frame_size=dc.get('frame_size', 224),
        sample_rate=dc.get('sample_rate', 16000),
        max_audio_seconds=dc.get('max_audio_seconds', 6.0),
        use_text=True, use_audio=True, use_video=True,
        context_turns=dc.get('context_turns', 5),
    )
    collate_fn = functools.partial(collate_meld, tokenizer=tok,
                                   use_text=True, use_audio=True, use_video=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                        collate_fn=collate_fn)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ti = {k: v.to(device) for k, v in batch.text_input.items()}
            aw = batch.audio_wave.to(device) if batch.audio_wave is not None else None
            vf = batch.video_frames.to(device) if batch.video_frames is not None else None
            out = model(text_input=ti, audio_wave=aw, video_frames=vf)
            all_probs.append(torch.softmax(out[0], dim=-1).cpu().numpy())
            all_labels.extend(batch.labels.tolist())
    return np.vstack(all_probs), np.array(all_labels)


# ── Cross-dataset evaluation ──────────────────────────────────────────────────
def eval_iemocap_on_meld(iemocap_probs, meld_labels):
    """
    Apply IEMOCAP-trained model on MELD shared classes.
    iemocap_probs: (N,4) softmax over [neutral, happy, anger, sadness]
    meld_labels: (N,) MELD labels (0-6), filter to shared only
    Map IEMOCAP pred space → MELD: neutral→0, happy→4, anger→6, sadness→3
    """
    shared_mask = np.array([l in MELD_SHARED for l in meld_labels])
    if shared_mask.sum() == 0:
        return {}
    sub_probs  = iemocap_probs[shared_mask]      # (M, 4)
    sub_labels = meld_labels[shared_mask]         # MELD ids
    # Map MELD labels back to IEMOCAP space for evaluation
    sub_labels_mapped = np.array([MELD_TO_IEMOCAP[l] for l in sub_labels])
    # Predictions: argmax over 4 IEMOCAP classes
    preds = sub_probs.argmax(1)
    acc  = accuracy_score(sub_labels_mapped, preds)
    mf1  = f1_score(sub_labels_mapped, preds, average='macro', zero_division=0)
    wf1  = f1_score(sub_labels_mapped, preds, average='weighted', zero_division=0)
    return {'acc': round(float(acc), 4), 'macro_f1': round(float(mf1), 4),
            'weighted_f1': round(float(wf1), 4),
            'n_shared': int(shared_mask.sum()), 'n_total': int(len(meld_labels))}


def eval_meld_on_iemocap(meld_probs, iemocap_labels):
    """
    Apply MELD-trained model on IEMOCAP.
    meld_probs: (N,7) softmax over MELD EMO_ORDER_7
    iemocap_labels: (N,) IEMOCAP labels 0-3
    Map MELD predictions → IEMOCAP space for shared classes
    """
    # For each IEMOCAP sample, pick the MELD column corresponding to shared class
    # Map iemocap label → meld target column
    ie_to_meld_col = {0: 0, 1: 4, 2: 6, 3: 3}  # neutral, joy, anger, sadness
    # Filter to valid labels
    valid = np.array([l in ie_to_meld_col for l in iemocap_labels])
    if valid.sum() == 0:
        return {}
    sub_probs  = meld_probs[valid]
    sub_labels = iemocap_labels[valid]
    # Restrict MELD probs to shared columns, renormalize
    shared_cols = [0, 4, 6, 3]  # MELD col indices for neutral, joy, anger, sadness
    shared_probs = sub_probs[:, shared_cols]
    shared_probs = shared_probs / (shared_probs.sum(1, keepdims=True) + 1e-10)
    # Map shared_cols indices → iemocap label ids
    col_to_ie = {0: 0, 4: 1, 6: 2, 3: 3}
    preds_shared_idx = shared_probs.argmax(1)  # index into shared_cols
    preds_ie = np.array([col_to_ie[shared_cols[i]] for i in preds_shared_idx])
    acc  = accuracy_score(sub_labels, preds_ie)
    mf1  = f1_score(sub_labels, preds_ie, average='macro', zero_division=0)
    wf1  = f1_score(sub_labels, preds_ie, average='weighted', zero_division=0)
    return {'acc': round(float(acc), 4), 'macro_f1': round(float(mf1), 4),
            'weighted_f1': round(float(wf1), 4),
            'n_valid': int(valid.sum()), 'n_total': int(len(iemocap_labels))}


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_cross_dataset_summary(results, save_prefix):
    """Bar chart comparing in-domain vs cross-dataset performance."""
    models = list(results.keys())
    metrics = ['acc', 'macro_f1']
    labels  = ['Accuracy', 'Macro-F1']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(C['bg'])

    for ax, metric, label in zip(axes, metrics, labels):
        x = np.arange(len(models))
        width = 0.35

        in_domain  = [results[m].get('in_domain', {}).get(metric, 0) for m in models]
        cross_d    = [results[m].get('cross_domain', {}).get(metric, 0) for m in models]

        bars1 = ax.bar(x - width/2, in_domain, width, label='In-Domain',
                       color=C['in_domain'], alpha=0.85, edgecolor='white')
        bars2 = ax.bar(x + width/2, cross_d, width, label='Cross-Domain (shared cls)',
                       color=C['cross'], alpha=0.85, edgecolor='white')

        # Drop arrows
        for xi, (ind, crd) in enumerate(zip(in_domain, cross_d)):
            if ind > 0 and crd > 0:
                drop = ind - crd
                ax.annotate('', xy=(xi + width/2, crd + 0.005),
                            xytext=(xi - width/2, ind - 0.005),
                            arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))
                ax.text(xi + 0.02, (ind + crd) / 2, f'-{drop:.2f}',
                        fontsize=8, color='#666', va='center')

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'In-Domain vs Cross-Domain {label}', fontweight='bold', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_facecolor(C['bg'])
        ax.grid(axis='y', color=C['grid'], lw=0.8)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_cross_domain.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_prefix}_cross_domain.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved cross-domain plot → {save_prefix}_cross_domain.png')


def plot_generalization_gap(results_iem2meld, results_meld2iem, save_path):
    """Side-by-side: transfer direction comparison."""
    methods = list(set(list(results_iem2meld.keys()) + list(results_meld2iem.keys())))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(C['bg'])

    for ax, (direction, res) in zip(axes, [
        ('IEMOCAP → MELD (shared classes)', results_iem2meld),
        ('MELD → IEMOCAP (shared classes)', results_meld2iem)
    ]):
        methods_present = [m for m in methods if m in res]
        f1_in   = [res[m].get('in_domain_f1', 0)    for m in methods_present]
        f1_cross= [res[m].get('cross_domain_f1', 0)  for m in methods_present]
        colors  = [C['methods'].get(m, '#9CA3AF') for m in methods_present]
        x = np.arange(len(methods_present))

        ax.scatter(f1_in, f1_cross, c=colors, s=100, zorder=5, edgecolors='#333', lw=1)
        for i, m in enumerate(methods_present):
            ax.annotate(m, (f1_in[i], f1_cross[i]),
                        textcoords='offset points', xytext=(5, 3), fontsize=8)
        mn = min(min(f1_in), min(f1_cross)) - 0.02
        mx = max(max(f1_in), max(f1_cross)) + 0.02
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1, alpha=0.5, label='y=x (no gap)')
        ax.set_xlabel('In-Domain Macro-F1', fontsize=10)
        ax.set_ylabel('Cross-Domain Macro-F1', fontsize=10)
        ax.set_title(direction, fontweight='bold', fontsize=10)
        ax.set_facecolor(C['bg'])
        ax.grid(True, color=C['grid'], lw=0.6)
        ax.legend(fontsize=9)

    fig.suptitle('Generalization Gap: In-Domain vs Cross-Domain', fontsize=12,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved generalization gap → {save_path}.png')


# ── Main ──────────────────────────────────────────────────────────────────────
MODELS_CROSS = [
    ('CHADO',    'chado'),
    ('MulT',     'mult'),
    ('CTNet',    'ctnet'),
    ('LF-LSTM',  'lflstm'),
    ('Baseline', 'baseline'),
]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    all_results  = {}
    iem2meld_res = {}
    meld2iem_res = {}

    print('\n' + '='*65)
    print('  CROSS-DATASET: IEMOCAP → MELD')
    print('='*65)

    for display_name, model_key in MODELS_CROSS:
        ie_ckpt   = f'{RES}/iemocap/{model_key}/best.pt'
        meld_ckpt = f'{RES}/meld/{model_key}/best.pt'

        if not (os.path.exists(ie_ckpt) and os.path.exists(meld_ckpt)):
            print(f'  [{display_name}] missing checkpoint(s), skip')
            continue

        print(f'\n  [{display_name}]')
        try:
            # Load IEMOCAP model
            ie_model, ie_text = load_iemocap_model(ie_ckpt, device, model_key)
            # In-domain: IEMOCAP test
            ie_probs, ie_labels = infer_iemocap_model(ie_model, ie_text, device)
            ie_in_acc = accuracy_score(ie_labels, ie_probs.argmax(1))
            ie_in_f1  = f1_score(ie_labels, ie_probs.argmax(1), average='macro', zero_division=0)
            print(f'    In-domain IEMOCAP: acc={ie_in_acc:.4f}  macro_f1={ie_in_f1:.4f}')

            # Cross-domain: run IEMOCAP model ON MELD test data
            meld_model, meld_text = load_meld_model(meld_ckpt, device, model_key)
            _, meld_labels_full = infer_meld_model(meld_model, meld_text, device)
            ie_probs_on_meld = _infer_iemocap_model_on_meld(ie_model, ie_text, device)
            cross_res = eval_iemocap_on_meld(ie_probs_on_meld, meld_labels_full)
            print(f'    Cross-domain MELD (shared {cross_res.get("n_shared",0)} samples): '
                  f'acc={cross_res.get("acc",0):.4f}  macro_f1={cross_res.get("macro_f1",0):.4f}')

            iem2meld_res[display_name] = {
                'in_domain_acc': round(ie_in_acc, 4),
                'in_domain_f1':  round(ie_in_f1, 4),
                'cross_domain_acc': cross_res.get('acc', 0),
                'cross_domain_f1':  cross_res.get('macro_f1', 0),
                'n_shared': cross_res.get('n_shared', 0),
            }

        except Exception as e:
            print(f'    ERROR: {e}')
            import traceback; traceback.print_exc()

    print('\n' + '='*65)
    print('  CROSS-DATASET: MELD → IEMOCAP')
    print('='*65)

    for display_name, model_key in MODELS_CROSS:
        ie_ckpt   = f'{RES}/iemocap/{model_key}/best.pt'
        meld_ckpt = f'{RES}/meld/{model_key}/best.pt'
        if not (os.path.exists(ie_ckpt) and os.path.exists(meld_ckpt)):
            continue

        print(f'\n  [{display_name}]')
        try:
            # In-domain MELD
            meld_model, meld_text = load_meld_model(meld_ckpt, device, model_key)
            meld_probs, meld_labels = infer_meld_model(meld_model, meld_text, device)
            meld_in_acc = accuracy_score(meld_labels, meld_probs.argmax(1))
            meld_in_f1  = f1_score(meld_labels, meld_probs.argmax(1), average='macro', zero_division=0)
            print(f'    In-domain MELD: acc={meld_in_acc:.4f}  macro_f1={meld_in_f1:.4f}')

            # Cross-domain: MELD model on IEMOCAP test data
            meld_probs_on_ie = _infer_meld_model_on_iemocap(meld_model, meld_text, device)
            ie_model_in_domain, ie_text2 = load_iemocap_model(ie_ckpt, device, model_key)
            _, ie_labels = infer_iemocap_model(ie_model_in_domain, ie_text2, device)
            cross_res = eval_meld_on_iemocap(meld_probs_on_ie, ie_labels)
            print(f'    Cross-domain IEMOCAP ({cross_res.get("n_valid",0)} samples): '
                  f'acc={cross_res.get("acc",0):.4f}  macro_f1={cross_res.get("macro_f1",0):.4f}')

            meld2iem_res[display_name] = {
                'in_domain_acc': round(meld_in_acc, 4),
                'in_domain_f1':  round(meld_in_f1, 4),
                'cross_domain_acc': cross_res.get('acc', 0),
                'cross_domain_f1':  cross_res.get('macro_f1', 0),
                'n_valid': cross_res.get('n_valid', 0),
            }

        except Exception as e:
            print(f'    ERROR: {e}')
            import traceback; traceback.print_exc()

    # ── Save and plot ─────────────────────────────────────────────────────────
    all_results = {'iemocap_to_meld': iem2meld_res, 'meld_to_iemocap': meld2iem_res}

    if iem2meld_res:
        plot_cross_dataset_summary(
            {m: {'in_domain': {'acc': v['in_domain_acc'], 'macro_f1': v['in_domain_f1']},
                 'cross_domain': {'acc': v['cross_domain_acc'], 'macro_f1': v['cross_domain_f1']}}
             for m, v in iem2meld_res.items()},
            f'{OUT_FIG}/iemocap_to_meld'
        )

    if iem2meld_res and meld2iem_res:
        plot_generalization_gap(iem2meld_res, meld2iem_res,
                                f'{OUT_FIG}/generalization_gap')

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('  CROSS-DATASET GENERALIZATION SUMMARY')
    print(f'{"="*70}')
    for direction, res in [('IEMOCAP→MELD', iem2meld_res),
                            ('MELD→IEMOCAP', meld2iem_res)]:
        print(f'\n  {direction}:')
        print(f'  {"Method":<12} {"In-Domain F1":>14} {"Cross-Domain F1":>16} {"Gap":>8}')
        print(f'  {"-"*55}')
        for m, v in res.items():
            gap = v['in_domain_f1'] - v['cross_domain_f1']
            print(f'  {m:<12} {v["in_domain_f1"]:14.4f} '
                  f'{v["cross_domain_f1"]:16.4f} {gap:8.4f}')

    with open(OUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved → {OUT_JSON}')


# ── Aux: run IEMOCAP model on MELD test CSV ────────────────────────────────────
def _infer_iemocap_model_on_meld(model, text_model, device):
    """Run IEMOCAP-trained model (4-class head) on MELD test data."""
    from datasets.meld.meld_dataset import MeldDataset, collate_meld, \
        build_label_map_from_order, EMO_ORDER_7
    from transformers import AutoTokenizer
    cfg = yaml.safe_load(open(f'{BASE}/configs/meld/chado_meld.yaml'))
    dc = cfg['data']
    tok = AutoTokenizer.from_pretrained(text_model, use_fast=True)
    label_map = build_label_map_from_order(EMO_ORDER_7)
    ds = MeldDataset(
        csv_path=dc['test_csv'], text_model_name=text_model,
        label_map=label_map, text_col=dc['text_col'], label_col=dc['label_col'],
        audio_path_col=dc['audio_path_col'], video_path_col=dc['video_path_col'],
        utt_id_col=dc.get('utt_id_col', 'utt_id'),
        num_frames=dc.get('num_frames', 8), frame_size=dc.get('frame_size', 224),
        sample_rate=dc.get('sample_rate', 16000),
        max_audio_seconds=dc.get('max_audio_seconds', 6.0),
        use_text=True, use_audio=True, use_video=True,
        context_turns=dc.get('context_turns', 5),
    )
    collate_fn = functools.partial(collate_meld, tokenizer=tok,
                                   use_text=True, use_audio=True, use_video=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                        collate_fn=collate_fn)
    from evaluation.inference_utils import forward_batch, _meld_batch_to_dict
    n_frames   = dc.get('num_frames', 8)
    frame_size = dc.get('frame_size', 224)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch_dict = _meld_batch_to_dict(batch)
            logits = forward_batch(model, batch_dict, device, n_frames, frame_size)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_probs)


def _infer_meld_model_on_iemocap(model, text_model, device):
    """Run MELD-trained model (7-class head) on IEMOCAP test data."""
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap
    cfg = yaml.safe_load(open(f'{BASE}/configs/iemocap/chado_iemocap.yaml'))
    dc = cfg['data']
    ds = IEMOCAPDataset(
        csv_path=dc['test_csv'], text_model_name=text_model,
        max_text_len=dc.get('max_text_len', 96),
        audio_sr=dc.get('sample_rate', 16000),
        audio_sec=dc.get('max_audio_seconds', 4.0),
        n_frames=dc.get('num_frames', 8),
        use_audio=True, use_video=True,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                        collate_fn=collate_iemocap)
    from evaluation.inference_utils import forward_batch
    n_frames   = dc.get('num_frames', 8)
    frame_size = dc.get('frame_size', 224)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            logits = forward_batch(model, batch, device, n_frames, frame_size)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_probs)


if __name__ == '__main__':
    main()
