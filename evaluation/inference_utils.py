"""
Shared inference utilities for evaluation scripts.
Handles model loading for CHADO and all baseline architectures,
and provides standardized dataset inference returning (probs, labels).
"""
import os, sys, yaml, functools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BASE = '/home/tahirahmad/Project_Code/CHADO_EMNLP'
sys.path.insert(0, BASE)

# ── State dict loader ─────────────────────────────────────────────────────────
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

def has_chado_components(sd):
    """Returns True if checkpoint was trained with causal/hyperbolic/OT."""
    return any('causal' in k or 'hyperbolic' in k or 'poincare' in k
               for k in sd.keys())

# ── Config path resolver ──────────────────────────────────────────────────────
_CONFIGS = {
    'iemocap': {
        'chado':    f'{BASE}/configs/iemocap/chado_iemocap.yaml',
        'mult':     f'{BASE}/configs/iemocap/mult_iemocap.yaml',
        'mmdfn':    f'{BASE}/configs/iemocap/mmdfn_iemocap.yaml',
        'ctnet':    f'{BASE}/configs/iemocap/ctnet_iemocap.yaml',
        'lflstm':   f'{BASE}/configs/iemocap/lflstm_iemocap.yaml',
        'bpmult':   f'{BASE}/configs/iemocap/bpmult_iemocap.yaml',
        'emoclip':  f'{BASE}/configs/iemocap/emoclip_iemocap.yaml',
        'ovmer':    f'{BASE}/configs/iemocap/ovmer_iemocap.yaml',
        'unimse':   f'{BASE}/configs/iemocap/unimse_iemocap.yaml',
        'aerllm':   f'{BASE}/configs/iemocap/aerllm_iemocap.yaml',
        'mesca':    f'{BASE}/configs/iemocap/mesca_iemocap.yaml',
        'mminter':  f'{BASE}/configs/iemocap/mminter_iemocap.yaml',
        'baseline': f'{BASE}/configs/iemocap/chado_iemocap.yaml',
    },
    'meld': {
        'chado':    f'{BASE}/configs/meld/chado_meld.yaml',
        'mult':     f'{BASE}/configs/meld/mult_meld.yaml',
        'mmdfn':    f'{BASE}/configs/meld/mmdfn_meld.yaml',
        'ctnet':    f'{BASE}/configs/meld/ctnet_meld.yaml',
        'lflstm':   f'{BASE}/configs/meld/lflstm_meld.yaml',
        'bpmult':   f'{BASE}/configs/meld/bpmult_meld.yaml',
        'emoclip':  f'{BASE}/configs/meld/emoclip_meld.yaml',
        'ovmer':    f'{BASE}/configs/meld/ovmer_meld.yaml',
        'unimse':   f'{BASE}/configs/meld/unimse_meld.yaml',
        'aerllm':   f'{BASE}/configs/meld/aerllm_meld.yaml',
        'mesca':    f'{BASE}/configs/meld/mesca_meld.yaml',
        'mminter':  f'{BASE}/configs/meld/mminter_meld.yaml',
        'baseline': f'{BASE}/configs/meld/chado_meld.yaml',
    },
    'mosei': {
        'chado':    f'{BASE}/configs/mosei/chado_mosei.yaml',
        'mult':     f'{BASE}/configs/mosei/mult_mosei.yaml',
        'mmdfn':    f'{BASE}/configs/mosei/mmdfn_mosei.yaml',
        'ctnet':    f'{BASE}/configs/mosei/ctnet_mosei.yaml',
        'lflstm':   f'{BASE}/configs/mosei/lflstm_mosei.yaml',
        'bpmult':   f'{BASE}/configs/mosei/bpmult_mosei.yaml',
        'emoclip':  f'{BASE}/configs/mosei/emoclip_mosei.yaml',
        'ovmer':    f'{BASE}/configs/mosei/ovmer_mosei.yaml',
        'unimse':   f'{BASE}/configs/mosei/unimse_mosei.yaml',
        'aerllm':   f'{BASE}/configs/mosei/aerllm_mosei.yaml',
        'mesca':    f'{BASE}/configs/mosei/mesca_mosei.yaml',
        'mminter':  f'{BASE}/configs/mosei/mminter_mosei.yaml',
        'baseline': f'{BASE}/configs/mosei/chado_mosei.yaml',
    }
}

def get_config(dataset, model_key):
    ds_cfgs = _CONFIGS.get(dataset, {})
    cfg_path = ds_cfgs.get(model_key)
    # Fallback: try the chado config (shares data section)
    if cfg_path is None or not os.path.exists(cfg_path):
        cfg_path = ds_cfgs.get('chado')
    if cfg_path is None or not os.path.exists(cfg_path):
        raise FileNotFoundError(f'No config for {dataset}/{model_key}')
    return yaml.safe_load(open(cfg_path))


# ── Model builder: uses train_baseline.py logic for non-CHADO ─────────────────
def build_model_for_inference(ckpt_path, dataset, model_key, device):
    """
    Build the correct model architecture and load weights.
    - CHADO variants (chado, chado_v2, chado_wo_*): CHADOTrimodal with full components
    - Baselines: use their native architecture via train_baseline.py build_model
    """
    sd = load_sd(ckpt_path)

    is_chado = (model_key.startswith('chado') or has_chado_components(sd))

    cfg = get_config(dataset, model_key)
    mc  = cfg['model']
    dc  = cfg['data']

    if is_chado:
        from models.chado.model import CHADOTrimodal
        text_model = detect_text_model(sd)
        cc = cfg.get('chado', {})
        model = CHADOTrimodal(
            text_model_name=text_model,
            audio_model_name=mc['audio_model_name'],
            video_model_name=mc.get('video_model_name', 'google/vit-base-patch16-224-in21k'),
            num_classes=dc['num_classes'],
            proj_dim=mc.get('proj_dim', mc.get('d_model', 256)),
            dropout=0.0,
            use_text=True,
            use_audio=mc.get('use_audio', True),
            use_video=mc.get('use_video', True),
            use_gated_fusion=mc.get('use_gated_fusion', True),
            backbone=cc.get('backbone', 'ctnet'),
            n_heads=mc.get('n_heads', 4),
            use_causal=cc.get('use_causal', True),
            use_hyperbolic=cc.get('use_hyperbolic', True),
            use_ot=cc.get('use_ot', True),
            use_mad=cc.get('use_mad', True),
            w_causal=cc.get('w_causal', 0.05),
            w_hyperbolic=cc.get('w_hyperbolic', 0.05),
            w_ot=cc.get('w_ot', 0.02),
        )
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model.to(device), text_model, cfg
    else:
        # Use baseline builder
        sys.path.insert(0, f'{BASE}/scripts/train')
        from train_baseline import build_model as _build
        # Override dropout for inference
        mc_copy = dict(mc)
        mc_copy['dropout'] = 0.0
        cfg_copy = dict(cfg)
        cfg_copy['model'] = mc_copy
        # Map inference model_key → train_baseline.py baseline key (some differ)
        _KEY_MAP = {'lflstm': 'lf_lstm', 'bpmult': 'bpmult', 'mmdfn': 'mmdfn',
                    'emoclip': 'emoclip', 'ovmer': 'ovmer', 'aerllm': 'aerllm',
                    'unimse': 'unimse', 'ctnet': 'ctnet', 'mult': 'mult',
                    'mesca': 'mesca', 'mminter': 'mminter'}
        cfg_copy['baseline'] = _KEY_MAP.get(model_key, model_key)
        model = _build(cfg_copy)
        model.load_state_dict(sd, strict=False)
        model.eval()
        text_model = mc.get('text_model_name', 'roberta-base')
        return model.to(device), text_model, cfg


# ── Forward pass: handles both CHADOTrimodal and baseline model APIs ───────────
def forward_batch(model, batch_data, device, n_frames=8, frame_size=224,
                  use_audio=True, use_video=True):
    """
    Unified forward pass returning logits.
    batch_data: dict with keys from collate functions.
    """
    import inspect
    sig = inspect.signature(model.forward)
    params = list(sig.parameters.keys())

    # Text inputs
    if 'input_ids' in batch_data:
        ti = {'input_ids':      batch_data['input_ids'].to(device),
              'attention_mask': batch_data['attention_mask'].to(device)}
    elif 'text_input' in batch_data:
        ti = {k: v.to(device) for k, v in batch_data['text_input'].items()}
    else:
        ti = None

    # Audio
    aw = batch_data.get('wav')
    if aw is None: aw = batch_data.get('audio_wave')
    if aw is None: aw = batch_data.get('audio')
    if aw is not None and use_audio: aw = aw.to(device)
    else: aw = None

    # Video — check multiple possible keys
    vf = None
    for key in ('video_frames', 'video', 'pixel_values'):
        val = batch_data.get(key)
        if val is not None:
            vf = val
            break
    if vf is not None and use_video:
        vf = vf.to(device)
    elif use_video:
        B = ti['input_ids'].shape[0] if ti else 1
        vf = torch.zeros(B, n_frames, 3, frame_size, frame_size, device=device)
    else:
        vf = None

    # Try CHADOTrimodal API first
    if 'text_input' in params or 'audio_wave' in params:
        out = model(text_input=ti, audio_wave=aw, video_frames=vf)
        logits = out[0] if isinstance(out, (list, tuple)) else out
    elif hasattr(model, 'forward'):
        # Try common baseline APIs
        try:
            out = model(text_input=ti, audio_wave=aw, video_frames=vf)
            logits = out[0] if isinstance(out, (list, tuple)) else out
        except TypeError:
            try:
                out = model(text_inputs=ti, audio=aw, video=vf)
                logits = out[0] if isinstance(out, (list, tuple)) else out
            except TypeError:
                out = model(ti, aw, vf)
                logits = out[0] if isinstance(out, (list, tuple)) else out
    return logits


# ── IEMOCAP dataset inference ─────────────────────────────────────────────────
IEMOCAP_LABEL2ID = {"neu": 0, "hap": 1, "ang": 2, "sad": 3}

def infer_iemocap(ckpt_path, dataset, model_key, device, csv_path=None):
    from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap

    model, text_model, cfg = build_model_for_inference(ckpt_path, dataset, model_key, device)
    dc = cfg['data']
    csv_path = csv_path or dc['test_csv']
    df = pd.read_csv(csv_path)
    lbl_col = 'label' if 'label' in df.columns else 'label_4'
    labels = np.array([IEMOCAP_LABEL2ID.get(str(v).strip(), 0)
                       if isinstance(v, str) else int(v) for v in df[lbl_col].values])

    ds = IEMOCAPDataset(
        csv_path=csv_path, text_model_name=text_model,
        max_text_len=dc.get('max_text_len', 96),
        audio_sr=dc.get('sample_rate', 16000),
        audio_sec=dc.get('max_audio_seconds', 4.0),
        n_frames=dc.get('num_frames', 8),
        use_audio=True, use_video=True,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                        collate_fn=collate_iemocap)

    n_frames   = dc.get('num_frames', 8)
    frame_size = dc.get('frame_size', 224)
    all_probs  = []
    with torch.no_grad():
        for batch in loader:
            logits = forward_batch(model, batch, device, n_frames, frame_size)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_probs), labels


# ── MELD dataset inference ────────────────────────────────────────────────────
def infer_meld(ckpt_path, dataset, model_key, device):
    from datasets.meld.meld_dataset import MeldDataset, collate_meld, \
        build_label_map_from_order, EMO_ORDER_7
    from transformers import AutoTokenizer

    model, text_model, cfg = build_model_for_inference(ckpt_path, dataset, model_key, device)
    dc = cfg['data']
    tok       = AutoTokenizer.from_pretrained(text_model, use_fast=True)
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
        context_turns=dc.get('context_turns', 3),  # default=3 matches MeldDataset default
    )
    collate_fn = functools.partial(collate_meld, tokenizer=tok,
                                   use_text=True, use_audio=True, use_video=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                        collate_fn=collate_fn)

    n_frames   = dc.get('num_frames', 8)
    frame_size = dc.get('frame_size', 224)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            # MeldDataset returns a namedtuple/object; convert to dict
            batch_dict = _meld_batch_to_dict(batch)
            logits = forward_batch(model, batch_dict, device, n_frames, frame_size)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(batch.labels.tolist())
    return np.vstack(all_probs), np.array(all_labels)


def _meld_batch_to_dict(batch):
    d = {}
    if hasattr(batch, 'text_input') and batch.text_input is not None:
        d.update(batch.text_input)
        d['text_input'] = batch.text_input
    if hasattr(batch, 'audio_wave') and batch.audio_wave is not None:
        d['audio_wave'] = batch.audio_wave
    if hasattr(batch, 'video_frames') and batch.video_frames is not None:
        d['video_frames'] = batch.video_frames
    # labels
    if hasattr(batch, 'labels'):
        d['labels'] = batch.labels
    elif hasattr(batch, 'label'):
        d['labels'] = batch.label
    return d


# ── CMU-MOSEI inference ───────────────────────────────────────────────────────
def build_mosei_model_for_inference(ckpt_path, model_key, device):
    """Build CHADOFeature or baseline model for CMU-MOSEI (feature-based, not raw AV)."""
    import yaml
    sd  = load_sd(ckpt_path)
    cfg_map = {
        'chado': f'{BASE}/configs/mosei/chado_mosei.yaml',
        'mult':  f'{BASE}/configs/mosei/mult_mosei.yaml',
        'mmdfn': f'{BASE}/configs/mosei/mmdfn_mosei.yaml',
        'ctnet': f'{BASE}/configs/mosei/ctnet_mosei.yaml',
        'lflstm': f'{BASE}/configs/mosei/lflstm_mosei.yaml',
        'bpmult': f'{BASE}/configs/mosei/bpmult_mosei.yaml',
        'emoclip': f'{BASE}/configs/mosei/emoclip_mosei.yaml',
        'baseline': f'{BASE}/configs/mosei/chado_mosei.yaml',
    }
    cfg_path = cfg_map.get(model_key) or cfg_map.get('chado')
    cfg = yaml.safe_load(open(cfg_path))
    mc  = cfg['model']
    dc  = cfg['data']

    is_chado = model_key.startswith('chado') or has_chado_components(sd)
    if is_chado:
        from models.chado.model import CHADOFeature
        cc = cfg.get('chado', {})
        model = CHADOFeature(
            num_classes=dc['num_classes'],
            d_model=mc.get('d_model', 256),
            use_audio=mc.get('use_audio', True),
            use_video=mc.get('use_video', True),
            text_model=mc.get('text_model_name', 'roberta-base'),
            modality_dropout=0.0,
            use_causal=cc.get('use_causal', True),
            use_hyperbolic=cc.get('use_hyperbolic', True),
            use_ot=cc.get('use_ot', True),
            use_mad=cc.get('use_mad', True),
        )
    else:
        # Baseline: try to import from train script
        try:
            import importlib
            baseline_mod = f'scripts.train.train_baseline_mosei'
            mod = importlib.import_module(baseline_mod)
            model = mod.build_model(cfg)
        except Exception:
            # Fallback: use BaselineFusion directly
            from models.fusion.mosei_fusion import BaselineFusion
            model = BaselineFusion(
                num_classes=dc['num_classes'],
                d_model=mc.get('d_model', 256),
                use_audio=mc.get('use_audio', True),
                use_video=mc.get('use_video', True),
                text_model=mc.get('text_model_name', 'roberta-base'),
            )
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model.to(device), cfg


def infer_mosei(ckpt_path, dataset, model_key, device):
    from datasets.mosei.mosei_utt_dataset import MoseiUttDataset, collate_mosei_utt

    model, cfg = build_mosei_model_for_inference(ckpt_path, model_key, device)
    dc = cfg['data']

    ds = MoseiUttDataset(
        manifest_path=dc['test_manifest'],
        max_text_len=dc.get('max_text_len', 96),
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=collate_mosei_utt)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            b = {
                'text':  batch['text'],
                'audio': batch['audio'].to(device),
                'video': batch['video'].to(device),
            }
            out = model(b)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            sig = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(sig)
            all_labels.extend(batch['label'].tolist())
    return np.vstack(all_probs), np.array(all_labels)
