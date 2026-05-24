"""
Run calibration for CMU-MOSEI only, appending to existing calibration_results.json.
"""
import os, sys, json, numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

sys.path.insert(0, '')
BASE   = ''
RES    = f'{BASE}/experiments/results'
OUT    = f'{BASE}/experiments/figures/calibration'
JSON_F = f'{RES}/calibration/calibration_results.json'
os.makedirs(OUT, exist_ok=True)

MODELS = [
    ('CHADO',    'chado'),
    ('MulT',     'mult'),
    ('MM-DFN',   'mmdfn'),
    ('CTNet',    'ctnet'),
    ('LF-LSTM',  'lflstm'),
    ('BP-MulT',  'bpmult'),
    ('EmoCLIP',  'emoclip'),
]

def ece_score(probs, labels, n_bins=15):
    """Expected Calibration Error for multi-label (per-class sigmoid outputs)."""
    # For multilabel: treat each class as binary
    n_classes = probs.shape[1]
    ece_vals = []
    for c in range(n_classes):
        p = probs[:, c]
        y = (labels[:, c] > 0).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        ece = 0.0
        for bl, bu in zip(bin_lowers, bin_uppers):
            mask = (p >= bl) & (p < bu)
            if mask.sum() == 0: continue
            acc  = y[mask].mean()
            conf = p[mask].mean()
            ece += mask.mean() * abs(acc - conf)
        ece_vals.append(ece)
    return float(np.mean(ece_vals))


def brier_score(probs, labels):
    return float(np.mean((probs - labels) ** 2))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load existing results
    results = json.load(open(JSON_F)) if os.path.exists(JSON_F) else {}
    if 'CMU-MOSEI' not in results:
        results['CMU-MOSEI'] = {}

    from evaluation.inference_utils import infer_mosei

    print('\n' + '='*60 + '  CMU-MOSEI  =====')
    for name, key in MODELS:
        if name in results['CMU-MOSEI']:
            print(f'  [{name}] already computed, skip')
            continue
        ckpt = f'{RES}/mosei/{key}/best.pt'
        if not os.path.exists(ckpt):
            print(f'  [{name}] no checkpoint, skip')
            continue
        print(f'  [{name}]...', end='', flush=True)
        try:
            probs, labels = infer_mosei(ckpt, 'mosei', key, device)
            # labels is float [N, 6], probs is sigmoid [N, 6]
            labels_bin = (labels > 0).astype(float)
            ece   = ece_score(probs, labels_bin)
            brier = brier_score(probs, labels_bin)
            # Subset accuracy (all labels match)
            preds_bin = (probs >= 0.5).astype(float)
            subset_acc = float(np.mean(np.all(preds_bin == labels_bin, axis=1)))
            print(f' {probs.shape[0]} samples  ECE={ece:.4f}  Brier={brier:.4f}  SubsetAcc={subset_acc:.4f}')
            results['CMU-MOSEI'][name] = {
                'ece': round(ece, 6), 'brier': round(brier, 6),
                'accuracy': round(subset_acc, 6), 'n': probs.shape[0]
            }
        except Exception as e:
            print(f' ERROR: {e}')
            import traceback; traceback.print_exc()

    with open(JSON_F, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {JSON_F}')


if __name__ == '__main__':
    main()
