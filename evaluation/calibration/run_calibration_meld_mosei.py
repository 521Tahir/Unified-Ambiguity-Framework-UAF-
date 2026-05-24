"""Finish calibration: run MELD + CMU-MOSEI, merge with existing IEMOCAP results."""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, '')
BASE    = ''
RES     = f'{BASE}/experiments/results'
OUT_DIR = f'{BASE}/experiments/figures/calibration'
OUT_JSON= f'{BASE}/experiments/results/calibration/calibration_results.json'
os.makedirs(OUT_DIR, exist_ok=True)

from evaluation.calibration.run_calibration import (
    MODELS, run_calibration, plot_reliability, plot_entropy_dist,
    plot_summary, C
)
from evaluation.inference_utils import infer_meld, infer_mosei

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load existing results
    all_results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            all_results = json.load(f)
    print(f'Loaded existing results for: {list(all_results.keys())}')

    # MELD
    meld_res, meld_probs = run_calibration('meld', infer_meld, device, 7)
    all_results['MELD'] = meld_res
    plot_reliability(meld_probs, 'MELD', f'{OUT_DIR}/meld')
    plot_entropy_dist(meld_probs, 'MELD', f'{OUT_DIR}/meld')

    # CMU-MOSEI
    print(f'\n{"="*60}  CMU-MOSEI (multilabel)  {"="*5}')
    mosei_res = {}
    for display_name, model_key in MODELS:
        ckpt = f'{RES}/mosei/{model_key}/best.pt'
        if not os.path.exists(ckpt): continue
        print(f'  [{display_name}]...', end='', flush=True)
        try:
            probs, labels = infer_mosei(ckpt, 'mosei', model_key, device)
            print(f' {len(labels)} samples', end='')
        except Exception as e:
            print(f' ERROR: {e}'); continue
        labels_arr = np.array(labels, dtype=np.float32)
        brier_per = [float(np.mean((probs[:,c] - labels_arr[:,c])**2))
                     for c in range(probs.shape[1])]
        avg_brier = float(np.mean(brier_per))
        em = float(np.mean(-np.sum(np.clip(probs,1e-10,1)*np.log(np.clip(probs,1e-10,1)),axis=1)))
        es = float(np.std(-np.sum(np.clip(probs,1e-10,1)*np.log(np.clip(probs,1e-10,1)),axis=1)))
        print(f'  Brier={avg_brier:.4f}  Entropy={em:.4f}')
        mosei_res[display_name] = {'brier': round(avg_brier,4),
                                    'brier_per_class': [round(v,4) for v in brier_per],
                                    'entropy_mean': round(em,4), 'entropy_std': round(es,4),
                                    'n': int(len(labels)),
                                    'note': 'multilabel sigmoid; ECE not computed'}
    all_results['CMU-MOSEI'] = mosei_res

    plot_summary(all_results, f'{OUT_DIR}/calibration_summary')

    with open(OUT_JSON, 'w') as f: json.dump(all_results, f, indent=2)
    print(f'\nSaved → {OUT_JSON}')

if __name__ == '__main__':
    main()
