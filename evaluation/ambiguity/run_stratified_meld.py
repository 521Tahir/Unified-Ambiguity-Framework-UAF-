"""Finish stratified: run MELD, merge with existing IEMOCAP results."""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, '/')
BASE    = ''
OUT_FIG = f'{BASE}/experiments/figures/stratified'
OUT_JSON= f'{BASE}/experiments/results/stratified/stratified_results.json'
os.makedirs(OUT_FIG, exist_ok=True)

from evaluation.ambiguity.run_stratified import (
    run_stratified, plot_bars, plot_drop, plot_heatmap
)
from evaluation.inference_utils import infer_meld

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    data = np.load(f'{BASE}/experiments/results/table3_samples.npz', allow_pickle=True)

    # Load existing
    all_results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            all_results = json.load(f)
    print(f'Loaded existing results for: {list(all_results.keys())}')

    if 'meld_amb' in data:
        amb = data['meld_amb']
        res = run_stratified('meld', infer_meld, amb, device)
        all_results['MELD'] = res
        plot_bars(res, 'MELD', f'{OUT_FIG}/meld')
        plot_drop(res, 'MELD', f'{OUT_FIG}/meld')

    plot_heatmap(all_results, f'{OUT_FIG}/stratified_heatmap')

    print(f'\n{"="*70}  SUMMARY')
    for ds, res in all_results.items():
        print(f'\n  {ds}:')
        print(f'  {"Method":<12} {"Low F1":>8} {"Med F1":>8} {"Hi F1":>8} {"Drop":>8}')
        print(f'  {"-"*45}')
        for m, strata in res.items():
            lo  = strata.get('Low',{}).get('macro_f1',0)
            med = strata.get('Medium',{}).get('macro_f1',0)
            hi  = strata.get('High',{}).get('macro_f1',0)
            print(f'  {m:<12} {lo:8.4f} {med:8.4f} {hi:8.4f} {lo-hi:8.4f}')

    with open(OUT_JSON,'w') as f: json.dump(all_results, f, indent=2)
    print(f'\nSaved → {OUT_JSON}')

if __name__ == '__main__':
    main()
