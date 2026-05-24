"""Finish significance: run MELD, merge with existing IEMOCAP results."""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, '')
BASE    = ''
OUT_FIG = f'{BASE}/experiments/figures/significance'
OUT_JSON= f'{BASE}/experiments/results/significance/significance_results.json'
os.makedirs(OUT_FIG, exist_ok=True)

from evaluation.significance.run_significance import (
    run_significance, plot_ci_bars, plot_heatmap, plot_delta
)
from evaluation.inference_utils import infer_meld

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load existing
    all_results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            all_results = json.load(f)
    print(f'Loaded existing results for: {list(all_results.keys())}')

    meld_res = run_significance('meld', infer_meld, device)
    if meld_res:
        all_results['MELD'] = meld_res
        plot_ci_bars(meld_res, 'MELD', f'{OUT_FIG}/meld')

    if all_results:
        plot_heatmap(all_results, f'{OUT_FIG}/mcnemar_heatmap')
        plot_delta(all_results, f'{OUT_FIG}/delta_lollipop')

    print(f'\n{"="*80}  SIGNIFICANCE SUMMARY')
    for ds, res in all_results.items():
        print(f'\n  {ds}:')
        for m, v in res.items():
            sig = v.get('mcnemar_sig', '—')
            delta = f'{v.get("delta_f1", 0):+.4f}' if 'delta_f1' in v else '—'
            print(f'  {m:<12} f1={v["macro_f1"]:.4f}  Δ={delta}  {sig}')

    with open(OUT_JSON,'w') as f: json.dump(all_results, f, indent=2)
    print(f'\nSaved → {OUT_JSON}')

if __name__ == '__main__':
    main()
