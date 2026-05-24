"""
Ambiguity-Stratified Performance Analysis.
Splits test samples into Low / Medium / High ambiguity terciles.

"""
import os, sys, json
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '')
BASE    = ''
RES     = f'{BASE}/experiments/results'
OUT_FIG = f'{BASE}/experiments/figures/stratified'
OUT_JSON= f'{BASE}/experiments/results/stratified/stratified_results.json'
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

from evaluation.inference_utils import infer_iemocap, infer_meld

C = {'bg': '#F7F9FB', 'grid': '#DDEAF7',
     'methods': {'CHADO': '#2E4057', 'MulT': '#048A81', 'MM-DFN': '#C25B5B',
                 'CTNet': '#E07B3F', 'LF-LSTM': '#6B8CAE', 'BP-MulT': '#8B5CF6',
                 'Baseline': '#9CA3AF'}}
matplotlib.rc('font', family='sans-serif', size=11)
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

MODELS = [
    ('CHADO',    'chado'),
    ('MulT',     'mult'),
    ('MM-DFN',   'mmdfn'),
    ('CTNet',    'ctnet'),
    ('LF-LSTM',  'lflstm'),
    ('BP-MulT',  'bpmult'),
]

def stratified_metrics(preds, labels, amb, n_strata=3):
    order  = np.argsort(amb)
    splits = np.array_split(order, n_strata)
    names  = ['Low', 'Medium', 'High']
    out    = {}
    for name, idx in zip(names, splits):
        p_s, l_s = preds[idx], labels[idx]
        out[name] = {
            'acc':         round(float(accuracy_score(l_s, p_s)), 4),
            'macro_f1':    round(float(f1_score(l_s, p_s, average='macro', zero_division=0)), 4),
            'weighted_f1': round(float(f1_score(l_s, p_s, average='weighted', zero_division=0)), 4),
            'n': int(len(idx)),
            'mean_amb': round(float(amb[idx].mean()), 4),
        }
    return out

def run_stratified(dataset, infer_fn, amb_scores, device):
    print(f'\n{"="*60}  {dataset.upper()}  (N={len(amb_scores)})  {"="*5}')
    ds_results = {}
    for display_name, model_key in MODELS:
        ckpt = f'{RES}/{dataset}/{model_key}/best.pt'
        if not os.path.exists(ckpt): continue
        print(f'  [{display_name}]...', end='', flush=True)
        try:
            probs, labels = infer_fn(ckpt, dataset, model_key, device)
            print(f' {len(labels)} samples')
        except Exception as e:
            print(f' ERROR: {e}'); continue
        N = min(len(labels), len(amb_scores))
        strata = stratified_metrics(probs[:N].argmax(1), labels[:N], amb_scores[:N])
        ds_results[display_name] = strata
        for sname, m in strata.items():
            print(f'    {sname:6s}(N={m["n"]:4d}, amb={m["mean_amb"]:.3f}): '
                  f'acc={m["acc"]:.4f}  f1={m["macro_f1"]:.4f}')
    return ds_results

def plot_bars(ds_results, dataset, save_prefix):
    strata  = ['Low', 'Medium', 'High']
    methods = list(ds_results.keys())
    if not methods: return
    x     = np.arange(len(strata))
    width = 0.8 / len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(C['bg'])
    for ax_i, metric in enumerate(['acc', 'macro_f1']):
        ax = axes[ax_i]
        for i, method in enumerate(methods):
            vals = [ds_results[method].get(s,{}).get(metric, np.nan) for s in strata]
            color = C['methods'].get(method, '#9CA3AF')
            ax.bar(x + i*width - 0.4 + width/2, vals, width*0.9,
                   label=method, color=color, alpha=0.85,
                   edgecolor='#1a2740' if method=='CHADO' else 'white',
                   linewidth=2.0 if method=='CHADO' else 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(['Low\nAmbiguity','Medium\nAmbiguity','High\nAmbiguity'], fontsize=10)
        ax.set_ylabel('Accuracy' if metric=='acc' else 'Macro-F1', fontsize=11)
        ax.set_title(f'{dataset.upper()} — {"Accuracy" if metric=="acc" else "Macro-F1"} by Ambiguity',
                     fontweight='bold', fontsize=11)
        ax.set_facecolor(C['bg']); ax.grid(axis='y', color=C['grid'], lw=0.8)
        ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_bars.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_prefix}_bars.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {save_prefix}_bars.png')

def plot_drop(ds_results, dataset, save_prefix):
    strata  = ['Low', 'Medium', 'High']
    methods = list(ds_results.keys())
    if not methods: return
    fig, ax = plt.subplots(figsize=(8,5))
    fig.patch.set_facecolor(C['bg']); ax.set_facecolor(C['bg'])
    for method in methods:
        vals = [ds_results[method].get(s,{}).get('macro_f1', np.nan) for s in strata]
        color = C['methods'].get(method, '#9CA3AF')
        ax.plot(strata, vals, 'o-', color=color,
                lw=2.5 if method=='CHADO' else 1.5,
                ms=8 if method=='CHADO' else 6,
                zorder=5 if method=='CHADO' else 3, label=method)
    ax.set_xlabel('Ambiguity Stratum', fontsize=11)
    ax.set_ylabel('Macro-F1', fontsize=11)
    ax.set_title(f'{dataset.upper()} — Performance Degradation vs. Ambiguity',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, color=C['grid'], lw=0.8)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_drop.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_prefix}_drop.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_heatmap(all_results, save_path):
    datasets    = list(all_results.keys())
    strata      = ['Low', 'Medium', 'High']
    all_methods = []
    for dr in all_results.values():
        for m in dr:
            if m not in all_methods: all_methods.append(m)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 5))
    if len(datasets) == 1: axes = [axes]
    for ax, ds in zip(axes, datasets):
        dr  = all_results[ds]
        mat = np.array([[dr.get(m,{}).get(s,{}).get('macro_f1', np.nan)
                         for s in strata] for m in all_methods])
        im = ax.imshow(mat, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(strata))); ax.set_xticklabels(strata, fontsize=10)
        ax.set_yticks(range(len(all_methods))); ax.set_yticklabels(all_methods, fontsize=10)
        ax.set_title(ds, fontweight='bold', fontsize=12)
        plt.colorbar(im, ax=ax, label='Macro-F1')
        for i in range(len(all_methods)):
            for j in range(len(strata)):
                v = mat[i,j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                            fontsize=8, color='white' if v>0.5 else 'black')
    plt.suptitle('Macro-F1 by Ambiguity Stratum', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved heatmap → {save_path}.png')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    data = np.load(f'{BASE}/experiments/results/table3_samples.npz', allow_pickle=True)
    print('NPZ keys:', list(data.keys()))
    all_results = {}

    if 'iemocap_amb' in data:
        amb = data['iemocap_amb']
        res = run_stratified('iemocap', infer_iemocap, amb, device)
        all_results['IEMOCAP'] = res
        plot_bars(res, 'IEMOCAP', f'{OUT_FIG}/iemocap')
        plot_drop(res, 'IEMOCAP', f'{OUT_FIG}/iemocap')

    if 'meld_amb' in data:
        amb = data['meld_amb']
        res = run_stratified('meld', infer_meld, amb, device)
        all_results['MELD'] = res
        plot_bars(res, 'MELD', f'{OUT_FIG}/meld')
        plot_drop(res, 'MELD', f'{OUT_FIG}/meld')

    if all_results:
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
