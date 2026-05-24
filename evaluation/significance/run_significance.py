
import os, sys, json
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import chi2_contingency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '')
BASE    = ''
RES     = f'{BASE}/experiments/results'
OUT_FIG = f'{BASE}/experiments/figures/significance'
OUT_JSON= f'{BASE}/experiments/results/significance/significance_results.json'
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

from evaluation.inference_utils import infer_iemocap, infer_meld

C = {'bg': '#F7F9FB', 'grid': '#DDEAF7', 'sig': '#2E4057', 'ns': '#9CA3AF',
     'methods': {'CHADO': '#2E4057', 'MulT': '#048A81', 'MM-DFN': '#C25B5B',
                 'CTNet': '#E07B3F', 'LF-LSTM': '#6B8CAE', 'BP-MulT': '#8B5CF6',
                 'EmoCLIP': '#7C3AED', 'UniMSE': '#065F46',
                 'AER-LLM': '#92400E', 'OVMER': '#9D174D', 'Baseline': '#9CA3AF'}}
matplotlib.rc('font', family='sans-serif', size=11)
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

# ── Tests ─────────────────────────────────────────────────────────────────────
def mcnemar_test(preds_a, preds_b, labels):
    correct_a = (preds_a == labels).astype(int)
    correct_b = (preds_b == labels).astype(int)
    b10 = int(np.sum((correct_a==1)&(correct_b==0)))
    b01 = int(np.sum((correct_a==0)&(correct_b==1)))
    table = np.array([[int(np.sum((correct_a==1)&(correct_b==1))), b10],
                       [b01, int(np.sum((correct_a==0)&(correct_b==0)))]])
    try:
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
        result = sm_mcnemar(table, exact=False, correction=True)
        return float(result.statistic), float(result.pvalue)
    except Exception:
        chi2, p, _, _ = chi2_contingency(table + 0.5)
        return float(chi2), float(p)

def bootstrap_ci(preds, labels, metric='macro_f1', n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n   = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if metric == 'macro_f1':
            vals.append(f1_score(labels[idx], preds[idx], average='macro', zero_division=0))
        else:
            vals.append(accuracy_score(labels[idx], preds[idx]))
    vals = np.array(vals)
    alpha = (1 - ci) / 2
    return float(np.mean(vals)), float(np.percentile(vals, alpha*100)), \
           float(np.percentile(vals, (1-alpha)*100))

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

MODELS = [
    ('MulT',     'mult'),
    ('MM-DFN',   'mmdfn'),
    ('CTNet',    'ctnet'),
    ('LF-LSTM',  'lflstm'),
    ('BP-MulT',  'bpmult'),
    ('EmoCLIP',  'emoclip'),
    ('UniMSE',   'unimse'),
    ('AER-LLM',  'aerllm'),
    ('OVMER',    'ovmer'),
    ('Baseline', 'baseline'),
]

def run_significance(dataset, infer_fn, device):
    print(f'\n{"="*65}  {dataset.upper()}')
    chado_ckpt = f'{RES}/{dataset}/chado/best.pt'
    if not os.path.exists(chado_ckpt): return {}

    print('  [CHADO]...', end='', flush=True)
    try:
        probs, labels = infer_fn(chado_ckpt, dataset, 'chado', device)
        print(f' {len(labels)} samples')
    except Exception as e:
        print(f' ERROR: {e}'); return {}

    chado_preds = probs.argmax(1)
    chado_acc   = float(accuracy_score(labels, chado_preds))
    chado_f1    = float(f1_score(labels, chado_preds, average='macro', zero_division=0))
    _, f1_lo, f1_hi = bootstrap_ci(chado_preds, labels, 'macro_f1')
    _, a_lo,  a_hi  = bootstrap_ci(chado_preds, labels, 'acc')

    print(f'  CHADO: acc={chado_acc:.4f}[{a_lo:.4f},{a_hi:.4f}]  '
          f'f1={chado_f1:.4f}[{f1_lo:.4f},{f1_hi:.4f}]')

    results = {'CHADO': {'acc': round(chado_acc,4), 'acc_lo': round(a_lo,4), 'acc_hi': round(a_hi,4),
                          'macro_f1': round(chado_f1,4), 'f1_lo': round(f1_lo,4), 'f1_hi': round(f1_hi,4),
                          'n': int(len(labels))}}

    for display_name, model_key in MODELS:
        ckpt = f'{RES}/{dataset}/{model_key}/best.pt'
        if not os.path.exists(ckpt): continue
        print(f'  [{display_name}]...', end='', flush=True)
        try:
            probs_b, _ = infer_fn(ckpt, dataset, model_key, device)
            print(f' OK', end='')
        except Exception as e:
            print(f' ERROR: {e}'); continue

        preds_b = probs_b.argmax(1)
        acc_b   = float(accuracy_score(labels, preds_b))
        f1_b    = float(f1_score(labels, preds_b, average='macro', zero_division=0))
        mn_chi, mn_p = mcnemar_test(chado_preds, preds_b, labels)
        _, bf1_lo, bf1_hi = bootstrap_ci(preds_b, labels, 'macro_f1')
        _, ba_lo,  ba_hi  = bootstrap_ci(preds_b, labels, 'acc')

        print(f'  acc={acc_b:.4f}  f1={f1_b:.4f}  Δf1={chado_f1-f1_b:+.4f}  '
              f'p={mn_p:.4f}{sig_stars(mn_p)}')
        results[display_name] = {
            'acc': round(acc_b,4), 'acc_lo': round(ba_lo,4), 'acc_hi': round(ba_hi,4),
            'macro_f1': round(f1_b,4), 'f1_lo': round(bf1_lo,4), 'f1_hi': round(bf1_hi,4),
            'mcnemar_chi2': round(mn_chi,4), 'mcnemar_p': round(mn_p,6),
            'mcnemar_sig': sig_stars(mn_p),
            'delta_acc': round(chado_acc-acc_b,4), 'delta_f1': round(chado_f1-f1_b,4),
        }
    return results

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_ci_bars(ds_results, dataset, save_prefix):
    methods = list(ds_results.keys())
    f1s     = [ds_results[m]['macro_f1'] for m in methods]
    lo_err  = [ds_results[m]['macro_f1'] - ds_results[m]['f1_lo'] for m in methods]
    hi_err  = [ds_results[m]['f1_hi'] - ds_results[m]['macro_f1'] for m in methods]
    colors  = [C['methods'].get(m,'#9CA3AF') for m in methods]
    fig, ax = plt.subplots(figsize=(max(8, len(methods)*1.3), 5))
    fig.patch.set_facecolor(C['bg']); ax.set_facecolor(C['bg'])
    x = np.arange(len(methods))
    ax.bar(x, f1s, color=colors, alpha=0.85,
           edgecolor=['#1a2740' if m=='CHADO' else 'white' for m in methods],
           linewidth=[2.0 if m=='CHADO' else 1.0 for m in methods])
    ax.errorbar(x, f1s, yerr=[lo_err,hi_err], fmt='none',
                color='#333', capsize=5, lw=2, capthick=2)
    chado_f1 = ds_results.get('CHADO',{}).get('macro_f1',0)
    for i, m in enumerate(methods):
        if m == 'CHADO': continue
        p = ds_results[m].get('mcnemar_p', 1.0)
        s = sig_stars(p)
        if s != 'ns':
            ax.text(i, f1s[i]+hi_err[i]+0.005, s, ha='center', va='bottom',
                    fontsize=9, color='#2E4057', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Macro-F1 (95% CI)', fontsize=11)
    ax.set_title(f'{dataset.upper()} — Macro-F1 with Bootstrap CIs',
                 fontweight='bold', fontsize=12)
    ax.grid(axis='y', color=C['grid'], lw=0.8)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_ci.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_prefix}_ci.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved CI bars → {save_prefix}_ci.png')

def plot_heatmap(all_results, save_path):
    datasets    = list(all_results.keys())
    all_methods = []
    for dr in all_results.values():
        for m in dr:
            if m != 'CHADO' and m not in all_methods: all_methods.append(m)
    mat = np.ones((len(all_methods), len(datasets)))
    for j, ds in enumerate(datasets):
        for i, m in enumerate(all_methods):
            if m in all_results[ds]:
                mat[i,j] = all_results[ds][m].get('mcnemar_p', 1.0)
    fig, ax = plt.subplots(figsize=(len(datasets)*3+2, len(all_methods)*0.6+2))
    fig.patch.set_facecolor(C['bg'])
    log_mat = -np.log10(np.clip(mat, 1e-10, 1.0))
    im = ax.imshow(log_mat, cmap='Blues', aspect='auto', vmin=0, vmax=4)
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(all_methods))); ax.set_yticklabels(all_methods, fontsize=10)
    for i in range(len(all_methods)):
        for j in range(len(datasets)):
            p = mat[i,j]
            ax.text(j, i, f'{sig_stars(p)}\np={p:.3f}', ha='center', va='center',
                    fontsize=7, color='white' if log_mat[i,j]>2 else 'black')
    plt.colorbar(im, ax=ax, label='-log10(p)', shrink=0.8)
    ax.set_title('McNemar Test: CHADO vs Baselines', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved McNemar heatmap → {save_path}.png')

def plot_delta(all_results, save_path):
    datasets = list(all_results.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets)+1, 5))
    if len(datasets)==1: axes=[axes]
    fig.patch.set_facecolor(C['bg'])
    for ax, ds in zip(axes, datasets):
        dr = all_results[ds]
        methods = [m for m in dr if m != 'CHADO']
        deltas  = [dr[m]['delta_f1'] for m in methods]
        pvals   = [dr[m].get('mcnemar_p',1.0) for m in methods]
        y = np.arange(len(methods))
        for i, (m, d, p) in enumerate(zip(methods, deltas, pvals)):
            color = C['sig'] if p<0.05 else C['ns']
            ax.hlines(y[i], 0, d, color=color, lw=2)
            ax.plot(d, y[i], 'o', color=color, ms=8)
            s = sig_stars(p)
            if s != 'ns':
                ax.text(d+0.001, y[i], s, va='center', fontsize=8,
                        color=color, fontweight='bold')
        ax.axvline(0, color='black', lw=1, ls='--')
        ax.set_yticks(y); ax.set_yticklabels(methods, fontsize=10)
        ax.set_xlabel('CHADO – Baseline (Macro-F1)', fontsize=10)
        ax.set_title(ds, fontweight='bold', fontsize=12)
        ax.set_facecolor(C['bg']); ax.grid(axis='x', color=C['grid'], lw=0.8)
    fig.suptitle('CHADO Performance Gains', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved delta lollipop → {save_path}.png')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    all_results = {}

    iem_res = run_significance('iemocap', infer_iemocap, device)
    if iem_res:
        all_results['IEMOCAP'] = iem_res
        plot_ci_bars(iem_res, 'IEMOCAP', f'{OUT_FIG}/iemocap')

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
        print(f'  {"Method":<12} {"Acc":>7} {"F1":>7} {"ΔF1":>7} {"McNemar p":>12} {"Sig":>5} {"95% CI":>16}')
        print(f'  {"-"*70}')
        for m, v in res.items():
            mc_p  = f'{v.get("mcnemar_p","—"):>12}' if isinstance(v.get("mcnemar_p"), float) else f'{"—":>12}'
            sig   = v.get('mcnemar_sig','—')
            delta = f'{v.get("delta_f1",0):+7.4f}' if 'delta_f1' in v else f'{"—":>7}'
            ci    = f'[{v["f1_lo"]:.4f},{v["f1_hi"]:.4f}]'
            print(f'  {m:<12} {v["acc"]:7.4f} {v["macro_f1"]:7.4f} {delta} {mc_p} {sig:>5} {ci:>16}')

    with open(OUT_JSON,'w') as f: json.dump(all_results, f, indent=2)
    print(f'\nSaved → {OUT_JSON}')

if __name__ == '__main__':
    main()
