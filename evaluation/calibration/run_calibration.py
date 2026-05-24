
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '')
BASE    = ''
RES     = f'{BASE}/experiments/results'
OUT_DIR = f'{BASE}/experiments/figures/calibration'
JSON_OUT= f'{BASE}/experiments/results/calibration/calibration_results.json'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)

from evaluation.inference_utils import infer_iemocap, infer_meld, infer_mosei

C = {'bg': '#F7F9FB', 'grid': '#DDEAF7',
     'methods': {'CHADO': '#2E4057', 'MulT': '#048A81', 'MM-DFN': '#C25B5B',
                 'CTNet': '#E07B3F', 'LF-LSTM': '#6B8CAE', 'BP-MulT': '#8B5CF6',
                 'Baseline': '#9CA3AF', 'EmoCLIP': '#7C3AED',
                 'UniMSE': '#065F46', 'AER-LLM': '#92400E', 'OVMER': '#9D174D'}}
matplotlib.rc('font', family='sans-serif', size=11)
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

# ── Calibration metrics ───────────────────────────────────────────────────────
def ece(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() == 0: continue
        avg_conf = confidences[mask].mean()
        avg_acc  = correct[mask].mean()
        ece_val += mask.sum() / len(labels) * abs(avg_conf - avg_acc)
    return float(ece_val)

def brier_score(probs, labels, n_classes):
    one_hot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

def nll(probs, labels):
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-10)))

def predictive_entropy(probs):
    H = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    return float(H.mean()), float(H.std())

def reliability_diagram_data(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() == 0:
            bin_accs.append(0); bin_confs.append((bins[i]+bins[i+1])/2); bin_counts.append(0)
        else:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(int(mask.sum()))
    return np.array(bin_confs), np.array(bin_accs), np.array(bin_counts)

# ── Models to evaluate ────────────────────────────────────────────────────────
MODELS = [
    ('CHADO',    'chado'),
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

# ── Per-dataset run ───────────────────────────────────────────────────────────
def run_calibration(dataset, infer_fn, device, n_classes):
    print(f'\n{"="*60}  {dataset.upper()}  {"="*10}')
    results, probs_dict = {}, {}
    for display_name, model_key in MODELS:
        ckpt = f'{RES}/{dataset}/{model_key}/best.pt'
        if not os.path.exists(ckpt): continue
        print(f'  [{display_name}]...', end='', flush=True)
        try:
            probs, labels = infer_fn(ckpt, dataset, model_key, device)
            print(f' {len(labels)} samples', end='')
        except Exception as e:
            print(f' ERROR: {e}'); continue

        ece_v  = ece(probs, labels)
        brier  = brier_score(probs, labels, n_classes)
        nll_v  = nll(probs, labels)
        em, es = predictive_entropy(probs)
        acc    = float((probs.argmax(1) == labels).mean())
        print(f'  ECE={ece_v:.4f}  Brier={brier:.4f}  Acc={acc:.4f}')
        results[display_name] = {'ece': round(ece_v, 4), 'brier': round(brier, 4),
                                  'nll': round(nll_v, 4), 'entropy_mean': round(em, 4),
                                  'entropy_std': round(es, 4), 'accuracy': round(acc, 4),
                                  'n': int(len(labels))}
        probs_dict[display_name] = (probs, labels)
    return results, probs_dict

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_reliability(probs_dict, dataset, save_prefix):
    n = len(probs_dict)
    if n == 0: return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.5))
    fig.patch.set_facecolor(C['bg'])
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (name, (probs, labels)) in zip(axes_flat, probs_dict.items()):
        bc, ba, _ = reliability_diagram_data(probs, labels)
        ece_v = ece(probs, labels)
        color = C['methods'].get(name, '#9CA3AF')
        ax.bar(bc, ba, width=0.055, color=color, alpha=0.8)
        ax.plot([0,1],[0,1], 'k--', lw=1.5, alpha=0.7)
        ax.fill_between(bc, bc, ba, where=(ba<bc), color=C['methods']['CHADO'], alpha=0.15)
        ax.fill_between(bc, bc, ba, where=(ba>bc), color='#E07B3F', alpha=0.15)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xlabel('Confidence', fontsize=9)
        ax.set_ylabel('Accuracy', fontsize=9)
        ax.set_title(f'{name}\nECE={ece_v:.3f}', fontsize=10, fontweight='bold')
        ax.set_facecolor(C['bg'])
        ax.grid(True, color=C['grid'], lw=0.6)

    for ax in axes_flat[n:]: ax.set_visible(False)
    fig.suptitle(f'{dataset.upper()} — Reliability Diagrams', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_reliability.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_prefix}_reliability.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {save_prefix}_reliability.png')

def plot_summary(all_results, save_path):
    ds_list = [k for k in all_results if k in ('IEMOCAP', 'MELD')]
    if not ds_list: return
    all_methods = []
    for dr in all_results.values():
        for m in dr:
            if m not in all_methods: all_methods.append(m)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(C['bg'])
    for ax_i, metric in enumerate(['ece', 'brier']):
        ax = axes[ax_i]
        x = np.arange(len(ds_list))
        width = 0.8 / len(all_methods)
        for i, method in enumerate(all_methods):
            vals = [all_results[ds].get(method, {}).get(metric, np.nan) for ds in ds_list]
            color = C['methods'].get(method, '#9CA3AF')
            ax.bar(x + i*width - 0.4 + width/2, vals, width*0.9,
                   label=method, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_list, fontsize=11)
        ax.set_ylabel(metric.upper() + ' (↓ better)', fontsize=11)
        ax.set_title(f'Calibration — {metric.upper()}', fontweight='bold', fontsize=12)
        ax.set_facecolor(C['bg'])
        ax.grid(axis='y', color=C['grid'], lw=0.8)
        ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved calibration summary → {save_path}.png')

def plot_entropy_dist(probs_dict, dataset, save_prefix):
    targets = [m for m in ['CHADO', 'CTNet', 'MulT'] if m in probs_dict]
    if len(targets) < 2: return
    fig, ax = plt.subplots(figsize=(7,4))
    fig.patch.set_facecolor(C['bg']); ax.set_facecolor(C['bg'])
    for name in targets:
        probs, labels = probs_dict[name]
        H = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        correct = probs.argmax(1) == labels
        color = C['methods'].get(name, '#9CA3AF')
        ax.hist(H[correct], bins=40, density=True, alpha=0.55, color=color, label=f'{name} ✓')
        ax.hist(H[~correct], bins=40, density=True, alpha=0.3, color=color,
                histtype='step', lw=2, label=f'{name} ✗')
    ax.set_xlabel('Predictive Entropy', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{dataset.upper()} — Entropy: Correct vs Wrong', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, color=C['grid'], lw=0.6)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_entropy.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_prefix}_entropy.png', dpi=200, bbox_inches='tight')
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    all_results = {}

    iem_res, iem_probs = run_calibration('iemocap', infer_iemocap, device, 4)
    all_results['IEMOCAP'] = iem_res
    plot_reliability(iem_probs, 'IEMOCAP', f'{OUT_DIR}/iemocap')
    plot_entropy_dist(iem_probs, 'IEMOCAP', f'{OUT_DIR}/iemocap')

    meld_res, meld_probs = run_calibration('meld', infer_meld, device, 7)
    all_results['MELD'] = meld_res
    plot_reliability(meld_probs, 'MELD', f'{OUT_DIR}/meld')
    plot_entropy_dist(meld_probs, 'MELD', f'{OUT_DIR}/meld')

    # MOSEI multilabel
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
        em, es = float(np.mean(-np.sum(np.clip(probs,1e-10,1)*np.log(np.clip(probs,1e-10,1)),axis=1))), \
                 float(np.std(-np.sum(np.clip(probs,1e-10,1)*np.log(np.clip(probs,1e-10,1)),axis=1)))
        print(f'  Brier={avg_brier:.4f}  Entropy={em:.4f}')
        mosei_res[display_name] = {'brier': round(avg_brier, 4),
                                    'brier_per_class': [round(v,4) for v in brier_per],
                                    'entropy_mean': round(em,4), 'entropy_std': round(es,4),
                                    'n': int(len(labels)),
                                    'note': 'multilabel sigmoid; ECE not computed'}
    all_results['CMU-MOSEI'] = mosei_res

    plot_summary(all_results, f'{OUT_DIR}/calibration_summary')

    # Print table
    print(f'\n{"="*70}')
    print('  CALIBRATION SUMMARY')
    print(f'{"="*70}')
    for ds, res in all_results.items():
        print(f'\n  {ds}:')
        print(f'  {"Method":<12} {"ECE":>8} {"Brier":>8} {"NLL":>8} {"Entropy":>10} {"Acc":>7}')
        print(f'  {"-"*55}')
        for m, v in res.items():
            print(f'  {m:<12} '
                  f'{str(round(v["ece"],4)) if "ece" in v else "N/A":>8} '
                  f'{v["brier"]:>8.4f} '
                  f'{str(round(v["nll"],4)) if "nll" in v else "N/A":>8} '
                  f'{v["entropy_mean"]:>10.4f} '
                  f'{v.get("accuracy",0):>7.4f}')

    with open(JSON_OUT, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved → {JSON_OUT}')

if __name__ == '__main__':
    main()
