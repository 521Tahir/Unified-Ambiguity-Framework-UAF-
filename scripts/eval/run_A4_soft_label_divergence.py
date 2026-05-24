#!/usr/bin/env python3
"""
A.4 — Soft-Label Distribution Evaluation

Metrics: KL(p_ann || q_model) ↓, JSD ↓, Wasserstein-1 ↓

IEMOCAP: annotator vote counts → Laplace-smoothed distribution (alpha=0.1).
  Laplace smoothing is standard Bayesian practice for sparse vote counts
  (3–5 annotators) to avoid zero-probability artifacts in KL computation.
  Each annotator independently labels one emotion → vote-count vector →
  a genuine per-sample categorical distribution over emotions.

CMU-MOSEI: excluded — provides averaged intensity ratings (0-3 Likert per
  annotator summed) rather than per-annotator discrete vote distributions.
  The aggregated intensity vector cannot be treated as a categorical
  distributional annotation equivalent to IEMOCAP's per-annotator votes.
  Soft-label KL/JSD/W1 metrics require proper per-annotator discrete
  categorical distributions to be methodologically valid.

MELD: excluded — no public annotator-level vote distributions.

Usage:
  python3 -u scripts/eval/run_A4_soft_label_divergence.py
"""
import os, sys, json, re, glob
import numpy as np
from scipy.stats import entropy as scipy_entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

BASE    = ROOT
OUT_DIR = f'{BASE}/experiments/results/A4_soft_label'
FIG_DIR = f'{BASE}/experiments/figures/A4_soft_label'
CACHE   = f'{BASE}/experiments/results/reviewer_issues/cache'
for d in (OUT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rc('font', family='sans-serif', size=11)
matplotlib.rcParams['axes.spines.top']   = False
matplotlib.rcParams['axes.spines.right'] = False

IEMOCAP_CAT = {
    'neutral':0,'neu':0,
    'happy':1,'hap':1,'excitement':1,'excited':1,'exc':1,
    'anger':2,'angry':2,'ang':2,'frustration':2,'fru':2,
    'sad':3,'sadness':3,
}
COLORS = {'AR-GOT':'#2E4057','OV-MER':'#048A81','AER-LLM':'#C25B5B','EmoCLIP':'#E07B3F'}

IEMOCAP_CACHES = {
    'AR-GOT':  f'{CACHE}/A1_iemocap_chado.npz',
    'OV-MER':  f'{CACHE}/A1_iemocap_ovmer.npz',
    'AER-LLM': f'{CACHE}/A1_iemocap_aerllm.npz',
    'EmoCLIP': f'{CACHE}/A1_iemocap_emoclip.npz',
}
MOSEI_CACHES = {
    'AR-GOT':  f'{CACHE}/A1v2_mosei_chado_best.npz',
    'OV-MER':  f'{CACHE}/A1v2_mosei_ovmer.npz',
    'AER-LLM': f'{CACHE}/A1v2_mosei_aerllm.npz',
    'EmoCLIP': f'{CACHE}/A1v2_mosei_emoclip.npz',
}

LAPLACE_ALPHA = 0.1   # Dirichlet pseudo-count added to each vote category
EPS           = 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# Annotator distribution parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_iemocap_annotator_info(eval_root):
    """
    Returns {uid: {'counts': np.array(4), 'dist_smooth': np.array(4)}}
    Laplace-smoothed dist: (counts + LAPLACE_ALPHA) / (n + 4*LAPLACE_ALPHA)
    """
    files  = glob.glob(os.path.join(eval_root,'Session*/dialog/EmoEvaluation/*.txt'))
    hdr    = re.compile(r'\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+\S+\s+\[')
    rtr    = re.compile(r'C-\w+:\s*([^;]+)')
    out    = {}
    for fp in files:
        content = open(fp, errors='ignore').read()
        pos     = [m.start() for m in re.finditer(r'\[[\d.]+\s*-', content)]
        pos.append(len(content))
        for i in range(len(pos)-1):
            blk = content[pos[i]:pos[i+1]]
            hm  = hdr.search(blk)
            if not hm: continue
            uid    = hm.group(3)
            labels = []
            for rm in rtr.finditer(blk):
                parts = [p.strip().rstrip(';').strip().lower()
                         for p in rm.group(1).split(';')]
                labels.extend([IEMOCAP_CAT[p] for p in parts if p in IEMOCAP_CAT])
            if len(labels) >= 2:
                cnt  = np.bincount(labels, minlength=4).astype(float)
                n    = cnt.sum()
                # Laplace (Dirichlet MAP) smoothed distribution
                smooth = (cnt + LAPLACE_ALPHA) / (n + 4 * LAPLACE_ALPHA)
                out[uid] = {'counts': cnt, 'dist_smooth': smooth}
    n_dis = sum(1 for v in out.values() if (v['counts'] > 0).sum() > 1)
    print(f'  IEMOCAP: {len(out)} utterances with ≥2 annotators '
          f'({n_dis} with annotator disagreement)')
    return out


def parse_mosei_annotator_info(manifest_path):
    """
    Returns {uid: {'raw': np.array(6), 'dist': np.array(6) normalized}}
    raw = per-emotion intensity sum across annotators.
    dist = raw / sum(raw) — used as intensity weights in weighted metrics.
    """
    out = {}
    for line in open(manifest_path):
        s   = json.loads(line)
        raw = np.array(s.get('label_raw', [0.0]*6), float).clip(min=0)
        tot = raw.sum()
        dist = raw / tot if tot > 1e-6 else np.ones(6)/6
        out[s['utt_id']] = {'raw': raw, 'dist': dist}
    n_amb = sum(1 for v in out.values() if (v['raw'] > 0).sum() > 1)
    print(f'  CMU-MOSEI: {len(out)} samples ({n_amb} with multi-emotion annotations)')
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Divergence primitives
# ═══════════════════════════════════════════════════════════════════════════════

def kl_div(p, q):
    p = np.asarray(p, float) + EPS; q = np.asarray(q, float) + EPS
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def jsd_val(p, q):
    """JSD in nats (range [0, ln2])."""
    p = np.asarray(p, float) + EPS; p /= p.sum()
    q = np.asarray(q, float) + EPS; q /= q.sum()
    return float(jensenshannon(p, q) ** 2)


def w1_ordinal(p, q):
    """W1 treating class indices as ordinal values 0…K-1."""
    p = np.asarray(p, float) + EPS; p /= p.sum()
    q = np.asarray(q, float) + EPS; q /= q.sum()
    vals = np.arange(len(p), dtype=float)
    return float(wasserstein_distance(vals, vals, u_weights=p, v_weights=q))


# ═══════════════════════════════════════════════════════════════════════════════
# IEMOCAP metrics  (Laplace-smoothed annotator distributions)
# ═══════════════════════════════════════════════════════════════════════════════

def _iemocap_entropy(counts):
    """Normalised annotator entropy for a vote-count array."""
    p = counts / (counts.sum() + EPS)
    pnz = p[p > EPS]
    if len(pnz) < 2:
        return 0.0
    return float(scipy_entropy(pnz)) / np.log(4)


def compute_iemocap_metrics(model_name, cache_path, ann_info,
                            entropy_threshold=0.0):
    """
    KL, JSD, W1 between Laplace-smoothed annotator distribution and
    model softmax.

    entropy_threshold: only include utterances whose annotator vote
      entropy ≥ this value (default 0 = all; >0 = disagreement-only).
    """
    d     = np.load(cache_path, allow_pickle=True)
    uids  = d['utt_ids'].tolist()
    probs = d['probs']    # (N, 4)

    kls, jsds, w1s = [], [], []
    for i, uid in enumerate(uids):
        if uid not in ann_info:
            continue
        info = ann_info[uid]
        if _iemocap_entropy(info['counts']) < entropy_threshold:
            continue
        p_smooth = info['dist_smooth']   # Laplace-smoothed
        q_mod    = probs[i]
        kls.append(kl_div(p_smooth, q_mod))
        jsds.append(jsd_val(p_smooth, q_mod))
        w1s.append(w1_ordinal(p_smooth, q_mod))

    n = len(kls)
    if n == 0:
        return None
    return {
        'model': model_name, 'n': n,
        'entropy_thr': entropy_threshold,
        'kl':  {'mean': float(np.mean(kls)),  'std': float(np.std(kls))},
        'jsd': {'mean': float(np.mean(jsds)), 'std': float(np.std(jsds))},
        'w1':  {'mean': float(np.mean(w1s)),  'std': float(np.std(w1s))},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CMU-MOSEI metrics  (intensity-weighted binary)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mosei_metrics(model_name, cache_path, ann_info,
                          multi_emotion_only=False):
    """
    Positive-emotion-restricted KL/JSD/W1 for CMU-MOSEI.

    For each sample, restrict both annotator intensity and model predictions
    to the emotions the annotator labeled non-zero (raw > 0).  Both
    distributions are renormalized over this subset.

    Model q = softmax(logit(sigmoid)) restricted to positive emotions.

    This directly measures whether the model correctly proportions
    probability among the emotions annotators considered relevant.
    Single-emotion samples get KL=JSD=W1=0 (trivially matched).
    Multi-emotion samples test ambiguity proportionality.

    multi_emotion_only: if True, exclude single-emotion samples entirely.
    """
    d    = np.load(cache_path, allow_pickle=True)
    uids = d['utt_ids'].tolist()
    sigs = d['sig_probs']   # (N, 6)

    kls, jsds, w1s = [], [], []

    for i, uid in enumerate(uids):
        if uid not in ann_info:
            continue
        info = ann_info[uid]
        raw  = info['raw']    # (6,) raw intensity sums

        ann_mask = raw > 0    # boolean — which emotions annotator labeled
        n_pos    = int(ann_mask.sum())
        if n_pos == 0:
            continue
        if multi_emotion_only and n_pos < 2:
            continue

        if n_pos == 1:
            # single emotion: any model trivially matches → contribution 0
            kls.append(0.0); jsds.append(0.0); w1s.append(0.0)
            continue

        # restrict annotator distribution to annotated emotions
        a_pos = raw[ann_mask]
        a_pos = a_pos / a_pos.sum()   # renormalized over positive subset

        # restrict model prediction to same subset, then softmax-of-logits
        sig_pos = np.clip(sigs[i][ann_mask], EPS, 1.0 - EPS)
        logits  = np.log(sig_pos / (1.0 - sig_pos))
        logits -= logits.max()
        q_pos   = np.exp(logits); q_pos /= q_pos.sum()

        kls.append(kl_div(a_pos, q_pos))
        jsds.append(jsd_val(a_pos, q_pos))
        w1s.append(w1_ordinal(a_pos, q_pos))

    n = len(kls)
    if n == 0:
        return None
    return {
        'model': model_name, 'n': n,
        'multi_only': multi_emotion_only,
        'kl':  {'mean': float(np.mean(kls)),  'std': float(np.std(kls))},
        'jsd': {'mean': float(np.mean(jsds)), 'std': float(np.std(jsds))},
        'w1':  {'mean': float(np.mean(w1s)),  'std': float(np.std(w1s))},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Print tables
# ═══════════════════════════════════════════════════════════════════════════════

def _print_table(title, subtitle, results, model_order):
    w = 18
    print('\n' + '='*72)
    print(f'A.4  {title}')
    print(f'  {subtitle}')
    print('='*72)
    hdr = f"  {'Model':<10} {'N':>5}  {'KL ↓':>{w}}  {'JSD ↓':>{w}}  {'W1 ↓':>{w}}"
    print(hdr)
    print('  ' + '-'*(len(hdr)-2))

    best = {}
    for m in ('kl','jsd','w1'):
        vals = [results[n][m]['mean'] for n in model_order if n in results]
        best[m] = min(vals) if vals else None

    for name in model_order:
        if name not in results:
            continue
        r = results[name]
        rows = {}
        for m in ('kl','jsd','w1'):
            val = r[m]['mean']; std = r[m]['std']
            marker = '▼' if best[m] is not None and abs(val - best[m]) < 1e-9 else ' '
            rows[m] = f"{val:.4f}±{std:.4f}{marker}"
        print(f"  {name:<10} {r['n']:>5}  {rows['kl']:>{w+1}}  {rows['jsd']:>{w+1}}  {rows['w1']:>{w+1}}")

    print()
    print('  ▼ = best (lowest) per metric column')


def print_all_tables(iemo_res, mosei_res):
    order = ['AR-GOT', 'OV-MER', 'AER-LLM', 'EmoCLIP']

    _print_table(
        'IEMOCAP — Soft-Label Distribution Divergence',
        f'Annotator: Laplace-smoothed vote dist (α={LAPLACE_ALPHA}, K=4) | '
        f'Model: softmax probabilities',
        iemo_res, order)

    _print_table(
        'CMU-MOSEI — Positive-Emotion-Restricted Divergence',
        'Distributions restricted to annotator-labeled emotions; softmax(logit(sig))',
        mosei_res, order)

    print('\n' + '-'*72)
    print('  MELD: excluded — annotator-level vote distributions are not publicly')
    print('  available for MELD. Only post-hoc KNN soft-label proxies exist,')
    print('  which do not represent true inter-annotator disagreement distributions.')
    print('-'*72)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary: best-model analysis
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(iemo_res, mosei_res):
    order = ['AR-GOT', 'OV-MER', 'AER-LLM', 'EmoCLIP']
    print('\n' + '='*72)
    print('A.4  COMBINED SUMMARY — Best Model per Metric')
    print('='*72)
    print(f"  {'Dataset':<14} {'Metric':<8} {'Best Model':<12} {'Value':>8}")
    print('  ' + '-'*46)
    for ds_name, res in [('IEMOCAP', iemo_res), ('CMU-MOSEI', mosei_res)]:
        for m, label in [('kl','KL'), ('jsd','JSD'), ('w1','W1')]:
            pairs = [(n, res[n][m]['mean']) for n in order if n in res]
            if not pairs: continue
            best_name, best_val = min(pairs, key=lambda x: x[1])
            mark = ' ★' if best_name == 'AR-GOT' else ''
            print(f"  {ds_name:<14} {label:<8} {best_name:<12} {best_val:>8.4f}{mark}")
    print()
    print('  ★ = AR-GOT wins')


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX table  (IEMOCAP only)
# ═══════════════════════════════════════════════════════════════════════════════

def _write_latex_iemocap(iemo_res):
    order   = ['AR-GOT', 'OV-MER', 'AER-LLM', 'EmoCLIP']
    primary = ['AR-GOT', 'AER-LLM', 'EmoCLIP']   # only bold within primary set
    best    = {m: min(iemo_res[n][m]['mean'] for n in primary if n in iemo_res)
               for m in ('kl','jsd','w1')}

    def fmt(name, val, metric):
        s = f'{val:.4f}'
        if name in primary and abs(val - best[metric]) < 1e-9:
            return f'\\textbf{{{s}}}'
        return s

    rows = []
    for name in order:
        r = iemo_res.get(name)
        if r is None: continue
        suppl = '\\textsuperscript{†}' if name not in primary else ''
        rows.append(
            f'  {name}{suppl:<2} & {fmt(name,r["kl"]["mean"],"kl")} '
            f'& {fmt(name,r["jsd"]["mean"],"jsd")} '
            f'& {fmt(name,r["w1"]["mean"],"w1")} \\\\'
        )

    table = (
        '\\begin{table}[t]\n'
        '\\centering\\small\n'
        '\\caption{Soft-label distribution divergence (A.4) on IEMOCAP. '
        'KL$(p_{\\text{ann}}\\|q_{\\text{model}})$, JSD, and Wasserstein-1 '
        'between Laplace-smoothed annotator vote distributions ($\\alpha=0.1$) '
        'and model softmax predictions ($\\downarrow$ lower is better). '
        'CMU-MOSEI and MELD are excluded: CMU-MOSEI provides averaged Likert '
        'intensity ratings rather than per-annotator discrete categorical votes; '
        'MELD has no public per-annotator vote distributions.}\n'
        '\\label{tab:soft_label_divergence}\n'
        '\\begin{tabular}{lccc}\n'
        '\\toprule\n'
        '\\textbf{Model} & KL$\\downarrow$ & JSD$\\downarrow$ & W1$\\downarrow$ \\\\\n'
        '\\midrule\n'
        + '\n'.join(rows) + '\n'
        '\\bottomrule\n'
        '\\end{tabular}\n'
        '\\end{table}\n'
    )
    tex_path = f'{OUT_DIR}/A4_soft_label_table.tex'
    with open(tex_path, 'w') as f:
        f.write(table)
    print(f'\n  LaTeX saved → {tex_path}')
    print(table)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure  (IEMOCAP only)
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_iemocap(iemo_res):
    order   = ['AR-GOT', 'OV-MER', 'AER-LLM', 'EmoCLIP']
    metrics = [('kl','KL Divergence ↓'), ('jsd','JSD ↓'), ('w1','Wasserstein-1 ↓')]
    names   = [n for n in order if n in iemo_res]
    x       = np.arange(len(names))
    bw      = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('A.4 — IEMOCAP Soft-Label Distribution Divergence\n'
                 'Laplace-smoothed annotator vote distributions vs. model softmax (α=0.1)',
                 fontsize=12, fontweight='bold')

    for col, (mkey, mlabel) in enumerate(metrics):
        ax = axes[col]
        ax.set_facecolor('#F7F9FB')
        ax.grid(axis='y', color='#DDEAF7', linewidth=0.8, zorder=0)
        vals   = [iemo_res[n][mkey]['mean'] for n in names]
        errs   = [iemo_res[n][mkey]['std']  for n in names]
        colors = [COLORS[n] for n in names]
        ax.bar(x, vals, width=bw, color=colors, zorder=3, alpha=0.9,
               yerr=errs, capsize=4, error_kw={'elinewidth':1.2,'ecolor':'#444'})
        best_idx = int(np.argmin(vals))
        for xi, (v, e) in enumerate(zip(vals, errs)):
            fw = 'bold' if xi == best_idx else 'normal'
            ax.text(xi, v + e + max(errs)*0.03, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8.5, fontweight=fw)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(mlabel, fontsize=11)

    from matplotlib.patches import Patch
    handles = [Patch(color=COLORS[n], label=n) for n in names]
    fig.legend(handles=handles, loc='lower center', ncol=len(names),
               bbox_to_anchor=(0.5, -0.05), fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    pdf = f'{FIG_DIR}/A4_soft_label_divergence.pdf'
    png = f'{FIG_DIR}/A4_soft_label_divergence.png'
    plt.savefig(pdf, dpi=200, bbox_inches='tight')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure → {pdf}')
    print(f'  Figure → {png}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    iemocap_root = '/home/tahirahmad/IEMOCAP_Final/dataset/IEMOCAP_full_release'

    print('\n[A.4] Parsing IEMOCAP annotator distributions …')
    iemocap_ann = parse_iemocap_annotator_info(iemocap_root) \
                  if os.path.isdir(iemocap_root) else {}

    # ── IEMOCAP ───────────────────────────────────────────────────────────────
    iemo_all = {}; iemo_amb = {}
    if iemocap_ann:
        for thr, store, label in [
            (0.0,  iemo_all, 'all samples'),
            (1e-9, iemo_amb, 'ambiguous-only (entropy>0)'),
        ]:
            print(f'\n[IEMOCAP] Laplace-smoothed KL/JSD/W1  [{label}] …')
            for name, cache in IEMOCAP_CACHES.items():
                if not os.path.exists(cache):
                    print(f'  [{name}] cache missing → skip'); continue
                r = compute_iemocap_metrics(name, cache, iemocap_ann,
                                            entropy_threshold=thr)
                if r is None:
                    print(f'  [{name}] no alignment → skip'); continue
                store[name] = r
                print(f'  [{name:8s}]  N={r["n"]}  '
                      f'KL={r["kl"]["mean"]:.4f}  '
                      f'JSD={r["jsd"]["mean"]:.4f}  '
                      f'W1={r["w1"]["mean"]:.4f}')

    iemo_res = iemo_all

    # ── Supplementary: ambiguous-only breakdown ───────────────────────────────
    if iemo_amb:
        _print_table('IEMOCAP — Ambiguous-only (entropy>0, supplementary)',
                     f'Subset: utterances where annotators disagreed  (α={LAPLACE_ALPHA})',
                     iemo_amb, list(COLORS.keys()))

    # ── PRIMARY TABLE — reviewer-requested models only ────────────────────────
    PRIMARY_ORDER = ['AR-GOT', 'AER-LLM', 'EmoCLIP']   # reviewer-requested
    EXTRA_ORDER   = ['OV-MER']                           # supplementary

    print('\n' + '★'*72)
    print('  A.4 PRIMARY TABLE  (reviewer-requested models — paper-ready)')
    print('  CMU-MOSEI and MELD: excluded — no per-annotator discrete vote')
    print('  distributions available (only averaged intensity ratings).')
    print('★'*72)

    if iemo_res:
        print(f'\n{"="*72}')
        print(f'A.4  IEMOCAP — Soft-Label Distribution Divergence')
        print(f'  Annotator: Laplace-smoothed per-annotator vote dist  (α={LAPLACE_ALPHA}, K=4)')
        print(f'  Model:     softmax probabilities over 4 emotion classes')
        print('='*72)
        hdr = f"  {'Model':<12} {'N':>5}  {'KL ↓':>14}  {'JSD ↓':>14}  {'W1 ↓':>14}"
        print(hdr); print('  '+'-'*(len(hdr)-2))
        best = {m: min(iemo_res[n][m]['mean'] for n in PRIMARY_ORDER if n in iemo_res)
                for m in ('kl','jsd','w1')}
        for name in PRIMARY_ORDER + EXTRA_ORDER:
            if name not in iemo_res: continue
            r = iemo_res[name]
            sep = ' (suppl.)' if name in EXTRA_ORDER else ''
            rows = {}
            for m in ('kl','jsd','w1'):
                val = r[m]['mean']; std = r[m]['std']
                mk = '▼' if name not in EXTRA_ORDER and abs(val-best[m])<1e-9 else ' '
                rows[m] = f"{val:.4f}±{std:.4f}{mk}"
            print(f"  {name+sep:<21} {r['n']:>5}  "
                  f"{rows['kl']:>16}  {rows['jsd']:>16}  {rows['w1']:>16}")
        print(); print('  ▼ = best among reviewer-requested models')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print('SUMMARY — AR-GOT ranking on IEMOCAP')
    print('='*72)
    if iemo_res:
        for m, label in [('kl','KL'), ('jsd','JSD'), ('w1','W1')]:
            pairs = [(n, iemo_res[n][m]['mean']) for n in PRIMARY_ORDER if n in iemo_res]
            if not pairs: continue
            ranked = sorted(pairs, key=lambda x: x[1])
            best_n, best_v = ranked[0]
            argot_val = iemo_res.get('AR-GOT',{}).get(m,{}).get('mean', float('nan'))
            star = ' ★ BEST' if best_n == 'AR-GOT' else \
                   f' (rank {next(i+1 for i,(n,_) in enumerate(ranked) if n=="AR-GOT")}/{len(ranked)})'
            print(f"  IEMOCAP  {label:<5} best={best_n:<10} {best_v:.4f}  AR-GOT={argot_val:.4f}{star}")
    print()
    print('  ★ = AR-GOT best  |  CMU-MOSEI & MELD excluded (see docstring)')
    print()
    print('  EXCLUSION NOTES:')
    print('  CMU-MOSEI — provides averaged 0-3 Likert intensity sums, NOT per-annotator')
    print('    discrete categorical votes. KL/JSD/W1 requires proper categorical p_ann.')
    print('  MELD — no public per-annotator vote distributions.')

    if iemo_res:
        _write_latex_iemocap(iemo_res)
        _plot_iemocap(iemo_res)

    out = {'IEMOCAP': iemo_all, 'IEMOCAP_ambiguous': iemo_amb,
           'CMU-MOSEI': 'excluded — averaged intensity ratings, not per-annotator discrete votes',
           'MELD':      'excluded — no public annotator-vote distributions'}
    with open(f'{OUT_DIR}/A4_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n  JSON → {OUT_DIR}/A4_results.json')
    print('Done.')


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
