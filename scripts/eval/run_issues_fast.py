#!/usr/bin/env python3
"""
Fast completion script for reviewer issues 1.1-1.4.
Issue 1.1 results are pre-loaded from the log (all 6 models × IEMOCAP + CMU-MOSEI,
MELD CHADO only). Issues 1.2-1.4 are run fresh (IEMOCAP + CMU-MOSEI only).

Usage:
  CUDA_VISIBLE_DEVICES=5 python3 -u scripts/eval/run_issues_fast.py
"""
import os, sys, json, types
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, entropy as scipy_entropy
from scipy.stats import wasserstein_distance
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
import yaml
import re, glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

BASE    = ROOT
RES     = f'{BASE}/experiments/results'
OUT_DIR = f'{BASE}/experiments/results/reviewer_issues'
FIG_DIR = f'{BASE}/experiments/figures/reviewer_issues'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Import helpers from the main eval script ──────────────────────────────────
sys.path.insert(0, f'{BASE}/scripts/eval')
from eval_reviewer_issues import (
    ent_norm, compute_ece, stratified_metrics,
    parse_iemocap_annotator_entropy, compute_mosei_label_entropy,
    compute_meld_knn_entropy,
    infer_iemocap_model, infer_mosei_model, infer_meld_model,
    extract_cls_norms_iemocap, extract_cls_norms_mosei,
    _make_ablated_iemocap_model, _make_ablated_mosei_model,
    _infer_iemocap_ablated, _infer_mosei_ablated,
    get_iemocap_annotator_distributions, get_mosei_annotator_distributions,
    jsd, kl_div,
    _plot_stratified, _plot_divergence,
    C,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Pre-built Issue 1.1 results (from completed inference runs)
# ═══════════════════════════════════════════════════════════════════════════════

IEMOCAP_MODELS = ['CHADO', 'MulT', 'MM-DFN', 'CTNet', 'LF-LSTM', 'BP-MulT']
MOSEI_MODELS   = ['CHADO', 'MulT', 'MM-DFN', 'CTNet', 'LF-LSTM', 'BP-MulT']

ISSUE11_PREBUILT = {
    'IEMOCAP': {
        'CHADO':   {'Low': {'acc':0.641,'macro_f1':0.569,'ece':0.270,'n':167,'mean_amb':0.000},
                    'Medium':{'acc':0.665,'macro_f1':0.630,'ece':0.257,'n':167,'mean_amb':0.112},
                    'High':  {'acc':0.497,'macro_f1':0.346,'ece':0.386,'n':167,'mean_amb':0.470}},
        'MulT':    {'Low': {'acc':0.695,'macro_f1':0.677,'ece':0.219,'n':167,'mean_amb':0.000},
                    'Medium':{'acc':0.683,'macro_f1':0.692,'ece':0.249,'n':167,'mean_amb':0.112},
                    'High':  {'acc':0.599,'macro_f1':0.474,'ece':0.302,'n':167,'mean_amb':0.470}},
        'MM-DFN':  {'Low': {'acc':0.749,'macro_f1':0.745,'ece':0.105,'n':167,'mean_amb':0.000},
                    'Medium':{'acc':0.671,'macro_f1':0.679,'ece':0.182,'n':167,'mean_amb':0.112},
                    'High':  {'acc':0.581,'macro_f1':0.452,'ece':0.277,'n':167,'mean_amb':0.470}},
        'CTNet':   {'Low': {'acc':0.695,'macro_f1':0.687,'ece':0.216,'n':167,'mean_amb':0.000},
                    'Medium':{'acc':0.713,'macro_f1':0.724,'ece':0.213,'n':167,'mean_amb':0.112},
                    'High':  {'acc':0.617,'macro_f1':0.465,'ece':0.291,'n':167,'mean_amb':0.470}},
        'LF-LSTM': {'Low': {'acc':0.731,'macro_f1':0.727,'ece':0.243,'n':167,'mean_amb':0.000},
                    'Medium':{'acc':0.701,'macro_f1':0.708,'ece':0.259,'n':167,'mean_amb':0.112},
                    'High':  {'acc':0.623,'macro_f1':0.494,'ece':0.336,'n':167,'mean_amb':0.470}},
        'BP-MulT': {'Low': {'acc':0.731,'macro_f1':0.734,'ece':0.246,'n':167,'mean_amb':0.000},
                    'Medium':{'acc':0.695,'macro_f1':0.701,'ece':0.298,'n':167,'mean_amb':0.112},
                    'High':  {'acc':0.545,'macro_f1':0.448,'ece':0.430,'n':167,'mean_amb':0.470}},
    },
    'CMU-MOSEI': {
        'CHADO':   {'Low': {'acc':0.807,'macro_f1':0.149,'ece':0.609,'n':726,'mean_amb':0.000},
                    'Medium':{'acc':0.722,'macro_f1':0.140,'ece':0.520,'n':726,'mean_amb':0.024},
                    'High':  {'acc':0.566,'macro_f1':0.145,'ece':0.362,'n':725,'mean_amb':0.450}},
        'MulT':    {'Low': {'acc':0.468,'macro_f1':0.188,'ece':0.209,'n':726,'mean_amb':0.000},
                    'Medium':{'acc':0.406,'macro_f1':0.226,'ece':0.155,'n':726,'mean_amb':0.024},
                    'High':  {'acc':0.247,'macro_f1':0.149,'ece':0.054,'n':725,'mean_amb':0.450}},
        'MM-DFN':  {'Low': {'acc':0.592,'macro_f1':0.199,'ece':0.246,'n':726,'mean_amb':0.000},
                    'Medium':{'acc':0.501,'macro_f1':0.239,'ece':0.194,'n':726,'mean_amb':0.024},
                    'High':  {'acc':0.374,'macro_f1':0.188,'ece':0.134,'n':725,'mean_amb':0.450}},
        'CTNet':   {'Low': {'acc':0.585,'macro_f1':0.215,'ece':0.297,'n':726,'mean_amb':0.000},
                    'Medium':{'acc':0.521,'macro_f1':0.249,'ece':0.241,'n':726,'mean_amb':0.024},
                    'High':  {'acc':0.357,'macro_f1':0.186,'ece':0.101,'n':725,'mean_amb':0.450}},
        'LF-LSTM': {'Low': {'acc':0.667,'macro_f1':0.242,'ece':0.252,'n':726,'mean_amb':0.000},
                    'Medium':{'acc':0.552,'macro_f1':0.247,'ece':0.187,'n':726,'mean_amb':0.024},
                    'High':  {'acc':0.428,'macro_f1':0.216,'ece':0.085,'n':725,'mean_amb':0.450}},
        'BP-MulT': {'Low': {'acc':0.705,'macro_f1':0.228,'ece':0.377,'n':726,'mean_amb':0.000},
                    'Medium':{'acc':0.602,'macro_f1':0.274,'ece':0.292,'n':726,'mean_amb':0.024},
                    'High':  {'acc':0.459,'macro_f1':0.216,'ece':0.174,'n':725,'mean_amb':0.450}},
    },
    'MELD': {
        'CHADO':   {'Low': {'acc':0.131,'macro_f1':0.033,'ece':0.049,'n':870,'mean_amb':0.267},
                    'Medium':{'acc':0.151,'macro_f1':0.037,'ece':0.031,'n':870,'mean_amb':0.504},
                    'High':  {'acc':0.180,'macro_f1':0.044,'ece':0.002,'n':870,'mean_amb':0.696}},
    }
}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Output: {OUT_DIR}')

    # ── Pre-compute ambiguity scores (fast) ──────────────────────────────────
    print('\n' + '='*70)
    print('Pre-computing Model-Independent Ambiguity Scores')
    print('='*70)

    eval_root = '/home/tahirahmad/IEMOCAP_Final/dataset/IEMOCAP_full_release'
    iemocap_amb = {}
    if os.path.isdir(eval_root):
        print('\n  IEMOCAP — parsing annotator vote entropy …')
        iemocap_amb = parse_iemocap_annotator_entropy(eval_root)

    mosei_manifest = f'{BASE}/data/processed/mosei/mosei_utt_test.jsonl'
    mosei_amb = {}
    if os.path.exists(mosei_manifest):
        print('\n  CMU-MOSEI — computing label_raw entropy …')
        mosei_amb = compute_mosei_label_entropy(mosei_manifest)

    all_output = {}

    # ── Issue 1.1: use pre-built results ─────────────────────────────────────
    print('\n' + '='*70)
    print('ISSUE 1.1 — Stratified Analysis (Model-Independent Ambiguity)')
    print('[Using pre-computed results from full inference run]')
    print('='*70)

    issue11 = ISSUE11_PREBUILT

    print('\n  IEMOCAP — Annotator Vote Entropy Strata')
    print(f'  {"Model":<12} {"Low F1":>8} {"Med F1":>8} {"High F1":>8} {"Drop":>7}')
    print('  ' + '-'*50)
    for m, strata in issue11['IEMOCAP'].items():
        lo = strata['Low']['macro_f1']
        md = strata['Medium']['macro_f1']
        hi = strata['High']['macro_f1']
        print(f'  {m:<12} {lo:>8.3f} {md:>8.3f} {hi:>8.3f} {lo-hi:>+7.3f}')

    print('\n  CMU-MOSEI — Label-Raw Entropy Strata')
    print(f'  {"Model":<12} {"Low F1":>8} {"Med F1":>8} {"High F1":>8} {"Drop":>7}')
    print('  ' + '-'*50)
    for m, strata in issue11['CMU-MOSEI'].items():
        lo = strata['Low']['macro_f1']
        md = strata['Medium']['macro_f1']
        hi = strata['High']['macro_f1']
        print(f'  {m:<12} {lo:>8.3f} {md:>8.3f} {hi:>8.3f} {lo-hi:>+7.3f}')

    print('\n  MELD — KNN Soft-Label Entropy Strata (CHADO only)')
    print(f'  {"Model":<12} {"Low F1":>8} {"Med F1":>8} {"High F1":>8} {"Drop":>7}')
    print('  ' + '-'*50)
    for m, strata in issue11['MELD'].items():
        lo = strata['Low']['macro_f1']
        md = strata['Medium']['macro_f1']
        hi = strata['High']['macro_f1']
        print(f'  {m:<12} {lo:>8.3f} {md:>8.3f} {hi:>8.3f} {lo-hi:>+7.3f}')

    _plot_stratified(issue11)
    all_output['issue_1.1'] = issue11
    with open(f'{OUT_DIR}/issue11_stratified.json', 'w') as f:
        json.dump(issue11, f, indent=2)
    print(f'\n  Saved → {OUT_DIR}/issue11_stratified.json')

    # ── Issues 1.2, 1.3, 1.4 ─────────────────────────────────────────────────
    # Import and run the individual functions from the main eval script
    from eval_reviewer_issues import run_issue12, run_issue13, run_issue14

    print('\n\nRunning Issue 1.2 …')
    all_output['issue_1.2'] = run_issue12(device, iemocap_amb, mosei_amb)

    print('\n\nRunning Issue 1.3 …')
    all_output['issue_1.3'] = run_issue13(device)

    print('\n\nRunning Issue 1.4 …')
    all_output['issue_1.4'] = run_issue14(device, iemocap_amb, mosei_amb)

    # ── Final summary ─────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('REVIEWER ISSUES — FULL SUMMARY')
    print('='*70)

    print('\n━━━ ISSUE 1.1: Independent-Ambiguity Stratified Analysis ━━━')
    for ds in ['IEMOCAP', 'CMU-MOSEI', 'MELD']:
        ds_res = issue11.get(ds, {})
        print(f'\n  {ds}:')
        for model, strata in ds_res.items():
            lo = strata.get('Low', {}).get('macro_f1', 0)
            hi = strata.get('High', {}).get('macro_f1', 0)
            drop = lo - hi
            mark = ' ←CHADO' if model == 'CHADO' else ''
            print(f'    {model:<12} Low={lo:.3f} → High={hi:.3f}  drop={drop:.3f}{mark}')

    print('\n━━━ ISSUE 1.2: Radius–Ambiguity Correlation ━━━')
    for key, r in all_output.get('issue_1.2', {}).items():
        if isinstance(r, dict) and 'spearman_r' in r:
            print(f'  {key}: Spearman ρ={r["spearman_r"]:+.4f}  '
                  f'p={r["spearman_p"]:.4f}  N={r["n"]}')

    print('\n━━━ ISSUE 1.3: Factor-Level Ablation (IEMOCAP) ━━━')
    iemo13 = all_output.get('issue_1.3', {}).get('IEMOCAP', {})
    print(f'  {"Variant":<22} {"Acc%":>7} {"MacF1%":>8}')
    print('  ' + '-'*42)
    for v, m in iemo13.items():
        if isinstance(m, dict) and 'acc' in m:
            print(f'  {v:<22} {m["acc"]*100:>6.2f}%  {m["macro_f1"]*100:>7.2f}%')

    print('\n━━━ ISSUE 1.4: Distribution Divergence ━━━')
    for ds, r in all_output.get('issue_1.4', {}).items():
        if isinstance(r, dict) and 'kl_div_mean' in r:
            print(f'  {ds}: KL={r["kl_div_mean"]:.4f}  '
                  f'JSD={r["jsd_mean"]:.4f}  Wass={r["wasserstein_mean"]:.4f}  (N={r["n"]})')

    out_path = f'{OUT_DIR}/all_reviewer_issues.json'
    with open(out_path, 'w') as f:
        json.dump(all_output, f, indent=2, default=str)
    print(f'\n  Full results saved → {out_path}')
    print(f'  Figures saved   → {FIG_DIR}/')


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
    main()
