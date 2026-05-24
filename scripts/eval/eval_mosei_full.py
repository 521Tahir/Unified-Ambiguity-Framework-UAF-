#!/usr/bin/env python3
"""
CMU-MOSEI CHADO full evaluation — 6-class multi-label.
Prints class-wise accuracy, precision, recall, F1 + overall metrics.

Usage:
  cd /home/tahirahmad/Project_Code/CHADO_EMNLP
  CUDA_VISIBLE_DEVICES=5 python3 scripts/eval/eval_mosei_full.py \
      --config configs/mosei/chado_mosei_best.yaml \
      --ckpt   experiments/results/mosei/chado_best/best.pt
"""
import argparse, os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report,
)

from datasets.mosei.mosei_utt_dataset import MoseiUttDataset as MoseiDataset, collate_mosei_utt as collate_mosei
from models.chado.model import CHADOFeature
from training.engine.trainer import load_checkpoint

EMOTIONS   = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
FULL_NAMES = ["Happy", "Sad", "Anger", "Surprise", "Disgust", "Fear"]
THR_GRID   = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mosei/chado_mosei_best.yaml")
    p.add_argument("--ckpt",   default="experiments/results/mosei/chado_best/best.pt")
    return p.parse_args()


def tune_thresholds_wacc(probs, y_true):
    """Per-class threshold that maximises binary accuracy (WAcc)."""
    C = probs.shape[1]
    thrs = np.full(C, 0.5)
    for c in range(C):
        best, best_t = -1.0, 0.5
        for t in THR_GRID:
            acc = ((probs[:, c] >= t) == y_true[:, c]).mean()
            if acc > best:
                best, best_t = acc, t
        thrs[c] = best_t
    return thrs


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))
    dc, mc, cc = cfg["data"], cfg["model"], cfg.get("chado", {})

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice  : {device}")
    print(f"Config  : {args.config}")
    print(f"Ckpt    : {args.ckpt}\n")

    def _make(manifest):
        return MoseiDataset(
            manifest_path=manifest,
            max_audio_len=dc.get("max_audio_len", 50),
            max_video_len=dc.get("max_video_len", 30),
            label_thr=dc.get("label_thr", 0.0),
        )

    val_loader  = DataLoader(_make(dc["val_manifest"]),  batch_size=64, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_mosei)
    test_loader = DataLoader(_make(dc["test_manifest"]), batch_size=64, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_mosei)

    model = CHADOFeature(
        num_classes     = dc["num_classes"],
        d_model         = mc.get("d_model", 256),
        use_audio       = mc.get("use_audio", True),
        use_video       = mc.get("use_video", True),
        text_model      = mc.get("text_model_name", "roberta-base"),
        modality_dropout= 0.0,
        use_causal      = cc.get("use_causal", True),
        use_hyperbolic  = cc.get("use_hyperbolic", True),
        use_ot          = cc.get("use_ot", True),
        use_mad         = cc.get("use_mad", True),
        w_causal        = cc.get("w_causal", 0.005),
        w_hyperbolic    = cc.get("w_hyperbolic", 0.01),
        w_ot            = cc.get("w_ot", 0.05),
        w_mad           = cc.get("w_mad", 0.3),
        mad_gamma       = cc.get("mad_gamma", 0.5),
        ot_sigma        = cc.get("ot_sigma", 0.05),
        ot_eps          = cc.get("ot_eps", 0.1),
        ot_iters        = cc.get("ot_iters", 20),
    ).to(device)

    load_checkpoint(model, args.ckpt, strict=False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded  : {n_params:.1f}M parameters\n")
    model.eval()

    @torch.no_grad()
    def collect(loader):
        ps, ls = [], []
        for batch in loader:
            bdev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits, _, _, _ = model(bdev)
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ls.append((batch["label"] > 0.5).int().numpy())
        return np.vstack(ps), np.vstack(ls)

    print("Running inference on val set (threshold tuning)...")
    val_probs, val_true = collect(val_loader)
    thresholds = tune_thresholds_wacc(val_probs, val_true)
    print("  Thresholds → " + "  ".join(f"{FULL_NAMES[i]}:{thresholds[i]:.2f}" for i in range(6)))

    print("Running inference on test set...")
    test_probs, y_true = collect(test_loader)
    y_pred = (test_probs >= thresholds).astype(int)

    N = len(y_true)

    # ── Per-class metrics ─────────────────────────────────────────────────────
    prec_arr = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_arr  = recall_score   (y_true, y_pred, average=None, zero_division=0)
    f1_arr   = f1_score       (y_true, y_pred, average=None, zero_division=0)
    support  = y_true.sum(axis=0)

    # Binary accuracy per class (WAcc)
    bin_acc = np.array([((y_pred[:,c]==y_true[:,c]).sum()/N) for c in range(6)])
    wacc    = bin_acc.mean()

    # Overall multi-label metrics
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec  = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    macro_rec   = recall_score   (y_true, y_pred, average="macro",    zero_division=0)
    subset_acc  = accuracy_score(y_true, y_pred)   # exact match

    # ── Print ─────────────────────────────────────────────────────────────────
    w = 70
    print()
    print("=" * w)
    print("  CHADO — CMU-MOSEI 6-Class Emotion Recognition (Test Split)")
    print("=" * w)
    print(f"\n  {'Overall Metrics':}")
    print(f"  {'WAcc (mean binary acc)':30s} {wacc*100:>8.2f}%")
    print(f"  {'Subset Accuracy (exact match)':30s} {subset_acc*100:>8.2f}%")
    print(f"  {'Macro-F1':30s} {macro_f1*100:>8.2f}%")
    print(f"  {'Weighted-F1':30s} {weighted_f1*100:>8.2f}%")
    print(f"  {'Macro-Precision':30s} {macro_prec*100:>8.2f}%")
    print(f"  {'Macro-Recall':30s} {macro_rec*100:>8.2f}%")
    print(f"  {'Test samples':30s} {N:>9d}")

    print(f"\n  {'Per-Class Results':}")
    print(f"  {'Emotion':<12} {'Support':>8} {'BinAcc':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 58)
    for i, name in enumerate(FULL_NAMES):
        print(f"  {name:<12} {int(support[i]):>8} {bin_acc[i]*100:>7.2f}% "
              f"{prec_arr[i]*100:>9.2f}% {rec_arr[i]*100:>7.2f}% {f1_arr[i]*100:>7.2f}%")
    print("  " + "-" * 58)
    print(f"  {'Macro avg':<12} {'':>8} {wacc*100:>7.2f}% "
          f"{macro_prec*100:>9.2f}% {macro_rec*100:>7.2f}% {macro_f1*100:>7.2f}%")

    print(f"\n  {'Per-Class Thresholds (val-tuned):':}")
    print("  " + "  ".join(f"{FULL_NAMES[i]}={thresholds[i]:.2f}" for i in range(6)))

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir  = os.path.dirname(args.ckpt)
    out_path = os.path.join(out_dir, "test_results.json")
    per_class = {}
    for i, e in enumerate(EMOTIONS):
        per_class[e] = {
            "precision": float(prec_arr[i]),
            "recall":    float(rec_arr[i]),
            "f1":        float(f1_arr[i]),
            "support":   int(support[i]),
            "binary_acc":float(bin_acc[i]),
            "threshold": float(thresholds[i]),
        }
    result = {
        "wacc":             float(wacc),
        "subset_accuracy":  float(subset_acc),
        "macro_f1":         float(macro_f1),
        "weighted_f1":      float(weighted_f1),
        "macro_precision":  float(macro_prec),
        "macro_recall":     float(macro_rec),
        "binary_acc_per_class": {e: float(bin_acc[i]) for i, e in enumerate(EMOTIONS)},
        "per_class":        per_class,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print("=" * w)


if __name__ == "__main__":
    main()
