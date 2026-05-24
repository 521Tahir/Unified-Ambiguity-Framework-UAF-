#!/usr/bin/env python3
"""
CMU-MOSEI CHADO per-class emotion accuracy evaluator.
6-class multi-label classification (happy/sad/anger/surprise/disgust/fear).

Workflow:
  1. Tune per-class thresholds on val set (maximise macro-F1)
  2. Apply tuned thresholds to test set
  3. Report per-class binary accuracy, P/R/F1, WAcc, Macro-F1

Usage (VS Code terminal):
  cd /home/tahirahmad/Project_Code/CHADO_EMNLP
  CUDA_VISIBLE_DEVICES=5 python3 scripts/eval/eval_mosei_perclass.py \
      --config configs/mosei/chado_mosei.yaml \
      --ckpt   experiments/results/mosei/chado/best.pt
"""
import argparse, os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.mosei.mosei_utt_dataset import (
    MoseiUttDataset as MoseiDataset, collate_mosei_utt as collate_mosei,
)
from models.chado.model import CHADOFeature
from models.chado.trainer_utils import tune_thresholds, apply_thresholds
from evaluation.metrics.classification import multilabel_metrics

MOSEI_EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mosei/chado_mosei.yaml")
    p.add_argument("--ckpt",   default="experiments/results/mosei/chado/best.pt")
    p.add_argument("--batch",  type=int, default=64)
    return p.parse_args()


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        labels = batch["label"].to(device)
        logits, _, _, _ = model(batch_dev)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Ckpt    : {args.ckpt}\n")

    dc = cfg["data"]
    mc = cfg["model"]
    cc = cfg.get("chado", {})

    # ── Build dataloaders ──────────────────────────────────────────────────
    def _make(manifest):
        return MoseiDataset(
            manifest_path=manifest,
            max_audio_len=dc.get("max_audio_len", 50),
            max_video_len=dc.get("max_video_len", 30),
            label_thr=dc.get("label_thr", 0.0),
        )

    val_loader  = DataLoader(
        _make(dc["val_manifest"]),
        batch_size=args.batch, shuffle=False, num_workers=4,
        pin_memory=True, collate_fn=collate_mosei,
    )
    test_loader = DataLoader(
        _make(dc["test_manifest"]),
        batch_size=args.batch, shuffle=False, num_workers=4,
        pin_memory=True, collate_fn=collate_mosei,
    )

    # ── Build model ────────────────────────────────────────────────────────
    model = CHADOFeature(
        num_classes     = dc["num_classes"],
        d_model         = mc.get("d_model", 256),
        use_audio       = mc.get("use_audio", True),
        use_video       = mc.get("use_video", True),
        text_model      = mc.get("text_model_name", "roberta-base"),
        modality_dropout= mc.get("modality_dropout", 0.1),
        use_causal      = cc.get("use_causal", True),
        use_hyperbolic  = cc.get("use_hyperbolic", True),
        use_ot          = cc.get("use_ot", True),
        use_mad         = cc.get("use_mad", True),
        w_causal        = cc.get("w_causal", 0.005),
        w_hyperbolic    = cc.get("w_hyperbolic", 0.005),
        w_ot            = cc.get("w_ot", 0.002),
        w_mad           = cc.get("w_mad", 0.02),
        mad_gamma       = cc.get("mad_gamma", 0.5),
        ot_sigma        = cc.get("ot_sigma", 0.05),
        ot_eps          = cc.get("ot_eps", 0.1),
        ot_iters        = cc.get("ot_iters", 20),
    ).to(device)

    ckpt  = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded  : {n_params:.1f}M parameters\n")

    # ── Tune thresholds on val set ─────────────────────────────────────────
    print("Tuning thresholds on validation set...")
    val_logits, val_labels = collect(model, val_loader, device)
    thresholds = tune_thresholds(val_logits, val_labels, objective="macro_f1")
    print(f"Tuned thresholds: { {e: f'{t:.3f}' for e, t in zip(MOSEI_EMOTIONS, thresholds)} }\n")

    # ── Evaluate on test set ───────────────────────────────────────────────
    print("Running inference on test set...")
    test_logits, test_labels = collect(model, test_loader, device)

    y_prob = torch.sigmoid(test_logits).numpy()
    y_pred = apply_thresholds(torch.sigmoid(test_logits), thresholds).numpy()
    y_true = (test_labels > 0.5).int().numpy()

    metrics = multilabel_metrics(y_true, y_pred, MOSEI_EMOTIONS)

    # ── Print results ──────────────────────────────────────────────────────
    print("=" * 65)
    print("  CMU-MOSEI CHADO — 6-Class Emotion Results (test split)")
    print("=" * 65)
    print(f"\n  Task:              Multi-label, 6 emotions (BCE + threshold)")
    print(f"  Subset Accuracy  : {metrics['subset_accuracy']*100:.2f}%  (all 6 labels exact match)")
    print(f"  WAcc (per-label) : {metrics['wacc']*100:.2f}%  (avg binary acc per emotion)")
    print(f"  Macro F1         : {metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted F1      : {metrics['weighted_f1']*100:.2f}%")
    print(f"  Macro Precision  : {metrics['macro_precision']*100:.2f}%")
    print(f"  Macro Recall     : {metrics['macro_recall']*100:.2f}%")

    print(f"\n  {'Emotion':<12} {'BinAcc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("  " + "-" * 58)
    bacc = metrics.get("binary_acc_per_class", {})
    for name, m in metrics["per_class"].items():
        ba = bacc.get(name, 0.0)
        print(f"  {name:<12} {ba*100:>7.2f}% {m['precision']*100:>7.2f}% "
              f"{m['recall']*100:>7.2f}% {m['f1']*100:>7.2f}% {m['support']:>9}")

    # ── Confidence analysis ────────────────────────────────────────────────
    print(f"\n  Average prediction confidence per emotion:")
    print(f"  {'Emotion':<12} {'Mean Prob':>10} {'Pred Rate':>10} {'True Rate':>10}")
    print("  " + "-" * 45)
    for i, name in enumerate(MOSEI_EMOTIONS):
        mp   = y_prob[:, i].mean()
        pr   = y_pred[:, i].mean()
        tr   = y_true[:, i].mean()
        print(f"  {name:<12} {mp:>9.3f}  {pr*100:>8.1f}%  {tr*100:>9.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(args.ckpt), "perclass_test_mosei.json")
    save_obj = {
        "subset_accuracy": metrics["subset_accuracy"],
        "wacc":            metrics["wacc"],
        "macro_f1":        metrics["macro_f1"],
        "weighted_f1":     metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall":    metrics["macro_recall"],
        "thresholds":      {e: float(t) for e, t in zip(MOSEI_EMOTIONS, thresholds)},
        "per_class":       metrics["per_class"],
        "binary_acc_per_class": metrics.get("binary_acc_per_class", {}),
    }
    with open(out_path, "w") as f:
        json.dump(save_obj, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
