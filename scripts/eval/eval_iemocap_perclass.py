#!/usr/bin/env python3
"""
IEMOCAP CHADO per-class emotion accuracy evaluator.
4-class: neutral / happy / angry / sad.

Usage (VS Code terminal):
  cd /home/tahirahmad/Project_Code/CHADO_EMNLP
  CUDA_VISIBLE_DEVICES=5 python3 scripts/eval/eval_iemocap_perclass.py \
      --config configs/iemocap/chado_iemocap.yaml \
      --ckpt   experiments/results/iemocap/chado/best.pt
"""
import argparse, os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
)

from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap, CLASS_NAMES
from models.chado.model import CHADOTrimodal
from training.engine.trainer import load_checkpoint

LABEL_NAMES = CLASS_NAMES   # ["neu", "hap", "ang", "sad"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/iemocap/chado_iemocap.yaml")
    p.add_argument("--ckpt",   default="experiments/results/iemocap/chado/best.pt")
    p.add_argument("--split",  default="test", choices=["train", "val", "test"])
    p.add_argument("--batch",  type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Ckpt    : {args.ckpt}")
    print(f"Split   : {args.split}\n")

    dc = cfg["data"]
    mc = cfg["model"]
    cc = cfg.get("chado", {})

    # ── Dataset ────────────────────────────────────────────────────────────
    csv_key = f"{args.split}_csv"
    ds = IEMOCAPDataset(
        csv_path         = dc[csv_key],
        text_model_name  = mc.get("text_model_name", "roberta-base"),
        image_model_name = mc.get("video_model_name", "google/vit-base-patch16-224-in21k"),
        max_text_len     = dc.get("max_text_len", 96),
        audio_sec        = dc.get("max_audio_seconds", 4.0),
        n_frames         = dc.get("num_frames", 8),
        use_audio        = mc.get("use_audio", True),
        use_video        = mc.get("use_video", True),
    )
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=collate_iemocap,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = CHADOTrimodal(
        text_model_name  = mc["text_model_name"],
        audio_model_name = mc["audio_model_name"],
        video_model_name = mc["video_model_name"],
        num_classes      = dc["num_classes"],
        proj_dim         = mc.get("proj_dim", 256),
        dropout          = mc.get("dropout", 0.2),
        use_text         = mc.get("use_text", True),
        use_audio        = mc.get("use_audio", True),
        use_video        = mc.get("use_video", True),
        use_gated_fusion = mc.get("use_gated_fusion", True),
        backbone         = cc.get("backbone", "ctnet"),
        n_heads          = mc.get("n_heads", 4),
        n_speakers       = cc.get("n_speakers", 10),
        use_causal       = cc.get("use_causal", True),
        use_hyperbolic   = cc.get("use_hyperbolic", True),
        use_ot           = cc.get("use_ot", True),
        use_mad          = cc.get("use_mad", False),
        w_causal         = cc.get("w_causal", 0.05),
        w_hyperbolic     = cc.get("w_hyperbolic", 0.05),
        w_ot             = cc.get("w_ot", 0.02),
        w_mad            = cc.get("w_mad", 0.0),
        w_speaker        = cc.get("w_speaker", 0.10),
        w_turn           = cc.get("w_turn", 0.05),
        w_modal          = cc.get("w_modal", 0.05),
        w_context        = cc.get("w_context", 0.05),
    ).to(device)

    load_checkpoint(model, args.ckpt, strict=False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded  : {n_params:.1f}M parameters\n")
    model.eval()

    # ── Inference ──────────────────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            wav   = batch["wav"].to(device)           if mc.get("use_audio", True) and "wav" in batch else None
            pixel = batch["pixel_values"].to(device)  if mc.get("use_video", True) and "pixel_values" in batch else None
            labels = batch["label"]

            logits, _, _ = model(
                text_input   = {"input_ids": input_ids, "attention_mask": attention_mask},
                audio_wave   = wav,
                video_frames = pixel,
            )
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    # ── Metrics ────────────────────────────────────────────────────────────
    acc        = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1= f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    macro_rec  = recall_score(y_true, y_pred,    average="macro",    zero_division=0)
    cm         = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES))))

    # ── Print results ──────────────────────────────────────────────────────
    print("=" * 62)
    print(f"  IEMOCAP CHADO — 4-Class Emotion Results ({args.split} split)")
    print("=" * 62)
    print(f"\n  Overall Accuracy   : {acc*100:.2f}%")
    print(f"  Macro F1           : {macro_f1*100:.2f}%")
    print(f"  Weighted F1        : {weighted_f1*100:.2f}%")
    print(f"  Macro Precision    : {macro_prec*100:.2f}%")
    print(f"  Macro Recall       : {macro_rec*100:.2f}%")

    print(f"\n  {'Emotion':<10} {'Correct':>8} {'Total':>8} {'Acc':>8} "
          f"{'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 64)

    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    prec_arr  = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_arr   = recall_score(y_true, y_pred,    average=None, zero_division=0)
    f1_arr    = f1_score(y_true, y_pred,        average=None, zero_division=0)

    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        correct = int(cm[i, i])
        total   = int(cm[i].sum())
        per_class[name] = {
            "correct": correct, "total": total,
            "accuracy": float(per_class_acc[i]),
            "precision": float(prec_arr[i]),
            "recall": float(rec_arr[i]),
            "f1": float(f1_arr[i]),
        }
        print(f"  {name:<10} {correct:>8} {total:>8} {per_class_acc[i]*100:>7.2f}% "
              f"{prec_arr[i]*100:>7.2f}% {rec_arr[i]*100:>7.2f}% {f1_arr[i]*100:>7.2f}%")

    print(f"\n  Confusion Matrix (rows=Actual, cols=Predicted):")
    header = "  " + " " * 10 + "".join(f"{n:>8}" for n in LABEL_NAMES)
    print(header)
    print("  " + "-" * (10 + 8 * len(LABEL_NAMES)))
    for i, name in enumerate(LABEL_NAMES):
        row = "  " + f"{name:<10}" + "".join(f"{cm[i,j]:>8}" for j in range(len(LABEL_NAMES)))
        print(row)

    print(f"\n  Full classification report:")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        digits=4, zero_division=0,
    ))

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(args.ckpt), f"perclass_{args.split}_iemocap.json")
    result = {
        "accuracy": float(acc), "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_prec), "macro_recall": float(macro_rec),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
