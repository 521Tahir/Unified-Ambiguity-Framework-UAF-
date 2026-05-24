#!/usr/bin/env python3
"""
Re-evaluate saved MOSEI baseline checkpoints to compute WAcc.
Loads best.pt and runs inference on test set only — no retraining.

Usage:
  python scripts/eval/eval_mosei_wacc.py --config configs/mosei/mult_mosei.yaml \
      --checkpoint experiments/results/mosei/mult/best.pt
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.mosei.mosei_utt_dataset import MoseiUttDataset, collate_mosei_utt
from evaluation.metrics.classification import multilabel_metrics, print_metrics
from models.chado.trainer_utils import tune_thresholds, apply_thresholds

MOSEI_EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def build_model(cfg):
    baseline = cfg.get("baseline", "baseline_fusion").lower()
    mc = cfg["model"]
    nc = cfg["data"]["num_classes"]
    common = dict(
        num_classes=nc,
        d_model=mc.get("d_model", 256),
        text_model=mc.get("text_model_name", "roberta-base"),
        use_audio=mc.get("use_audio", True),
        use_video=mc.get("use_video", True),
        dropout=mc.get("dropout", 0.2),
    )
    if baseline == "mult":
        from baselines.mult.mult_mosei import MulTMosei
        return MulTMosei(**common, n_layers=mc.get("n_layers", 3), n_heads=mc.get("n_heads", 4))
    elif baseline == "mmdfn":
        from baselines.mmdfn.mmdfn_mosei import MMDFNMosei
        return MMDFNMosei(**common, n_heads=mc.get("n_heads", 4), n_graph_layers=mc.get("n_graph_layers", 2))
    elif baseline == "ctnet":
        from baselines.ctnet.ctnet_mosei import CTNetMosei
        return CTNetMosei(**common, n_heads=mc.get("n_heads", 4))
    elif baseline == "lf_lstm":
        from baselines.lflstm.lflstm_mosei import LFLSTMMosei
        return LFLSTMMosei(**common)
    elif baseline == "bpmult":
        from baselines.bpmult.bpmult_mosei import BPMulTMosei
        return BPMulTMosei(**common, n_layers=mc.get("n_layers", 3), n_heads=mc.get("n_heads", 4))
    else:
        from models.fusion.mosei_fusion import BaselineFusion
        return BaselineFusion(
            num_classes=nc, d_model=mc.get("d_model", 256),
            use_audio=mc.get("use_audio", True), use_video=mc.get("use_video", True),
        )


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dc = cfg["data"]

    # Build val + test datasets for threshold tuning
    val_ds = MoseiUttDataset(dc["val_manifest"], max_text_len=dc.get("max_text_len", 96))
    test_ds = MoseiUttDataset(dc["test_manifest"], max_text_len=dc.get("max_text_len", 96))

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=4, collate_fn=collate_mosei_utt, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=4, collate_fn=collate_mosei_utt, pin_memory=True)

    # Load model
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Handle different checkpoint formats: {"model": state} or {"model_state": state} or raw state
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    # Strip DDP prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    def run_inference(loader):
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                logits, _ = model(batch)
                all_logits.append(logits.cpu())
                all_labels.append(batch["label"].cpu())
        return torch.cat(all_logits), torch.cat(all_labels)

    print("Running val inference for threshold tuning...")
    val_logits, val_labels = run_inference(val_loader)
    val_labels_np = val_labels.numpy()

    # Tune thresholds: pass raw logits + labels (tune_thresholds applies sigmoid internally)
    thresholds = tune_thresholds(val_logits, val_labels, objective="macro_f1")
    print(f"Tuned thresholds: {[f'{t:.3f}' for t in thresholds]}")

    print("Running test inference...")
    test_logits, test_labels = run_inference(test_loader)
    test_labels_np = test_labels.numpy()

    # apply_thresholds expects probabilities
    test_preds = apply_thresholds(torch.sigmoid(test_logits), thresholds).numpy()
    metrics = multilabel_metrics(test_labels_np, test_preds, MOSEI_EMOTIONS)

    print_metrics(metrics, prefix="[test] ")
    print(f"\n>>> WAcc = {metrics['wacc']:.4f}  |  Macro-F1 = {metrics['macro_f1']:.4f}")

    # Save updated results
    out_dir = os.path.dirname(args.checkpoint)
    out_path = os.path.join(out_dir, "test_results_wacc.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
