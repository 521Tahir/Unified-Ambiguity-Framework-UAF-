#!/usr/bin/env python3
"""
MELD per-class emotion accuracy evaluator.

Loads the CHADO checkpoint, runs inference on the test set,
and prints a full per-class breakdown.

Usage (VS Code terminal):
  cd /home/tahirahmad/Project_Code/CHADO_EMNLP
  CUDA_VISIBLE_DEVICES=5 python3 scripts/eval/eval_meld_perclass.py \
      --config configs/meld/chado_meld.yaml \
      --ckpt   experiments/results/meld/chado/best.pt
"""
import argparse, os, sys, json, functools
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from datasets.meld.meld_dataset import (
    MeldDataset, collate_meld, build_label_map_from_order, EMO_ORDER_7,
)
from models.chado.model import CHADOTrimodal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/meld/chado_meld.yaml")
    p.add_argument("--ckpt",   default="experiments/results/meld/chado/best.pt")
    p.add_argument("--split",  default="test", choices=["train","val","test"])
    p.add_argument("--batch",  type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Ckpt   : {args.ckpt}")
    print(f"Split  : {args.split}")

    dc  = cfg["data"]
    mc  = cfg["model"]
    cc  = cfg.get("chado", {})

    label_map = build_label_map_from_order(EMO_ORDER_7)
    csv_key   = f"{args.split}_csv"
    ds = MeldDataset(
        csv_path        = dc[csv_key],
        label_col       = dc.get("label_col","emotion"),
        text_col        = dc.get("text_col","text"),
        audio_path_col  = dc.get("audio_path_col","audio_path"),
        video_path_col  = dc.get("video_path_col","video_path"),
        utt_id_col      = dc.get("utt_id_col","utt_id"),
        label_map       = label_map,
        text_model_name = mc.get("text_model_name","roberta-base"),
        sample_rate     = dc.get("sample_rate",16000),
        max_audio_seconds = dc.get("max_audio_seconds",6.0),
        num_frames      = dc.get("num_frames",8),
        frame_size      = dc.get("frame_size",224),
        context_turns   = dc.get("context_turns",5),
        use_text        = mc.get("use_text",True),
        use_audio       = mc.get("use_audio",True),
        use_video       = mc.get("use_video",True),
    )
    tok = ds.tokenizer
    loader = DataLoader(
        ds,
        batch_size  = args.batch,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = functools.partial(
            collate_meld,
            tokenizer = tok,
            use_text  = mc.get("use_text", True),
            use_audio = mc.get("use_audio", True),
            use_video = mc.get("use_video", True),
        ),
    )

    # Build model
    model = CHADOTrimodal(
        num_classes     = dc["num_classes"],
        text_model_name = mc.get("text_model_name","roberta-base"),
        audio_model_name= mc.get("audio_model_name","facebook/wav2vec2-base"),
        video_model_name= mc.get("video_model_name","google/vit-base-patch16-224-in21k"),
        proj_dim        = mc.get("proj_dim",256),
        dropout         = mc.get("dropout",0.2),
        use_text        = mc.get("use_text",True),
        use_audio       = mc.get("use_audio",True),
        use_video       = mc.get("use_video",True),
        use_gated_fusion= mc.get("use_gated_fusion",True),
        n_heads         = mc.get("n_heads",4),
        n_speakers      = cc.get("n_speakers",7),
        backbone        = cc.get("backbone","ctnet"),
        use_causal      = cc.get("use_causal",True),
        use_hyperbolic  = cc.get("use_hyperbolic",True),
        use_ot          = cc.get("use_ot",True),
        use_mad         = cc.get("use_mad",False),
        w_causal        = cc.get("w_causal",0.01),
        w_hyperbolic    = cc.get("w_hyperbolic",0.01),
        w_ot            = cc.get("w_ot",0.005),
        w_mad           = cc.get("w_mad",0.0),
        w_speaker       = cc.get("w_speaker",0.1),
        w_turn          = cc.get("w_turn",0.05),
        w_modal         = cc.get("w_modal",0.05),
        w_context       = cc.get("w_context",0.05),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)\n")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            text_in = {k: batch.text_input[k].to(device)
                       for k in ("input_ids","attention_mask")} if batch.text_input else None
            wav     = batch.audio_wave.to(device)   if batch.audio_wave   is not None else None
            frames  = batch.video_frames.to(device) if batch.video_frames is not None else None
            labels  = batch.labels

            logits, _, _ = model(text_input=text_in, audio_wave=wav, video_frames=frames)
            preds = logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    # ── Print results ──────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    print("=" * 62)
    print(f"  MELD Per-Class Emotion Accuracy  ({args.split} split)")
    print("=" * 62)
    print(f"  Overall Accuracy : {acc*100:.2f}%")
    print()

    cls_names = EMO_ORDER_7
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(cls_names))),
        target_names=cls_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    # Per-class accuracy (recall per class)
    print("  Per-class Accuracy (Recall):")
    print(f"  {'Emotion':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("  " + "-" * 42)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(cls_names))))
    for i, name in enumerate(cls_names):
        total   = cm[i].sum()
        correct = cm[i, i]
        cls_acc = correct / total if total > 0 else 0.0
        print(f"  {name:<12} {correct:>8} {total:>8} {cls_acc*100:>7.2f}%")
    print()

    # Save
    out_path = os.path.join(
        os.path.dirname(args.ckpt),
        f"perclass_{args.split}.json"
    )
    result = {
        "overall_accuracy": float(acc),
        "per_class": {
            name: {
                "correct": int(cm[i, i]),
                "total":   int(cm[i].sum()),
                "accuracy": float(cm[i,i]/cm[i].sum()) if cm[i].sum()>0 else 0.0,
            }
            for i, name in enumerate(cls_names)
        }
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
