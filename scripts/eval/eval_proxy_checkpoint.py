"""
Evaluate a saved best.pt checkpoint on the test set.
Works for both IEMOCAP and MELD proxy runs.

Usage:
  python3 scripts/eval/eval_proxy_checkpoint.py \
      --dataset iemocap --proxy mad --seed 42
"""
import argparse, json, os, sys
import numpy as np
import torch
import yaml
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def load_cfg(dataset):
    cfgs = {
        "iemocap": "configs/iemocap/chado_iemocap_v2.yaml",
        "meld":    "configs/meld/chado_meld_final.yaml",
    }
    with open(cfgs[dataset]) as f:
        return yaml.safe_load(f)


def build_model(cfg, dataset, proxy, ckpt_path=None):
    mc = cfg["model"]
    # infer n_speakers from checkpoint to avoid size mismatch
    n_speakers = 10
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        w = ck["model"].get("causal.speaker_head.weight")
        if w is not None:
            n_speakers = w.shape[0]
    if dataset == "iemocap":
        from models.chado.model import CHADOTrimodal
        cc = cfg.get("chado", {})
        model = CHADOTrimodal(
            text_model_name=mc["text_model_name"],
            audio_model_name=mc["audio_model_name"],
            video_model_name=mc["video_model_name"],
            num_classes=cfg["data"]["num_classes"],
            proj_dim=mc.get("proj_dim", 256),
            dropout=mc.get("dropout", 0.15),
            backbone=mc.get("backbone", "ctnet"),
            n_heads=mc.get("n_heads", 4),
            use_causal=cc.get("use_causal", True),
            use_hyperbolic=cc.get("use_hyperbolic", True),
            use_ot=cc.get("use_ot", True),
            use_mad=cc.get("use_mad", True),
            proxy_type=proxy,
            n_speakers=n_speakers,
            w_causal=cc.get("w_causal", 0.015),
            w_hyperbolic=cc.get("w_hyperbolic", 0.015),
            w_ot=cc.get("w_ot", 0.008),
            w_mad=cc.get("w_mad", 0.5),
        )
    else:  # meld
        from models.chado.model import CHADOTrimodal
        cc = cfg.get("chado", {})
        model = CHADOTrimodal(
            text_model_name=mc["text_model_name"],
            audio_model_name=mc["audio_model_name"],
            video_model_name=mc["video_model_name"],
            num_classes=cfg["data"]["num_classes"],
            proj_dim=mc.get("proj_dim", 256),
            dropout=mc.get("dropout", 0.15),
            backbone=mc.get("backbone", "ctnet"),
            n_heads=mc.get("n_heads", 4),
            use_causal=cc.get("use_causal", True),
            use_hyperbolic=cc.get("use_hyperbolic", True),
            use_ot=cc.get("use_ot", True),
            use_mad=cc.get("use_mad", True),
            proxy_type=proxy,
            n_speakers=n_speakers,
            w_causal=cc.get("w_causal", 0.015),
            w_hyperbolic=cc.get("w_hyperbolic", 0.015),
            w_ot=cc.get("w_ot", 0.008),
            w_mad=cc.get("w_mad", 0.5),
        )
    return model


def make_loader(cfg, dataset, split):
    dc = cfg["data"]
    mc = cfg["model"]
    tc = cfg["train"]
    nw = min(4, tc.get("num_workers", 4))
    bs = tc.get("batch_size_per_gpu", 16) * 2

    if dataset == "iemocap":
        from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap
        csv_key = {"train": "train_csv", "val": "val_csv", "test": "test_csv"}[split]
        cache_dir = os.path.join(dc.get("feature_cache_dir", ""), split) if dc.get("feature_cache_dir") else None
        ds = IEMOCAPDataset(
            csv_path=dc[csv_key],
            text_model_name=mc["text_model_name"],
            image_model_name=mc["video_model_name"],
            max_text_len=dc.get("max_text_len", 96),
            audio_sr=dc.get("sample_rate", 16000),
            audio_sec=dc.get("max_audio_seconds", 8.0),
            n_frames=dc.get("num_frames", 8),
            feature_cache_dir=cache_dir,
        )
        return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_iemocap)
    else:
        import functools
        from transformers import AutoTokenizer
        from datasets.meld.meld_dataset import (MeldDataset, collate_meld,
                                                 build_label_map_from_order, EMO_ORDER_7)
        csv_key = {"train": "train_csv", "val": "val_csv", "test": "test_csv"}[split]
        cache_dir = os.path.join(dc.get("feature_cache_dir", ""), split) if dc.get("feature_cache_dir") else None
        label_map = build_label_map_from_order(EMO_ORDER_7)
        tok = AutoTokenizer.from_pretrained(mc["text_model_name"], use_fast=True)
        ds = MeldDataset(
            csv_path=dc[csv_key],
            text_model_name=mc["text_model_name"],
            label_map=label_map,
            text_col=dc["text_col"],
            label_col=dc["label_col"],
            audio_path_col=dc["audio_path_col"],
            video_path_col=dc["video_path_col"],
            utt_id_col=dc.get("utt_id_col", "utt_id"),
            num_frames=dc.get("num_frames", 8),
            frame_size=dc.get("frame_size", 224),
            sample_rate=dc.get("sample_rate", 16000),
            max_audio_seconds=dc.get("max_audio_seconds", 6.0),
            feature_cache_dir=cache_dir,
        )
        collate_fn = functools.partial(collate_meld, tokenizer=tok,
                                       use_text=True, use_audio=True, use_video=True)
        return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_fn)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        # support both dict (IEMOCAP) and MeldBatch dataclass (MELD)
        if hasattr(batch, "labels"):   # MeldBatch
            text   = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
            wav    = batch.audio_wave.to(device)  if batch.audio_wave   is not None else None
            pv     = batch.video_frames.to(device) if batch.video_frames is not None else None
            labels = batch.labels.to(device)
        else:                          # IEMOCAP dict
            text = {"input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device)}
            wav = batch.get("wav");  wav = wav.to(device) if wav is not None else None
            pv  = batch.get("pixel_values"); pv = pv.to(device) if pv is not None else None
            labels = batch["label"].to(device)

        out = model(text_input=text, audio_wave=wav, video_frames=pv)
        logits = out[0]
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    classes = sorted(np.unique(y_true))
    n_cls = len(classes)

    acc      = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    w_f1     = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_p  = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r  = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": w_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "n_samples": int(len(y_true)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["iemocap", "meld"])
    p.add_argument("--proxy",   required=True, choices=["mad", "entropy", "margin"])
    p.add_argument("--seed",    default="42")
    p.add_argument("--gpu",     type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt_path = f"experiments/results/{args.dataset}/proxy_comparison/{args.proxy}/seed_{args.seed}/best.pt"
    out_path  = f"experiments/results/{args.dataset}/proxy_comparison/{args.proxy}/seed_{args.seed}/test_results.json"

    print(f"Dataset: {args.dataset}  Proxy: {args.proxy}  Seed: {args.seed}")
    print(f"Checkpoint: {ckpt_path}")

    cfg = load_cfg(args.dataset)
    model = build_model(cfg, args.dataset, args.proxy, ckpt_path=ckpt_path)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    meta = ck.get("meta", {})
    print(f"  Checkpoint epoch: {meta.get('epoch', '?')}")
    model.eval().to(device)

    loader = make_loader(cfg, args.dataset, "test")
    print(f"  Test samples: {len(loader.dataset)}")

    metrics = evaluate(model, loader, device)
    metrics["checkpoint_epoch"] = meta.get("epoch", -1)
    metrics["proxy"] = args.proxy
    metrics["dataset"] = args.dataset

    print(f"\n  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1: {metrics['weighted_f1']:.4f}")

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
