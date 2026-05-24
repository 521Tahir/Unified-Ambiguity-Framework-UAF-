"""
Extract AR-GOT (CHADO) 5-factor embeddings from the IEMOCAP dataset.

Saves: experiments/results/iemocap/factor_analysis/argot_factor_embeddings.npz

Arrays:
  z_c, z_u, z_tau, z_m, z_e   — factor embeddings
  gender                        — 0=Male, 1=Female
  speaker_session_id            — int 0-9  (Ses01M=0 ... Ses05F=9)
  duration_sec                  — float (end - start)
  age_proxy_label               — 0=Short (<=3.6s), 1=Long (>3.6s)
  label_4                       — emotion label int
  split                         — 0=train, 1=val, 2=test
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datasets.iemocap.iemocap_dataset import IEMOCAPDataset, collate_iemocap, LABEL2ID
from models.chado.model import CHADOTrimodal


# ─── speaker map ──────────────────────────────────────────────────────────────
_SPEAKER_MAP = {
    f"Ses0{s}{g}": (s - 1) * 2 + (0 if g == "M" else 1)
    for s in range(1, 6) for g in ("M", "F")
}   # Ses01M=0, Ses01F=1, ..., Ses05M=8, Ses05F=9

AGE_THRESHOLD = 3.6   # seconds


def _gender_from_utt(utt_id: str) -> int:
    """0=Male, 1=Female from utt_id like Ses02M_..."""
    return 0 if utt_id[5] == "M" else 1


def _spk_id_from_utt(utt_id: str) -> int:
    key = utt_id[:6]
    return _SPEAKER_MAP.get(key, -1)


# ─── dataset wrapper that also returns row metadata ──────────────────────────
class IEMOCAPWithMeta(IEMOCAPDataset):
    """Thin subclass that appends metadata fields to each sample."""

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        r = self.df.iloc[idx]
        utt_id = str(r["utt_id"])
        start   = float(r.get("start", 0.0) or 0.0)
        end     = float(r.get("end",   0.0) or 0.0)
        dur     = max(0.0, end - start)
        item["_gender"]     = _gender_from_utt(utt_id)
        item["_spk_id"]     = _spk_id_from_utt(utt_id)
        item["_dur"]        = dur
        item["_age_proxy"]  = 1 if dur > AGE_THRESHOLD else 0
        item["_idx"]        = idx
        return item


def collate_with_meta(batch):
    base = collate_iemocap(batch)
    base["_gender"]    = torch.tensor([b["_gender"]    for b in batch], dtype=torch.long)
    base["_spk_id"]    = torch.tensor([b["_spk_id"]    for b in batch], dtype=torch.long)
    base["_dur"]       = torch.tensor([b["_dur"]       for b in batch], dtype=torch.float32)
    base["_age_proxy"] = torch.tensor([b["_age_proxy"] for b in batch], dtype=torch.long)
    base["_idx"]       = torch.tensor([b["_idx"]       for b in batch], dtype=torch.long)
    return base


# ─── model builder ────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device) -> CHADOTrimodal:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # infer roberta version from layer count
    keys = list(ck["model"].keys())
    layer_ids = [int(k.split("encoder.layer.")[1].split(".")[0])
                 for k in keys if "encoder.layer." in k]
    n_layers = max(layer_ids) + 1 if layer_ids else 12
    text_model = "roberta-large" if n_layers >= 24 else "roberta-base"

    model = CHADOTrimodal(
        text_model_name=text_model,
        audio_model_name="facebook/wav2vec2-base",
        video_model_name="google/vit-base-patch16-224-in21k",
        num_classes=4,
        proj_dim=256,
        backbone="ctnet",
        use_causal=True,
        use_hyperbolic=False,   # not needed for extraction
        use_ot=False,
        use_mad=False,
        n_speakers=10,
    )
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}")
    model.eval().to(device)
    print(f"  Loaded {ckpt_path}  (roberta={text_model}, "
          f"val_macro_f1={ck.get('meta', {}).get('val_macro_f1', '?'):.4f})")
    return model


# ─── extraction loop ─────────────────────────────────────────────────────────
@torch.no_grad()
def extract_split(model, loader, device):
    all_z_c, all_z_u, all_z_tau, all_z_m, all_z_e = [], [], [], [], []
    all_gender, all_spk, all_dur, all_age, all_label = [], [], [], [], []
    all_idx = []

    for batch in tqdm(loader, leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        wav            = batch.get("wav")
        if wav is not None:
            wav = wav.to(device)
        pv = batch.get("pixel_values")
        if pv is not None:
            pv = pv.to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            _, _, aux = model(
                text_input={"input_ids": input_ids, "attention_mask": attention_mask},
                audio_wave=wav,
                video_frames=pv,
            )

        factors = aux.get("factors")
        if factors is None:
            raise RuntimeError("Checkpoint has no causal module — wrong checkpoint?")

        z_c, z_u, z_t, z_m, z_e = [f.float().cpu() for f in factors]
        all_z_c.append(z_c);  all_z_u.append(z_u)
        all_z_tau.append(z_t); all_z_m.append(z_m); all_z_e.append(z_e)

        all_gender.append(batch["_gender"].cpu())
        all_spk.append(batch["_spk_id"].cpu())
        all_dur.append(batch["_dur"].cpu())
        all_age.append(batch["_age_proxy"].cpu())
        all_label.append(batch["label"].cpu())
        all_idx.append(batch["_idx"].cpu())

    return {
        "z_c":   torch.cat(all_z_c).numpy(),
        "z_u":   torch.cat(all_z_u).numpy(),
        "z_tau": torch.cat(all_z_tau).numpy(),
        "z_m":   torch.cat(all_z_m).numpy(),
        "z_e":   torch.cat(all_z_e).numpy(),
        "gender":            torch.cat(all_gender).numpy(),
        "speaker_session_id": torch.cat(all_spk).numpy(),
        "duration_sec":      torch.cat(all_dur).numpy(),
        "age_proxy_label":   torch.cat(all_age).numpy(),
        "label_4":           torch.cat(all_label).numpy(),
        "idx":               torch.cat(all_idx).numpy(),
    }


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   default="experiments/results/iemocap/chado_wo_mad/best.pt")
    p.add_argument("--cache",  default="data/cache/iemocap")
    p.add_argument("--train-csv", default="data/manifests/iemocap/train.csv")
    p.add_argument("--val-csv",   default="data/manifests/iemocap/val.csv")
    p.add_argument("--test-csv",  default="data/manifests/iemocap/test.csv")
    p.add_argument("--out",    default="experiments/results/iemocap/factor_analysis/argot_factor_embeddings.npz")
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--workers",type=int, default=4)
    p.add_argument("--gpu",    type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.ckpt, device)

    splits_data = []
    for split_name, csv_path, split_id in [
        ("train", args.train_csv, 0),
        ("val",   args.val_csv,   1),
        ("test",  args.test_csv,  2),
    ]:
        print(f"\nExtracting {split_name} ...")
        cache_dir = os.path.join(args.cache, split_name) if args.cache else None
        ds = IEMOCAPWithMeta(
            csv_path=csv_path,
            text_model_name="roberta-base",   # tokeniser only — weights from ckpt
            image_model_name="google/vit-base-patch16-224-in21k",
            feature_cache_dir=cache_dir,
            use_audio=True,
            use_video=True,
        )
        loader = DataLoader(
            ds, batch_size=args.batch, shuffle=False,
            num_workers=args.workers, collate_fn=collate_with_meta,
            pin_memory=(device.type == "cuda"),
        )
        data = extract_split(model, loader, device)
        data["split"] = np.full(len(data["label_4"]), split_id, dtype=np.int8)
        splits_data.append(data)
        print(f"  {len(data['label_4'])} utterances extracted")

    # concatenate all splits
    combined = {}
    for key in splits_data[0]:
        combined[key] = np.concatenate([d[key] for d in splits_data], axis=0)

    n = len(combined["label_4"])
    print(f"\nTotal utterances: {n}")
    print(f"  z_c shape: {combined['z_c'].shape}")
    print(f"  Gender distribution: {np.bincount(combined['gender'])}")
    print(f"  Speaker distribution: {np.bincount(combined['speaker_session_id'])}")
    print(f"  Age-proxy distribution: {np.bincount(combined['age_proxy_label'])}")
    print(f"  Duration range: {combined['duration_sec'].min():.2f}–{combined['duration_sec'].max():.2f}s, median={np.median(combined['duration_sec']):.2f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **combined)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
