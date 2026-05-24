"""
Pre-extract and cache video frames + audio for MELD to eliminate per-epoch AVI decoding.
Saves {idx:07d}.pt per sample with {"video": float16[T,3,H,W], "audio": float32[L]}.
Run on CPU only; does not interfere with GPU training jobs.

Usage:
    python scripts/preprocess/cache_features_meld.py --workers 16
"""
import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from datasets.meld.meld_dataset import load_video_frames_opencv, load_audio_waveform

SPLITS = {
    "train": "data/processed/meld/meld_train.csv",
    "val":   "data/processed/meld/meld_val.csv",
    "test":  "data/processed/meld/meld_test.csv",
}
CACHE_BASE = "data/cache/meld"

NUM_FRAMES    = 8
FRAME_SIZE    = 224
SAMPLE_RATE   = 16000
MAX_AUDIO_SEC = 6.0


def process_one(args):
    idx, video_path, audio_path, out_path = args
    if os.path.exists(out_path):
        return idx, True, "cached"
    try:
        video = load_video_frames_opencv(video_path, NUM_FRAMES, FRAME_SIZE)  # [T,3,H,W] f32
        audio = load_audio_waveform(audio_path, SAMPLE_RATE, MAX_AUDIO_SEC)  # [L] f32
        torch.save(
            {"video": video.to(torch.float16), "audio": audio},
            out_path,
        )
        return idx, True, "ok"
    except Exception as e:
        return idx, False, str(e)


def cache_split(split, csv_path, workers):
    df = pd.read_csv(csv_path)
    out_dir = os.path.join(CACHE_BASE, split)
    os.makedirs(out_dir, exist_ok=True)

    tasks = []
    for idx, row in df.iterrows():
        out_path = os.path.join(out_dir, f"{idx:07d}.pt")
        tasks.append((idx, str(row["video_path"]), str(row["audio_path"]), out_path))

    print(f"\n[{split}] {len(tasks)} samples → {out_dir}")

    errors = 0
    from multiprocessing import Pool
    with Pool(processes=workers) as pool:
        for _, ok, msg in tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks), desc=split):
            if not ok:
                errors += 1

    print(f"[{split}] done. errors={errors}/{len(tasks)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    for split in args.splits:
        csv_path = SPLITS[split]
        if not os.path.exists(csv_path):
            print(f"[SKIP] {csv_path} not found")
            continue
        cache_split(split, csv_path, args.workers)

    print("\nAll splits cached. Add to config:")
    print(f"  feature_cache_dir: {os.path.abspath(CACHE_BASE)}")


if __name__ == "__main__":
    main()
