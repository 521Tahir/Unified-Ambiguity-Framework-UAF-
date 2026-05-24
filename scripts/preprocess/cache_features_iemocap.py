"""
Pre-extract and cache video frames + audio for IEMOCAP to eliminate per-epoch AVI decoding.
Saves {idx:07d}.pt per sample with {"video": float16[T,3,H,W], "audio": float32[L]}.
IEMOCAP uses session-level AVI files with start/end timestamps — seeking is the bottleneck.

Usage:
    python scripts/preprocess/cache_features_iemocap.py --workers 12
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

SPLITS = {
    "train": "data/manifests/iemocap/train.csv",
    "val":   "data/manifests/iemocap/val.csv",
    "test":  "data/manifests/iemocap/test.csv",
}
CACHE_BASE = "data/cache/iemocap"

NUM_FRAMES    = 8
FRAME_SIZE    = 224
AUDIO_SR      = 16000
MAX_AUDIO_SEC = 4.0


def _load_video(avi_path, start, end):
    import cv2
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        return torch.zeros((NUM_FRAMES, 3, FRAME_SIZE, FRAME_SIZE), dtype=torch.float32)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    s_f = int(max(0.0, float(start)) * fps)
    e_f = int(max(0.0, float(end)) * fps)
    if e_f <= s_f:
        e_f = s_f + int(fps)
    idxs = np.linspace(s_f, e_f, num=NUM_FRAMES, dtype=np.int64)
    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [T,H,W,3]
    return torch.from_numpy(arr).permute(0, 3, 1, 2)  # [T,3,H,W]


def _load_audio(wav_path, start, end):
    import soundfile as sf
    import librosa
    max_len = int(AUDIO_SR * MAX_AUDIO_SEC)
    try:
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != AUDIO_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_SR)
        s = int(max(0.0, float(start)) * AUDIO_SR)
        e = int(max(0.0, float(end)) * AUDIO_SR)
        if e <= s:
            e = min(len(audio), s + max_len)
        seg = audio[s:e]
        if len(seg) < max_len:
            seg = np.pad(seg, (0, max_len - len(seg)))
        else:
            seg = seg[:max_len]
        return torch.from_numpy(seg.astype(np.float32))
    except Exception:
        return torch.zeros(max_len, dtype=torch.float32)


def process_one(args):
    idx, avi_path, wav_path, start, end, out_path = args
    if os.path.exists(out_path):
        return idx, True, "cached"
    try:
        video = _load_video(avi_path, start, end)   # [T,3,H,W] f32
        audio = _load_audio(wav_path, start, end)   # [L] f32
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
        tasks.append((
            idx,
            str(row["avi_path"]),
            str(row["wav_path"]),
            float(row["start"]),
            float(row["end"]),
            out_path,
        ))

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
    parser.add_argument("--workers", type=int, default=12)
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
