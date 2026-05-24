#!/usr/bin/env python3
"""
Build meld_train.csv / meld_val.csv / meld_test.csv from raw MELD data.
Each row: text, audio_path (mp4), video_path (mp4), emotion, dialogue_id, utt_id

The MELD dataset stores each utterance as a separate .mp4 file.
Audio is extracted on-the-fly by the dataset loader via torchaudio.

Usage:
  python scripts/preprocess/build_meld_csv.py \
      --raw_root /home/tahirahmad/Project_Code/CHADO_MELD/data/raw/MELD.Raw/MELD.Raw \
      --out_dir  /home/tahirahmad/Project_Code/CHADO_EMNLP/data/processed/meld
"""
import argparse
import os

import pandas as pd


# MELD emotion label mapping (lowercase, strip spaces)
EMO_MAP = {
    "neutral":   "neutral",
    "joy":       "joy",
    "surprise":  "surprise",
    "anger":     "anger",
    "sadness":   "sadness",
    "disgust":   "disgust",
    "fear":      "fear",
}


def build_split(csv_path: str, video_dir: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []

    for _, r in df.iterrows():
        dia = int(r["Dialogue_ID"])
        utt = int(r["Utterance_ID"])
        text = str(r["Utterance"]).strip()
        emotion = str(r["Emotion"]).strip().lower()

        if emotion not in EMO_MAP:
            continue

        # MP4 filename pattern: dia{D}_utt{U}.mp4
        mp4_name = f"dia{dia}_utt{utt}.mp4"
        mp4_path = os.path.join(video_dir, mp4_name)

        if not os.path.exists(mp4_path):
            continue  # skip missing files

        rows.append({
            "text":        text,
            "audio_path":  mp4_path,  # same file used for audio (torchaudio reads mp4)
            "video_path":  mp4_path,
            "emotion":     EMO_MAP[emotion],
            "dialogue_id": dia,
            "utt_id":      utt,
        })

    result = pd.DataFrame(rows)
    print(f"  {split}: {len(result)} samples ({len(df) - len(result)} skipped)")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", default="/home/tahirahmad/Project_Code/CHADO_MELD/data/raw/MELD.Raw/MELD.Raw")
    p.add_argument("--out_dir",  default="/home/tahirahmad/Project_Code/CHADO_EMNLP/data/processed/meld")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    root = args.raw_root

    splits = [
        ("train", os.path.join(root, "train_sent_emo.csv"),
                  os.path.join(root, "train_splits")),
        ("val",   os.path.join(root, "dev_sent_emo.csv"),
                  os.path.join(root, "dev_splits_complete")),
        ("test",  os.path.join(root, "test_sent_emo.csv"),
                  os.path.join(root, "output_repeated_splits_test")),
    ]

    for split, csv_p, vid_dir in splits:
        if not os.path.exists(csv_p):
            print(f"[WARN] CSV not found: {csv_p}")
            continue
        if not os.path.isdir(vid_dir):
            print(f"[WARN] Video dir not found: {vid_dir}")
            continue
        df = build_split(csv_p, vid_dir, split)
        out = os.path.join(args.out_dir, f"meld_{split}.csv")
        df.to_csv(out, index=False)
        print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
