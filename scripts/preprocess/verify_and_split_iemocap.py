#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLS = [
    "utt_id", "session", "transcript", "wav_path", "avi_path",
    "start", "end", "label_4"
]

def infer_speaker_id(utt_id: str) -> str:
    """
    IEMOCAP utt_id patterns often begin with Ses01F / Ses01M / ...
    We treat that prefix as a speaker id proxy for speaker-disjoint splitting.
    Example: Ses01F_impro01_F000 -> speaker = Ses01F
    """
    m = re.match(r"^(Ses\d{2}[FM])_", str(utt_id))
    return m.group(1) if m else "UNK"

def validate_rows(df: pd.DataFrame, check_audio_header: bool = False) -> pd.DataFrame:
    """
    - Ensures required columns
    - Ensures wav_path/avi_path exist
    - Ensures label_4 non-empty
    - Ensures start/end are numeric and end>start
    - Optionally checks WAV header bytes (cheap sanity check, not full decode)
    """
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Basic cleanup
    df = df.copy()
    df["label_4"] = df["label_4"].astype(str).str.strip()
    df["transcript"] = df["transcript"].astype(str)

    # Numeric sanity
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")

    # File existence checks
    wav_exists = df["wav_path"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))
    avi_exists = df["avi_path"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))

    # Additional sanity
    label_ok = df["label_4"].notna() & (df["label_4"] != "") & (df["label_4"] != "nan")
    time_ok = df["start"].notna() & df["end"].notna() & (df["end"] > df["start"]) & (df["start"] >= 0)

    keep = wav_exists & avi_exists & label_ok & time_ok
    bad = df.loc[~keep].copy()

    # Optional: quick WAV header check (RIFF/WAVE)
    if check_audio_header:
        def wav_header_ok(path: str) -> bool:
            try:
                with open(path, "rb") as f:
                    head = f.read(12)
                return head[:4] == b"RIFF" and head[8:12] == b"WAVE"
            except Exception:
                return False
        wav_hdr = df.loc[keep, "wav_path"].apply(wav_header_ok)
        keep2 = wav_hdr
        bad2 = df.loc[keep].loc[~keep2].copy()
        bad = pd.concat([bad, bad2], ignore_index=True)
        df = df.loc[keep].loc[keep2].copy()
    else:
        df = df.loc[keep].copy()

    # Add speaker_id proxy
    df["speaker_id"] = df["utt_id"].apply(infer_speaker_id)

    return df, bad

def stratified_random_split(df: pd.DataFrame, seed: int, test_size: float, val_size: float):
    """
    Utterance-level stratified split:
      - split off test
      - split remaining into train/val
    """
    y = df["label_4"].values
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=y
    )
    y_tv = df_trainval["label_4"].values
    # val_size is fraction of full dataset; convert to fraction of trainval
    val_frac_of_trainval = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_frac_of_trainval, random_state=seed, stratify=y_tv
    )
    return df_train, df_val, df_test

def speaker_disjoint_split(df: pd.DataFrame, seed: int, test_size: float, val_size: float):
    """
    Speaker-disjoint split approximation using speaker_id.
    We do group splitting by selecting speakers into test/val/train, while trying to keep class balance.
    This is a heuristic (fast, simple), not perfect stratified group split.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    speakers = df["speaker_id"].unique().tolist()
    rng.shuffle(speakers)

    n = len(speakers)
    n_test = max(1, int(round(test_size * n)))
    n_val  = max(1, int(round(val_size * n)))

    test_spk = set(speakers[:n_test])
    val_spk  = set(speakers[n_test:n_test + n_val])
    train_spk= set(speakers[n_test + n_val:])

    df_test = df[df["speaker_id"].isin(test_spk)].copy()
    df_val  = df[df["speaker_id"].isin(val_spk)].copy()
    df_train= df[df["speaker_id"].isin(train_spk)].copy()
    return df_train, df_val, df_test

def print_distribution(name: str, df: pd.DataFrame):
    dist = df["label_4"].value_counts(dropna=False)
    total = len(df)
    print(f"\n[{name}] rows={total}")
    for k, v in dist.items():
        print(f"  {k}: {v} ({(v/total*100):.2f}%)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to iemocap_full_4class.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory for cleaned CSV + splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.10)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--split_mode", choices=["utterance", "speaker"], default="utterance",
                    help="utterance=stratified random; speaker=approx speaker-disjoint")
    ap.add_argument("--check_wav_header", action="store_true",
                    help="Optional quick sanity check for WAV RIFF/WAVE header")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)

    print(f"[INFO] Loaded rows: {len(df)}")
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing columns: {missing_cols}")

    # Validate
    df_clean, df_bad = validate_rows(df, check_audio_header=args.check_wav_header)

    print(f"[INFO] Valid rows: {len(df_clean)}")
    print(f"[INFO] Dropped/invalid rows: {len(df_bad)}")

    # Save invalid rows for inspection
    bad_path = os.path.join(args.out_dir, "invalid_rows.csv")
    df_bad.to_csv(bad_path, index=False)
    print(f"[OK] Saved invalid rows to: {bad_path}")

    # Save cleaned master CSV
    clean_path = os.path.join(args.out_dir, "iemocap_4class_clean.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"[OK] Saved cleaned CSV to: {clean_path}")

    # Print distributions
    print_distribution("CLEAN_ALL", df_clean)

    # Split
    if args.split_mode == "utterance":
        df_train, df_val, df_test = stratified_random_split(
            df_clean, seed=args.seed, test_size=args.test_size, val_size=args.val_size
        )
    else:
        df_train, df_val, df_test = speaker_disjoint_split(
            df_clean, seed=args.seed, test_size=args.test_size, val_size=args.val_size
        )

    # Save splits
    split_dir = os.path.join(args.out_dir, f"splits_{args.split_mode}_seed{args.seed}")
    os.makedirs(split_dir, exist_ok=True)

    train_path = os.path.join(split_dir, "train.csv")
    val_path   = os.path.join(split_dir, "val.csv")
    test_path  = os.path.join(split_dir, "test.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"\n[OK] Saved splits to:\n  {split_dir}")
    print_distribution("TRAIN", df_train)
    print_distribution("VAL", df_val)
    print_distribution("TEST", df_test)

    # Quick leakage warning (only relevant in utterance mode)
    if args.split_mode == "utterance":
        # check speaker overlap
        tr_spk = set(df_train["speaker_id"].unique())
        te_spk = set(df_test["speaker_id"].unique())
        overlap = tr_spk.intersection(te_spk)
        print(f"\n[NOTE] Speaker overlap between train and test (utterance split) = {len(overlap)} speakers")
        if len(overlap) > 0:
            print("       This is expected for utterance-level split and may increase accuracy (easier).")
            print("       If you want stricter evaluation, rerun with --split_mode speaker.")

if __name__ == "__main__":
    main()
