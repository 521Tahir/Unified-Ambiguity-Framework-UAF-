#!/usr/bin/env python3
"""
Build utterance-level CMU-MOSEI manifests (metadata only — no inline features).
Each row = one utterance: video_id, utt_idx, t_start, t_end, label, text.
Features are loaded at runtime from CSD files using video_id + time interval.

Usage:
  conda run -n chado_mm python3 scripts/preprocess/build_mosei_utt_manifests.py
"""
import json, hashlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = Path("/home/tahirahmad/CHADO_CLAUDECODE/data/raw/CMU-MOSEI")
OUT_DIR   = ROOT / "data/processed/mosei"

CSD_PATHS = {
    "labels": str(DATA_ROOT / "labels/CMU_MOSEI_Labels.csd"),
    "audio":  str(DATA_ROOT / "acoustics/CMU_MOSEI_COVAREP.csd"),
    "visual": str(DATA_ROOT / "visuals/CMU_MOSEI_VisualFacet42.csd"),
    "words":  str(DATA_ROOT / "languages/CMU_MOSEI_TimestampedWords.csd"),
}

EMOTION_COLS  = [1, 2, 3, 4, 5, 6]
EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
LABEL_THR = 0.0

def _words_in_interval(wrd_feat, wrd_int, t_start, t_end):
    words = []
    for i, iv in enumerate(wrd_int):
        if float(iv[1]) > t_start and float(iv[0]) < t_end:
            w = wrd_feat[i]
            if hasattr(w, '__iter__'): w = w[0] if len(w) == 1 else str(w)
            if hasattr(w, 'decode'): w = w.decode('utf-8', errors='ignore')
            s = str(w).strip().replace("b'", "").replace("'", "").strip()
            if s and s.lower() not in ('sp', '', 'b', ''):
                words.append(s)
    return " ".join(words)

def _det_split(vid_id, seed="CHADO_MOSEI_UTT_V2"):
    h = int(hashlib.md5((seed + vid_id).encode()).hexdigest(), 16)
    r = h % 100
    if r < 80: return "train"
    if r < 90: return "val"
    return "test"

def main():
    from mmsdk import mmdatasdk

    print("Loading CSD files...")
    ds_lab = mmdatasdk.mmdataset({"All Labels": CSD_PATHS["labels"]})
    ds_cov = mmdatasdk.mmdataset({"COVAREP":    CSD_PATHS["audio"]})
    ds_fac = mmdatasdk.mmdataset({"FACET 4.2":  CSD_PATHS["visual"]})
    ds_wrd = mmdatasdk.mmdataset({"words":       CSD_PATHS["words"]})

    lab_keys = list(ds_lab["All Labels"].keys())
    cov_keys = set(ds_cov["COVAREP"].keys())
    fac_keys = set(ds_fac["FACET 4.2"].keys())
    wrd_keys = set(ds_wrd["words"].keys())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": [], "test": []}
    skipped = 0

    print(f"Processing {len(lab_keys)} videos into utterance-level manifests...")

    for vid_idx, vid_id in enumerate(lab_keys):
        if vid_idx % 500 == 0:
            print(f"  {vid_idx}/{len(lab_keys)} videos processed...")

        lab_feat = np.array(ds_lab["All Labels"][vid_id]["features"])  # [N_utt, 7]
        lab_int  = np.array(ds_lab["All Labels"][vid_id]["intervals"]) # [N_utt, 2]
        n_utt = lab_feat.shape[0]
        split = _det_split(vid_id)

        # Load words once per video (lightweight)
        wrd_feat = np.array(ds_wrd["words"][vid_id]["features"]) if vid_id in wrd_keys else None
        wrd_int  = np.array(ds_wrd["words"][vid_id]["intervals"]) if vid_id in wrd_keys else None

        for i in range(n_utt):
            t_start = float(lab_int[i][0])
            t_end   = float(lab_int[i][1])
            if t_end <= t_start:
                skipped += 1
                continue

            emo_raw = [float(lab_feat[i, c]) for c in EMOTION_COLS]
            emo_bin = [1 if v > LABEL_THR else 0 for v in emo_raw]

            text = ""
            if wrd_feat is not None:
                text = _words_in_interval(wrd_feat, wrd_int, t_start, t_end)

            rec = {
                "utt_id":    f"{vid_id}__{i:03d}",
                "video_id":  vid_id,
                "utt_idx":   i,
                "t_start":   round(t_start, 4),
                "t_end":     round(t_end, 4),
                "text":      text,
                "label":     emo_bin,
                "label_raw": [round(v, 4) for v in emo_raw],
                "has_audio": vid_id in cov_keys,
                "has_visual": vid_id in fac_keys,
                "has_text":  len(text.strip()) > 0,
                "csd": CSD_PATHS,
            }
            splits[split].append(rec)

    print(f"Skipped {skipped} utterances (invalid time intervals)")

    for split_name, rows in splits.items():
        out_path = OUT_DIR / f"mosei_utt_{split_name}.jsonl"
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_pos = [sum(r["label"]) for r in rows]
        print(f"[{split_name:5s}] {len(rows):6d} utterances | "
              f"avg_emotions={np.mean(n_pos):.2f} | "
              f"any_emotion={sum(any(r['label']) for r in rows)} | "
              f"has_audio={sum(r['has_audio'] for r in rows)} | "
              f"has_text={sum(r['has_text'] for r in rows)}")

    print("\nDone. Output:", OUT_DIR)

if __name__ == "__main__":
    main()
