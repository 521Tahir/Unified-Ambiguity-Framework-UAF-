#!/usr/bin/env python3
"""
Build Leave-One-Environment-Out (LOEO) speaker-disjoint splits.

IEMOCAP  : 5 sessions → 5 environments (sessions never share speakers)
MELD     : 1039 dialogues → 5 contiguous blocks of episodes
CMU-MOSEI: 3293 video_ids → 5 alphabetically-sorted chunks (one video = one speaker/channel)

For each fold i:
  test  = environment i
  val   = environment (i+1) % 5
  train = remaining 3 environments

Outputs: data/loeo/{dataset}/fold_{0..4}/{train,val,test}.(csv|jsonl)
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LOEO_DIR = ROOT / "data" / "loeo"
N_FOLDS  = 5


# ─────────────────────────────────────────────────────────────────────────────
# IEMOCAP
# ─────────────────────────────────────────────────────────────────────────────

def build_iemocap_splits():
    """5 sessions → 5 folds. Session column already in CSV."""
    dfs = [pd.read_csv(ROOT / "data/manifests/iemocap" / f)
           for f in ("train.csv", "val.csv", "test.csv")]
    full = pd.concat(dfs, ignore_index=True)
    full["session"] = full["session"].astype(int)

    sessions = sorted(full["session"].unique())   # [1,2,3,4,5]
    assert len(sessions) == 5, f"Expected 5 sessions, got {sessions}"

    print(f"IEMOCAP: {len(full)} utterances across sessions {sessions}")
    for sess, cnt in full.groupby("session").size().items():
        print(f"  Session {sess}: {cnt} utts")

    out_base = LOEO_DIR / "iemocap"
    for fold in range(N_FOLDS):
        test_sess  = sessions[fold]
        val_sess   = sessions[(fold + 1) % N_FOLDS]
        train_sess = [s for s in sessions if s not in (test_sess, val_sess)]

        df_test  = full[full["session"] == test_sess]
        df_val   = full[full["session"] == val_sess]
        df_train = full[full["session"].isin(train_sess)]

        fold_dir = out_base / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(fold_dir / "train.csv", index=False)
        df_val.to_csv(fold_dir  / "val.csv",   index=False)
        df_test.to_csv(fold_dir / "test.csv",  index=False)
        print(f"  Fold {fold}: test=Ses{test_sess}({len(df_test)})  "
              f"val=Ses{val_sess}({len(df_val)})  "
              f"train=Ses{train_sess}({len(df_train)})")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# MELD
# ─────────────────────────────────────────────────────────────────────────────

def build_meld_splits():
    """
    1039 dialogues sorted by dialogue_id → 5 contiguous episode blocks.
    Full conversations preserved (no utterance split across blocks).
    """
    dfs = [pd.read_csv(ROOT / "data/processed/meld" / f)
           for f in ("meld_train.csv", "meld_val.csv", "meld_test.csv")]
    full = pd.concat(dfs, ignore_index=True)
    full["dialogue_id"] = full["dialogue_id"].astype(int)

    dialogues = sorted(full["dialogue_id"].unique())
    n_dia = len(dialogues)
    chunk = n_dia // N_FOLDS
    groups = []
    for i in range(N_FOLDS):
        start = i * chunk
        end   = (i + 1) * chunk if i < N_FOLDS - 1 else n_dia
        groups.append(set(dialogues[start:end]))

    print(f"MELD: {len(full)} utterances, {n_dia} dialogues → {N_FOLDS} groups of {chunk}±1")
    for i, g in enumerate(groups):
        cnt = full[full["dialogue_id"].isin(g)].shape[0]
        print(f"  Group {i}: {len(g)} dialogues, {cnt} utterances")

    out_base = LOEO_DIR / "meld"
    for fold in range(N_FOLDS):
        test_grp  = groups[fold]
        val_grp   = groups[(fold + 1) % N_FOLDS]
        train_grp = set()
        for i, g in enumerate(groups):
            if i not in (fold, (fold + 1) % N_FOLDS):
                train_grp |= g

        df_test  = full[full["dialogue_id"].isin(test_grp)]
        df_val   = full[full["dialogue_id"].isin(val_grp)]
        df_train = full[full["dialogue_id"].isin(train_grp)]

        fold_dir = out_base / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(fold_dir / "train.csv", index=False)
        df_val.to_csv(fold_dir  / "val.csv",   index=False)
        df_test.to_csv(fold_dir / "test.csv",  index=False)
        print(f"  Fold {fold}: test={len(df_test)}  val={len(df_val)}  train={len(df_train)}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CMU-MOSEI
# ─────────────────────────────────────────────────────────────────────────────

def build_mosei_splits():
    """
    3293 video_ids (one per speaker/channel) sorted alphabetically → 5 groups.
    Each video_id ≈ one content creator (different demographics/cultures).
    """
    rows = []
    for fname in ("mosei_utt_train.jsonl", "mosei_utt_val.jsonl", "mosei_utt_test.jsonl"):
        for line in open(ROOT / "data/processed/mosei" / fname):
            rows.append(json.loads(line))

    video_ids = sorted(set(r["video_id"] for r in rows))
    n_vids    = len(video_ids)
    chunk     = n_vids // N_FOLDS
    groups    = []
    for i in range(N_FOLDS):
        start = i * chunk
        end   = (i + 1) * chunk if i < N_FOLDS - 1 else n_vids
        groups.append(set(video_ids[start:end]))

    print(f"CMU-MOSEI: {len(rows)} utterances, {n_vids} video_ids → {N_FOLDS} groups of {chunk}±1")
    for i, g in enumerate(groups):
        cnt = sum(1 for r in rows if r["video_id"] in g)
        print(f"  Group {i}: {len(g)} video_ids, {cnt} utterances")

    out_base = LOEO_DIR / "mosei"
    for fold in range(N_FOLDS):
        test_grp  = groups[fold]
        val_grp   = groups[(fold + 1) % N_FOLDS]
        train_grp = set()
        for i, g in enumerate(groups):
            if i not in (fold, (fold + 1) % N_FOLDS):
                train_grp |= g

        split = {"train": [], "val": [], "test": []}
        for r in rows:
            vid = r["video_id"]
            if vid in test_grp:
                split["test"].append(r)
            elif vid in val_grp:
                split["val"].append(r)
            elif vid in train_grp:
                split["train"].append(r)

        fold_dir = out_base / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for sname, srows in split.items():
            with open(fold_dir / f"{sname}.jsonl", "w") as f:
                for r in srows:
                    f.write(json.dumps(r) + "\n")
        print(f"  Fold {fold}: test={len(split['test'])}  "
              f"val={len(split['val'])}  train={len(split['train'])}")

    print()


# ─────────────────────────────────────────────────────────────────────────────

def verify_speaker_disjoint():
    """Sanity check: no speaker/video_id appears in both train and test."""
    print("=== Disjoint verification ===")

    # IEMOCAP — check session
    for fold in range(N_FOLDS):
        base = LOEO_DIR / "iemocap" / f"fold_{fold}"
        tr = set(pd.read_csv(base / "train.csv")["session"].unique())
        te = set(pd.read_csv(base / "test.csv")["session"].unique())
        va = set(pd.read_csv(base / "val.csv")["session"].unique())
        overlap = (tr & te) | (tr & va) | (te & va)
        assert not overlap, f"IEMOCAP fold {fold}: session overlap {overlap}"
    print("  IEMOCAP: PASS (no session overlap across folds)")

    # MELD — check dialogue_id
    for fold in range(N_FOLDS):
        base = LOEO_DIR / "meld" / f"fold_{fold}"
        tr = set(pd.read_csv(base / "train.csv")["dialogue_id"].unique())
        te = set(pd.read_csv(base / "test.csv")["dialogue_id"].unique())
        va = set(pd.read_csv(base / "val.csv")["dialogue_id"].unique())
        overlap = (tr & te) | (tr & va) | (te & va)
        assert not overlap, f"MELD fold {fold}: dialogue overlap {overlap}"
    print("  MELD   : PASS (no dialogue overlap across folds)")

    # CMU-MOSEI — check video_id
    for fold in range(N_FOLDS):
        base = LOEO_DIR / "mosei" / f"fold_{fold}"
        def vids(fname):
            return {json.loads(l)["video_id"] for l in open(base / fname)}
        tr, te, va = vids("train.jsonl"), vids("test.jsonl"), vids("val.jsonl")
        overlap = (tr & te) | (tr & va) | (te & va)
        assert not overlap, f"MOSEI fold {fold}: video_id overlap {overlap}"
    print("  MOSEI  : PASS (no video_id overlap across folds)")
    print()


if __name__ == "__main__":
    print("Building LOEO speaker-disjoint splits...\n")
    build_iemocap_splits()
    build_meld_splits()
    build_mosei_splits()
    verify_speaker_disjoint()
    print("All splits written to data/loeo/")
