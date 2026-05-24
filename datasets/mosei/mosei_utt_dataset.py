"""
Utterance-level CMU-MOSEI dataset.
Each sample is one utterance; audio/visual features extracted from CSD on-the-fly.
Text encoded via RoBERTa tokenizer.
Labels: 6-class multi-label binary [happy, sad, anger, surprise, disgust, fear].
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset

EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

# Per-process cache: avoid re-loading CSD files for every sample
_SDK_CACHE: Dict[int, Any] = {}


def _get_pid():
    import os
    from torch.utils.data import get_worker_info
    wi = get_worker_info()
    return wi.id if wi is not None else -1


def _load_sdk(csd_paths: Dict[str, str]):
    from mmsdk import mmdatasdk
    ds_cov = mmdatasdk.mmdataset({"COVAREP":   csd_paths["audio"]})
    ds_fac = mmdatasdk.mmdataset({"FACET 4.2": csd_paths["visual"]})
    return ds_cov, ds_fac


def _frames_for_interval(feat, intervals, t_start, t_end, max_len: int):
    """Extract feature frames within [t_start, t_end], pad/truncate to max_len."""
    feat = np.array(feat, dtype=np.float32)
    intervals = np.array(intervals)
    i_s = intervals[:, 0]
    i_e = intervals[:, 1]
    mask = (i_e > t_start) & (i_s < t_end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros((max_len, feat.shape[1]), dtype=np.float32)
    seg = feat[idx]
    if len(seg) >= max_len:
        seg = seg[:max_len]
    else:
        pad = np.zeros((max_len - len(seg), feat.shape[1]), dtype=np.float32)
        seg = np.vstack([seg, pad])
    seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
    return seg


class MoseiUttDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        text_model_name: str = "roberta-base",
        max_text_len: int = 96,
        max_audio_len: int = 50,   # frames per utterance (~1-5 sec @ COVAREP rate)
        max_video_len: int = 30,   # frames per utterance
        label_thr: float = 0.0,
    ):
        self.rows: List[Dict] = []
        with open(manifest_path) as f:
            for line in f:
                self.rows.append(json.loads(line))

        self.max_audio_len = max_audio_len
        self.max_video_len = max_video_len
        self.label_thr = label_thr

        # Store CSD paths from first row
        self.csd_paths = self.rows[0]["csd"] if self.rows else {}

    def __len__(self):
        return len(self.rows)

    def _get_sdk(self):
        pid = _get_pid()
        if pid not in _SDK_CACHE:
            _SDK_CACHE[pid] = _load_sdk(self.csd_paths)
        return _SDK_CACHE[pid]

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        vid_id  = row["video_id"]
        t_start = row["t_start"]
        t_end   = row["t_end"]
        text    = row.get("text", "") or ""

        # Load audio/visual from CSD at runtime
        ds_cov, ds_fac = self._get_sdk()
        try:
            cov_feat = ds_cov["COVAREP"][vid_id]["features"]
            cov_int  = ds_cov["COVAREP"][vid_id]["intervals"]
            audio = _frames_for_interval(cov_feat, cov_int, t_start, t_end, self.max_audio_len)
        except Exception:
            audio = np.zeros((self.max_audio_len, 74), dtype=np.float32)
        try:
            fac_feat = ds_fac["FACET 4.2"][vid_id]["features"]
            fac_int  = ds_fac["FACET 4.2"][vid_id]["intervals"]
            video = _frames_for_interval(fac_feat, fac_int, t_start, t_end, self.max_video_len)
        except Exception:
            video = np.zeros((self.max_video_len, 35), dtype=np.float32)

        label = torch.tensor(row["label"], dtype=torch.float32)  # [6]

        return {
            "utt_id": row["utt_id"],
            "text":   text,                          # string — tokenized by model encoder
            "audio":  torch.from_numpy(audio),       # [max_audio_len, 74]
            "video":  torch.from_numpy(video),       # [max_video_len, 35]
            "label":  label,                         # [6] binary
        }


def collate_mosei_utt(batch: List[Dict]) -> Dict:
    return {
        "utt_id": [b["utt_id"] for b in batch],
        "text":   [b["text"]   for b in batch],
        "audio":  torch.stack([b["audio"]  for b in batch]),  # [B, T, 74]
        "video":  torch.stack([b["video"]  for b in batch]),  # [B, T, 35]
        "label":  torch.stack([b["label"]  for b in batch]),  # [B, 6]
    }
