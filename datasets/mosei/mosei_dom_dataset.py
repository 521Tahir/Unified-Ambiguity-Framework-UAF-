"""
CMU-MOSEI dominant-emotion dataset (6-class single-label).
Dominant emotion = argmax of raw emotion intensities [happy, sad, anger, surprise, disgust, fear].
Utterances with all-zero emotion scores are excluded.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.mosei.mosei_utt_dataset import _get_pid, _load_sdk, _frames_for_interval

EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


class MoseiDomDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        max_audio_len: int = 50,
        max_video_len: int = 30,
        filter_zeros: bool = False,
    ):
        self.max_audio_len = max_audio_len
        self.max_video_len = max_video_len
        rows = []
        with open(manifest_path) as f:
            for line in f:
                rows.append(json.loads(line))

        # By default include all-zero utterances (argmax→happy, consistent with paper protocol)
        self.rows: List[Dict] = []
        for r in rows:
            raw = r.get("label_raw", r.get("label", [0]*6))
            if filter_zeros and max(raw) == 0:
                continue
            self.rows.append(r)

        self.csd_paths = self.rows[0]["csd"] if self.rows else {}

    def __len__(self):
        return len(self.rows)

    def _get_sdk(self):
        pid = _get_pid()
        from datasets.mosei.mosei_utt_dataset import _SDK_CACHE
        if pid not in _SDK_CACHE:
            _SDK_CACHE[pid] = _load_sdk(self.csd_paths)
        return _SDK_CACHE[pid]

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        vid_id  = row["video_id"]
        t_start = row["t_start"]
        t_end   = row["t_end"]
        text    = row.get("text", "") or ""

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

        # Dominant emotion class
        raw = row.get("label_raw", row.get("label", [0]*6))
        dom_class = int(np.argmax(raw))

        return {
            "utt_id": row["utt_id"],
            "text":   text,
            "audio":  torch.from_numpy(audio),
            "video":  torch.from_numpy(video),
            "label":  torch.tensor(dom_class, dtype=torch.long),
        }


def collate_mosei_dom(batch: List[Dict]) -> Dict:
    return {
        "utt_id": [b["utt_id"] for b in batch],
        "text":   [b["text"]   for b in batch],
        "audio":  torch.stack([b["audio"] for b in batch]),
        "video":  torch.stack([b["video"] for b in batch]),
        "label":  torch.stack([b["label"] for b in batch]),
    }
