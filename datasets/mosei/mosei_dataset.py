import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


# ---------------- Text cleanup (manifest text_raw contains [b'...'] tokens) ----------------
_TOKEN_RE = re.compile(r"\[b'([^']*)'\]")


def clean_mosei_text(text_raw: str) -> str:
    """
    Convert:
      "[b'sp'] [b'i'] [b'see'] ..."
    -> "i see ..."
    """
    if not text_raw:
        return ""
    toks = _TOKEN_RE.findall(text_raw)
    if toks:
        toks = [t for t in toks if t and t.lower() != "sp"]
        return " ".join(toks).strip()
    return text_raw.replace("[b'sp']", " ").strip()


# ---------------- Per-process cache for SDK datasets ----------------
@dataclass
class _SDKCache:
    labels_ds: Any = None
    words_ds: Any = None
    audio_ds: Any = None
    visual_ds: Any = None
    comps: Dict[str, str] = None


_SDK_CACHE: Dict[int, _SDKCache] = {}


def _get_cache_key() -> int:
    wi = get_worker_info()
    return wi.id if wi is not None else -1


def _init_sdk_datasets(csd_paths: Dict[str, str]):
    from mmsdk import mmdatasdk  # from cmu-multimodal-sdk

    labels_path = csd_paths["labels"]
    words_path = csd_paths["words"]
    audio_path = csd_paths["audio"]
    visual_path = csd_paths["visual"]

    labels_ds = mmdatasdk.mmdataset({Path(labels_path).stem: labels_path})
    words_ds = mmdatasdk.mmdataset({Path(words_path).stem: words_path})
    audio_ds = mmdatasdk.mmdataset({Path(audio_path).stem: audio_path})
    visual_ds = mmdatasdk.mmdataset({Path(visual_path).stem: visual_path})

    comps = {
        "labels": Path(labels_path).stem,
        "words": Path(words_path).stem,
        "audio": Path(audio_path).stem,
        "visual": Path(visual_path).stem,
    }
    return labels_ds, words_ds, audio_ds, visual_ds, comps


def _safe_fetch(ds, comp: str, utt_id: str) -> Optional[Dict[str, Any]]:
    try:
        return ds[comp][utt_id]
    except Exception:
        return None


def _h5_or_array_to_np(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.array(x, dtype=np.float32)
    except Exception:
        try:
            arr = np.array(x[:], dtype=np.float32)
        except Exception:
            return None

    # SANITIZE: remove NaN/Inf and clip extreme values
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # COVAREP/Facet should not have absurd magnitudes; clip to stabilize training
    arr = np.clip(arr, -10.0, 10.0).astype(np.float32)
    return arr



def _label_vec_to_emo6_multilabel(label_vec_1x7: Optional[np.ndarray], thr: float = 0.0) -> np.ndarray:
    """
    MOSEI Labels typically: [sentiment, happy, sad, angry, fearful, disgust, surprise]
    We use the 6 emotions and binarize:
      y_i = 1 if emo_i > thr else 0
    Output: float32 shape [6] in order:
      [happy, sad, angry, fearful, disgust, surprise]
    """
    if label_vec_1x7 is None:
        return np.zeros((6,), dtype=np.float32)
    v = label_vec_1x7.reshape(-1)
    if v.shape[0] < 7:
        return np.zeros((6,), dtype=np.float32)
    emo6 = v[1:7].astype(np.float32)
    y = (emo6 > thr).astype(np.float32)
    return y


class MoseiCSDDataset(Dataset):
    """
    Multi-label 6-emotion dataset:
      label: float tensor [6] (0/1)
    Ablations: T, TA, TV, TAV
    """

    def __init__(
        self,
        manifest_path: str,
        ablation: str = "TAV",
        use_manifest_text: bool = True,
        max_audio_len: int = 400,
        max_video_len: int = 400,
        label_thr: float = 0.0,
    ):
        self.manifest_path = Path(manifest_path)
        self.ablation = ablation.upper()
        assert self.ablation in ("T", "TA", "TV", "TAV")

        self.use_manifest_text = use_manifest_text
        self.max_audio_len = max_audio_len
        self.max_video_len = max_video_len
        self.label_thr = float(label_thr)

        self.rows: List[Dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        if not self.rows:
            raise RuntimeError(f"Empty manifest: {self.manifest_path}")

        self.csd_paths = self.rows[0]["csd"]

    def __len__(self) -> int:
        return len(self.rows)

    def _get_sdk(self) -> _SDKCache:
        k = _get_cache_key()
        if k not in _SDK_CACHE:
            labels_ds, words_ds, audio_ds, visual_ds, comps = _init_sdk_datasets(self.csd_paths)
            _SDK_CACHE[k] = _SDKCache(labels_ds=labels_ds, words_ds=words_ds, audio_ds=audio_ds, visual_ds=visual_ds, comps=comps)
        return _SDK_CACHE[k]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        utt_id = row["utt_id"]

        cache = self._get_sdk()
        labels = _safe_fetch(cache.labels_ds, cache.comps["labels"], utt_id)
        words = _safe_fetch(cache.words_ds, cache.comps["words"], utt_id)
        audio = _safe_fetch(cache.audio_ds, cache.comps["audio"], utt_id)
        visual = _safe_fetch(cache.visual_ds, cache.comps["visual"], utt_id)

        # ----- label: 6-emotion multi-label -----
        label_vec = None
        if labels is not None:
            label_vec = _h5_or_array_to_np(labels.get("features", None))
        y = _label_vec_to_emo6_multilabel(label_vec, thr=self.label_thr)  # [6]
        y_t = torch.from_numpy(y).float()

        # ----- text -----
        if self.use_manifest_text:
            text = clean_mosei_text(row.get("text_raw", ""))
        else:
            feats = None if words is None else words.get("features", None)
            if feats is None:
                text = ""
            else:
                toks = [str(t) for t in feats]
                toks = [t for t in toks if t and t.lower() != "sp"]
                text = " ".join(toks).strip()

        out: Dict[str, Any] = {"utt_id": utt_id, "text": text, "label": y_t}

        # ----- audio / video according to ablation -----
        if self.ablation in ("TA", "TAV"):
            a = None if audio is None else _h5_or_array_to_np(audio.get("features", None))
            out["audio"] = self._pad_trunc_time(a, self.max_audio_len, expected_dim=74)
        if self.ablation in ("TV", "TAV"):
            v = None if visual is None else _h5_or_array_to_np(visual.get("features", None))
            out["video"] = self._pad_trunc_time(v, self.max_video_len, expected_dim=35)

        return out

    @staticmethod
    def _pad_trunc_time(x: Optional[np.ndarray], max_len: int, expected_dim: int) -> torch.Tensor:
        """
        Pads/truncates to [max_len, expected_dim].
        If x is missing/wrong-dim -> zeros.
        Also sanitizes non-finite and clips.
        """
        if x is None:
            return torch.zeros((max_len, expected_dim), dtype=torch.float32)

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        T, D = x.shape[0], x.shape[1]
        if D != expected_dim:
            return torch.zeros((max_len, expected_dim), dtype=torch.float32)

        # sanitize again (double safety)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -10.0, 10.0).astype(np.float32)

        if T >= max_len:
            x2 = x[:max_len]
        else:
            pad = np.zeros((max_len - T, expected_dim), dtype=np.float32)
            x2 = np.concatenate([x, pad], axis=0)

        return torch.from_numpy(x2)




def mosei_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    utt_ids = [b["utt_id"] for b in batch]
    texts = [b["text"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)  # [B, 6]

    out = {"utt_id": utt_ids, "text": texts, "label": labels}

    if "audio" in batch[0]:
        out["audio"] = torch.stack([b["audio"] for b in batch], dim=0)  # [B, T, D]
    if "video" in batch[0]:
        out["video"] = torch.stack([b["video"] for b in batch], dim=0)  # [B, T, D]

    return out


# Aliases for unified train script
MoseiDataset = MoseiCSDDataset
collate_mosei = mosei_collate_fn
