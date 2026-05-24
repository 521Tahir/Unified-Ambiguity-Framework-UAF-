"""
IEMOCAP trimodal dataset.
CSV columns: transcript, wav_path, avi_path, start, end, label_4
label_4: neu | hap | ang | sad

Causal supervision signals derived from utt_id:
  speaker_id : int 0-9  (Ses01M=0, Ses01F=1, ..., Ses05F=9)
  turn_norm  : float [0,1] — turn index / max turn in dialogue
"""
import os
import re
from typing import Dict

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoTokenizer

LABEL2ID = {"neu": 0, "hap": 1, "ang": 2, "sad": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES = [ID2LABEL[i] for i in range(len(LABEL2ID))]

# Ses01M=0, Ses01F=1, ..., Ses05M=8, Ses05F=9
_SPEAKER_MAP = {f"Ses0{s}{g}": (s - 1) * 2 + (0 if g == "M" else 1)
                for s in range(1, 6) for g in ("M", "F")}


def _parse_iemocap_utt(utt_id: str):
    """
    Returns (speaker_id, dialogue_id, turn_idx) from utt_id like Ses02M_script03_2_M018.
    speaker_id: int 0-9; turn_idx: int (the trailing number).
    """
    spk_key = utt_id[:6]   # e.g. "Ses02M"
    speaker_id = _SPEAKER_MAP.get(spk_key, -1)
    # dialogue = everything except the last _X### suffix
    m = re.match(r"^(.*?)_[MF](\d+)$", utt_id)
    if m:
        dialogue_id = m.group(1)
        turn_idx = int(m.group(2))
    else:
        dialogue_id = utt_id
        turn_idx = 0
    return speaker_id, dialogue_id, turn_idx


class IEMOCAPDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        text_model_name: str = "roberta-base",
        image_model_name: str = "google/vit-base-patch16-224-in21k",
        max_text_len: int = 96,
        audio_sr: int = 16000,
        audio_sec: float = 4.0,
        n_frames: int = 8,
        use_audio: bool = True,
        use_video: bool = True,
        context_cache: str | None = None,
        feature_cache_dir: str | None = None,  # if set, load precomputed av tensors from here
    ):
        df = pd.read_csv(csv_path)

        # --- parse causal supervision signals from utt_id ---
        parsed = df["utt_id"].apply(_parse_iemocap_utt)
        df["_speaker_id"] = parsed.apply(lambda x: x[0])
        df["_dialogue_id"] = parsed.apply(lambda x: x[1])
        df["_turn_idx"]   = parsed.apply(lambda x: x[2])
        # normalise turn within dialogue
        max_turn = df.groupby("_dialogue_id")["_turn_idx"].transform("max").clip(lower=1)
        df["_turn_norm"] = (df["_turn_idx"] / max_turn).clip(0.0, 1.0)

        self.df = df
        self.tok = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
        self.img_proc = AutoImageProcessor.from_pretrained(image_model_name) if use_video else None
        self.max_text_len = max_text_len
        self.audio_sr = audio_sr
        self.audio_sec = audio_sec
        self.n_frames = n_frames
        self.use_audio = use_audio
        self.use_video = use_video

        # optional pre-computed context embeddings {utt_id -> np.float32 [768]}
        self._ctx_emb: dict[str, np.ndarray] | None = None
        if context_cache and os.path.exists(context_cache):
            cache = np.load(context_cache, allow_pickle=True)
            self._ctx_emb = {
                str(uid): emb
                for uid, emb in zip(cache["utt_ids"], cache["embs"])
            }
        self._ctx_dim = 768
        self.feature_cache_dir = feature_cache_dir

    def __len__(self):
        return len(self.df)

    def _load_audio(self, wav_path: str, start: float, end: float) -> np.ndarray:
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != self.audio_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_sr)
        s = int(max(0.0, float(start)) * self.audio_sr)
        e = int(max(0.0, float(end)) * self.audio_sr)
        if e <= s:
            e = min(len(audio), s + int(self.audio_sec * self.audio_sr))
        seg = audio[s:e]
        target_len = int(self.audio_sec * self.audio_sr)
        if len(seg) < target_len:
            seg = np.pad(seg, (0, target_len - len(seg)))
        else:
            seg = seg[:target_len]
        return seg.astype(np.float32)

    def _load_video(self, avi_path: str, start: float, end: float) -> np.ndarray:
        cap = cv2.VideoCapture(avi_path)
        if not cap.isOpened():
            return np.zeros((self.n_frames, 224, 224, 3), dtype=np.uint8)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        s_f = int(max(0.0, float(start)) * fps)
        e_f = int(max(0.0, float(end)) * fps)
        if e_f <= s_f:
            e_f = s_f + int(fps)
        idxs = np.linspace(s_f, e_f, num=self.n_frames, dtype=np.int64)
        frames = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()
        return np.stack(frames, axis=0)

    def __getitem__(self, idx: int) -> Dict:
        r = self.df.iloc[idx]
        y = LABEL2ID[str(r["label_4"]).strip()]

        enc = self.tok(
            str(r["transcript"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        if self.feature_cache_dir:
            cache_path = os.path.join(self.feature_cache_dir, f"{idx:07d}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                wav = cached["audio"] if self.use_audio else None
                pixel_values = cached["video"].to(torch.float32) if self.use_video else None
            else:
                wav = self._load_audio(r["wav_path"], r["start"], r["end"]) if self.use_audio else None
                pixel_values = None
                if self.use_video:
                    frames = self._load_video(r["avi_path"], r["start"], r["end"])
                    pixel_values = self.img_proc(list(frames), return_tensors="pt")["pixel_values"]
        else:
            wav = self._load_audio(r["wav_path"], r["start"], r["end"]) if self.use_audio else None
            if self.use_video:
                frames = self._load_video(r["avi_path"], r["start"], r["end"])
                pixel_values = self.img_proc(list(frames), return_tensors="pt")["pixel_values"]
            else:
                pixel_values = None

        # context embedding (prior-turn mean) — zeros if unavailable / first turn
        if self._ctx_emb is not None:
            ctx = self._ctx_emb.get(str(r["utt_id"]), np.zeros(self._ctx_dim, dtype=np.float32))
        else:
            ctx = np.zeros(self._ctx_dim, dtype=np.float32)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "wav":            (wav if isinstance(wav, torch.Tensor) else torch.from_numpy(wav)) if wav is not None else None,
            "pixel_values":   pixel_values,
            "label":          torch.tensor(y, dtype=torch.long),
            # causal supervision signals
            "speaker_id":     torch.tensor(int(r["_speaker_id"]), dtype=torch.long),
            "turn_norm":      torch.tensor(float(r["_turn_norm"]), dtype=torch.float32),
            "context_emb":    torch.from_numpy(ctx),
        }


def collate_iemocap(batch):
    out = {
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label":          torch.stack([b["label"] for b in batch]),
        "speaker_id":     torch.stack([b["speaker_id"] for b in batch]),
        "turn_norm":      torch.stack([b["turn_norm"] for b in batch]),
        "context_emb":    torch.stack([b["context_emb"] for b in batch]),
    }
    if batch[0]["wav"] is not None:
        out["wav"] = torch.stack([b["wav"] for b in batch])
    if batch[0]["pixel_values"] is not None:
        out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
    return out
