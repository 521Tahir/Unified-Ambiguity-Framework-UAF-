import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# MELD 7-class canonical order
EMO_ORDER_7 = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]

# Main 6 Friends characters + "other" → 7 speaker classes
_MELD_MAIN_SPEAKERS = {"joey": 0, "monica": 1, "chandler": 2, "ross": 3, "rachel": 4, "phoebe": 5}
_MELD_OTHER_SPEAKER = 6

def _meld_speaker_id(name: str) -> int:
    return _MELD_MAIN_SPEAKERS.get(str(name).strip().lower(), _MELD_OTHER_SPEAKER)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_label_map_from_order(order=EMO_ORDER_7) -> Dict[str, int]:
    return {k: i for i, k in enumerate(order)}


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def _frames_to_tensor(frames: List[np.ndarray], num_frames: int, size: int) -> torch.Tensor:
    out = []
    for fr in frames[:num_frames]:
        if fr is None:
            out.append(torch.zeros((3, size, size), dtype=torch.float32))
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_LINEAR)
        fr = fr.astype(np.float32) / 255.0
        fr = torch.from_numpy(fr).permute(2, 0, 1)  # 3,H,W
        out.append(fr)
    while len(out) < num_frames:
        out.append(torch.zeros((3, size, size), dtype=torch.float32))
    return torch.stack(out, dim=0)  # T,3,H,W


def load_video_frames_opencv(video_path: str, num_frames: int, size: int) -> torch.Tensor:
    """
    Returns [T, 3, H, W] float32 in [0,1]. If fails -> zeros.
    """
    if not video_path or (not os.path.exists(video_path)):
        return torch.zeros((num_frames, 3, size, size), dtype=torch.float32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return torch.zeros((num_frames, 3, size, size), dtype=torch.float32)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length <= 0:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return _frames_to_tensor(frames, num_frames, size)

    idxs = np.linspace(0, max(length - 1, 0), num=num_frames).astype(int).tolist()

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        frames.append(frame if ok else None)
    cap.release()

    # fill missing with last good
    last_good = None
    for i in range(len(frames)):
        if frames[i] is None:
            frames[i] = last_good
        else:
            last_good = frames[i]
    if frames[0] is None:
        return torch.zeros((num_frames, 3, size, size), dtype=torch.float32)

    return _frames_to_tensor(frames, num_frames, size)


def load_audio_waveform(wav_path: str, target_sr: int, max_seconds: float) -> torch.Tensor:
    """
    SAFE WAV loader using soundfile (bypasses torchaudio->torchcodec libtorchcodec errors).
    Returns [L] float32 at target_sr padded/truncated to max_seconds.
    """
    import soundfile as sf

    max_len = int(target_sr * max_seconds)
    if not wav_path or (not os.path.exists(wav_path)):
        return torch.zeros((max_len,), dtype=torch.float32)

    try:
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception:
        return torch.zeros((max_len,), dtype=torch.float32)

    if isinstance(wav, np.ndarray) and wav.ndim == 2:
        wav = wav.mean(axis=1)  # to mono

    wav = torch.from_numpy(np.asarray(wav, dtype=np.float32))

    if sr != target_sr:
        # tensor-only resample (no torchcodec required)
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    if wav.numel() > max_len:
        wav = wav[:max_len]
    elif wav.numel() < max_len:
        wav = torch.nn.functional.pad(wav, (0, max_len - wav.numel()))

    return wav.to(torch.float32)


@dataclass
class MeldBatch:
    utt_id: List[str]
    labels: torch.Tensor
    text_input: Optional[Dict[str, torch.Tensor]]
    audio_wave: Optional[torch.Tensor]
    video_frames: Optional[torch.Tensor]
    speaker_id: Optional[torch.Tensor] = None   # [B] long, -1 = unknown
    turn_norm: Optional[torch.Tensor]  = None   # [B] float [0,1]


class MeldDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        text_model_name: str,
        label_map: Dict[str, int],
        text_col: str,
        label_col: str,
        audio_path_col: str,
        video_path_col: str,
        utt_id_col: str,
        num_frames: int,
        frame_size: int,
        sample_rate: int,
        max_audio_seconds: float,
        use_text: bool = True,
        use_audio: bool = True,
        use_video: bool = True,
        context_turns: int = 3,
        seed: int = 42,
        speaker_csv: Optional[str] = None,   # path to raw MELD CSV with Speaker column
        feature_cache_dir: Optional[str] = None,  # if set, load precomputed av tensors from here
    ):
        df = pd.read_csv(csv_path)
        # Sort by dialogue then utterance so context lookup is correct
        df["_uid_int"] = pd.to_numeric(df[utt_id_col], errors="coerce").fillna(0).astype(int)
        df = df.sort_values(["dialogue_id", "_uid_int"]).reset_index(drop=True)

        # --- causal signals: turn_norm ---
        max_turn = df.groupby("dialogue_id")["_uid_int"].transform("max").clip(lower=1)
        df["_turn_norm"] = (df["_uid_int"] / max_turn).clip(0.0, 1.0)

        # --- causal signals: speaker_id (from raw MELD CSV if provided) ---
        if speaker_csv and os.path.exists(speaker_csv):
            raw = pd.read_csv(speaker_csv)[["Dialogue_ID", "Utterance_ID", "Speaker"]]
            raw = raw.rename(columns={"Dialogue_ID": "dialogue_id", "Utterance_ID": "_uid_int"})
            df = df.merge(raw, on=["dialogue_id", "_uid_int"], how="left")
            df["_speaker_id"] = df["Speaker"].apply(
                lambda s: _meld_speaker_id(s) if pd.notna(s) else -1
            )
        else:
            df["_speaker_id"] = -1   # unknown — supervision skipped

        self.df = df

        self.text_col = text_col
        self.label_col = label_col
        self.audio_path_col = audio_path_col
        self.video_path_col = video_path_col
        self.utt_id_col = utt_id_col
        self.context_turns = context_turns

        self.label_map = label_map
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_video = use_video

        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds

        # Build (dialogue_id, utt_id_int) -> text lookup for dialogue context
        self._ctx: Dict[tuple, str] = {}
        if use_text and context_turns > 0:
            for _, row in df.iterrows():
                self._ctx[(row["dialogue_id"], int(row["_uid_int"]))] = safe_text(row[text_col])

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name) if use_text else None
        self.feature_cache_dir = feature_cache_dir
        set_seed(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        utt_id = str(r[self.utt_id_col])

        text = safe_text(r[self.text_col])
        emo = str(r[self.label_col]).strip().lower()
        label = self.label_map.get(emo, self.label_map.get("neutral", 0))

        item: Dict[str, Any] = {
            "utt_id":     utt_id,
            "label":      label,
            "speaker_id": int(r["_speaker_id"]),
            "turn_norm":  float(r["_turn_norm"]),
        }

        if self.use_text:
            target_text = text  # always store raw target utterance
            context_str = None
            if self.context_turns > 0 and self._ctx:
                did = r["dialogue_id"]
                uid = int(r["_uid_int"])
                prev = []
                for k in range(max(0, uid - self.context_turns), uid):
                    t = self._ctx.get((did, k), "")
                    if t:
                        prev.append(t)
                if prev:
                    context_str = " | ".join(prev)
            item["text"] = target_text
            item["context"] = context_str  # None if no context

        if self.feature_cache_dir:
            cache_path = os.path.join(self.feature_cache_dir, f"{idx:07d}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                if self.use_audio:
                    item["audio"] = cached["audio"]
                if self.use_video:
                    item["video"] = cached["video"].to(torch.float32)
                return item

        if self.use_audio:
            ap = str(r[self.audio_path_col])
            item["audio"] = load_audio_waveform(ap, self.sample_rate, self.max_audio_seconds)

        if self.use_video:
            vp = str(r[self.video_path_col])
            item["video"] = load_video_frames_opencv(vp, self.num_frames, self.frame_size)

        return item


def collate_meld(
    batch: List[Dict[str, Any]],
    tokenizer,
    use_text: bool,
    use_audio: bool,
    use_video: bool,
) -> MeldBatch:
    utt_id = [b["utt_id"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    text_input = None
    if use_text:
        targets = [b.get("text", "") for b in batch]
        contexts = [b.get("context", None) for b in batch]
        # Use RoBERTa sentence-pair format if any context exists: text_a=context, text_b=target
        if any(c is not None for c in contexts):
            ctx_strs = [c if c is not None else "" for c in contexts]
            text_input = tokenizer(
                ctx_strs,
                targets,
                padding=True,
                truncation="only_first",   # truncate context (A) if too long, preserve target (B)
                max_length=256,
                return_tensors="pt",
            )
        else:
            text_input = tokenizer(
                targets,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )

    audio_wave = None
    if use_audio:
        audio_wave = torch.stack([b["audio"] for b in batch], dim=0)  # [B,L]

    video_frames = None
    if use_video:
        video_frames = torch.stack([b["video"] for b in batch], dim=0)  # [B,T,3,H,W]

    return MeldBatch(
        utt_id=utt_id,
        labels=labels,
        text_input=text_input,
        audio_wave=audio_wave,
        video_frames=video_frames,
        speaker_id=torch.tensor([b["speaker_id"] for b in batch], dtype=torch.long),
        turn_norm=torch.tensor([b["turn_norm"]   for b in batch], dtype=torch.float32),
    )
