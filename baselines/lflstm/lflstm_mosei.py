"""
LF-LSTM baseline for CMU-MOSEI.
Ref: Wei et al. "FV2ES" (IEEE Trans. Broadcast. 2023) — Late Fusion LSTM.
Each modality encoded independently then concatenated for classification.
Text: RoBERTa CLS token.
Audio: BiLSTM over COVAREP (74-dim) → mean pool.
Video: BiLSTM over VisualFacet42 (35-dim) → mean pool.
"""
import torch
import torch.nn as nn

from baselines.mosei_feature_encoders import MOSEIFeatureEncoders


class LFLSTMMosei(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        d_model: int = 256,
        text_model: str = "roberta-base",
        use_audio: bool = True,
        use_video: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_audio = use_audio
        self.use_video = use_video

        # Shared text encoder (RoBERTa CLS) + linear projections for audio/video
        self.encoders = MOSEIFeatureEncoders(
            text_model=text_model, d_model=d_model,
            use_audio=use_audio, use_video=use_video, dropout=dropout,
        )

        # BiLSTM for temporal audio/video sequences (applied after projection)
        if use_audio:
            self.audio_lstm = nn.LSTM(
                d_model, d_model // 2, num_layers=2, batch_first=True,
                bidirectional=True, dropout=dropout,
            )
        if use_video:
            self.video_lstm = nn.LSTM(
                d_model, d_model // 2, num_layers=2, batch_first=True,
                bidirectional=True, dropout=dropout,
            )

        fusion_dim = d_model * (1 + int(use_audio) + int(use_video))
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, batch):
        texts = batch["text"]
        audio = batch.get("audio", None)
        video = batch.get("video", None)

        pooled, seqs = self.encoders(texts, audio, video)
        feats = [pooled["text"]]

        if self.use_audio and "audio" in seqs:
            lstm_out, _ = self.audio_lstm(seqs["audio"])   # [B, T_a, d_model]
            feats.append(lstm_out.mean(dim=1))

        if self.use_video and "video" in seqs:
            lstm_out, _ = self.video_lstm(seqs["video"])   # [B, T_v, d_model]
            feats.append(lstm_out.mean(dim=1))

        z = torch.cat(feats, dim=-1)
        logits = self.classifier(z)
        return logits, z
