"""
LF-LSTM baseline for IEMOCAP and MELD.
Uses TrimodalEncoders (RoBERTa + wav2vec2 + ViT) for fair comparison.
Architecture: Each modality encoded, then BiLSTM over audio/video sequences, late fusion concat.
"""
import torch
import torch.nn as nn

from baselines.shared_encoders import TrimodalEncoders


class LFLSTMModel(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        audio_model_name: str,
        video_model_name: str,
        num_classes: int,
        proj_dim: int = 256,
        dropout: float = 0.2,
        use_text: bool = True,
        use_audio: bool = True,
        use_video: bool = True,
    ):
        super().__init__()
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_video = use_video

        self.encoders = TrimodalEncoders(
            text_model_name=text_model_name,
            audio_model_name=audio_model_name,
            video_model_name=video_model_name,
            proj_dim=proj_dim,
            dropout=dropout,
            use_text=use_text,
            use_audio=use_audio,
            use_video=use_video,
        )

        if use_audio:
            self.audio_lstm = nn.LSTM(
                proj_dim, proj_dim // 2, num_layers=2, batch_first=True,
                bidirectional=True, dropout=dropout,
            )
        if use_video:
            self.video_lstm = nn.LSTM(
                proj_dim, proj_dim // 2, num_layers=2, batch_first=True,
                bidirectional=True, dropout=dropout,
            )

        fusion_dim = proj_dim * (int(use_text) + int(use_audio) + int(use_video))
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, text_input=None, audio_wave=None, video_frames=None):
        pooled, seqs = self.encoders(text_input, audio_wave, video_frames)
        feats = []

        if self.use_text and "text" in pooled:
            feats.append(pooled["text"])

        if self.use_audio and "audio" in seqs:
            lstm_out, _ = self.audio_lstm(seqs["audio"])
            feats.append(lstm_out.mean(dim=1))

        if self.use_video and "video" in seqs:
            lstm_out, _ = self.video_lstm(seqs["video"])
            feats.append(lstm_out.mean(dim=1))

        z = torch.cat(feats, dim=-1)
        return self.classifier(z), z
