from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as functional


VARIANTS = (
    "temporal_only",
    "spectral_only",
    "no_derivative",
    "no_lstm",
    "mean_pool",
    "full",
)


@dataclass
class ModelConfig:
    num_classes: int
    sample_rate: float
    variant: str = "full"
    n_fft: int = 256
    hop_length: int = 64
    channels: int = 16
    hidden_size: int = 32
    fusion_size: int = 64
    smoothing_kernel: int = 5
    dropout: float = 0.2

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"未知配置：{self.variant}")
        if self.num_classes < 2 or self.sample_rate <= 0:
            raise ValueError("类别数至少为 2，采样率必须为正数")
        if self.n_fft < 4 or self.n_fft % 2:
            raise ValueError("n_fft 必须是至少为 4 的偶数")
        if not 1 <= self.hop_length <= self.n_fft:
            raise ValueError("hop_length 必须位于 [1, n_fft]")
        if self.smoothing_kernel < 1 or self.smoothing_kernel % 2 == 0:
            raise ValueError("平滑窗口必须为正奇数")
        if min(self.channels, self.hidden_size, self.fusion_size) < 1:
            raise ValueError("特征维度必须为正整数")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1)")

    def to_dict(self):
        return asdict(self)


class TemporalAttentionPool(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.Tanh(),
            nn.Linear(feature_size, 1),
        )

    def forward(self, sequence):
        weights = self.score(sequence).softmax(dim=1)
        return (sequence * weights).sum(dim=1), weights.squeeze(-1)


class PGDTAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_temporal = config.variant != "spectral_only"
        self.use_spectral = config.variant != "temporal_only"
        self.use_derivative = config.variant != "no_derivative"
        self.register_buffer("window", torch.hann_window(config.n_fft))
        feature_size = 0
        if self.use_spectral:
            self.spectral_encoder = nn.Sequential(
                nn.Conv2d(1, config.channels, 3, padding=1),
                nn.GELU(),
                nn.AvgPool2d((2, 1)),
                nn.Conv2d(config.channels, config.channels * 2, 3, padding=1),
                nn.GELU(),
            )
            feature_size += config.channels * 2
        if self.use_temporal:
            self.temporal_encoder = nn.Sequential(
                nn.Conv1d(2 if self.use_derivative else 1, config.channels, 7, padding=3),
                nn.GELU(),
                nn.Conv1d(config.channels, config.channels * 2, 5, padding=2),
                nn.GELU(),
            )
            if config.variant != "no_lstm":
                self.lstm = nn.LSTM(
                    config.channels * 2, config.hidden_size,
                    batch_first=True, bidirectional=True,
                )
                feature_size += config.hidden_size * 2
            else:
                self.lstm = None
                feature_size += config.channels * 2
        self.projection = nn.Sequential(
            nn.Linear(feature_size, config.fusion_size),
            nn.LayerNorm(config.fusion_size),
            nn.GELU(),
        )
        self.pool = (
            None if config.variant == "mean_pool"
            else TemporalAttentionPool(config.fusion_size)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_size, config.num_classes),
        )

    def temporal_input(self, voltage):
        if not self.use_derivative:
            return voltage.unsqueeze(1)
        padding = self.config.smoothing_kernel // 2
        smoothed = functional.avg_pool1d(
            functional.pad(voltage.unsqueeze(1), (padding, padding), mode="replicate"),
            self.config.smoothing_kernel, stride=1,
        ).squeeze(1)
        derivative = torch.diff(smoothed, dim=-1)
        derivative = functional.pad(derivative, (1, 0))
        return torch.stack((voltage, derivative), dim=1)

    def forward(self, voltage, return_details=False):
        if voltage.ndim != 2 or voltage.shape[1] < self.config.n_fft:
            raise ValueError("输入应为 [batch, samples]，且长度至少为 n_fft")
        frame_indices = torch.arange(
            0, voltage.shape[1], self.config.hop_length, device=voltage.device,
        )
        sequences = []
        if self.use_spectral:
            spectrum = torch.stft(
                voltage, n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                window=self.window, center=True, pad_mode="constant",
                return_complex=True,
            )[..., :frame_indices.numel()]
            representation = torch.log1p(spectrum.abs() / self.window.sum())
            spectral = self.spectral_encoder(representation.unsqueeze(1))
            sequences.append(spectral.mean(dim=2).transpose(1, 2))
        if self.use_temporal:
            temporal = self.temporal_encoder(self.temporal_input(voltage))
            temporal = temporal.index_select(2, frame_indices).transpose(1, 2)
            if self.lstm is not None:
                temporal, _ = self.lstm(temporal)
            sequences.append(temporal)
        fused = self.projection(torch.cat(sequences, dim=-1))
        if self.pool is None:
            pooled = fused.mean(dim=1)
            weights = voltage.new_full(fused.shape[:2], 1 / fused.shape[1])
        else:
            pooled, weights = self.pool(fused)
        logits = self.classifier(pooled)
        if return_details:
            return {
                "logits": logits,
                "attention": weights,
                "frame_times_seconds": frame_indices / self.config.sample_rate,
            }
        return logits
