from typing import Union, List, Dict
import numpy as np
import torch
from torch import nn
from torchaudio import transforms


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._device: Union[str, torch.device] = "cpu"

    def to(self, device: Union[str, torch.device]):
        self._device = device
        return super().to(device)

    @property
    def device(self):
        return self._device


class WakeWordDetector(BaseModule):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 400,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.output_size = 2

        self.rnn = nn.GRU(
            input_size=n_mels, hidden_size=hidden_dim, batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.output_size),
        )
        self.layernorm = nn.LayerNorm(self.n_mels)

    def extract_features(
        self, waveforms: List[torch.Tensor], sample_rate: int = 16000, n_fft: int = 400
    ):
        """
        Args:
            waveforms (List[torch.Tensor]): List of waveforms of shape (n_channels, n_frames).
            sample_rate (int): Sample rate of the waveforms. Defaults to 16000.

        Returns:
            features (torch.Tensor): (batch_size, n_features, n_frames)
            lengths (List[int]): List of lengths of each feature sequence before padded.
        """
        extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, n_mels=self.n_mels, normalized=False
        )

        # feats.shape = (batch_size, n_mels, n_frames)
        # NOTE: only take the first channel
        feats: List[torch.Tensor] = []
        lengths: List[int] = []
        for waveform in waveforms:
            # Extract log mel spectrogram features
            spec = extractor(waveform)[0]  # (n_mels, n_frames)
            # log_spec = torch.log(spec + 1e-9)
            log_spec = spec

            feats.append(log_spec)
            lengths.append(log_spec.size(1))

        # Pad to the same length
        max_len = max(lengths)
        for i in range(len(feats)):
            feats[i] = nn.functional.pad(feats[i], (0, max_len - lengths[i]), value=0)
        feats = torch.stack(feats, dim=0)  # (batch_size, n_mels, n_frames)

        return feats, lengths

    def _forward(
        self,
        waveforms: Union[List[np.ndarray], List[torch.Tensor]],
    ):
        """
        Args:
            waveforms (Union[List[np.ndarray], List[torch.Tensor]]): List of waveforms of shape (n_channels, n_frames).
            sample_rate (int): Sample rate of the waveforms. Defaults to 16000.

        Returns:
            logits (torch.Tensor): (batch_size, 2)
        """

        if isinstance(waveforms[0], np.ndarray):
            waveforms = [torch.from_numpy(waveform).float() for waveform in waveforms]
        for wav in waveforms:
            wav.to(self.device)

        feats, _ = self.extract_features(waveforms, self.sample_rate, self.n_fft)
        feats = feats.permute(0, 2, 1).contiguous()  # (batch_size, n_frames, n_feats)

        feats = self.layernorm(feats)
        _, hs = self.rnn(feats)
        hs = torch.squeeze(hs, 0)  # (batch_size, hidden_dim)
        logits = self.fc(hs)  # (batch_size, 2)

        return logits

    def forward(self, waveforms, labels=None):
        if self.training:
            logits = self._forward(waveforms)
            loss = nn.functional.cross_entropy(logits, labels)
            return loss
        else:
            with torch.no_grad():
                logits = self._forward(waveforms)
            probs = torch.softmax(logits, dim=1)[:, 1]
            predicted = torch.argmax(logits, dim=1)

            return predicted, probs
