import math
import os
import random
import torch
import torch.utils.data
import numpy as np
import librosa
from typing import Tuple, List
from pathlib import Path


def read_audio(
    file_path: str, sampling_rate: int, subtract_dc: bool = False
) -> torch.Tensor:
    # Load the audio file
    y, _ = librosa.load(file_path, sr=sampling_rate)
    y = torch.from_numpy(y).float()

    # Subtract the mean to remove DC offset
    if subtract_dc:
        y = y - y.mean()

    # Compute the RMS of the current signal
    rms = torch.sqrt(torch.mean(y**2))

    # Desired RMS in linear scale for -20 dBFS
    desired_rms = 10 ** (-20 / 20)

    # Compute the required gain to reach the desired RMS
    gain = desired_rms / rms

    # Constrain the gain within -3 to 3 dB
    gain = torch.clamp(gain, 10 ** (-3 / 20), 10 ** (3 / 20))

    # Apply the gain
    y = y * gain

    # Normalize the waveform to range between -1 and 1
    y = y / y.abs().max()

    # Set the sample width to 16-bit
    y = (y * 32767).short().float() / 32767

    return y.unsqueeze(0)


class MelSpectrogramConverter:
    def __init__(
        self,
        n_fft: int,
        num_mels: int,
        sampling_rate: int,
        hop_size: int,
        win_size: int,
        fmin: float,
        fmax: float,
    ):
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax

        # Precompute mel basis
        self.mel_basis = torch.from_numpy(
            librosa.filters.mel(
                sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
            )
        ).float()

        # Precompute Hanning window
        self.hann_window = torch.hann_window(win_size)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        # Ensure input is 2D
        if y.dim() == 1:
            y = y.unsqueeze(0)

        # Pad the signal
        pad_length = int((self.n_fft - self.hop_size) / 2)
        if y.size(1) > pad_length:
            y = torch.nn.functional.pad(y, (pad_length, pad_length), mode="reflect")
        else:
            y = torch.nn.functional.pad(y, (0, self.n_fft), mode="constant")

        # Compute STFT
        D = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window.to(y.device),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Convert to magnitude spectrogram and add small epsilon
        S = torch.sqrt(D.real.pow(2) + D.imag.pow(2) + 1e-9)

        # Apply mel filterbank
        S = torch.matmul(self.mel_basis.to(y.device), S)

        # Convert to log scale
        S = torch.log(torch.clamp(S, min=1e-5))

        return S


class MelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files,
        segment_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
        fmax_loss=None,
        fine_tuning=False,
        base_mels_path=None,
    ):
        self.audio_files = training_files
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

        self.mel_converter = MelSpectrogramConverter(
            n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax
        )
        self.mel_converter_loss = MelSpectrogramConverter(
            n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax_loss
        )

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio = read_audio(filename, self.sampling_rate)
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        if not self.fine_tuning:
            if self.split:
                if audio.shape[1] >= self.segment_size:
                    max_audio_start = audio.shape[1] - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start : audio_start + self.segment_size]
                else:
                    audio = torch.nn.functional.pad(
                        audio, (0, self.segment_size - audio.shape[1]), "constant"
                    )

            mel = self.mel_converter(audio)
        else:
            mel = torch.from_numpy(
                np.load(
                    os.path.join(
                        self.base_mels_path,
                        os.path.splitext(os.path.split(filename)[-1])[0] + ".npy",
                    )
                )
            ).float()

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.shape[1] >= self.segment_size:
                    mel_start = random.randint(0, mel.shape[2] - frames_per_seg - 1)
                    mel = mel[:, :, mel_start : mel_start + frames_per_seg]
                    audio = audio[
                        :,
                        mel_start * self.hop_size : (mel_start + frames_per_seg)
                        * self.hop_size,
                    ]
                else:
                    mel = torch.nn.functional.pad(
                        mel, (0, frames_per_seg - mel.shape[2]), "constant"
                    )
                    audio = torch.nn.functional.pad(
                        audio, (0, self.segment_size - audio.shape[1]), "constant"
                    )

        mel_loss = self.mel_converter_loss(audio)

        return mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze()

    def __len__(self):
        return len(self.audio_files)


def collect_audio_files(
    root_dir: str, train_ratio: float = 0.8, audio_extension: str = ".wav"
) -> Tuple[List[str], List[str]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"The directory {root_dir} does not exist")

    speaker_files = {}

    # Collect all audio files and group them by speaker ID
    for audio_file in root_path.rglob(f"*{audio_extension}"):
        speaker_id = audio_file.parent.name
        file_path = str(audio_file)
        if speaker_id not in speaker_files:
            speaker_files[speaker_id] = []
        speaker_files[speaker_id].append(file_path)

    train_set = []
    val_set = []

    # Split files into training and validation sets for each speaker
    for files in speaker_files.values():
        random.shuffle(files)
        split_index = int(len(files) * train_ratio)
        train_set.extend(files[:split_index])
        val_set.extend(files[split_index:])

    return train_set, val_set
