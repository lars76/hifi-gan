import math
import random
import torch
import torch.utils.data
import librosa
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm


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
            y = torch.nn.functional.pad(y, (pad_length, pad_length), mode="reflect")####################
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
            pad_mode="reflect",#######################
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
        fmax_loss=None,
        fine_tuning=False,
    ):
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
        self.fine_tuning = fine_tuning

        self.audio_files = []
        for filename in tqdm(training_files):
            audio = read_audio(filename, self.sampling_rate)
            if audio.shape[1] < self.segment_size:
                print(f"Skipping {filename} because {audio.shape[1]} < {self.segment_size}")
                continue
            self.audio_files.append(filename)

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
        audio = read_audio(filename, self.sampling_rate)

        if not self.fine_tuning:
            if self.split:
                max_audio_start = audio.shape[1] - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start : audio_start + self.segment_size]

            mel = self.mel_converter(audio)
        else:
            mel = torch.load(filename.replace(".wav", ".pt"))

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                mel_start = random.randint(0, mel.shape[2] - frames_per_seg - 1)
                mel = mel[:, :, mel_start : mel_start + frames_per_seg]
                audio = audio[
                    :,
                    mel_start * self.hop_size : (mel_start + frames_per_seg)
                    * self.hop_size,
                ]

        mel_loss = self.mel_converter_loss(audio)

        return {
            "mel": mel.squeeze(0),
            "audio": audio.squeeze(0),
            "mel_for_loss": mel_loss.squeeze(0),
        }

    def __len__(self):
        return len(self.audio_files)

    @staticmethod
    def pad_tensors(data: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
        if not data:
            raise ValueError("Data must contain at least one tensor.")

        # Get the shape of the first tensor
        first_shape = data[0].shape

        # Ensure all tensors have the same number of dimensions and same first dimension (if 2D)
        if not all(tensor.dim() == len(first_shape) for tensor in data):
            raise ValueError("All tensors must have the same number of dimensions.")
        if len(first_shape) == 2 and not all(tensor.shape[0] == first_shape[0] for tensor in data):
            raise ValueError("All 2D tensors must have the same first dimension.")

        # Find the maximum length of the last dimension
        max_len = max(tensor.shape[-1] for tensor in data)

        # Pad the last dimension of each tensor
        padded_data = [
            torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[-1]), value=pad_value)
            for tensor in data
        ]

        # Stack the padded tensors
        return torch.stack(padded_data)

    def collate_fn(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mel = self.pad_tensors([b["mel"] for b in batch])
        audio = self.pad_tensors([b["audio"] for b in batch])
        mel_for_loss = self.pad_tensors([b["mel_for_loss"] for b in batch])
        return mel, audio, mel_for_loss


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
        split_index = int(len(files) * train_ratio)
        train_set.extend(files[:split_index])
        val_set.extend(files[split_index:])

    return train_set, val_set
