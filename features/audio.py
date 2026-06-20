"""Audio front-end shared by both stages: wav -> log-mel frames + event/frame mapping."""
from __future__ import annotations

import numpy as np

SR = 22050
HOP = 512          # 512 / 22050 = 23.2 ms per frame  (~43 fps)
N_FFT = 2048
N_MELS = 80


def load_audio(path, sr: int = SR) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def log_mel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """-> (n_frames, N_MELS) float32 log-mel spectrogram."""
    import librosa
    m = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    m = librosa.power_to_db(m, ref=np.max).astype(np.float32)
    return m.T  # (T, mel)


def frame_times(n_frames: int, sr: int = SR, hop: int = HOP) -> np.ndarray:
    return (np.arange(n_frames) * hop) / sr


def time_to_frame(t: float, sr: int = SR, hop: int = HOP) -> int:
    return int(round(t * sr / hop))


def action_frame_labels(times_sec, n_frames: int) -> np.ndarray:
    """Binary per-frame label: 1 where a note action occurs (Stage 1 target)."""
    lab = np.zeros(n_frames, dtype=np.float32)
    for t in times_sec:
        f = time_to_frame(float(t))
        if 0 <= f < n_frames:
            lab[f] = 1.0
    return lab
