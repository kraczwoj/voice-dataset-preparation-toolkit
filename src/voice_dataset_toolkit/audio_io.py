from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate_hz: int
    bit_depth: int

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[1])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])


def _decode_pcm24(raw: bytes) -> np.ndarray:
    octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    values = (
        octets[:, 0].astype(np.int32)
        | (octets[:, 1].astype(np.int32) << 8)
        | (octets[:, 2].astype(np.int32) << 16)
    )
    values = np.where(values & 0x800000, values - 0x1000000, values)
    return values.astype(np.float64) / 8388608.0


def read_wav(path: Path) -> AudioBuffer:
    try:
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frames = stream.getnframes()
            compression = stream.getcomptype()
            raw = stream.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Unsupported or invalid WAV file: {path}") from exc

    if compression != "NONE":
        raise ValueError(f"Compressed WAV is not supported: {path}")
    if channels < 1:
        raise ValueError(f"Invalid channel count in {path}")

    if sample_width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        data = _decode_pcm24(raw)
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM bit depth: {sample_width * 8} in {path}")

    if data.size % channels:
        raise ValueError(f"Corrupt interleaved PCM data in {path}")
    data = data.reshape(-1, channels)
    if channels == 1:
        data = data[:, 0]
    return AudioBuffer(data.astype(np.float64, copy=False), sample_rate, sample_width * 8)


def _encode_pcm24(samples: np.ndarray) -> bytes:
    integers = np.rint(np.clip(samples, -1.0, 1.0 - 1.0 / 8388608.0) * 8388608.0).astype(np.int32)
    unsigned = integers & 0xFFFFFF
    out = np.empty((unsigned.size, 3), dtype=np.uint8)
    out[:, 0] = unsigned & 0xFF
    out[:, 1] = (unsigned >> 8) & 0xFF
    out[:, 2] = (unsigned >> 16) & 0xFF
    return out.tobytes()


def write_wav(path: Path, buffer: AudioBuffer, bit_depth: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(buffer.samples, dtype=np.float64)
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    flat = samples.reshape(-1)

    if bit_depth == 16:
        pcm = np.rint(np.clip(flat, -1.0, 1.0 - 1.0 / 32768.0) * 32768.0).astype("<i2").tobytes()
        width = 2
    elif bit_depth == 24:
        pcm = _encode_pcm24(flat)
        width = 3
    elif bit_depth == 32:
        pcm = np.rint(
            np.clip(flat, -1.0, 1.0 - 1.0 / 2147483648.0) * 2147483648.0
        ).astype("<i4").tobytes()
        width = 4
    else:
        raise ValueError("bit_depth must be 16, 24, or 32")

    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(int(channels))
        stream.setsampwidth(width)
        stream.setframerate(int(buffer.sample_rate_hz))
        stream.writeframes(pcm)
