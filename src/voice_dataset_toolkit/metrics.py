from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, resample_poly

from .audio_io import AudioBuffer
from .models import AudioMetrics

EPSILON = 1e-12


def amplitude_to_db(value: float) -> float:
    return float(20.0 * math.log10(max(abs(value), EPSILON)))


def _k_weight(samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """Apply the two-stage BS.1770 K-weighting filter."""
    data = samples if samples.ndim == 2 else samples[:, None]

    def high_shelf() -> tuple[np.ndarray, np.ndarray]:
        frequency = 1681.974450955533
        gain_db = 3.999843853973347
        q = 0.7071752369554196
        k = math.tan(math.pi * frequency / sample_rate_hz)
        vh = 10.0 ** (gain_db / 20.0)
        vb = vh ** 0.4996667741545416
        a0 = 1.0 + k / q + k * k
        b = np.array([
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
        ])
        a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
        return b, a

    def high_pass() -> tuple[np.ndarray, np.ndarray]:
        frequency = 38.13547087602444
        q = 0.5003270373238773
        k = math.tan(math.pi * frequency / sample_rate_hz)
        a0 = 1.0 + k / q + k * k
        b = np.array([1.0 / a0, -2.0 / a0, 1.0 / a0])
        a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
        return b, a

    filtered = data
    for b, a in (high_shelf(), high_pass()):
        filtered = lfilter(b, a, filtered, axis=0)
    return filtered


def integrated_loudness_lufs(samples: np.ndarray, sample_rate_hz: int) -> float | None:
    """Estimate gated integrated loudness using ITU-R BS.1770 block gating.

    The implementation targets mono/stereo voice assets. LFE and surround channel
    mapping are intentionally outside the scope of this preparation toolkit.
    """
    if samples.size == 0 or sample_rate_hz <= 0:
        return None
    data = samples if samples.ndim == 2 else samples[:, None]
    weighted = _k_weight(data, sample_rate_hz)
    block_size = max(1, int(round(0.400 * sample_rate_hz)))
    hop_size = max(1, int(round(0.100 * sample_rate_hz)))
    if weighted.shape[0] < block_size:
        weighted = np.pad(weighted, ((0, block_size - weighted.shape[0]), (0, 0)))

    block_energies: list[float] = []
    for start in range(0, weighted.shape[0] - block_size + 1, hop_size):
        block = weighted[start : start + block_size]
        block_energies.append(float(np.sum(np.mean(block * block, axis=0))))
    if not block_energies:
        return None

    energies = np.asarray(block_energies)
    loudness = -0.691 + 10.0 * np.log10(np.maximum(energies, EPSILON))
    absolute = energies[loudness > -70.0]
    if absolute.size == 0:
        return None
    relative_threshold = -0.691 + 10.0 * math.log10(float(np.mean(absolute))) - 10.0
    gated = energies[(loudness > -70.0) & (loudness > relative_threshold)]
    if gated.size == 0:
        return None
    return float(-0.691 + 10.0 * math.log10(float(np.mean(gated))))


def true_peak_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -240.0
    data = samples if samples.ndim == 2 else samples[:, None]
    oversampled = resample_poly(data, up=4, down=1, axis=0, window=("kaiser", 8.6))
    return amplitude_to_db(float(np.max(np.abs(oversampled))))


def _silence_statistics(
    mono: np.ndarray,
    sample_rate_hz: int,
    threshold_dbfs: float,
) -> tuple[float, float, float]:
    if mono.size == 0:
        return 1.0, 0.0, 0.0
    frame = max(1, int(round(0.020 * sample_rate_hz)))
    hop = max(1, int(round(0.010 * sample_rate_hz)))
    if mono.size < frame:
        mono = np.pad(mono, (0, frame - mono.size))
    starts = list(range(0, mono.size - frame + 1, hop))
    rms = np.array([math.sqrt(float(np.mean(mono[s : s + frame] ** 2))) for s in starts])
    silent = np.array([amplitude_to_db(x) < threshold_dbfs for x in rms], dtype=bool)
    active = np.flatnonzero(~silent)
    if active.size == 0:
        duration = mono.size / sample_rate_hz
        return 1.0, duration, duration
    leading = starts[int(active[0])] / sample_rate_hz
    last_end = min(mono.size, starts[int(active[-1])] + frame)
    trailing = (mono.size - last_end) / sample_rate_hz
    return float(np.mean(silent)), float(leading), float(max(0.0, trailing))


def analyze_buffer(
    buffer: AudioBuffer,
    path: Path | str,
    silence_threshold_dbfs: float = -45.0,
) -> AudioMetrics:
    samples = np.asarray(buffer.samples, dtype=np.float64)
    mono = samples if samples.ndim == 1 else np.mean(samples, axis=1)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = math.sqrt(float(np.mean(samples * samples))) if samples.size else 0.0
    dc = float(np.mean(mono)) if mono.size else 0.0
    clipped = int(np.count_nonzero(np.abs(samples) >= 0.999))
    silence_ratio, leading, trailing = _silence_statistics(
        mono, buffer.sample_rate_hz, silence_threshold_dbfs
    )
    loudness = integrated_loudness_lufs(samples, buffer.sample_rate_hz)
    warnings: list[str] = []
    if buffer.sample_rate_hz < 16000:
        warnings.append("low_sample_rate")
    if buffer.channels > 1:
        warnings.append("multichannel_source")
    if clipped:
        warnings.append("clipping_detected")
    if abs(dc) > 0.005:
        warnings.append("dc_offset")
    if silence_ratio > 0.40:
        warnings.append("high_silence_ratio")
    if buffer.frames / buffer.sample_rate_hz < 1.0:
        warnings.append("very_short_file")
    if true_peak_dbfs(samples) > -1.0:
        warnings.append("true_peak_above_minus_1_dbfs")
    if loudness is None:
        warnings.append("loudness_below_gate")

    return AudioMetrics(
        path=str(path),
        sample_rate_hz=buffer.sample_rate_hz,
        channels=buffer.channels,
        bit_depth=buffer.bit_depth,
        frames=buffer.frames,
        duration_seconds=round(buffer.frames / buffer.sample_rate_hz, 6),
        rms_dbfs=round(amplitude_to_db(rms), 4),
        sample_peak_dbfs=round(amplitude_to_db(peak), 4),
        true_peak_dbfs=round(true_peak_dbfs(samples), 4),
        integrated_loudness_lufs=None if loudness is None else round(loudness, 4),
        dc_offset=round(dc, 8),
        crest_factor_db=round(amplitude_to_db(peak) - amplitude_to_db(rms), 4),
        clipped_samples=clipped,
        clipped_ratio=round(clipped / max(1, samples.size), 8),
        silence_ratio=round(silence_ratio, 6),
        leading_silence_seconds=round(leading, 6),
        trailing_silence_seconds=round(trailing, 6),
        warnings=warnings,
    )
