from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .audio_io import AudioBuffer, read_wav, write_wav
from .metrics import analyze_buffer, integrated_loudness_lufs, true_peak_dbfs
from .models import ProcessedAsset, ProcessingOptions


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.copy()
    return np.mean(samples, axis=1)


def remove_dc_offset(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples.copy()
    return samples - np.mean(samples, axis=0, keepdims=samples.ndim == 2)


def _active_bounds(
    samples: np.ndarray,
    sample_rate_hz: int,
    threshold_dbfs: float,
    padding_ms: int,
) -> tuple[int, int]:
    mono = samples if samples.ndim == 1 else np.mean(samples, axis=1)
    if mono.size == 0:
        return 0, 0
    frame = max(1, int(round(0.020 * sample_rate_hz)))
    hop = max(1, int(round(0.010 * sample_rate_hz)))
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    starts = range(0, max(1, mono.size - frame + 1), hop)
    active_starts = []
    for start in starts:
        chunk = mono[start : min(mono.size, start + frame)]
        rms = math.sqrt(float(np.mean(chunk * chunk))) if chunk.size else 0.0
        if rms >= threshold:
            active_starts.append(start)
    if not active_starts:
        return 0, mono.size
    padding = int(round(padding_ms * sample_rate_hz / 1000.0))
    first = max(0, active_starts[0] - padding)
    last = min(mono.size, active_starts[-1] + frame + padding)
    return first, last


def trim_silence(
    samples: np.ndarray,
    sample_rate_hz: int,
    threshold_dbfs: float,
    padding_ms: int,
) -> tuple[np.ndarray, int]:
    first, last = _active_bounds(samples, sample_rate_hz, threshold_dbfs, padding_ms)
    return samples[first:last].copy(), first


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.copy()
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(
        samples,
        up=target_rate // divisor,
        down=source_rate // divisor,
        axis=0,
        window=("kaiser", 8.6),
    ).astype(np.float64, copy=False)


def _quiet_cut(
    mono: np.ndarray,
    sample_rate_hz: int,
    earliest: int,
    desired: int,
    silence_threshold_dbfs: float,
) -> int:
    search_radius = int(round(2.0 * sample_rate_hz))
    search_start = max(earliest, desired - search_radius)
    search_end = min(mono.size, desired + int(round(0.5 * sample_rate_hz)))
    frame = max(1, int(round(0.020 * sample_rate_hz)))
    hop = max(1, int(round(0.010 * sample_rate_hz)))
    best_position = desired
    best_rms = float("inf")
    for start in range(search_start, max(search_start + 1, search_end - frame + 1), hop):
        chunk = mono[start : start + frame]
        if not chunk.size:
            continue
        rms = float(np.mean(chunk * chunk))
        if rms < best_rms:
            best_rms = rms
            best_position = start + frame // 2
    silence_energy = (10.0 ** (silence_threshold_dbfs / 20.0)) ** 2
    if best_rms > silence_energy:
        return desired
    return max(earliest, min(search_end, best_position))


def segment_audio(
    samples: np.ndarray,
    sample_rate_hz: int,
    min_seconds: float,
    max_seconds: float,
    silence_threshold_dbfs: float = -45.0,
) -> list[tuple[int, int, np.ndarray]]:
    if samples.size == 0:
        return []
    mono = samples if samples.ndim == 1 else np.mean(samples, axis=1)
    minimum = max(1, int(round(min_seconds * sample_rate_hz)))
    maximum = max(minimum, int(round(max_seconds * sample_rate_hz)))
    segments: list[tuple[int, int, np.ndarray]] = []
    start = 0
    while mono.size - start > maximum:
        desired = start + maximum
        end = _quiet_cut(
            mono,
            sample_rate_hz,
            start + minimum,
            desired,
            silence_threshold_dbfs,
        )
        if end <= start:
            end = desired
        segments.append((start, end, samples[start:end].copy()))
        start = end
    if mono.size - start >= minimum:
        segments.append((start, mono.size, samples[start:].copy()))
    elif segments and start < mono.size:
        prior_start, _, prior = segments[-1]
        segments[-1] = (prior_start, mono.size, np.concatenate([prior, samples[start:]], axis=0))
    elif start < mono.size:
        segments.append((start, mono.size, samples[start:].copy()))
    return segments


def normalize_loudness(
    samples: np.ndarray,
    sample_rate_hz: int,
    target_lufs: float | None,
    peak_ceiling_dbfs: float,
) -> tuple[np.ndarray, float, float | None, bool]:
    if target_lufs is None or samples.size == 0:
        measured = integrated_loudness_lufs(samples, sample_rate_hz)
        return samples.copy(), 0.0, measured, False
    measured = integrated_loudness_lufs(samples, sample_rate_hz)
    if measured is None:
        return samples.copy(), 0.0, None, False
    desired_gain = target_lufs - measured
    current_true_peak = true_peak_dbfs(samples)
    maximum_gain = peak_ceiling_dbfs - current_true_peak
    applied_gain = min(desired_gain, maximum_gain)
    peak_limited = applied_gain < desired_gain - 1e-6
    output = samples * (10.0 ** (applied_gain / 20.0))
    achieved = integrated_loudness_lufs(output, sample_rate_hz)
    return output, float(applied_gain), achieved, peak_limited


def _atomic_write(path: Path, buffer: AudioBuffer, bit_depth: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        write_wav(temporary, buffer, bit_depth)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def process_file(
    source: Path,
    output_dir: Path,
    options: ProcessingOptions,
    first_index: int,
) -> list[ProcessedAsset]:
    options.validate()
    source_buffer = read_wav(source)
    samples = source_buffer.samples.copy()
    original_rate = source_buffer.sample_rate_hz
    trim_offset = 0

    if options.mono:
        samples = to_mono(samples)
    if options.remove_dc:
        samples = remove_dc_offset(samples)
    if options.trim_silence:
        samples, trim_offset = trim_silence(
            samples,
            original_rate,
            options.silence_threshold_dbfs,
            options.trim_padding_ms,
        )
    samples = resample_audio(samples, original_rate, options.sample_rate_hz)
    segments = segment_audio(
        samples,
        options.sample_rate_hz,
        options.min_segment_seconds,
        options.max_segment_seconds,
        options.silence_threshold_dbfs,
    )

    assets: list[ProcessedAsset] = []
    for local_index, (start, end, segment) in enumerate(segments):
        normalized, gain_db, achieved_lufs, peak_limited = normalize_loudness(
            segment,
            options.sample_rate_hz,
            options.target_lufs,
            options.peak_ceiling_dbfs,
        )
        output_path = output_dir / f"{options.prefix}_{first_index + local_index:05d}.wav"
        output_buffer = AudioBuffer(normalized, options.sample_rate_hz, options.bit_depth)
        _atomic_write(output_path, output_buffer, options.bit_depth)
        metrics = analyze_buffer(output_buffer, output_path, options.silence_threshold_dbfs)
        source_start = trim_offset / original_rate + start / options.sample_rate_hz
        source_end = trim_offset / original_rate + end / options.sample_rate_hz
        assets.append(
            ProcessedAsset(
                source=str(source),
                output=str(output_path),
                segment_index=local_index,
                source_start_seconds=round(source_start, 6),
                source_end_seconds=round(source_end, 6),
                applied_gain_db=round(gain_db, 4),
                target_lufs=options.target_lufs,
                achieved_lufs=None if achieved_lufs is None else round(achieved_lufs, 4),
                peak_limited_normalization=peak_limited,
                metrics=metrics,
            )
        )
    return assets
