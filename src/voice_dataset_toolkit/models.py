from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any


@dataclass(slots=True)
class AudioMetrics:
    path: str
    sample_rate_hz: int
    channels: int
    bit_depth: int
    frames: int
    duration_seconds: float
    rms_dbfs: float
    sample_peak_dbfs: float
    true_peak_dbfs: float
    integrated_loudness_lufs: float | None
    dc_offset: float
    crest_factor_db: float
    clipped_samples: int
    clipped_ratio: float
    silence_ratio: float
    leading_silence_seconds: float
    trailing_silence_seconds: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProcessedAsset:
    source: str
    output: str
    segment_index: int
    source_start_seconds: float
    source_end_seconds: float
    applied_gain_db: float
    target_lufs: float | None
    achieved_lufs: float | None
    peak_limited_normalization: bool
    metrics: AudioMetrics

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data


@dataclass(slots=True)
class ProcessingOptions:
    sample_rate_hz: int = 24000
    bit_depth: int = 24
    mono: bool = True
    remove_dc: bool = True
    trim_silence: bool = True
    silence_threshold_dbfs: float = -45.0
    trim_padding_ms: int = 150
    min_segment_seconds: float = 1.0
    max_segment_seconds: float = 30.0
    target_lufs: float | None = -23.0
    peak_ceiling_dbfs: float = -1.0
    prefix: str = "voice"

    def validate(self) -> None:
        if self.sample_rate_hz < 8000:
            raise ValueError("sample_rate_hz must be at least 8000")
        if self.bit_depth not in (16, 24, 32):
            raise ValueError("bit_depth must be 16, 24, or 32")
        if self.min_segment_seconds <= 0:
            raise ValueError("min_segment_seconds must be positive")
        if self.max_segment_seconds < self.min_segment_seconds:
            raise ValueError("max_segment_seconds must be >= min_segment_seconds")
        if not -90.0 <= self.silence_threshold_dbfs <= -10.0:
            raise ValueError("silence_threshold_dbfs must be between -90 and -10")
        if not -12.0 <= self.peak_ceiling_dbfs <= 0.0:
            raise ValueError("peak_ceiling_dbfs must be between -12 and 0")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.prefix):
            raise ValueError(
                "prefix must start with an alphanumeric character and contain only letters, "
                "numbers, dots, underscores, or hyphens"
            )


SUPPORTED_SUFFIXES = {".wav", ".wave"}


def discover_audio_files(root: Path, recursive: bool = True) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
