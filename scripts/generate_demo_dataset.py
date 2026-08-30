from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_dataset_toolkit.audio_io import AudioBuffer, write_wav


def main() -> None:
    output = ROOT / "demo_input"
    output.mkdir(exist_ok=True)
    sample_rate = 48000
    rng = np.random.default_rng(7)
    for index, frequency in enumerate((155.0, 185.0, 220.0), start=1):
        duration = 8.0 + index
        time = np.arange(int(sample_rate * duration)) / sample_rate
        envelope = 0.55 + 0.45 * np.sin(2 * math.pi * 1.8 * time) ** 2
        voice_like = envelope * (
            0.075 * np.sin(2 * math.pi * frequency * time)
            + 0.025 * np.sin(2 * math.pi * frequency * 2.1 * time)
            + 0.008 * rng.normal(size=time.size)
        )
        padded = np.concatenate([
            np.zeros(sample_rate // 2),
            voice_like,
            np.zeros(sample_rate // 3),
        ])
        write_wav(output / f"speaker_take_{index}.wav", AudioBuffer(padded, sample_rate, 24), 24)
    print(f"Demo files written to {output}")


if __name__ == "__main__":
    main()
