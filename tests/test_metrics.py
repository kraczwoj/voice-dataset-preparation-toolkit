from __future__ import annotations

import math
import unittest

import numpy as np

from voice_dataset_toolkit.audio_io import AudioBuffer
from voice_dataset_toolkit.metrics import analyze_buffer, integrated_loudness_lufs


class MetricsTests(unittest.TestCase):
    def test_loudness_tracks_gain(self) -> None:
        sample_rate = 48000
        time = np.arange(sample_rate * 2) / sample_rate
        tone = 0.1 * np.sin(2 * math.pi * 440 * time)
        first = integrated_loudness_lufs(tone, sample_rate)
        second = integrated_loudness_lufs(tone * 0.5, sample_rate)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertAlmostEqual(first - second, 6.0206, places=2)

    def test_analysis_flags_clipping_and_stereo(self) -> None:
        samples = np.column_stack([np.ones(48000), np.ones(48000) * 0.5])
        result = analyze_buffer(AudioBuffer(samples, 48000, 24), "synthetic.wav")
        self.assertIn("clipping_detected", result.warnings)
        self.assertIn("multichannel_source", result.warnings)
        self.assertGreater(result.clipped_samples, 0)

    def test_silence_statistics_detect_padding(self) -> None:
        sample_rate = 24000
        samples = np.concatenate([
            np.zeros(sample_rate),
            np.ones(sample_rate) * 0.1,
            np.zeros(sample_rate),
        ])
        result = analyze_buffer(AudioBuffer(samples, sample_rate, 24), "silence.wav")
        self.assertGreater(result.leading_silence_seconds, 0.9)
        self.assertGreater(result.trailing_silence_seconds, 0.9)
        self.assertGreater(result.silence_ratio, 0.60)


if __name__ == "__main__":
    unittest.main()
