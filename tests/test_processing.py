from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from voice_dataset_toolkit.audio_io import AudioBuffer, read_wav, write_wav
from voice_dataset_toolkit.models import ProcessingOptions
from voice_dataset_toolkit.processing import (
    normalize_loudness,
    process_file,
    resample_audio,
    segment_audio,
    trim_silence,
)


class ProcessingTests(unittest.TestCase):
    def test_trim_and_resample(self) -> None:
        source_rate = 48000
        samples = np.concatenate([
            np.zeros(source_rate),
            np.ones(source_rate) * 0.1,
            np.zeros(source_rate),
        ])
        trimmed, offset = trim_silence(samples, source_rate, -45.0, 100)
        self.assertGreater(offset, int(source_rate * 0.85))
        self.assertLess(trimmed.size, int(source_rate * 1.3))
        converted = resample_audio(trimmed, source_rate, 24000)
        self.assertAlmostEqual(converted.size / trimmed.size, 0.5, places=2)

    def test_segment_audio_obeys_target_size(self) -> None:
        sample_rate = 1000
        samples = np.ones(9500) * 0.1
        segments = segment_audio(samples, sample_rate, 1.0, 3.0)
        self.assertGreaterEqual(len(segments), 3)
        self.assertEqual(sum(item[2].size for item in segments), samples.size)
        self.assertTrue(all(item[2].size >= 1000 for item in segments))

    def test_normalization_respects_peak_ceiling(self) -> None:
        sample_rate = 24000
        time = np.arange(sample_rate * 2) / sample_rate
        samples = 0.8 * np.sin(2 * math.pi * 200 * time)
        output, _, _, limited = normalize_loudness(samples, sample_rate, -1.0, -3.0)
        self.assertTrue(limited)
        self.assertLessEqual(np.max(np.abs(output)), 10 ** (-2.8 / 20))

    def test_end_to_end_processing(self) -> None:
        sample_rate = 48000
        time = np.arange(sample_rate * 5) / sample_rate
        voice_like = 0.08 * (
            np.sin(2 * math.pi * 180 * time) + 0.3 * np.sin(2 * math.pi * 360 * time)
        )
        source = np.concatenate([np.zeros(sample_rate // 2), voice_like, np.zeros(sample_rate // 2)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "source.wav"
            output_dir = root / "prepared"
            write_wav(input_path, AudioBuffer(source, sample_rate, 24), 24)
            options = ProcessingOptions(max_segment_seconds=3.0, prefix="demo")
            assets = process_file(input_path, output_dir, options, 1)
            self.assertEqual(len(assets), 2)
            for asset in assets:
                output = read_wav(Path(asset.output))
                self.assertEqual(output.sample_rate_hz, 24000)
                self.assertEqual(output.bit_depth, 24)
                self.assertEqual(output.channels, 1)
                self.assertLessEqual(asset.metrics.true_peak_dbfs, -0.9)


if __name__ == "__main__":
    unittest.main()
