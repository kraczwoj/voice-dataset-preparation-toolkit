from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from voice_dataset_toolkit.audio_io import AudioBuffer, read_wav, write_wav


class AudioIOTests(unittest.TestCase):
    def test_pcm_round_trip_supported_depths(self) -> None:
        source = np.linspace(-0.9, 0.9, 2000, dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            for depth, tolerance in ((16, 4e-5), (24, 2e-7), (32, 2e-9)):
                path = Path(directory) / f"roundtrip_{depth}.wav"
                write_wav(path, AudioBuffer(source, 24000, depth), depth)
                restored = read_wav(path)
                self.assertEqual(restored.bit_depth, depth)
                self.assertEqual(restored.sample_rate_hz, 24000)
                np.testing.assert_allclose(restored.samples, source, atol=tolerance)

    def test_stereo_shape_is_preserved(self) -> None:
        source = np.column_stack([np.zeros(100), np.ones(100) * 0.2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            write_wav(path, AudioBuffer(source, 48000, 24), 24)
            restored = read_wav(path)
            self.assertEqual(restored.samples.shape, (100, 2))
            self.assertEqual(restored.channels, 2)


if __name__ == "__main__":
    unittest.main()
