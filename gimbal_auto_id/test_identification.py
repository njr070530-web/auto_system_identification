from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from identification import identify, load_raw_csv


class IdentificationSyntheticTest(unittest.TestCase):
    def test_load_skips_malformed_row_without_compressing_time(self) -> None:
        raw_csv = """time_ms,yaw_pos_ref_rad,yaw_pos_fdb_rad,yaw_speed_fdb_rad_s,yaw_current_cmd,yaw_current_saturated
2594,0.000000,-0.000201,-0.001223,-125,0
2595,0.000000,-0.000204,-0.003353,-264,0
2596,0.000000,-0.000202,0.001973,133,0
2597,0.000000,-0.0000000,-0.000204,0.008897,-31,0
2601,0.000000,-0.000205,-0.000690,-492,0
2602,0.000000,-0.000215,-0.009745,-661,0
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "raw.csv"
            raw_path.write_text(raw_csv, encoding="utf-8")
            warning_output = StringIO()
            with redirect_stderr(warning_output):
                frame = load_raw_csv(raw_path)

        self.assertEqual(frame["time_ms"].tolist(), [2594, 2595, 2596, 2601, 2602])
        self.assertAlmostEqual(
            (frame["time_ms"].iloc[3] - frame["time_ms"].iloc[2]) / 1000.0,
            0.005,
        )
        self.assertEqual(frame.attrs["sample_index_gaps"], [(2596.0, 2601.0)])
        warning = warning_output.getvalue()
        self.assertIn("line 5", warning)
        self.assertIn(
            "2597,0.000000,-0.0000000,-0.000204,0.008897,-31,0",
            warning,
        )
        self.assertIn("skipped 1 malformed telemetry row", warning)
        self.assertIn("Sample-index gap detected: 2596 -> 2601", warning)
        self.assertIn("Continuing identification", warning)

    def test_recovers_negative_gain_time_constant_and_delay(self) -> None:
        rng = np.random.default_rng(20260927)
        dt_s = 0.001
        time_s = np.arange(0.0, 40.0, dt_s)
        current = np.zeros_like(time_s)
        excitation = rng.choice([-1.0, 1.0], size=time_s.size)
        excitation = np.convolve(excitation, np.ones(20) / 20.0, mode="same")
        current[time_s >= 8.0] = 5000.0 * excitation[time_s >= 8.0]

        expected_K = -1.7e-4
        expected_T_s = 0.052
        expected_delay_s = 0.017
        delayed_current = np.interp(
            time_s - expected_delay_s,
            time_s,
            current,
            left=current[0],
        )
        speed = np.zeros_like(time_s)
        pole = np.exp(-dt_s / expected_T_s)
        for index in range(1, len(time_s)):
            speed[index] = (
                pole * speed[index - 1]
                + expected_K * (1.0 - pole) * delayed_current[index - 1]
            )
        speed += rng.normal(0.0, 0.003, size=speed.size)

        frame = pd.DataFrame(
            {
                "time_ms": np.arange(time_s.size),
                "yaw_pos_ref_rad": np.zeros_like(time_s),
                "yaw_pos_fdb_rad": np.zeros_like(time_s),
                "yaw_speed_fdb_rad_s": speed,
                "yaw_current_cmd": current,
                "yaw_current_saturated": np.zeros_like(time_s, dtype=int),
            }
        )
        metadata = {"T": 40.0, "buffer_ratio": 0.2, "pitch_rad": 0.0}
        result, _ = identify(frame, metadata)

        self.assertLess(abs((result.K - expected_K) / expected_K), 0.08)
        self.assertLess(abs((result.T_s - expected_T_s) / expected_T_s), 0.15)
        self.assertLess(abs(result.Td_s - expected_delay_s), 0.004)
        self.assertGreater(result.fit_percent, 90.0)


if __name__ == "__main__":
    unittest.main()
