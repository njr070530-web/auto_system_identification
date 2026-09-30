#!/usr/bin/env python3
"""Run one MCU yaw experiment and identify a first-order-plus-delay model."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from identification import identify, load_raw_csv
from plotting import plot_fit, plot_signals
from serial_io import abort, open_serial, start_and_capture


V1_METADATA = {
    "pitch_rad": 0.0,
    "T": 40.0,
    "buffer_ratio": 0.2,
    "f_start": 1.0,
    "f_end": 50.0,
    "amp_max": 0.05,
    "omega_max": 1.7,
    "alpha_max": 100.0,
    "current_limit": 25000.0,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yaw Gimbal Automatic System Identification V1")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--port", help="UART port, for example COM8 or /dev/ttyUSB0")
    source.add_argument(
        "--analyze-raw",
        type=Path,
        help="analyze an existing V1 raw.csv without opening the UART",
    )
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--k-tau",
        type=float,
        default=None,
        help="Signed torque per current-command unit; enables J/B calculation",
    )
    parser.add_argument("--start-timeout", type=float, default=5.0)
    parser.add_argument("--experiment-timeout", type=float, default=60.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "output",
    )
    return parser.parse_args()


def _write_summary_csv(path: Path, summary: dict) -> None:
    field_names = [
        "pitch_rad",
        "K",
        "T_s",
        "Td_s",
        "fit_percent",
        "rmse",
        "J",
        "B",
        "sample_count",
        "dropped_samples",
        "saturation_sample_count",
        "saturation_ratio_percent",
        "identification_sample_count",
        "full_run_saturation_sample_count",
        "full_run_saturation_ratio_percent",
        "baseline_speed_mean_rad_s",
        "baseline_speed_std_rad_s",
        "baseline_current_mean",
        "optimizer_success",
        "optimizer_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerow({name: summary.get(name) for name in field_names})


def main() -> int:
    args = _arguments()
    if args.analyze_raw is not None:
        raw_path = args.analyze_raw.resolve()
        if not raw_path.is_file():
            raise FileNotFoundError(f"raw CSV does not exist: {raw_path}")
        run_directory = raw_path.parent
        metadata = V1_METADATA.copy()
        stats: dict[str, int] = {}
        with raw_path.open("r", encoding="utf-8", newline="") as raw_file:
            received_samples = max(sum(1 for _ in raw_file) - 1, 0)
        dropped_samples = None
        print(f"Analyzing existing raw CSV: {raw_path}")
        print(
            "Warning: raw.csv does not contain UART metadata or MCU drop "
            "statistics; using the fixed Auto-ID V1 experiment metadata.",
            file=sys.stderr,
        )
    else:
        run_directory = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_directory.mkdir(parents=True, exist_ok=False)
        raw_path = run_directory / "raw.csv"

        print(f"Opening {args.port} at {args.baud} baud")
        with open_serial(args.port, args.baud) as serial_port:
            try:
                capture = start_and_capture(
                    serial_port,
                    raw_path,
                    start_timeout_s=args.start_timeout,
                    experiment_timeout_s=args.experiment_timeout,
                )
            except KeyboardInterrupt:
                abort(serial_port)
                print("Interrupted; AUTO_ID_ABORT sent to MCU.", file=sys.stderr)
                return 130

        if capture.status != "finished":
            print(
                f"Experiment {capture.status}; raw samples kept in {raw_path}",
                file=sys.stderr,
            )
            return 2
        metadata = capture.metadata
        stats = capture.stats
        received_samples = capture.received_samples
        dropped_samples = int(stats.get("dropped", 0))

    frame = load_raw_csv(raw_path)
    result, fit_signals = identify(
        frame,
        metadata,
        dropped_samples=dropped_samples,
        k_tau=args.k_tau,
    )
    summary = result.to_dict()
    summary["metadata"] = metadata
    summary["mcu_stats"] = stats
    summary["received_samples"] = received_samples
    summary["k_tau"] = args.k_tau
    summary["historical_sanity_reference"] = {
        "K": -0.0001479917,
        "T_s": 0.044374897,
        "Td_s": 0.014585,
        "usage": "display-only sanity check; not used by the optimizer",
    }

    mcu_samples = stats.get("samples")
    if mcu_samples is not None and int(mcu_samples) != received_samples:
        print(
            "Warning: MCU reported "
            f"{mcu_samples} samples but the host received {received_samples}.",
            file=sys.stderr,
        )

    _write_summary_csv(run_directory / "summary.csv", summary)
    with (run_directory / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False, allow_nan=False)
        summary_file.write("\n")
    plot_signals(frame, metadata, run_directory / "signals.png")
    plot_fit(fit_signals, result, run_directory / "fit.png")

    print("\nIdentified model: Omega/I_cmd = K/(T*s+1) * exp(-Td*s)")
    print(f"  K  = {result.K:.9g}")
    print(f"  T  = {result.T_s:.6f} s")
    print(f"  Td = {result.Td_s:.6f} s")
    print(f"  Fit = {result.fit_percent:.2f} %")
    print(f"  RMSE = {result.rmse:.6g} rad/s")
    print("Current saturation:")
    print(
        f"  {result.saturation_sample_count} / "
        f"{result.identification_sample_count} samples"
    )
    print(f"  {result.saturation_ratio_percent:.2f} %")
    if result.dropped_samples is None:
        print("Dropped samples reported by MCU: unavailable in offline raw.csv")
    else:
        print(f"Dropped samples reported by MCU: {result.dropped_samples}")
    if result.J is None:
        print("J = unavailable")
        print("B = unavailable")
    else:
        print(f"J = {result.J:.9g}")
        print(f"B = {result.B:.9g}")
    print(f"Outputs: {run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
