"""Plots for gimbal Auto-ID output."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_signals(frame, metadata: dict[str, float], output_path: Path) -> None:
    time_s = frame["time_ms"] / 1000.0
    buffer_time_s = float(metadata["T"]) * float(metadata["buffer_ratio"])
    current_limit = float(metadata["current_limit"])

    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(time_s, frame["yaw_pos_ref_rad"], linewidth=0.8)
    axes[0].set_ylabel("Yaw ref [rad]")
    axes[1].plot(time_s, frame["yaw_pos_fdb_rad"], linewidth=0.8)
    axes[1].set_ylabel("Yaw fdb [rad]")
    axes[2].plot(time_s, frame["yaw_speed_fdb_rad_s"], linewidth=0.8)
    axes[2].set_ylabel("Yaw speed [rad/s]")
    axes[3].plot(time_s, frame["yaw_current_cmd"], linewidth=0.7)
    axes[3].axhline(current_limit, color="tab:red", linestyle="--", linewidth=0.8)
    axes[3].axhline(-current_limit, color="tab:red", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel("Current cmd")
    axes[3].set_xlabel("Experiment time [s]")

    for axis in axes:
        axis.axvline(buffer_time_s, color="0.35", linestyle=":", linewidth=1.0)
        axis.grid(True, alpha=0.25)
    figure.suptitle("Yaw Auto-ID signals (dotted line: identification window start)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_fit(signals, result, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        signals.time_s,
        signals.measured_speed_rad_s,
        label="Measured yaw speed",
        linewidth=0.8,
    )
    axis.plot(
        signals.time_s,
        signals.model_speed_rad_s,
        label="Identified model",
        linewidth=1.0,
    )
    axis.set_xlabel("Experiment time [s]")
    axis.set_ylabel("Yaw speed [rad/s]")
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.set_title(
        f"K={result.K:.8g}, T={result.T_s:.6f} s, "
        f"Td={result.Td_s:.6f} s, fit={result.fit_percent:.2f} %, "
        f"RMSE={result.rmse:.6g}, saturation={result.saturation_ratio_percent:.2f} %"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
