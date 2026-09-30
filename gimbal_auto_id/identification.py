"""Time-domain first-order-plus-dead-time identification."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize


REQUIRED_COLUMNS = [
    "time_ms",
    "yaw_pos_ref_rad",
    "yaw_pos_fdb_rad",
    "yaw_speed_fdb_rad_s",
    "yaw_current_cmd",
    "yaw_current_saturated",
]

DELAY_MIN_S = 0.0
DELAY_MAX_S = 0.050
TIME_CONSTANT_MIN_S = 1.0e-5
TIME_CONSTANT_MAX_S = 5.0


@dataclass
class IdentificationResult:
    pitch_rad: float
    K: float
    T_s: float
    Td_s: float
    fit_percent: float
    rmse: float
    J: float | None
    B: float | None
    sample_count: int
    identification_sample_count: int
    dropped_samples: int | None
    saturation_sample_count: int
    saturation_ratio_percent: float
    full_run_saturation_sample_count: int
    full_run_saturation_ratio_percent: float
    baseline_speed_mean_rad_s: float
    baseline_speed_std_rad_s: float
    baseline_current_mean: float
    optimizer_success: bool
    optimizer_message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FitSignals:
    time_s: np.ndarray
    measured_speed_rad_s: np.ndarray
    model_speed_rad_s: np.ndarray


def load_raw_csv(path) -> pd.DataFrame:
    rows: list[list[str]] = []
    malformed_rows: list[tuple[int, str]] = []

    with open(path, "r", encoding="utf-8", newline="") as raw_file:
        header_line = raw_file.readline()
        header = next(csv.reader([header_line])) if header_line else []
        if header != REQUIRED_COLUMNS:
            raise ValueError(
                "Unexpected raw.csv columns. Expected: " + ",".join(REQUIRED_COLUMNS)
            )

        for line_number, raw_line in enumerate(raw_file, start=2):
            raw_content = raw_line.rstrip("\r\n")
            try:
                fields = next(csv.reader([raw_content]))
            except csv.Error:
                fields = []
            if len(fields) != len(REQUIRED_COLUMNS):
                malformed_rows.append((line_number, raw_content))
                print(
                    "Warning: skipped malformed telemetry row at "
                    f"line {line_number}: {raw_content}",
                    file=sys.stderr,
                )
                continue
            rows.append(fields)

    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    if malformed_rows:
        noun = "row" if len(malformed_rows) == 1 else "rows"
        print(
            f"Warning: skipped {len(malformed_rows)} malformed telemetry {noun}.",
            file=sys.stderr,
        )
    if frame.empty:
        raise ValueError("raw.csv contains no valid samples")

    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.all(np.isfinite(frame[REQUIRED_COLUMNS].to_numpy(dtype=float))):
        raise ValueError("raw.csv contains NaN or infinite values")
    if not frame["time_ms"].is_monotonic_increasing:
        raise ValueError("time_ms must be monotonically increasing")

    time_ms = frame["time_ms"].to_numpy(dtype=float)
    gaps = [
        (float(previous), float(current))
        for previous, current in zip(time_ms[:-1], time_ms[1:])
        if current - previous > 1.0
    ]
    for previous, current in gaps:
        print(
            f"Sample-index gap detected: {previous:g} -> {current:g}.",
            file=sys.stderr,
        )
    if malformed_rows:
        print(
            "Continuing identification with remaining valid data.",
            file=sys.stderr,
        )

    frame.attrs["malformed_rows"] = malformed_rows
    frame.attrs["sample_index_gaps"] = gaps
    return frame


def _delayed_input(
    query_time_s: np.ndarray,
    delay_s: float,
    source_time_s: np.ndarray,
    source_input: np.ndarray,
) -> np.ndarray:
    return np.interp(
        query_time_s - delay_s,
        source_time_s,
        source_input,
        left=float(source_input[0]),
        right=float(source_input[-1]),
    )


def _simulate_at_samples(
    time_s: np.ndarray,
    delayed_current: np.ndarray,
    initial_speed: float,
    K: float,
    T_s: float,
) -> np.ndarray:
    output = np.empty_like(time_s)
    output[0] = initial_speed
    for index in range(1, len(time_s)):
        dt_s = max(float(time_s[index] - time_s[index - 1]), 0.0)
        pole = np.exp(-dt_s / T_s)
        output[index] = (
            pole * output[index - 1]
            + K * (1.0 - pole) * delayed_current[index - 1]
        )
    return output


def _simulate_contiguous_segments(
    time_s: np.ndarray,
    delayed_current: np.ndarray,
    measured_speed: np.ndarray,
    K: float,
    T_s: float,
    maximum_step_s: float,
) -> np.ndarray:
    """Simulate each measured segment without crossing a telemetry gap."""

    output = np.empty_like(time_s)
    segment_starts = np.concatenate(
        ([0], np.flatnonzero(np.diff(time_s) > maximum_step_s) + 1)
    )
    segment_stops = np.concatenate((segment_starts[1:], [len(time_s)]))
    for start, stop in zip(segment_starts, segment_stops):
        output[start:stop] = _simulate_at_samples(
            time_s[start:stop],
            delayed_current[start:stop],
            float(measured_speed[start]),
            K,
            T_s,
        )
    return output


def _initial_delay_and_gain(
    fit_time_s: np.ndarray,
    measured_speed: np.ndarray,
    source_time_s: np.ndarray,
    source_current: np.ndarray,
    dt_s: float,
) -> tuple[float, float]:
    delay_grid = np.arange(DELAY_MIN_S, DELAY_MAX_S + 0.5 * dt_s, dt_s)
    centered_speed = measured_speed - np.mean(measured_speed)
    best_score = -np.inf
    best_delay = 0.0
    best_gain = 0.0

    for delay_s in delay_grid:
        shifted_current = _delayed_input(
            fit_time_s, delay_s, source_time_s, source_current
        )
        centered_current = shifted_current - np.mean(shifted_current)
        denominator = float(np.dot(centered_current, centered_current))
        if denominator <= np.finfo(float).eps:
            continue
        covariance = float(np.dot(centered_current, centered_speed))
        gain = covariance / denominator
        score = abs(covariance) / np.sqrt(
            denominator * max(float(np.dot(centered_speed, centered_speed)), 1.0e-30)
        )
        if score > best_score:
            best_score = score
            best_delay = float(delay_s)
            best_gain = float(gain)

    if not np.isfinite(best_score) or abs(best_gain) < np.finfo(float).eps:
        raise ValueError("Insufficient current excitation for identification")
    return best_delay, best_gain


def identify(
    frame: pd.DataFrame,
    metadata: dict[str, float],
    dropped_samples: int | None = 0,
    k_tau: float | None = None,
) -> tuple[IdentificationResult, FitSignals]:
    duration_s = float(metadata["T"])
    buffer_time_s = duration_s * float(metadata["buffer_ratio"])
    pitch_rad = float(metadata.get("pitch_rad", 0.0))

    time_s = frame["time_ms"].to_numpy(dtype=float) / 1000.0
    speed = frame["yaw_speed_fdb_rad_s"].to_numpy(dtype=float)
    current = frame["yaw_current_cmd"].to_numpy(dtype=float)
    saturated = frame["yaw_current_saturated"].to_numpy(dtype=int) != 0

    baseline_mask = (time_s >= 0.0) & (time_s < buffer_time_s)
    identification_mask = (time_s >= buffer_time_s) & (time_s < duration_s)
    if np.count_nonzero(baseline_mask) < 2:
        raise ValueError("Baseline region contains fewer than two samples")
    if np.count_nonzero(identification_mask) < 100:
        raise ValueError("Identification region contains too few samples")

    positive_steps = np.diff(time_s)
    positive_steps = positive_steps[positive_steps > 0.0]
    if positive_steps.size == 0:
        raise ValueError("Identification timestamps do not advance")
    dt_s = float(np.median(positive_steps))

    # A delayed input lookup can otherwise interpolate through a missing UART
    # interval. Exclude the short delay horizon after each gap, then fit the
    # remaining contiguous segments independently.
    fit_mask = identification_mask.copy()
    gap_indices = np.flatnonzero(np.diff(time_s) > 1.5 * dt_s)
    for gap_index in gap_indices:
        first_sample_after_gap = time_s[gap_index + 1]
        fit_mask &= ~(
            (time_s >= first_sample_after_gap)
            & (time_s <= first_sample_after_gap + DELAY_MAX_S)
        )

    fit_time = time_s[fit_mask]
    fit_speed = speed[fit_mask]
    if fit_time.size < 100:
        raise ValueError("Too few valid samples remain after telemetry gaps")
    maximum_contiguous_step_s = 1.5 * dt_s

    initial_delay, initial_gain = _initial_delay_and_gain(
        fit_time, fit_speed, time_s, current, dt_s
    )
    initial_T_s = 0.05
    speed_scale = max(float(np.std(fit_speed)), 1.0e-6)

    def residual(parameters: np.ndarray) -> np.ndarray:
        K, T_s, delay_s = parameters
        delayed_current = _delayed_input(
            fit_time, delay_s, time_s, current
        )
        predicted = _simulate_contiguous_segments(
            fit_time,
            delayed_current,
            fit_speed,
            float(K),
            float(T_s),
            maximum_contiguous_step_s,
        )
        return (predicted - fit_speed) / speed_scale

    fit = optimize.least_squares(
        residual,
        x0=np.array([initial_gain, initial_T_s, initial_delay]),
        bounds=(
            np.array([-np.inf, max(TIME_CONSTANT_MIN_S, dt_s / 10.0), DELAY_MIN_S]),
            np.array([np.inf, TIME_CONSTANT_MAX_S, DELAY_MAX_S]),
        ),
        x_scale=np.array([max(abs(initial_gain), 1.0e-8), 0.05, 0.01]),
        max_nfev=250,
    )
    K, T_s, delay_s = (float(value) for value in fit.x)

    delayed_at_samples = _delayed_input(fit_time, delay_s, time_s, current)
    predicted_at_samples = _simulate_contiguous_segments(
        fit_time,
        delayed_at_samples,
        fit_speed,
        K,
        T_s,
        maximum_contiguous_step_s,
    )
    error = fit_speed - predicted_at_samples
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = float(np.linalg.norm(fit_speed - np.mean(fit_speed)))
    fit_percent = (
        float(100.0 * (1.0 - np.linalg.norm(error) / denominator))
        if denominator > np.finfo(float).eps
        else float("nan")
    )

    identification_saturated = int(np.count_nonzero(saturated[identification_mask]))
    identification_count = int(np.count_nonzero(identification_mask))
    full_saturated = int(np.count_nonzero(saturated))
    full_count = len(frame)

    B = None if k_tau is None else float(k_tau / K)
    J = None if B is None else float(T_s * B)
    baseline_speed = speed[baseline_mask]
    baseline_current = current[baseline_mask]

    result = IdentificationResult(
        pitch_rad=pitch_rad,
        K=K,
        T_s=T_s,
        Td_s=delay_s,
        fit_percent=fit_percent,
        rmse=rmse,
        J=J,
        B=B,
        sample_count=full_count,
        identification_sample_count=identification_count,
        dropped_samples=None if dropped_samples is None else int(dropped_samples),
        saturation_sample_count=identification_saturated,
        saturation_ratio_percent=100.0 * identification_saturated / identification_count,
        full_run_saturation_sample_count=full_saturated,
        full_run_saturation_ratio_percent=100.0 * full_saturated / full_count,
        baseline_speed_mean_rad_s=float(np.mean(baseline_speed)),
        baseline_speed_std_rad_s=float(np.std(baseline_speed)),
        baseline_current_mean=float(np.mean(baseline_current)),
        optimizer_success=bool(fit.success),
        optimizer_message=str(fit.message),
    )
    signals = FitSignals(fit_time, fit_speed, predicted_at_samples)
    return result, signals
