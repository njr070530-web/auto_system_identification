"""UART transport for the one-shot gimbal Auto-ID experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any


CSV_HEADER = (
    "time_ms,yaw_pos_ref_rad,yaw_pos_fdb_rad,yaw_speed_fdb_rad_s,"
    "yaw_current_cmd,yaw_current_saturated"
)


@dataclass
class CaptureResult:
    metadata: dict[str, float]
    stats: dict[str, int]
    received_samples: int
    status: str


def _parse_key_values(line: str, prefix: str, value_type: type) -> dict[str, Any]:
    if not line.startswith(prefix):
        raise ValueError(f"Expected marker {prefix!r}, got {line!r}")
    values: dict[str, Any] = {}
    for field in line[len(prefix) :].lstrip(",").split(","):
        if not field:
            continue
        key, separator, value = field.partition("=")
        if not separator:
            continue
        values[key] = value_type(value)
    return values


def open_serial(port: str, baud: int):
    try:
        import serial
    except ImportError as exc:  # pragma: no cover - depends on host setup
        raise RuntimeError("pyserial is required: python -m pip install pyserial") from exc

    return serial.Serial(
        port=port,
        baudrate=baud,
        timeout=0.2,
        write_timeout=1.0,
    )


def start_and_capture(
    serial_port,
    raw_csv_path: Path,
    start_timeout_s: float = 5.0,
    experiment_timeout_s: float = 60.0,
) -> CaptureResult:
    """Start the MCU experiment and stream its CSV payload directly to disk."""

    serial_port.reset_input_buffer()
    serial_port.write(b"AUTO_ID_START\r\n")
    serial_port.flush()

    metadata: dict[str, float] | None = None
    stats: dict[str, int] = {}
    received_samples = 0
    header_seen = False
    start_deadline = time.monotonic() + start_timeout_s
    experiment_deadline: float | None = None

    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_csv_path.open("w", encoding="utf-8", newline="") as raw_file:
        while True:
            now = time.monotonic()
            if metadata is None and now > start_deadline:
                raise TimeoutError("Timed out waiting for #AUTO_ID_START")
            if experiment_deadline is not None and now > experiment_deadline:
                raise TimeoutError("Timed out waiting for #AUTO_ID_END")

            payload = serial_port.readline()
            if not payload:
                continue
            line = payload.decode("ascii", errors="replace").strip()
            if not line:
                continue

            if line.startswith("#AUTO_ID_ERROR,"):
                raise RuntimeError(f"MCU rejected Auto-ID: {line.split(',', 1)[1]}")

            if line == "#AUTO_ID_ABORTED":
                raw_file.flush()
                return CaptureResult(metadata or {}, stats, received_samples, "aborted")

            if line.startswith("#AUTO_ID_START,"):
                metadata = _parse_key_values(line, "#AUTO_ID_START", float)
                experiment_deadline = time.monotonic() + experiment_timeout_s
                continue

            if metadata is None:
                continue

            if line == CSV_HEADER:
                if not header_seen:
                    raw_file.write(line + "\n")
                    raw_file.flush()
                    header_seen = True
                continue

            if line.startswith("#AUTO_ID_STATS,"):
                stats = _parse_key_values(line, "#AUTO_ID_STATS", int)
                continue

            if line == "#AUTO_ID_END":
                if not header_seen:
                    raise RuntimeError("MCU ended the experiment without a CSV header")
                raw_file.flush()
                return CaptureResult(metadata, stats, received_samples, "finished")

            if line.startswith("#"):
                continue

            if not header_seen:
                raise RuntimeError(f"Received sample before CSV header: {line!r}")

            # Preserve the MCU text exactly. Numeric validation happens later;
            # raw.csv is never rewritten or filtered.
            raw_file.write(line + "\n")
            received_samples += 1
            if received_samples % 1000 == 0:
                raw_file.flush()


def abort(serial_port) -> None:
    serial_port.write(b"AUTO_ID_ABORT\r\n")
    serial_port.flush()
