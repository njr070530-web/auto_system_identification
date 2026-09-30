# Yaw Gimbal Automatic System Identification V1

This tool runs one commanded 40 s yaw experiment and identifies

\[
\frac{\Omega(s)}{I_{cmd}(s)}=\frac{K}{Ts+1}e^{-T_ds}.
\]

The MCU does not chirp at normal startup. The gimbal must be enabled and its IMU
control loop must be ready; otherwise it returns `#AUTO_ID_ERROR,gimbal_not_ready`.
UART7 is configured for 921600 baud on PE7 (RX) and PE8 (TX).

Install the host dependencies:

```bash
python -m pip install numpy scipy pandas matplotlib pyserial
```

Run on Windows:

```bash
python tools/gimbal_auto_id/auto_id.py --port COM8 --baud 921600
```

Run on Linux:

```bash
python3 tools/gimbal_auto_id/auto_id.py --port /dev/ttyUSB0 --baud 921600
```

To finish analysis from an existing V1 capture without opening the UART or
running the gimbal again:

```bash
python tools/gimbal_auto_id/auto_id.py --analyze-raw tools/gimbal_auto_id/output/20260930_224139/raw.csv
```

Offline analysis writes `summary.csv`, `summary.json`, `signals.png`, and
`fit.png` next to that `raw.csv`. Since `raw.csv` contains only the CSV header
and telemetry rows, this mode uses the fixed V1 experiment metadata; MCU-side
drop statistics are reported as unavailable.

The script flushes stale input, sends `AUTO_ID_START`, waits for the MCU start
marker, saves every received sample verbatim to `raw.csv`, waits for the end
marker, then performs the fit. Pressing Ctrl+C sends `AUTO_ID_ABORT` before the
PC closes the port.

Each run creates `output/YYYYMMDD_HHMMSS/` containing:

- `raw.csv`: the complete received 0–40 s sample stream, unchanged and unfiltered
- `summary.csv` and `summary.json`: model parameters and quality metrics
- `signals.png`: references, feedback, speed, current, and current limits
- `fit.png`: measured and modeled yaw speed in the identification window

The start marker supplies `T` and `buffer_ratio`. The host derives the baseline
window as `0 <= t < T*buffer_ratio` and the identification window as
`T*buffer_ratio <= t < T`; there is no hard-coded 8 s analysis boundary. Fade
samples and saturated-current samples remain in the fit. Saturation is reported
as an experiment quality metric.

The firmware logs the final limited `yaw_cur` sent to the downstream chassis and
derives the saturation flag from the speed PID sum immediately before its
±25000 limiter. This is not a current sensor reading. Samples enter a 256-record
RAM queue and are formatted into 16-record DMA batches. Two 32-byte-aligned AXI
SRAM TX buffers are alternated and cleaned from DCache before DMA. Queue overflow
increments `dropped` rather than overwriting unread samples.

`K_tau` cannot be established from this repository with confirmed units and
sign, so J and B are unavailable by default. If a separately calibrated signed
torque-per-command conversion is known, pass `--k-tau VALUE`; the tool then uses
`B=K_tau/K` and `J=T*B`.

Run the deterministic synthetic test with:

```bash
python -m unittest tools/gimbal_auto_id/test_identification.py
```
