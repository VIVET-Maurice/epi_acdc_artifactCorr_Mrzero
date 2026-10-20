"""Write one waveform CSV containing time in seconds and currents in amperes.

Channel numbers in the settings are 1-based. Columns after time follow physical
output order, starting with channel 0 in the CSV header. This module creates
data only: it does not upload a waveform or control the FPGA.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np


def _array(value, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite numbers.")
    return result


def _positive(value, name: str) -> float:
    result = _array(value, name)
    if result.ndim or result <= 0:
        raise ValueError(f"{name} must be a positive scalar.")
    return float(result)


def quantize_currents(currents_A) -> np.ndarray:
    """Estimate nominal DAC rounding for numerical field verification.

    The CSV itself contains the original currents in amperes. This calculation
    follows the downstream waveform reader's signed 16-bit conversion; it does
    not apply electrical calibration or communicate with hardware.
    """
    currents = _array(currents_A, "currents_A")
    if np.any(np.abs(currents) > 3):
        raise ValueError("waveform accepts currents only within [-3, 3] A.")
    codes = np.trunc((currents / 5.0) * 32767.0)
    return (codes / 32767.0) * 5.0


def validate_waveform_settings(settings: dict, n_channels: int) -> dict:
    """Validate channel order, polarity and current limits for the CSV."""
    required = ("channel_map", "channel_polarity", "max_current_A", "total_current_limit_A")
    missing = [name for name in required if name not in settings]
    if missing:
        raise ValueError("Missing waveform settings: " + ", ".join(missing))
    channels = _array(settings["channel_map"], "channel_map")
    if (channels.shape != (n_channels,) or (channels != np.floor(channels)).any()
            or (channels < 1).any() or (channels > 64).any()
            or np.unique(channels).size != n_channels):
        raise ValueError("channel_map needs one distinct integer in 1..64 per calibrated channel.")
    polarity = _array(settings["channel_polarity"], "channel_polarity")
    if polarity.shape != (n_channels,) or not np.isin(polarity, [-1, 1]).all():
        raise ValueError("channel_polarity needs one +1 or -1 per calibrated channel.")
    limits = _array(settings["max_current_A"], "max_current_A")
    if limits.ndim == 0:
        limits = np.full(n_channels, float(limits))
    if limits.shape != (n_channels,) or (limits <= 0).any() or (limits > 3).any():
        raise ValueError("max_current_A must be positive, at most 3 A, scalar or one per channel.")
    clean = {
        "channel_map": channels.astype(int).tolist(),
        "channel_polarity": polarity.astype(int).tolist(),
        "max_current_A": limits.tolist(),
        "total_current_limit_A": _positive(settings["total_current_limit_A"], "total_current_limit_A"),
    }
    if settings.get("min_dac_interval_s") is not None:
        clean["min_dac_interval_s"] = _positive(settings["min_dac_interval_s"], "min_dac_interval_s")
    return clean


def export_waveform(currents_A, state_edges_s, settings: dict, output_dir, *,
                    include_header: bool = True,
                    include_zero_preamble: bool = True,
                    include_final_zero_row: bool = True,
                    zero_time_column: bool = False,
                    numbered_filename: bool = False) -> dict:
    """Write a waveform CSV, with optional zero-current boundary rows.

    ``currents_A`` has shape [calibrated channels, states]. ``state_edges_s``
    contains n_states+1 nonnegative, strictly increasing times relative to the
    selected time origin. State k lasts from edge k until edge k+1. A positive
    first edge emits a zero-current preamble at time zero when
    ``include_zero_preamble`` is true. ``include_final_zero_row`` controls the
    final all-zero row. ``include_header`` controls the commented column header,
    and ``zero_time_column`` writes zero instead of each timestamp. No clipping,
    interpolation, timing shift or current-to-DAC conversion is applied.

    By default, write ``waveform.csv`` in an empty folder. With
    ``numbered_filename=True``, write ``waveform1.csv``, ``waveform2.csv``, etc.
    in the same folder, using one more than the largest existing numeric suffix.
    Existing files are never overwritten.
    """
    currents = _array(currents_A, "currents_A")
    if currents.ndim != 2 or not 1 <= currents.shape[0] <= 64 or currents.shape[1] < 1:
        raise ValueError("currents_A must have shape [1..64 channels, at least one state].")
    n_channels, n_states = currents.shape
    config = validate_waveform_settings(settings, n_channels)
    edges = _array(state_edges_s, "state_edges_s")
    if edges.shape != (n_states + 1,) or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("state_edges_s must contain n_states+1 increasing nonnegative times.")
    if np.any(np.abs(currents) > np.asarray(config["max_current_A"])[:, None]):
        raise ValueError("Optimized currents exceed max_current_A; currents are never clipped.")
    total = np.sum(np.abs(currents), axis=0)
    if np.any(total >= config["total_current_limit_A"]):
        raise ValueError("A state reaches/exceeds total_current_limit_A; a strict margin is required.")

    has_zero_preamble = edges[0] > 0 and include_zero_preamble
    state_times = edges if include_final_zero_row else edges[:-1]
    times = np.r_[0.0, state_times] if has_zero_preamble else state_times
    minimum = config.get("min_dac_interval_s", 0.0)
    if np.any(np.diff(times) < minimum):
        raise ValueError(f"Waveform times contain an interval below {minimum:g} s.")
    if zero_time_column:
        times = np.zeros_like(times)

    # Include complete eight-output boards and set every unused output to zero.
    channel_map = np.asarray(config["channel_map"]) - 1
    physical_channels = ((int(channel_map.max()) + 8) // 8) * 8
    mapped = np.zeros((times.size, physical_channels))
    first_state_row = int(has_zero_preamble)
    mapped[first_state_row:first_state_row + n_states, channel_map] = (
        currents.T * np.asarray(config["channel_polarity"])
    )

    output = Path(output_dir)
    if numbered_filename:
        output.mkdir(parents=True, exist_ok=True)
        numbers = [int(match.group(1)) for entry in output.iterdir()
                   if (match := re.fullmatch(r"waveform([0-9]+)\.csv", entry.name))]
        number = max(numbers, default=0) + 1
        while True:
            csv_file = output / f"waveform{number}.csv"
            try:
                stream = csv_file.open("x")
            except FileExistsError:
                number += 1
            else:
                break
    else:
        csv_file = output / "waveform.csv"
        if csv_file.exists():
            raise FileExistsError("waveform.csv already exists; use a new output directory.")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Output folder must be empty so it contains only waveform.csv.")
        output.mkdir(parents=True, exist_ok=True)
        stream = csv_file.open("x")
    header = "time_s," + ",".join(f"ch{k}_A" for k in range(physical_channels))
    with stream:
        np.savetxt(stream, np.column_stack([times, mapped]),
                   delimiter=",", fmt="%.17g", header=header if include_header else "")
    return {
        "csv_file": str(csv_file),
        "physical_channels": physical_channels,
        "states": n_states,
        "state_edges_s": edges.tolist(),
        "maximum_abs_current_A": float(np.max(np.abs(currents))),
        "maximum_total_abs_current_A": float(np.max(total)),
        "final_zero_s": float(edges[-1]),
        "settings": config,
    }
