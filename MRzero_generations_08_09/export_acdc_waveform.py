"""Write one waveform CSV containing time in seconds and currents in amperes.

Channel numbers in the settings are 1-based. Columns after time follow physical
output order, starting with channel 0 in the CSV header. This module creates
data only: it does not upload a waveform or control the FPGA.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
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
    """Normalize output mapping and current bounds without changing settings.

    ``disabled_channels`` lists physical output numbers (1-based), not indices
    into the calibrated basis. Their bounds become exactly zero in basis
    order. Explicit zero bounds also mark their mapped outputs as disabled.
    Use the returned bounds for optimization as well as for CSV validation.
    """
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
    limits = _array(settings["max_current_A"], "max_current_A").copy()
    if limits.ndim == 0:
        limits = np.full(n_channels, float(limits))
    if limits.shape != (n_channels,) or (limits < 0).any() or (limits > 3).any():
        raise ValueError("max_current_A must be nonnegative, at most 3 A, scalar or one per channel.")
    disabled = np.asarray(settings.get("disabled_channels", []), dtype=object)
    if (disabled.ndim != 1 or any(isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) for value in disabled)):
        raise ValueError("disabled_channels must be a list of integer physical output numbers, not booleans.")
    disabled = [int(value) for value in disabled]
    if len(set(disabled)) != len(disabled):
        raise ValueError("disabled_channels must contain unique physical output numbers.")
    if not set(disabled).issubset(channels.astype(int).tolist()):
        raise ValueError("Every disabled_channels output must be present in channel_map.")
    limits[np.isin(channels, disabled)] = 0.
    clean = {
        "channel_map": channels.astype(int).tolist(),
        "channel_polarity": polarity.astype(int).tolist(),
        "max_current_A": limits.tolist(),
        "disabled_channels": sorted(channels[limits == 0].astype(int).tolist()),
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
                    numbered_filename: bool = False,
                    filename_prefix: str | None = None,
                    start_at_first_nonzero: bool = False) -> dict:
    """Write a waveform CSV, with optional zero-current boundary rows.

    ``currents_A`` has shape [calibrated channels, states]. ``state_edges_s``
    contains n_states+1 nonnegative, strictly increasing times relative to the
    selected time origin. State k lasts from edge k until edge k+1. A positive
    first edge emits a zero-current preamble at time zero when
    ``include_zero_preamble`` is true. ``include_final_zero_row`` controls the
    final all-zero row. ``include_header`` controls the commented column header,
    and ``zero_time_column`` writes zero instead of each timestamp. No clipping,
    interpolation or current-to-DAC conversion is applied.

    With ``start_at_first_nonzero=True``, omit initial states whose currents
    are all exactly zero and subtract the first remaining state's original
    start time from every exported edge. No state is duplicated, and no zero
    preamble is added. If all commands are zero, retain every state and subtract
    the original first edge, reporting ``first_nonzero_found=False``. Original
    model edges and state count remain in ``state_edges_s`` and ``states``;
    ``csv_state_edges_s`` and ``csv_states`` describe the exported layout.

    By default, write ``waveform.csv`` in an empty folder. With
    ``numbered_filename=True``, write ``waveform1.csv``, ``waveform2.csv``, etc.
    in the same folder, using one more than the largest existing numeric suffix.
    Existing files are never overwritten.

    ``filename_prefix`` selects a simple filename stem in a shared output
    folder. For example, ``"waveform_static"`` writes ``waveform_static.csv``;
    with numbering enabled, later exports become ``waveform_static2.csv``,
    ``waveform_static3.csv``, etc. The default naming remains unchanged.
    """
    if filename_prefix is not None and (not isinstance(filename_prefix, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", filename_prefix) is None):
        raise ValueError("filename_prefix must be a simple filename stem using letters, digits, '_' or '-'.")
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

    nonzero_states = np.flatnonzero(np.any(currents != 0, axis=0))
    first_nonzero_found = bool(nonzero_states.size)
    leading_zero_states = (int(nonzero_states[0])
                           if start_at_first_nonzero and first_nonzero_found else 0)
    time_origin_shift_s = float(edges[leading_zero_states]) if start_at_first_nonzero else 0.
    csv_edges = edges[leading_zero_states:] - time_origin_shift_s
    csv_currents = currents[:, leading_zero_states:]
    csv_states = csv_currents.shape[1]
    has_zero_preamble = bool(csv_edges[0] > 0 and include_zero_preamble)
    state_times = csv_edges if include_final_zero_row else csv_edges[:-1]
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
    mapped[first_state_row:first_state_row + csv_states, channel_map] = (
        csv_currents.T * np.asarray(config["channel_polarity"])
    )

    output = Path(output_dir)
    if filename_prefix is not None:
        output.mkdir(parents=True, exist_ok=True)
        if numbered_filename:
            pattern = re.compile(re.escape(filename_prefix) + r"([0-9]*)\.csv")
            numbers = [int(match.group(1) or 1) for entry in output.iterdir()
                       if (match := pattern.fullmatch(entry.name))]
            number = max(numbers, default=0) + 1
        else:
            number = 1
        while True:
            suffix = "" if number == 1 else str(number)
            csv_file = output / f"{filename_prefix}{suffix}.csv"
            try:
                stream = csv_file.open("x")
            except FileExistsError:
                if not numbered_filename:
                    raise FileExistsError(f"{csv_file.name} already exists; existing waveforms are never overwritten.") from None
                number += 1
            else:
                break
    elif numbered_filename:
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
        "csv_states": csv_states,
        "csv_state_edges_s": csv_edges.tolist(),
        "time_origin_shift_s": time_origin_shift_s,
        "leading_zero_states": leading_zero_states,
        "first_nonzero_found": first_nonzero_found,
        "start_at_first_nonzero": bool(start_at_first_nonzero),
        "csv_includes_zero_preamble": has_zero_preamble,
        "csv_final_zero_s": float(csv_edges[-1]) if include_final_zero_row else None,
        "maximum_abs_current_A": float(np.max(np.abs(currents))),
        "maximum_total_abs_current_A": float(np.max(total)),
        "final_zero_s": float(edges[-1]),
        "settings": config,
    }


def export_zero_time_copy(csv_file, output_path=None) -> dict:
    """Copy a waveform CSV, replacing only its numeric time fields with zero.

    The default companion of ``waveform1.csv`` is
    ``waveform1_zero_time.csv``. Headers, comments, blank lines, current fields,
    row order and line endings are preserved byte for byte. In particular, no
    current is parsed and reformatted or quantized. This companion does not
    contain playback chronology; the original CSV remains the timing source.

    An existing identical companion is accepted. A different existing file is
    never overwritten, and the source cannot be used as the destination.
    """
    source = Path(csv_file)
    output = (source.with_name(source.stem + "_zero_time.csv")
              if output_path is None else Path(output_path))
    if source.resolve() == output.resolve():
        raise ValueError("The zero-time companion must differ from the source CSV.")
    original = source.read_bytes()
    converted = []
    rows, column_count = 0, None
    for line in original.splitlines(keepends=True):
        stripped = line.lstrip()
        if not stripped.strip() or stripped.startswith(b"#"):
            converted.append(line)
            continue
        _, separator, remainder = line.partition(b",")
        fields = line.split(b",")
        try:
            values = np.asarray([float(field) for field in fields])
        except (ValueError, OverflowError) as error:
            raise ValueError("Source CSV needs finite numeric time and current fields.") from error
        if (not separator or not 2 <= len(fields) <= 65
                or not np.isfinite(values).all()
                or (column_count is not None and len(fields) != column_count)):
            raise ValueError("Source CSV needs consistent finite rows: time and 1..64 currents.")
        column_count = len(fields)
        rows += 1
        converted.append(b"0," + remainder)
    if rows == 0:
        raise ValueError("Source CSV contains no numeric waveform rows.")
    data = b"".join(converted)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(data)
        already_present = False
    except FileExistsError:
        if output.read_bytes() != data:
            raise FileExistsError(
                f"{output.name} already exists with different contents; existing waveforms are never overwritten."
            ) from None
        already_present = True
    return {
        "csv_file": str(output),
        "source_csv_file": str(source),
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "csv_sha256": hashlib.sha256(data).hexdigest(),
        "time_column_all_zero": True,
        "rows": rows,
        "physical_channels": column_count - 1,
        "already_present": already_present,
    }
