"""Write two identical static current states without controlling ACDC hardware."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from export_acdc_waveform import validate_waveform_settings


def export_static_currents(currents_A, settings: dict, output_dir,
                           hold_timestamp_s: float = 1.0) -> dict:
    """Write only ``static_currents.csv`` in an empty output directory.

    ``currents_A`` contains one current per calibrated channel. The CSV has no
    header: its first column is time in seconds, followed by physical output
    currents in amperes. Rows at zero and ``hold_timestamp_s`` contain the
    same currents. There is no zero-current preamble or terminal zero state.

    Mapping, polarity and limits use the dynamic waveform settings. Complete
    eight-output boards are included, with unused outputs set to zero. Currents
    are never clipped or converted to DAC codes. The final row does not itself
    guarantee that controller software keeps the currents applied afterward.
    """
    if np.iscomplexobj(currents_A):
        raise ValueError("currents_A must be real.")
    currents = np.asarray(currents_A, dtype=float)
    if currents.ndim != 1 or not 1 <= currents.size <= 64:
        raise ValueError("currents_A must contain one current per channel, for 1..64 channels.")
    if not np.isfinite(currents).all():
        raise ValueError("currents_A must contain only finite numbers.")
    config = validate_waveform_settings(settings, currents.size)
    if np.any(np.abs(currents) > np.asarray(config["max_current_A"])):
        raise ValueError("Optimized currents exceed max_current_A; currents are never clipped.")
    total = float(np.sum(np.abs(currents)))
    if total >= config["total_current_limit_A"]:
        raise ValueError("Static currents reach/exceed total_current_limit_A; a strict margin is required.")

    if np.iscomplexobj(hold_timestamp_s):
        raise ValueError("hold_timestamp_s must be a finite positive scalar in seconds.")
    hold = np.asarray(hold_timestamp_s, dtype=float)
    if hold.ndim or not np.isfinite(hold) or hold <= 0:
        raise ValueError("hold_timestamp_s must be a finite positive scalar in seconds.")
    hold = float(hold)
    if hold < config.get("min_dac_interval_s", 0.0):
        raise ValueError("hold_timestamp_s is below min_dac_interval_s.")

    channel_map = np.asarray(config["channel_map"]) - 1
    physical_channels = ((int(channel_map.max()) + 8) // 8) * 8
    physical_currents = np.zeros(physical_channels)
    physical_currents[channel_map] = currents * np.asarray(config["channel_polarity"])
    data = np.column_stack(([0.0, hold], np.tile(physical_currents, (2, 1))))

    output = Path(output_dir)
    csv_file = output / "static_currents.csv"
    if csv_file.exists():
        raise FileExistsError("static_currents.csv already exists; use a new output directory.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output folder must be empty so it contains only static_currents.csv.")
    output.mkdir(parents=True, exist_ok=True)
    with csv_file.open("x") as stream:
        np.savetxt(stream, data, delimiter=",", fmt="%.17g")
    return {
        "csv_file": str(csv_file),
        "physical_channels": physical_channels,
        "states": 1,
        "timestamps_s": [0.0, hold],
        "maximum_abs_current_A": float(np.max(np.abs(currents))),
        "total_abs_current_A": total,
        "settings": config,
    }
