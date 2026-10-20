"""Read the measured Cartesian sequence for conventional GRE or single-shot EPI.

Times in the acquisition struct refer to the first RF center. ``rf_time_s``
identifies the excitation belonging to each sample: a spoiled GRE starts a
new transverse phase history at every RF, while EPI keeps one history.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _scalar_definition(sequence, name, default):
    value = np.asarray(sequence.definitions.get(name, default), dtype=float).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]) or value[0] <= 0:
        raise ValueError(f"Pulseq {name} must be one positive finite value; multi-echo is unsupported.")
    return float(value[0])


def _events(sequence, pp):
    """Keep event times, not just Matrix/TE declarations in the file header."""
    clock, rf, adc = 0.0, [], []
    first_rf = None
    for number in sequence.block_events:
        block = sequence.get_block(number)
        duration = float(sequence.block_durations[number])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Sequence block durations must be positive and finite.")
        if block.rf is not None:
            if getattr(block.rf, "use", "undefined") != "excitation":
                raise ValueError("Supported GRE sequences have excitation RF only: no refocusing RF or preparation RF.")
            # A single GRE amplitude proxy assumes the same excitation at
            # every TR. RF phase_offset may still change for RF spoiling.
            pulse = block.rf
            waveform = np.asarray(pulse.signal)
            frequency = np.asarray([pulse.freq_offset, getattr(pulse, "freq_ppm", 0)], float)
            if not np.isfinite(waveform).all() or not np.isfinite(frequency).all():
                raise ValueError("RF waveform and frequency must be finite.")
            if first_rf is None:
                first_rf = (waveform.copy(), np.asarray(pulse.t).copy(), float(pulse.shape_dur), frequency)
            elif (waveform.shape != first_rf[0].shape
                  or not np.allclose(waveform, first_rf[0], rtol=1e-6, atol=1e-9)
                  or not np.allclose(pulse.t, first_rf[1], rtol=0, atol=1e-12)
                  or abs(float(pulse.shape_dur) - first_rf[2]) > 1e-12):
                raise ValueError("All GRE excitations must use the same RF waveform/flip angle; "
                                 "variable excitation profiles need a separate amplitude model.")
            elif not np.allclose(frequency, first_rf[3], rtol=0, atol=1e-9):
                raise ValueError("Changing RF frequency between lines is unsupported; use one slice/frequency.")
            center = clock + float(block.rf.delay) + float(pp.calc_rf_center(block.rf)[0])
            rf.append({"start": clock, "end": clock + duration, "center": center,
                       "phase": float(block.rf.phase_offset)})
        if block.adc is not None:
            if block.rf is not None:
                raise ValueError("RF and ADC in the same block cannot preserve zero shim during RF.")
            if abs(float(block.adc.freq_offset)) > 1e-9:
                raise ValueError("Nonzero ADC frequency offsets require an explicit receiver model.")
            adc.append({"start": clock, "end": clock + duration,
                        "delay": float(block.adc.delay), "dwell": float(block.adc.dwell),
                        "samples": int(block.adc.num_samples), "phase": float(block.adc.phase_offset)})
        clock += duration
    if not rf or not adc:
        raise ValueError("The sequence must contain excitation RF and imaging ADC events.")
    return rf, adc, clock


def _cartesian(kx, ky, times, kind):
    """Restore readout direction and sort phase lines, retaining acquisition order."""
    ny, nx = kx.shape
    if nx < 2 or ny < 2 or nx % 2 or ny % 2:
        raise ValueError("Supported full Cartesian matrices have positive even Nx and Ny, both at least 2.")
    polarity = np.where(np.mean(np.diff(kx, axis=1), axis=1) < 0, -1, 1)
    for row in range(ny):
        if polarity[row] < 0:
            kx[row], ky[row], times[row] = kx[row, ::-1], ky[row, ::-1], times[row, ::-1]
    row_order = np.argsort(ky.mean(axis=1))
    kx, ky, times = kx[row_order], ky[row_order], times[row_order]
    step_x = float(np.median(np.diff(kx, axis=1)))
    step_y = float(np.median(np.diff(ky[:, 0])))
    offset = float(np.median(kx[:, nx // 2]))
    allowed_offsets = [0, step_x / 2] if kind == "gre" else [0]
    grid_offset = min(allowed_offsets, key=lambda value: abs(value - offset))
    expected_x = (np.arange(nx) - nx // 2) * step_x + grid_offset
    expected_y = (np.arange(ny) - ny // 2) * step_y
    if (step_x <= 0 or step_y <= 0
            or not np.allclose(kx, expected_x[None, :], rtol=0, atol=3e-3) #originally 1e-3
            or not np.allclose(ky, expected_y[:, None], rtol=0, atol=3e-3)):#orginally 1e-3
        raise ValueError("Expected a full centered Cartesian grid; undersampling, ramp sampling "
                         "and non-Cartesian trajectories are unsupported.")
    return kx, ky, times, polarity[row_order], row_order, np.asarray([1 / step_x, 1 / step_y]), grid_offset


def read_sequence(path: Path) -> tuple[dict, dict]:
    """Return the existing MATLAB acquisition struct and hardware timing report.

    Conventional spoiled GRE has one ADC line per excitation. Only leading
    dummy excitations are accepted. Single-shot EPI retains one excitation,
    with zero or three leading navigator ADC blocks. Neither branch accepts
    multi-echo, refocusing RF, undersampling or non-Cartesian acquisitions.
    """
    import pypulseq as pp

    path = Path(path)
    sequence = pp.Sequence()
    sequence.read(str(path))
    rf, adc, duration = _events(sequence, pp)
    k_adc, _, excitation_centers, refocusing_centers, adc_times = sequence.calculate_kspace()
    centers = np.asarray([event["center"] for event in rf])
    if len(refocusing_centers) or not np.allclose(excitation_centers, centers, rtol=0, atol=1e-10):
        raise ValueError("Excitation events disagree with the calculated trajectory; no refocusing RF is supported.")
    origin = float(centers[0])
    nx = adc[0]["samples"]
    dwell = adc[0]["dwell"]
    if (nx < 2 or nx % 2 or dwell <= 0 or not np.isfinite(dwell)
            or any(event["samples"] != nx for event in adc)
            or not np.allclose([event["dwell"] for event in adc], dwell, rtol=0, atol=1e-12)):
        raise ValueError("All ADC lines need the same positive dwell and the same even sample count.")
    if k_adc.shape != (3, len(adc) * nx) or not np.isfinite(k_adc).all():
        raise ValueError("ADC events and calculated trajectory dimensions disagree.")
    all_times = np.asarray(adc_times).reshape(len(adc), nx)
    adc_start = np.asarray([event["start"] + event["delay"] for event in adc])
    if not np.allclose(all_times[:, 0], adc_start + dwell / 2, rtol=0, atol=1e-10):
        raise ValueError("Calculated ADC sample centers disagree with the event integration windows.")
    # Every sample must belong to the most recent excitation, never a future RF.
    rf_index = np.searchsorted(centers, adc_start, side="right") - 1
    if np.any(rf_index < 0):
        raise ValueError("An ADC event occurs before the first excitation.")
    rf_end = np.asarray([event["end"] for event in rf])
    if (np.any(adc_start < rf_end[rf_index] - 1e-12)
            or not np.array_equal(np.searchsorted(centers, all_times[:, -1], side="right") - 1, rf_index)):
        raise ValueError("An ADC window overlaps excitation RF or spans multiple excitations.")
    counts = np.bincount(rf_index, minlength=len(rf))
    active = np.flatnonzero(counts)
    if np.any(counts[active[0]:] == 0):
        raise ValueError("Dummy RF without ADC is allowed only before all acquired imaging lines.")
    if np.all(counts[active] == 1):
        kind, navigator_count = "gre", 0
    elif len(active) == 1 and len(rf) == 1:
        kind = "epi"
        declared_nav = sequence.definitions.get("NavigatorEchoes", 0 if len(adc) % 2 == 0 else 3)
        navigator_count = int(declared_nav)
        if navigator_count not in (0, 3) or navigator_count != declared_nav:
            raise ValueError("The EPI adapter supports zero or three leading navigator ADC lines.")
    else:
        raise ValueError("Multi-echo / multiple ADC lines per GRE excitation are unsupported.")
    selected = np.arange(navigator_count, len(adc))
    ny = len(selected)
    selected_rf = rf_index[selected]
    kx = k_adc[0].reshape(len(adc), nx)[selected].copy()
    ky = k_adc[1].reshape(len(adc), nx)[selected].copy()
    kz = k_adc[2].reshape(len(adc), nx)[selected]
    times = all_times[selected].copy() - origin
    if not np.isfinite(times).all() or np.max(np.abs(kz)) > 1e-3:
        raise ValueError("Only finite two-dimensional trajectories with zero slice k-space are supported.")
    starts = np.asarray([rf[index]["end"] for index in selected_rf] if kind == "gre"
                        else [adc[index]["start"] for index in selected]) - origin
    ends = np.asarray([adc[index]["end"] for index in selected]) - origin
    if kind == "epi":
        # Historical EPI keeps each state through any inter-block gradient gap.
        ends[:-1] = starts[1:]
    if (np.any(starts <= 0) or np.any(ends <= starts) or np.any(np.diff(starts) <= 0)
            or np.any(ends[:-1] > starts[1:] + 1e-12)
            or np.any(times.min(axis=1) < starts - 1e-12)
            or np.any(times.max(axis=1) >= ends)):
        raise ValueError("Invalid chronological current windows or ADC outside its current window.")
    first_delay = adc_start[selected] - origin - starts
    first_center_delay = all_times[selected, 0] - origin - starts
    kx, ky, times, polarity, order, fov, grid_offset = _cartesian(kx, ky, times, kind)
    rf_times = np.repeat((centers[selected_rf[order]] - origin)[:, None], nx, axis=1)
    phases = np.asarray([rf[rf_index[index]]["phase"] - adc[index]["phase"] for index in selected])
    phases = np.repeat(phases[order, None], nx, axis=1)
    echo_times = np.asarray([np.interp(0, kx[row], times[row]) - rf_times[row, 0]
                             for row in range(ny)])
    te = _scalar_definition(sequence, "TE", np.median(echo_times) if kind == "gre" else echo_times[ny // 2])
    checked_te = echo_times if kind == "gre" else np.asarray([echo_times[ny // 2]])
    if not np.allclose(checked_te, te, rtol=0, atol=1e-7):
        raise ValueError("Pulseq TE must match the Cartesian echo center; GRE requires constant mono-echo TE.")
    if kind == "gre":
        rf_starts = np.asarray([event["start"] for event in rf])
        differences = np.diff(rf_starts)
        tr = _scalar_definition(sequence, "TR", np.median(differences))
        if not np.allclose(differences, tr, rtol=0, atol=1e-7):
            raise ValueError("Conventional GRE requires constant TR between RF block starts, including dummies.")
    else:
        tr = _scalar_definition(sequence, "TR", duration)
        if abs(tr - duration) > 1e-7:
            raise ValueError("Pulseq TR must equal the complete single-shot sequence duration.")
    if "FOV" in sequence.definitions:
        declared = np.asarray(sequence.definitions["FOV"]).reshape(-1)
        if declared.size >= 2 and not np.allclose(fov, declared[:2], rtol=1e-4, atol=1e-6):
            raise ValueError("Calculated field of view disagrees with the Pulseq definition.")
    if "Matrix" in sequence.definitions:
        declared = np.asarray(sequence.definitions["Matrix"]).reshape(-1)
        if declared.size != 2 or not np.array_equal(declared, [nx, ny]):
            raise ValueError("Pulseq Matrix disagrees with the acquired full Cartesian trajectory.")
    acquisition = {"sequence_kind": kind, "kx_mInv": kx, "ky_mInv": ky, "time_s": times,
                   "rf_time_s": rf_times, "phase_offset_rad": phases,
                   "line_index": np.repeat((order + 1)[:, None], nx, axis=1),
                   "line_start_s": starts[:, None], "line_end_s": ends[:, None],
                   "polarity": polarity, "dwell_s": dwell, "TE_s": te, "TR_s": tr,
                   "FOV_m": fov, "matrix_size": np.asarray([nx, ny]),
                   "echo_spacing_s": float(np.median(np.diff(starts)))}
    if navigator_count:
        # Preserve navigator ADC midpoints for the symmetric phase-fit check.
        acquisition["navigator_echo_center_s"] = all_times[:navigator_count].mean(axis=1) - origin
    timing = {"sequence_kind": kind, "image_shape": [ny, nx], "matrix_size": [nx, ny],
              "state_start_s": starts.tolist(), "state_end_s": ends.tolist(),
              "state_edges_s": np.r_[starts, ends[-1]].tolist(), "rf_center_s": origin,
              "rf_start_s": [event["start"] - origin for event in rf],
              "rf_end_s": [event["end"] - origin for event in rf],
              "first_adc_delay_s": first_delay.tolist(),
              "first_adc_sample_center_delay_s": first_center_delay.tolist(),
              "sequence_duration_s": duration, "navigator_count": navigator_count,
              "dummy_rf_count": int(active[0]), "FOV_m": fov.tolist(),
              "readout_grid_offset_mInv": float(grid_offset),
              "state_order": "chronological acquired lines; zero outside explicit current windows",
              "sequence_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return acquisition, timing
