"""Read a single-shot spiral and its actual per-turn hardware output pulses.

The sampling arrays are chronological ``(1, sample)`` arrays. Their shape is
independent of ``image_shape``. ``line_index`` is the historical model key for
the one-based current-state index; here each state is a spiral turn, not a
Cartesian line. Times are relative to the excitation centre.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from read_acdc_sequence import _events, _scalar_definition


def _positive_integer(sequence, name):
    value = _scalar_definition(sequence, name, np.nan)
    if value != int(value):
        raise ValueError(f"Pulseq {name} must be a positive integer.")
    return int(value)


def read_spiral_sequence(path: Path) -> tuple[dict, dict]:
    """Return a verified 2D spiral-out acquisition and per-turn timing report.

    Supported acquisitions have one excitation, zero or three leading
    navigators, one continuous imaging ADC, and one output pulse per turn.
    The pulses, rather than equal sample groups or Twix segment boundaries,
    define the shim states. ADC samples and k coordinates are never reordered.
    """
    import pypulseq as pp

    path = Path(path)
    sequence = pp.Sequence()
    sequence.read(str(path))
    if sequence.definitions.get("TrajectoryType") != "spiral_out":
        raise ValueError("The spiral adapter requires TrajectoryType spiral_out.")
    rf, adc, duration = _events(sequence, pp)
    navigator_count = int(sequence.definitions.get("NavigatorEchoes", 0))
    if navigator_count not in (0, 3) or len(rf) != 1 or len(adc) != navigator_count + 1:
        raise ValueError("Spiral input requires one excitation, one imaging ADC and zero or three leading navigators.")
    imaging = adc[-1]
    dwell, samples = imaging["dwell"], imaging["samples"]
    if samples < 2 or not np.isfinite(dwell) or dwell <= 0:
        raise ValueError("Spiral imaging ADC needs at least two samples and positive finite dwell.")
    if navigator_count and (len({event["samples"] for event in adc[:-1]}) != 1
                            or len({event["dwell"] for event in adc[:-1]}) != 1):
        raise ValueError("Spiral navigator lengths and dwell times must match each other.")
    for name, actual, tolerance in (("SpiralADCSamples", samples, 0),
                                     ("SpiralADCDwell", dwell, 1e-12)):
        declared = _scalar_definition(sequence, name, actual)
        if not np.isclose(declared, actual, rtol=0, atol=tolerance):
            raise ValueError(f"Pulseq {name} disagrees with the imaging ADC event.")
    matrix = np.asarray(sequence.definitions.get("ReconstructionMatrix",
                        sequence.definitions.get("Matrix", [])), float).reshape(-1)
    fov = np.asarray(sequence.definitions.get("FOV", []), float).reshape(-1)[:2]
    if (matrix.size != 2 or not np.isfinite(matrix).all() or np.any(matrix < 2)
            or np.any(matrix != np.round(matrix)) or fov.size != 2
            or not np.isfinite(fov).all() or np.any(fov <= 0)):
        raise ValueError("Spiral reconstruction requires a positive 2D Matrix and FOV definition.")
    matrix = matrix.astype(int)
    if "Matrix" in sequence.definitions and not np.array_equal(sequence.definitions["Matrix"], matrix):
        raise ValueError("Matrix and ReconstructionMatrix disagree.")
    if _positive_integer(sequence, "Shots") != 1 or _positive_integer(sequence, "SpiralInterleaves") != 1:
        raise ValueError("This adapter supports one spiral shot/interleaf only.")

    kspace, _, excitation, refocusing, adc_times = sequence.calculate_kspace()
    total_samples = sum(event["samples"] for event in adc)
    if (np.asarray(kspace).shape != (3, total_samples)
            or not np.isfinite(kspace).all() or len(refocusing)
            or not np.allclose(excitation, [rf[0]["center"]], rtol=0, atol=1e-10)):
        raise ValueError("Spiral ADC trajectory and excitation events disagree.")
    adc_times = np.asarray(adc_times, float)
    if adc_times.shape != (total_samples,) or not np.isfinite(adc_times).all():
        raise ValueError("Spiral ADC sample times are invalid.")
    offset = 0
    for event in adc:
        selected = adc_times[offset:offset + event["samples"]]
        expected = event["start"] + event["delay"] + (np.arange(event["samples"]) + .5) * event["dwell"]
        if not np.allclose(selected, expected, rtol=0, atol=1e-10):
            raise ValueError("Calculated ADC times disagree with the integration-window centres.")
        offset += event["samples"]
    if adc_times[0] < rf[0]["end"] or np.max(np.abs(kspace[2])) > 1e-3:
        raise ValueError("Spiral ADC must follow RF and have a finite two-dimensional trajectory.")

    channel = str(sequence.definitions.get("ImagingTurnOutputChannel", "ext1"))
    clock, triggers, imaging_block, adc_index = 0., [], None, 0
    for number in sequence.block_events:
        block = sequence.get_block(number)
        if block.adc is not None:
            labels = {event.label: event.value for event in (block.label or {}).values()}
            expected_nav = adc_index < navigator_count
            if "NAV" in labels and bool(labels["NAV"]) != expected_nav:
                raise ValueError("Pulseq NAV labels disagree with the leading navigator selection.")
            if np.asarray(getattr(block.adc, "phase_modulation", [])).size:
                raise ValueError("Sample-dependent ADC phase modulation needs an explicit receiver model.")
            if not expected_nav:
                imaging_block = number
            adc_index += 1
        for event in (getattr(block, "trigger", None) or {}).values():
            if event.type == "output" and event.channel == channel:
                if number != imaging_block:
                    raise ValueError("Turn output pulses must occur only in the spiral imaging ADC block.")
                triggers.append((clock + float(event.delay), float(event.duration)))
        clock += float(sequence.block_durations[number])
    triggers.sort()
    turns = _positive_integer(sequence, "SpiralTurns")
    declared_outputs = _positive_integer(sequence, "ImagingTurnOutputCount")
    if len(triggers) != turns or declared_outputs != turns:
        raise ValueError("There must be exactly one actual imaging output pulse per declared spiral turn.")
    trigger_times = np.asarray([event[0] for event in triggers])
    trigger_duration = np.asarray([event[1] for event in triggers])
    declared_output_duration = _scalar_definition(sequence, "ImagingTurnOutputDuration", trigger_duration[0])
    if (not np.isfinite(trigger_times).all() or np.any(np.diff(trigger_times) <= 0)
            or np.any(trigger_duration <= 0) or np.any(trigger_times + trigger_duration > imaging["end"] + 1e-12)
            or np.any(trigger_times[:-1] + trigger_duration[:-1] > trigger_times[1:])
            or not np.allclose(trigger_duration, declared_output_duration, rtol=0, atol=1e-12)):
        raise ValueError("Spiral output pulses must be positive, chronological and non-overlapping.")

    times = adc_times[-samples:]
    trajectory = np.asarray(kspace[:, -samples:])
    state_index = np.searchsorted(trigger_times, times, side="right")
    if (np.any(state_index < 1) or np.any(state_index > turns)
            or not np.array_equal(np.unique(state_index), np.arange(1, turns + 1))
            or trigger_times[0] < rf[0]["end"] or times[-1] >= imaging["end"]):
        raise ValueError("Each imaging sample must belong to one populated, post-RF turn window.")
    # The first samples are close to k=0: their angle is sensitive to text
    # rounding. Check complete rotations between subsequent pulse locations.
    angles = np.unwrap(np.arctan2(trajectory[1], trajectory[0]))
    pulse_angles = np.interp(trigger_times[1:], times, angles)
    angular_step = np.diff(pulse_angles)
    if angular_step.size and not np.allclose(angular_step, 2 * np.pi, rtol=0, atol=.08):
        raise ValueError("Measured output pulses do not delimit consecutive full spiral turns.")
    radius = np.hypot(trajectory[0], trajectory[1])
    if not np.isclose(radius[-1], np.min(matrix / (2 * fov)), rtol=.02, atol=.01):
        raise ValueError("Spiral outer radius disagrees with the declared reconstruction matrix/FOV.")
    origin = float(rf[0]["center"])
    starts = trigger_times - origin
    ends = np.r_[starts[1:], imaging["end"] - origin]
    te = _scalar_definition(sequence, "TE", times[0] - origin)
    tr = _scalar_definition(sequence, "TR", duration)
    if not np.isclose(te, times[0] - origin, rtol=0, atol=1e-7):
        raise ValueError("Spiral-out TE must identify the first imaging ADC sample centre.")
    if not np.isclose(tr, duration, rtol=0, atol=1e-7):
        raise ValueError("Spiral TR must equal the complete single-shot sequence duration.")
    shape = (1, samples)
    acquisition = dict(sequence_kind="spiral", trajectory_type="spiral_out",
        kx_mInv=trajectory[0].reshape(shape), ky_mInv=trajectory[1].reshape(shape),
        time_s=(times - origin).reshape(shape), rf_time_s=np.zeros(shape),
        phase_offset_rad=np.full(shape, rf[0]["phase"] - imaging["phase"]),
        line_index=state_index.reshape(shape), line_start_s=starts[:, None], line_end_s=ends[:, None],
        dwell_s=dwell, TE_s=te, TR_s=tr, FOV_m=fov,
        flip_deg=_scalar_definition(sequence, "FlipAngle", np.nan),
        matrix_size=matrix, image_shape=matrix[::-1].tolist(),
        navigator_count=navigator_count, navigator_sample_count=[event["samples"] for event in adc[:-1]],
        navigator_dwell_s=[event["dwell"] for event in adc[:-1]], turn_count=turns,
        uniform_radial_density=bool(sequence.definitions.get("UniformRadialDensity", 0)),
        max_adc_segment_length=_positive_integer(sequence, "MaxAdcSegmentLength"),
        echo_spacing_s=float(np.median(np.diff(starts))))
    first_indices = np.searchsorted(times - origin, starts)
    first_indices = np.minimum(first_indices, samples - 1)
    timing = dict(sequence_kind="spiral", trajectory_type="spiral_out",
        image_shape=matrix[::-1].tolist(), sampling_shape=list(shape), matrix_size=matrix.tolist(),
        sample_count=samples, turn_count=turns, state_start_s=starts.tolist(), state_end_s=ends.tolist(),
        state_edges_s=np.r_[starts, ends[-1]].tolist(), rf_center_s=origin,
        rf_start_s=[rf[0]["start"] - origin], rf_end_s=[rf[0]["end"] - origin],
        first_adc_delay_s=((times - dwell / 2 - origin)[first_indices] - starts).tolist(),
        first_adc_sample_center_delay_s=((times - origin)[first_indices] - starts).tolist(),
        sequence_duration_s=duration, navigator_count=navigator_count, dummy_rf_count=0,
        navigator_samples=[event["samples"] for event in adc[:-1]],
        navigator_dwell_s=[event["dwell"] for event in adc[:-1]],
        output_channel=channel, output_count=turns, output_duration_s=trigger_duration.tolist(),
        output_times_s=starts.tolist(), output_pulses_verified=True,
        angular_step_between_pulses_rad=angular_step.tolist(), FOV_m=fov.tolist(),
        state_order="Chronological spiral turns delimited by actual Pulseq output pulses; zero outside current windows.",
        trajectory_source="Pulseq calculate_kspace at actual imaging ADC centres; no Cartesian sorting.",
        sequence_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return acquisition, timing


__all__ = ["read_spiral_sequence"]
