"""Verified sequence-trigger origins for contiguous EPI command CSVs.

This module only prepares timing data. It never loads a waveform, configures
a controller, sends a trigger or changes currents on hardware.
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    result = np.asarray(value, dtype=float)
    if result.ndim != 0 or not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar.")
    return float(result)


def sequence_trigger_timing(sequence_path, timing, channel='ext1'):
    """Verify a first-imaging trigger or one trigger per imaging command.

    ``timing`` is the report returned by ``read_acdc_sequence.read_sequence``.
    Trigger time is measured from actual Pulseq block durations and event
    delays, then expressed relative to that report's first RF center. A file
    hash, when available in timing, must agree. The first imaging trigger is
    never inferred just from the current-window start or the sequence name.
    A train of triggers is accepted when every imaging-command start has
    exactly one matching event. The first event defines the CSV clock origin;
    all events and their state indices remain recorded in the report. This
    validates sequence timing, without assuming how a controller handles the
    subsequent triggers.
    """
    import pypulseq as pp

    if timing.get('sequence_kind', 'epi') != 'epi':
        raise ValueError("First-imaging-trigger export currently supports contiguous EPI only.")
    if not isinstance(channel, str) or not channel:
        raise ValueError("channel must name an output-trigger channel.")
    rf_center = _finite_scalar(timing['rf_center_s'], 'rf_center_s')
    if np.iscomplexobj(timing['state_start_s']):
        raise ValueError("timing must include finite, strictly increasing real imaging command starts.")
    starts = np.asarray(timing['state_start_s'], dtype=float).reshape(-1)
    if (not starts.size or not np.isfinite(starts).all()
            or np.any(np.diff(starts) <= 0)):
        raise ValueError("timing must include finite, strictly increasing real imaging command starts.")
    path = Path(sequence_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if timing.get('sequence_sha256') not in (None, digest):
        raise ValueError("Sequence content has changed since acquisition timing was validated.")
    sequence = pp.Sequence()
    sequence.read(str(path))
    events, elapsed = [], 0.
    for number, duration in sequence.block_durations.items():
        block = sequence.get_block(number)
        for event in (getattr(block, 'trigger', None) or {}).values():
            if event.type == 'output':
                absolute = elapsed + float(event.delay)
                events.append(dict(block=int(number), channel=str(event.channel),
                                   time_sequence_s=absolute, time_rf_s=absolute - rf_center,
                                   duration_s=float(event.duration)))
        elapsed += float(duration)
    selected = sorted((event for event in events if event['channel'] == channel),
                      key=lambda event: event['time_sequence_s'])
    if not selected:
        raise ValueError(f"No {channel} output trigger found; require a first-imaging trigger or one per imaging command.")
    event = selected[0]
    tolerance = 1e-9
    if abs(event['time_rf_s'] - starts[0]) > tolerance:
        raise ValueError("The selected trigger is not at the first imaging-command start.")
    trigger_times = np.asarray([item['time_rf_s'] for item in selected])
    if len(selected) == 1:
        mode, indices = 'single_trigger', [0]
        alignment_error = abs(float(trigger_times[0] - starts[0]))
    else:
        if len(selected) != starts.size:
            raise ValueError(f"Need one first-imaging {channel} trigger or exactly one per imaging command "
                             f"({starts.size}), found {len(selected)}.")
        if (np.any(np.diff(trigger_times) <= 0)
                or not np.allclose(trigger_times, starts, rtol=0, atol=tolerance)):
            raise ValueError(f"The {channel} trigger train must match every imaging-command start "
                             "in order, without duplicates or additional navigator triggers.")
        mode, indices = 'per_line', list(range(int(starts.size)))
        alignment_error = float(np.max(np.abs(trigger_times - starts)))
    return dict(trigger_rf_s=event['time_rf_s'], trigger_sequence_s=event['time_sequence_s'],
                channel=channel, verified_first_imaging_trigger=True,
                mode=mode, trigger_count=len(selected),
                verified_per_line_triggers=mode == 'per_line',
                trigger_times_rf_s=trigger_times.tolist(),
                trigger_times_sequence_s=[item['time_sequence_s'] for item in selected],
                trigger_state_indices=indices,
                maximum_trigger_alignment_error_s=alignment_error,
                trigger_alignment_tolerance_s=tolerance,
                first_imaging_start_rf_s=float(starts[0]), output_events=events,
                sequence_duration_s=float(elapsed),
                sequence_file=str(path.resolve()), sequence_sha256=digest,
                method='Actual Pulseq block clock plus output-event delay; first imaging start and optional per-line train verified',
                hardware_latency_measured=False, no_hardware_action_performed=True)


def waveform_edges_for_origin(model_or_starts, ends=None, *, origin='first_imaging_trigger',
                              trigger_rf_s=None, min_update_interval_s=8e-6):
    """Return (edges, report) for held EPI commands followed by a final zero.

    Pass either a model with line_start_s/line_end_s or the two arrays in
    RF-relative seconds. For trigger-relative export, trigger_rf_s must be
    explicitly measured or obtained from sequence_trigger_timing. It may be
    signed: the origin can be before or after RF. No trigger is assumed from
    a filename or silently replaced with the first current start.

    Commands are contiguous EPI states. Gaps require explicit zero states and
    are rejected here. Use export_waveform(..., zero_time_column=False) with
    the returned edges, preserving its initial zero preamble when needed and
    its final zero row. Zeroing the timestamps would discard this timing.
    """
    if hasattr(model_or_starts, 'line_start_s'):
        if ends is not None:
            raise ValueError("Do not supply ends when passing a model.")
        starts, ends = model_or_starts.line_start_s, model_or_starts.line_end_s
    else:
        starts = model_or_starts
        if ends is None:
            raise ValueError("Provide both starts and ends, or a model containing them.")
    if np.iscomplexobj(starts) or np.iscomplexobj(ends):
        raise ValueError("Current windows must be real.")
    starts, ends = np.asarray(starts, float).reshape(-1), np.asarray(ends, float).reshape(-1)
    if (not starts.size or starts.shape != ends.shape
            or not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or np.any(starts < 0) or np.any(ends <= starts)
            or np.any(np.diff(starts) <= 0)
            or not np.allclose(ends[:-1], starts[1:], rtol=0, atol=1e-12)):
        raise ValueError("Need finite ordered, contiguous EPI command windows.")
    minimum = _finite_scalar(min_update_interval_s, 'min_update_interval_s')
    if minimum <= 0:
        raise ValueError("min_update_interval_s must be positive.")
    if origin == 'rf':
        origin_rf_s = 0.
    elif origin == 'first_imaging_trigger':
        if trigger_rf_s is None:
            raise ValueError("Explicit trigger_rf_s is required; verify the actual sequence trigger.")
        origin_rf_s = _finite_scalar(trigger_rf_s, 'trigger_rf_s')
    else:
        raise ValueError("origin must be 'rf' or 'first_imaging_trigger'.")
    edges = np.r_[starts, ends[-1]] - origin_rf_s
    # Only remove floating-point cancellation at a verified coincident origin.
    edges[np.abs(edges) < 1e-12] = 0.
    if np.any(edges < 0):
        raise ValueError("A command would start before the selected CSV origin.")
    has_preamble = bool(edges[0] > 0)
    csv_times = np.r_[0., edges] if has_preamble else edges
    intervals = np.diff(csv_times)
    if np.any(intervals + 1e-12 < minimum):
        raise ValueError("A CSV row interval is shorter than the configured DAC update minimum.")
    report = dict(origin=origin, origin_rf_s=origin_rf_s,
                  first_command_time_s=float(edges[0]), final_zero_time_s=float(edges[-1]),
                  command_state_count=int(starts.size),
                  csv_rows_with_final_zero_and_preamble=int(csv_times.size),
                  includes_zero_preamble=has_preamble, final_zero_required=True,
                  minimum_row_interval_s=float(intervals.min()),
                  minimum_allowed_interval_s=minimum,
                  zero_time_column_must_be_false=True,
                  timing_operation='CSV time = model RF-relative time minus selected origin',
                  no_hardware_action_performed=True)
    return edges, report


__all__ = ['sequence_trigger_timing', 'waveform_edges_for_origin']
