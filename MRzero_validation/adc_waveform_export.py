"""Export current-monitor sampling times, resetting the clock at each trigger.

The CSV is a single numeric column in seconds, without a header. Its JSON
sidecar retains segment boundaries so the repeated local times can also be
placed on the actual sequence clock. This module only writes local files.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

from shim_export_timing import sequence_trigger_timing


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a positive finite real scalar.")
    array = np.asarray(value, dtype=float)
    if array.ndim or not np.isfinite(array) or array <= 0:
        raise ValueError(f"{name} must be a positive finite real scalar.")
    return float(array)


def _sample_count(duration, interval, clock_scale):
    # Differences of absolute Pulseq times can move an exact multiple a few
    # ulps above its integer ratio. Do not create an endpoint sample for that
    # subtraction roundoff; every segment is the half-open interval [0, end).
    ratio = duration / interval
    if not np.isfinite(ratio) or ratio > np.iinfo(np.intp).max:
        raise ValueError('Requested sampling interval produces too many ADC samples.')
    nearest = np.rint(ratio)
    tolerance = 8 * np.finfo(float).eps * max(interval, clock_scale) / interval
    if abs(ratio - nearest) <= tolerance:
        ratio = nearest
    return max(1, int(np.ceil(ratio)))


def export_adc_waveform(sequence_path, timing, output_dir, *, interval_s=50e-6,
                        trigger_channel='ext1', duration_s=None,
                        min_interval_s=None):
    """Write ``waveform_adc.csv`` and its timing metadata without overwriting.

    Sampling starts at the first verified imaging trigger and repeats every
    ``interval_s``. Each subsequent verified trigger starts a fresh grid at
    zero. Trigger boundaries and the final endpoint are excluded from the
    preceding segment. By default the acquisition stops at the last imaging
    command's end. ``duration_s`` overrides the total duration measured from
    the first trigger, and must finish within the actual sequence duration.

    ``timing`` is the report from ``read_acdc_sequence.read_sequence``; trigger
    events are verified against the actual ``sequence_path``. Additional
    exports use ``waveform_adc2.csv``, ``waveform_adc3.csv``, etc. The matching
    JSON records the file hash, timing origin and each segment's row indices.
    An existing identical CSV/metadata pair is reused. ``min_interval_s`` is
    optional and only applies when an actual sampling limit is known.
    No currents, optimizer or MRzero model are required.
    """
    interval = _positive(interval_s, 'interval_s')
    minimum = None if min_interval_s is None else _positive(min_interval_s, 'min_interval_s')
    if minimum is not None and interval < minimum:
        raise ValueError(f"interval_s must be at least min_interval_s ({minimum:g} s).")
    trigger = sequence_trigger_timing(sequence_path, timing, channel=trigger_channel)
    origin_rf = float(trigger['trigger_rf_s'])
    origin_sequence = float(trigger['trigger_sequence_s'])
    sequence_duration = _positive(trigger['sequence_duration_s'], 'sequence_duration_s')
    if 'sequence_duration_s' in timing:
        timing_duration = _positive(timing['sequence_duration_s'], 'timing.sequence_duration_s')
        if abs(sequence_duration - timing_duration) > 1e-9:
            raise ValueError('Timing sequence_duration_s disagrees with the actual sequence duration.')
    rf_center = float(timing['rf_center_s'])
    sequence_end_rf = sequence_duration - rf_center
    if np.iscomplexobj(timing['state_end_s']):
        raise ValueError('state_end_s must contain finite ordered real imaging ends.')
    ends = np.asarray(timing['state_end_s'], dtype=float).reshape(-1)
    starts = np.asarray(timing['state_start_s'], dtype=float).reshape(-1)
    if (ends.shape != starts.shape or not ends.size or not np.isfinite(ends).all()
            or np.any(ends <= starts) or np.any(np.diff(ends) <= 0)
            or ends[-1] > sequence_end_rf + 1e-12):
        raise ValueError('state_end_s must contain finite ordered imaging ends within the sequence.')
    if duration_s is None:
        stop_rf = float(ends[-1])
        duration_source = 'last_imaging_command_end'
    else:
        requested_duration = _positive(duration_s, 'duration_s')
        stop_rf = origin_rf + requested_duration
        duration_source = 'explicit_duration_from_first_trigger'
    if stop_rf > sequence_end_rf + 1e-12:
        raise ValueError('Requested ADC acquisition extends beyond the sequence end.')
    if stop_rf <= origin_rf:
        raise ValueError('ADC acquisition must finish after the first imaging trigger.')
    if stop_rf > sequence_end_rf:
        stop_rf = sequence_end_rf
    duration = stop_rf - origin_rf
    all_triggers = np.asarray(trigger['trigger_times_rf_s'], dtype=float)
    # A trigger at the requested stop is an endpoint, not a new segment.
    active = all_triggers[all_triggers < stop_rf - 1e-12]
    if not active.size:
        active = all_triggers[:1]
    boundaries = np.r_[active, stop_rf]
    rows, segments, offset = [], [], 0
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segment_duration = float(stop - start)
        count = _sample_count(segment_duration, interval, max(abs(start), abs(stop)))
        local = np.arange(count, dtype=float) * interval
        rows.append(local)
        segments.append(dict(trigger_index=index, csv_row_start=offset,
            csv_row_stop=offset + count, sample_count=count,
            start_time_rf_s=float(start),
            start_time_sequence_s=float(start + rf_center),
            start_time_from_first_trigger_s=float(start - origin_rf),
            duration_s=segment_duration, final_sample_local_s=float(local[-1])))
        offset += count
    samples = np.concatenate(rows)
    stream = io.StringIO()
    np.savetxt(stream, samples[:, None], fmt='%.17g')
    csv_bytes = stream.getvalue().encode('ascii')
    report = dict(csv_sha256=hashlib.sha256(csv_bytes).hexdigest(), csv_columns=1,
        csv_header=False, column_units='seconds', time_origin='latest_imaging_trigger',
        interval_s=interval, min_interval_s=minimum, samples=int(samples.size),
        trigger_count=int(active.size), sequence_trigger_count=int(all_triggers.size),
        segment_sample_counts=[item['sample_count'] for item in segments],
        duration_s=float(duration), duration_source=duration_source,
        first_trigger_rf_s=origin_rf, first_trigger_sequence_s=origin_sequence,
        stop_rf_s=float(stop_rf), stop_sequence_s=float(stop_rf + rf_center),
        sequence_duration_s=sequence_duration, segments=segments,
        trigger=trigger, endpoint_included=False,
        no_hardware_action_performed=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r'waveform_adc([0-9]*)\.(?:csv|json)')
    numbers = [int(match.group(1) or 1) for entry in output.iterdir()
               if (match := pattern.fullmatch(entry.name))]
    for existing in sorted(set(numbers)):
        stem = 'waveform_adc' + ('' if existing == 1 else str(existing))
        existing_csv, existing_metadata = output / f'{stem}.csv', output / f'{stem}.json'
        try:
            saved = json.loads(existing_metadata.read_text(encoding='utf-8'))
            same_metadata = isinstance(saved, dict) and {key: saved.get(key) for key in report} == report
            if same_metadata and existing_csv.read_bytes() == csv_bytes:
                return dict(report, csv_file=str(existing_csv),
                            metadata_file=str(existing_metadata), already_present=True)
        except (OSError, ValueError, TypeError):
            pass
    number = max(numbers, default=0) + 1
    while True:
        stem = 'waveform_adc' + ('' if number == 1 else str(number))
        csv_file, metadata_file = output / f'{stem}.csv', output / f'{stem}.json'
        try:
            csv_stream = csv_file.open('xb')
        except FileExistsError:
            number += 1
            continue
        try:
            metadata_stream = metadata_file.open('x', encoding='utf-8')
        except FileExistsError:
            csv_stream.close()
            csv_file.unlink()
            number += 1
            continue
        except BaseException:
            csv_stream.close()
            csv_file.unlink(missing_ok=True)
            raise
        break
    report.update(csv_file=str(csv_file), metadata_file=str(metadata_file), already_present=False)
    try:
        with csv_stream:
            csv_stream.write(csv_bytes)
        with metadata_stream:
            json.dump(report, metadata_stream, indent=2, allow_nan=False)
            metadata_stream.write('\n')
    except BaseException:
        csv_stream.close()
        metadata_stream.close()
        csv_file.unlink(missing_ok=True)
        metadata_file.unlink(missing_ok=True)
        raise
    return report
