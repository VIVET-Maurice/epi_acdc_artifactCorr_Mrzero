"""Export independently fitted static/dynamic field shims for one acquisition.

Fine-raster dynamic commands are checked with event recurrence, without a
quadratic time-by-time response matrix. The continuous first-order current,
switch-off tail and optional electrical limits follow acdc_hardware.py.
No function communicates with a waveform generator or scanner.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from export_acdc_waveform import export_waveform, quantize_currents
from shim_export_timing import sequence_trigger_timing, waveform_edges_for_origin


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _channel(value, channels, name, *, allow_zero=False):
    raw = np.asarray(_numpy(value))
    if np.iscomplexobj(raw) or raw.dtype.kind == 'b':
        raise ValueError(f'{name} must contain real values.')
    array = np.asarray(raw, float)
    array = np.full(channels, array.item()) if array.size == 1 else array.reshape(-1)
    if (array.shape != (channels,) or not np.isfinite(array).all()
            or np.any(array < 0 if allow_zero else array <= 0)):
        qualifier = 'nonnegative' if allow_zero else 'positive'
        raise ValueError(f'{name} needs {qualifier} finite channel values.')
    return array


def _field_peaks(model, command, first, last, common_tau, chunk=64):
    """Bound every calibrated quality/corner field using small GPU blocks."""
    basis = model.B_quality
    if not isinstance(basis, torch.Tensor):
        basis = torch.as_tensor(basis, device=model.device, dtype=model.dtype)
    if basis.ndim != 2 or basis.shape[1] != model.nc or not bool(torch.isfinite(basis).all()):
        raise ValueError('Finite calibrated quality field maps are required.')
    peak_command = peak_actual = 0.
    positive = negative = None
    if not common_tau:
        positive, negative = basis.clamp_min(0), basis.clamp_max(0)
    with torch.no_grad():
        for start in range(0, command.shape[1], chunk):
            sl = slice(start, start+chunk)
            c, a, b = [torch.as_tensor(v[:, sl], device=basis.device, dtype=basis.dtype)
                       for v in (command, first, last)]
            peak_command = max(peak_command, float((basis @ c).abs().max()))
            if common_tau:
                peak_actual = max(peak_actual, float((basis @ a).abs().max()),
                                  float((basis @ b).abs().max()))
            else:
                low, high = torch.minimum(a, b), torch.maximum(a, b)
                peak_actual = max(peak_actual,
                    float((positive @ low + negative @ high).abs().max()),
                    float((positive @ high + negative @ low).abs().max()))
    return peak_command, peak_actual


def waveform_hardware_diagnostics(model, currents_A, state_edges_rf_s, *,
                                  total_current_limit_A, field_limit_Hz):
    """Check arbitrary contiguous commands with continuous-current metrics.

    The model's quality maps include the finite-voxel corners when enabled.
    An explicit zero-command interval covers the preamble and final TR tail.
    The final switch-off derivative is checked even at an exact TR boundary.
    """
    if np.iscomplexobj(currents_A) or np.iscomplexobj(state_edges_rf_s):
        raise ValueError('Commands and event times must be real.')
    currents, edges = np.asarray(currents_A, float), np.asarray(state_edges_rf_s, float)
    if (currents.ndim != 2 or currents.shape[0] != model.nc or currents.shape[1] < 1
            or edges.shape != (currents.shape[1]+1,) or not np.isfinite(currents).all()
            or not np.isfinite(edges).all() or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError('Expected finite channel-by-state currents and increasing RF-relative edges.')
    if (not np.isfinite(total_current_limit_A) or total_current_limit_A <= 0
            or np.isnan(field_limit_Hz) or field_limit_Hz <= 0):
        raise ValueError('Positive total-current and field limits are required.')
    config = dict(model.hardware.config)
    minimum = float(config.get('min_update_interval_s', 8e-6))
    if not np.isfinite(minimum) or minimum <= 0 or np.min(np.diff(edges))+1e-12 < minimum:
        raise ValueError('Waveform update intervals violate the configured DAC minimum.')
    if not model.current_response['enabled']:
        raise ValueError('An explicit first-order current response is required.')
    tau = _channel(model.current_response['tau_s'], model.nc, 'tau_s')[:, None]
    limits = _channel(model.max_current_A, model.nc, 'max_current_A', allow_zero=True)
    cycle = float(config.get('cycle_duration_s') or (edges[-1]+12*float(tau.max())))
    if not np.isfinite(cycle) or cycle < edges[-1]:
        raise ValueError('The configured TR must include the complete waveform.')
    durations = np.diff(edges)
    command = currents
    if edges[0] > 0:
        durations = np.r_[edges[0], durations]
        command = np.column_stack((np.zeros(model.nc), command))
    durations = np.r_[durations, cycle-edges[-1]]
    command = np.column_stack((command, np.zeros(model.nc)))
    first, last = np.empty_like(command), np.empty_like(command)
    value = np.zeros(model.nc)
    for index, duration in enumerate(durations):
        first[:, index] = value
        value = command[:, index] + (value-command[:, index])*np.exp(-duration/tau[:, 0])
        last[:, index] = value
    dt = durations[None]
    decay = np.exp(-dt/tau)
    difference = first-command
    slope_first = -difference/tau
    slope_last = slope_first*decay
    peak_channel = np.maximum(abs(first), abs(last)).max(axis=1)
    common = bool(np.all(tau == tau[0]))
    total_bound = (np.maximum(abs(first).sum(0), abs(last).sum(0)).max() if common
                   else np.maximum(abs(first), abs(last)).sum(0).max())
    command_field, actual_field = _field_peaks(model, command, first, last, common)
    decay_integral = -tau*np.expm1(-dt/tau)
    signed = command*dt+difference*decay_integral
    square = (command**2*dt+2*command*difference*decay_integral
              -difference**2*tau/2*np.expm1(-2*dt/tau))
    safe_command = np.where(command == 0, 1., command)
    ratio = np.maximum(-difference/safe_command, np.finfo(float).tiny)
    crossing = (first*last < 0) & (command != 0)
    crossing_time = np.where(crossing, tau*np.log(ratio), 0.)
    crossing_time = np.minimum(np.maximum(crossing_time, 0), dt)
    signed_before = command*crossing_time-difference*tau*np.expm1(-crossing_time/tau)
    absolute = np.where(crossing, abs(signed_before)+abs(signed-signed_before), abs(signed))
    metrics = dict(peak_actual_channel_current_A=peak_channel,
        peak_slew_A_per_s=abs(slope_first).max(1), peak_total_current_bound_A=float(total_bound),
        peak_command_field_Hz=command_field, peak_field_bound_Hz=actual_field,
        avg_abs_current_A=absolute.sum(1)/cycle,
        mean_square_current_A2=np.maximum(square, 0).sum(1)/cycle,
        end_current_A=last[:, -1], cycle_duration_s=cycle,
        minimum_command_interval_s=float(np.diff(edges).min()),
        field_constraint_domain='all calibrated quality points and supplied finite-voxel corners',
        response_method='exact first-order event recurrence, zero initial current and final switch-off tail',
        response_time_matrix_allocated=False)
    checks = dict(command_channel=bool(np.all(abs(currents) <= limits[:, None]+1e-12)),
        command_total=bool(np.all(abs(currents).sum(0) < total_current_limit_A)),
        command_field=command_field <= field_limit_Hz+1e-8,
        actual_channel=bool(np.all(peak_channel <= limits+1e-12)),
        actual_total_bound=float(total_bound) < total_current_limit_A,
        actual_field_bound=actual_field <= field_limit_Hz+1e-8,
        update_intervals_met=True)
    if np.any(limits == 0):
        checks['disabled_channels_zero'] = bool(np.all(currents[limits == 0] == 0)
                                                and np.all(peak_channel[limits == 0] == 0))
    resistance = None if config.get('coil_R_ohm') is None else _channel(config['coil_R_ohm'], model.nc, 'coil_R_ohm')
    inductance = None if config.get('coil_L_H') is None else _channel(config['coil_L_H'], model.nc, 'coil_L_H', allow_zero=True)
    if config.get('max_voltage_V') is not None and (resistance is None or inductance is None):
        raise ValueError('Voltage limits require the configured coil resistance and inductance.')
    if config.get('max_coil_power_W') is not None and resistance is None:
        raise ValueError('Power limits require the configured coil resistance.')
    if resistance is not None:
        metrics['average_coil_power_W'] = resistance*metrics['mean_square_current_A2']
    if resistance is not None and inductance is not None:
        v_first = resistance[:, None]*first+inductance[:, None]*slope_first
        v_last = resistance[:, None]*last+inductance[:, None]*slope_last
        metrics['peak_coil_voltage_V'] = np.maximum(abs(v_first), abs(v_last)).max(1)
    for key, metric in (('max_slew_A_per_s', 'peak_slew_A_per_s'),
                        ('avg_abs_current_A', 'avg_abs_current_A'),
                        ('max_voltage_V', 'peak_coil_voltage_V'),
                        ('max_coil_power_W', 'average_coil_power_W')):
        if config.get(key) is not None:
            if key in ('avg_abs_current_A', 'max_coil_power_W') and config.get('cycle_duration_s') is None:
                raise ValueError('Average/power limits require the complete configured TR.')
            limit = _channel(config[key], model.nc, key)
            checks[key] = bool(np.all(np.asarray(metrics[metric]) <= limit+1e-12))
    metrics.update(checks=checks, all_limits_met=all(checks.values()))
    return metrics


def _contraction(model, commands, report, total, field):
    tiny = 1e-30
    limit = _channel(model.max_current_A, model.nc, 'max_current_A', allow_zero=True)
    active = limit > 0
    ratios = [1.,
        (total-min(1e-4, total*1e-3))/max(float(abs(commands).sum(0).max()), tiny),
        (total-min(1e-4, total*1e-3))/max(report['peak_total_current_bound_A'], tiny)]
    if active.any():
        ratios += [float(np.min(limit[active]/np.maximum(abs(commands[active]).max(1), tiny))),
                   float(np.min(limit[active]/np.maximum(
                       np.asarray(report['peak_actual_channel_current_A'])[active], tiny)))]
    if np.isfinite(field):
        budget = field-min(.05, field*1e-3)
        ratios += [budget/max(report['peak_command_field_Hz'], tiny),
                   budget/max(report['peak_field_bound_Hz'], tiny)]
    config = model.hardware.config
    for key, metric in (('max_slew_A_per_s', 'peak_slew_A_per_s'),
                        ('avg_abs_current_A', 'avg_abs_current_A'),
                        ('max_voltage_V', 'peak_coil_voltage_V'),
                        ('max_coil_power_W', 'average_coil_power_W')):
        if config.get(key) is not None:
            limit = _channel(config[key], model.nc, key)
            ratio = float(np.min(limit*(1-1e-5)/np.maximum(report[metric], tiny)))
            ratios.append(np.sqrt(ratio) if key == 'max_coil_power_W' else ratio)
    return max(0., min(ratios))


def prepare_field_waveform(model, currents_A, edges_rf_s, *, total_current_limit_A, field_limit_Hz):
    """Uniformly contract if needed, then independently check nominal DAC values."""
    if np.iscomplexobj(currents_A):
        raise ValueError('Waveform commands must be real.')
    original = np.asarray(currents_A, float)
    report = waveform_hardware_diagnostics(model, original, edges_rf_s,
        total_current_limit_A=total_current_limit_A, field_limit_Hz=field_limit_Hz)
    if not report['checks'].get('disabled_channels_zero', True):
        raise ValueError('Disabled channels must have exactly zero commands; refit with those channels excluded.')
    scale = _contraction(model, original, report, total_current_limit_A, field_limit_Hz)
    commands = original*scale
    for attempt in range(8):
        projected = waveform_hardware_diagnostics(model, commands, edges_rf_s,
            total_current_limit_A=total_current_limit_A, field_limit_Hz=field_limit_Hz)
        quantized = quantize_currents(commands)
        applied = waveform_hardware_diagnostics(model, quantized, edges_rf_s,
            total_current_limit_A=total_current_limit_A, field_limit_Hz=field_limit_Hz)
        if projected['all_limits_met'] and applied['all_limits_met']:
            return commands, quantized, dict(uniform_contraction=scale,
                raw=report, commands=projected, nominal_DAC=applied,
                quantization_rechecks=attempt, solver_optimum_preserved=bool(scale == 1.))
        extra = min(_contraction(model, commands, projected, total_current_limit_A, field_limit_Hz),
                    _contraction(model, quantized, applied, total_current_limit_A, field_limit_Hz))*.999
        commands *= extra
        scale *= extra
    raise ValueError('The field waveform could not satisfy configured limits before and after DAC conversion.')


def export_field_shim_waveforms(model, output_dir, settings, *, total_current_limit_A,
                               field_limit_Hz, timing, sequence_path,
                               csv_time_origin='first_imaging_trigger', trigger_channel='ext1',
                               raster_s=10e-6, max_iterations=80, static_ridge=1e-6,
                               dynamic_ridge=1e-3, provenance=None):
    """Fit upstream QP/FISTA field shims, check and export both named waveforms."""
    from field_shim_waveforms import solve_field_shim_waveforms
    minimum = float(model.hardware.config.get('min_update_interval_s', 8e-6))
    trigger = sequence_trigger_timing(sequence_path, timing, channel=trigger_channel)
    fits = solve_field_shim_waveforms(model, total_current_limit_A=total_current_limit_A,
        raster_s=raster_s, max_iterations=max_iterations, static_ridge=static_ridge,
        dynamic_ridge=dynamic_ridge, min_update_interval_s=minimum)
    exports = {}
    for name in ('static', 'dynamic'):
        fit = fits[name]
        edges = np.asarray(fit['state_edges_rf_s'])
        commands, quantized, hardware = prepare_field_waveform(model, fit['currents_A'], edges,
            total_current_limit_A=total_current_limit_A, field_limit_Hz=field_limit_Hz)
        csv_edges, csv_timing = waveform_edges_for_origin(edges[:-1], edges[1:],
            origin=csv_time_origin, trigger_rf_s=trigger['trigger_rf_s'], min_update_interval_s=minimum)
        exported = export_waveform(commands, csv_edges, settings, output_dir,
            include_header=True, include_zero_preamble=True, include_final_zero_row=True,
            zero_time_column=False, numbered_filename=True, filename_prefix=f'waveform_{name}')
        path = Path(exported['csv_file'])
        table = np.loadtxt(path, delimiter=',', ndmin=2)
        first_row = int(csv_edges[0] > 0)
        # CSV column zero is time; 1-based physical output numbers therefore
        # directly index the corresponding current columns.
        selected = np.asarray(exported['settings']['channel_map'])
        polarity = np.asarray(exported['settings']['channel_polarity'])
        recovered = table[first_row:first_row+commands.shape[1], selected].T*polarity[:, None]
        if not np.array_equal(quantize_currents(recovered), quantized):
            path.unlink()
            raise ValueError('Reloaded field-shim CSV does not reproduce the checked DAC currents.')
        report = dict(method=name, waveform_export=exported, trigger=trigger, CSV_timing=csv_timing,
            source_sha256=dict(provenance or {}), current_response=model.current_response,
            upstream=fits['report'], field_solver=fit['report'], hardware=hardware,
            csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            csv_DAC_roundtrip_verified=True, image_optimization_required=False,
            no_hardware_action_performed=True)
        metadata, archive = path.with_suffix('.json'), path.with_suffix('.npz')
        if metadata.exists() or archive.exists():
            path.unlink()
            raise FileExistsError('Field waveform sidecars already exist; existing exports are preserved.')
        with archive.open('xb') as stream:
            np.savez_compressed(stream, currents_A=commands, nominal_DAC_currents_A=quantized,
                solver_currents_A=fit['currents_A'], state_edges_rf_s=edges, csv_state_edges_s=csv_edges)
        with metadata.open('x', encoding='utf-8') as stream:
            json.dump(_safe(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
        exports[name] = dict(csv_file=str(path), metadata_file=str(metadata), archive_file=str(archive),
            csv_sha256=report['csv_sha256'], csv_DAC_roundtrip_verified=True,
            uniform_contraction=hardware['uniform_contraction'],
            hardware_limits_met=True, solver=fit['report'])
    return _safe(dict(exports=exports, report=fits['report']))


__all__ = ['export_field_shim_waveforms', 'waveform_hardware_diagnostics', 'prepare_field_waveform']
