"""Constrained 3D image optimization and explicit full-volume waveform export.

One chronological line waveform is repeated at every acquired kz partition.
The volume model supplies the full 3D image objective; residual-field guidance
uses exact integrated amplifier response between consecutive readout centers.
No acquisition, controller or other hardware is commanded by this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

EXPERIMENT = Path(__file__).resolve().parent.parent/'Experiment'
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))
from acdc_gpu import project_currents_tensor
from acdc_gpu_field import solve_field_gpu
from export_acdc_waveform import export_waveform, quantize_currents, validate_waveform_settings
from acdc_hardware import HardwareConstraints
from field_guidance import FieldGuidance


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _guide(model, scale_Hz):
    # Only the guide's sequence label is adapted. Its spatial domain contains
    # every active 3D voxel; all ADC times and command windows remain measured.
    names = ('nc', 'nstates', 'time_s', 'rf_time_s', 'line_index', 'line_start_s',
             'line_end_s', 'current_response', 'quality_mask', 'maps_numpy', 'b0_numpy', 'tensor')
    adapter = SimpleNamespace(**{name: getattr(model, name) for name in names}, sequence_kind='epi')
    guide = FieldGuidance(adapter, scale_Hz=scale_Hz)
    guide.report.update(sequence_kind='epi3d', domain='All active full-volume 3D object centers; equal spatial weights.',
                        repeated_partition_policy='Identical command waveform and RF-relative timing for every kz partition.')
    return guide


def optimize_volume_image(model, initial_currents_A=None, *, initialization='static',
                          total_current_limit_A=30., field_limit_Hz=1000.,
                          field_guidance_weight=.05, field_guidance_scale_Hz=100.,
                          max_iterations=1000, learning_rate_A=.01, patience=100,
                          hardware=None, callback=None, callback_every=10):
    """Projected Adam, retaining the best combined image/field objective.

    Exact zero channel bounds disable those commands throughout optimization,
    including static initialization. Limits apply to the DAC commands and to
    the continuous first-order physical response through ``HardwareConstraints``.
    A finite optimization result is not automatically eligible for export:
    the separate export step also requires temporal-interpolation convergence.

    The optional callback receives a separate display snapshot containing the
    current complex image, scalar loss history and maximum command field. The
    image comes from the objective's own forward evaluation. Display arrays
    are independent copies; only scalar rows remain in the returned history.
    """
    for name, value in (('max_iterations', max_iterations), ('patience', patience), ('callback_every', callback_every)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if (not np.isfinite(learning_rate_A) or learning_rate_A <= 0
            or not np.isfinite(field_guidance_weight) or field_guidance_weight < 0):
        raise ValueError('Learning rate must be positive and field guidance weight nonnegative, both finite.')
    if initialization not in ('static', 'zero'):
        raise ValueError('initialization must be static or zero.')
    if not model.current_response['enabled']:
        raise ValueError('Hardware-checked volume optimization requires the enabled physical current-response model.')
    config = dict(hardware or {})
    config.setdefault('cycle_duration_s', float(model.TR_s))
    constraints = HardwareConstraints(model, config)
    guide = _guide(model, field_guidance_scale_Hz) if field_guidance_weight else None

    @torch.no_grad()
    def project(value):
        command = project_currents_tensor(model.tensor(value), model.max_current_A,
            total_current_limit_A, model.B_quality, field_limit_Hz)
        return constraints.additional_projection(command, total_current_limit_A, field_limit_Hz)

    static_report = None
    if initial_currents_A is None:
        initial_currents_A = np.zeros((model.nc, model.nstates))
        if initialization == 'static':
            mask = np.asarray(model.quality_mask, dtype=bool)
            static = solve_field_gpu(np.asarray(model.maps_numpy)[mask], -np.asarray(model.b0_numpy)[mask],
                device=model.device, dtype=model.dtype, max_current_A=model.numpy(model.max_current_A),
                total_current_limit_A=total_current_limit_A, field_limit_Hz=field_limit_Hz,
                max_iterations=150, tolerance=1e-6)
            initial_currents_A = np.repeat(static['currents_A'], model.nstates, axis=1)
            static_report = {key: static[key] for key in ('loss', 'iterations', 'converged', 'seconds')}
    initial = project(model._check_currents(initial_currents_A)).detach()
    commands = torch.nn.Parameter(initial.clone())
    adam = torch.optim.Adam([commands], lr=float(learning_rate_A))
    best, best_objective = initial.clone(), np.inf
    initial_objective = initial_mse = None
    history, failures, stale = [], [], 0
    last_callback_iteration = None

    @torch.no_grad()
    def notify(iteration, row, image):
        nonlocal last_callback_iteration
        if callback is None or last_callback_iteration == iteration:
            return
        maximum_field = commands.new_tensor(0.)
        # Bound the temporary field array by a few chronological states rather
        # than constructing all volume voxels x all waveform states at once.
        for first_state in range(0, model.nstates, 4):
            fields = model.B_quality @ commands[:, first_state:first_state+4]
            maximum_field = torch.maximum(maximum_field, fields.abs().amax())
        snapshot = dict(row)
        snapshot.update(image=np.array(model.numpy(image.detach()), copy=True),
            history={'iteration': [item['iteration'] for item in history],
                     'image_mse': [item['image_mse'] for item in history]},
            max_field_Hz=float(maximum_field))
        callback(iteration, snapshot, np.array(model.numpy(commands.detach()), copy=True))
        last_callback_iteration = iteration

    stop_reason = 'maximum iterations reached'
    for iteration in range(int(max_iterations)+1):
        adam.zero_grad(set_to_none=True)
        image = model.forward(commands)[0]
        image_value = model._image_loss(image)
        if guide is None:
            penalty = image_value.new_tensor(0.)
            field_gradient = None
            residual_rms = None
        else:
            penalty, field_gradient, residual = guide.value_gradient(commands, gradient=True)
            residual_rms = float(residual)
        objective = float(image_value.detach()) + float(field_guidance_weight)*float(penalty)
        if not np.isfinite(objective):
            failures.append('nonfinite objective')
            stop_reason = failures[-1]
            break
        if initial_objective is None:
            initial_objective, initial_mse = objective, float(image_value.detach())
        threshold = 1e-12*max(1., abs(best_objective)) if np.isfinite(best_objective) else 0.
        improved = objective < best_objective-threshold
        if improved:
            best, best_objective, stale = commands.detach().clone(), objective, 0
        else:
            stale += 1
        row = dict(iteration=iteration, image_mse=float(image_value.detach()), objective=objective,
                   best_objective=float(best_objective), field_penalty=float(penalty),
                   residual_field_rms_Hz=residual_rms)
        history.append(row)
        if callback is not None and (iteration % callback_every == 0 or iteration == max_iterations):
            notify(iteration, row, image)
        if iteration == max_iterations:
            break
        if stale >= patience:
            stop_reason = 'patience reached without improvement of combined objective'
            notify(iteration, row, image)
            break
        image_value.backward()
        if field_gradient is not None:
            commands.grad.add_(field_gradient, alpha=float(field_guidance_weight))
        if commands.grad is None or not bool(torch.isfinite(commands.grad).all()):
            failures.append('nonfinite or missing image gradient')
            stop_reason = failures[-1]
            notify(iteration, row, image)
            break
        adam.step()
        with torch.no_grad():
            commands.copy_(project(commands))
    if initial_objective is None:
        raise RuntimeError('The initial full-volume objective is nonfinite; no optimization candidate is available.')
    currents = model.numpy(best)
    quantized = quantize_currents(currents)
    command_report = constraints.diagnostics(best, total_current_limit_A, field_limit_Hz)
    quantized_report = constraints.diagnostics(quantized, total_current_limit_A, field_limit_Hz)
    with torch.no_grad():
        quantized_tensor = model.tensor(quantized)
        quantized_mse = float(model.torch_objective(quantized_tensor))
        quantized_penalty = 0. if guide is None else float(guide.value_gradient(quantized_tensor, gradient=False)[0])
    quantized_objective = quantized_mse+float(field_guidance_weight)*quantized_penalty
    improved = bool(best_objective < initial_objective-1e-12*max(1., abs(initial_objective)))
    quantized_passed = bool(np.isfinite(quantized_objective) and quantized_objective <= initial_objective)
    return dict(currents_A=currents, quantized_currents_A=quantized,
        initial_currents_A=model.numpy(initial), channel_max_current_A=model.numpy(model.max_current_A),
        initial_objective=float(initial_objective), initial_image_mse=float(initial_mse),
        best_objective=float(best_objective), quantized_objective=float(quantized_objective),
        quantized_image_mse=quantized_mse, quantized_field_penalty=quantized_penalty,
        improved=improved, quantized_objective_not_worse_than_initial=quantized_passed,
        finite_optimization=not failures, stop_reason=stop_reason, failures=failures,
        history=history, hardware=command_report, hardware_after_DAC=quantized_report,
        static_initialization=static_report, field_guidance_weight=float(field_guidance_weight),
        field_guidance_scale_Hz=float(field_guidance_scale_Hz),
        field_guidance=None if guide is None else guide.report,
        total_current_limit_A=float(total_current_limit_A), field_limit_Hz=float(field_limit_Hz),
        repeated_waveform_per_partition=True,
        temporal_interpolation_validation_required_before_export=True)


def export_volume_waveform(solution, timing, settings, output_dir, *, metadata,
                           time_interpolation_report, filename_prefix='waveform_volume',
                           zero_time_column=False):
    """Export a checked *timed* waveform including every inter-partition zero.

    Zero-time, trigger-stepped playback cannot represent the inserted off rows:
    the source sequence has no trigger at those switch-off times. It is refused
    here rather than shifting subsequent currents onto the wrong line/partition.
    """
    if zero_time_column:
        raise ValueError('3D zero-time playback is unsupported: inter-partition zero-current events have no matching sequence triggers; preserve timed playback.')
    if not isinstance(time_interpolation_report, dict) or time_interpolation_report.get('passed') is not True:
        raise ValueError('Full-volume temporal interpolation convergence must pass before waveform export.')
    required = ('improved', 'quantized_objective_not_worse_than_initial', 'finite_optimization')
    if not all(solution.get(key) is True for key in required):
        raise ValueError('A finite improvement surviving DAC quantization is required before full-volume export.')
    if (not solution.get('hardware', {}).get('all_limits_met')
            or not solution.get('hardware_after_DAC', {}).get('all_limits_met')):
        raise ValueError('Full-volume waveform fails command or continuous physical-current limits.')
    currents = np.asarray(solution['currents_A'], dtype=float)
    clean = validate_waveform_settings(settings, currents.shape[0])
    if not np.array_equal(np.asarray(clean['max_current_A']), np.asarray(solution['channel_max_current_A'])):
        raise ValueError('Export channel limits differ from the limits used during optimization.')
    if float(clean['total_current_limit_A']) != float(solution['total_current_limit_A']):
        raise ValueError('Export total-current limit differs from the optimization configuration.')
    if (timing.get('repeated_waveform_per_partition') is not True
            or timing.get('verified_per_line_triggers') is not True):
        raise ValueError('Export needs verified repeated-partition waveform and exact per-line trigger timing.')
    times = np.asarray(timing['waveform_event_time_first_trigger_s'], dtype=float)
    labels = np.asarray(timing['waveform_event_state_index'])
    if (times.ndim != 1 or labels.shape != times.shape or len(times) < 2
            or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0)
            or not np.issubdtype(labels.dtype, np.integer) or labels[-1] != -1
            or np.any(labels < -1) or np.any(labels >= currents.shape[1])):
        raise ValueError('Full-volume event times and command-state labels are inconsistent.')
    nz, ny = int(timing['partition_count']), currents.shape[1]
    if not np.array_equal(labels, np.tile(np.r_[np.arange(ny), -1], nz)):
        raise ValueError('Full-volume waveform must repeat all line commands and then zero after every partition.')
    expanded = np.zeros((currents.shape[0], len(times)-1))
    active = labels[:-1] >= 0
    expanded[:, active] = currents[:, labels[:-1][active]]
    # All checks precede the first write, including metadata serialization.
    serial_metadata = _json_safe(metadata)
    json.dumps(serial_metadata, allow_nan=False)
    report = export_waveform(expanded, times, clean, output_dir,
        filename_prefix=filename_prefix, numbered_filename=True,
        include_final_zero_row=True, zero_time_column=False, start_at_first_nonzero=False)
    csv_path = Path(report['csv_file'])
    summary = dict(format='full_3d_epi_timed_waveform_v1', csv=report,
        csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        playback='Timed full-volume playback from first imaging trigger, including off events between partitions.',
        zero_time_export_supported=False, repeated_partition_count=nz,
        line_command_count_per_partition=ny, explicit_zero_events=nz,
        command_event_state_indices=labels.tolist(),
        sequence_sha256=timing.get('sequence_sha256'), sequence_signature_md5=timing.get('sequence_signature_md5'),
        temporal_interpolation=time_interpolation_report,
        hardware=solution['hardware'], hardware_after_DAC=solution['hardware_after_DAC'],
        initial_objective=solution['initial_objective'], best_objective=solution['best_objective'],
        quantized_objective=solution['quantized_objective'],
        field_guidance_weight=solution['field_guidance_weight'],
        field_guidance_scale_Hz=solution['field_guidance_scale_Hz'], metadata=serial_metadata,
        no_hardware_action_performed=True)
    sidecar = csv_path.with_suffix('.json')
    with sidecar.open('x', encoding='utf-8') as stream:
        json.dump(_json_safe(summary), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    summary['sidecar_json'] = str(sidecar)
    return summary


__all__ = ['optimize_volume_image', 'export_volume_waveform']
