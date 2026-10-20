"""Plot held ACDC commands and the model's filtered physical currents.

This is a display helper: it reuses model.sample_waveform, changes no physics
or commands, and never communicates with a waveform controller.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def read_waveform_csv(path):
    """Read the existing CSV format: seconds, then physical-output amperes.

    Commented headers from export_waveform and headerless tables are accepted.
    A zeroed time column contains no playback chronology and is rejected.
    The data remain exactly those stored in the file; no quantization is added.
    """
    path = Path(path)
    table = np.loadtxt(path, delimiter=',', ndmin=2)
    if (table.shape[0] < 2 or table.shape[1] < 2 or table.shape[1] > 65
            or not np.isfinite(table).all()):
        raise ValueError("CSV needs finite rows: time in seconds and 1..64 physical currents.")
    if np.any(table[:, 0] < 0) or np.any(np.diff(table[:, 0]) <= 0):
        raise ValueError("CSV timestamps must increase strictly; a zeroed time column cannot be plotted as playback timing.")
    return dict(file=str(path.resolve()), time_s=table[:, 0].copy(),
                commanded_A=table[:, 1:].T.copy())


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    result = np.asarray(value, dtype=float)
    if result.ndim != 0 or not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar.")
    return float(result)


def _output_mapping(channel_count, channel_map, channel_polarity):
    mapping = np.arange(1, channel_count + 1) if channel_map is None else np.asarray(channel_map)
    polarity = np.ones(channel_count) if channel_polarity is None else np.asarray(channel_polarity)
    if (np.iscomplexobj(mapping) or mapping.shape != (channel_count,)
            or not np.isfinite(mapping).all() or np.any(mapping != np.floor(mapping))
            or np.any(mapping < 1) or np.any(mapping > 64) or len(np.unique(mapping)) != channel_count):
        raise ValueError("channel_map needs one distinct physical output number in 1..64 per model channel.")
    if polarity.shape != (channel_count,) or not np.isin(polarity, [-1, 1]).all():
        raise ValueError("channel_polarity needs one +1 or -1 per model channel.")
    return mapping.astype(int), polarity.astype(float)


def _map_outputs(values, mapping, polarity):
    count = ((int(mapping.max()) + 7) // 8) * 8
    result = np.zeros((count, values.shape[1]))
    result[mapping - 1] = np.asarray(values) * polarity[:, None]
    return result


def waveform_plot_data(model, currents_A, *, origin='rf', trigger_rf_s=None,
                       channel_map=None, channel_polarity=None, channel=None,
                       csv_file=None, csv_origin_rf_s=None):
    """Prepare full traces and an exact-response transition zoom for plotting.

    ``channel`` is a 1-based physical output number; None selects the largest
    command change, including startup and final switch-off. Trigger-relative
    plots retain the earlier RF history as negative display times.

    For a CSV, csv_origin_rf_s explicitly locates its time zero relative to RF.
    Its actual commands can be overlaid, while the physical prediction remains
    based on currents_A and the model's command windows, not on measured current
    telemetry. The report checks CSV/model values and transition timestamps.
    """
    if origin == 'rf':
        origin_rf_s, origin_label = 0., 'centre RF'
    elif origin == 'first_imaging_trigger':
        if trigger_rf_s is None:
            raise ValueError("trigger_rf_s is required for a trigger-relative plot.")
        origin_rf_s = _finite_scalar(trigger_rf_s, 'trigger_rf_s')
        origin_label = "trigger d'imagerie"
    else:
        raise ValueError("origin must be 'rf' or 'first_imaging_trigger'.")
    mapping, polarity = _output_mapping(model.nc, channel_map, channel_polarity)
    final_zero_rf_s = float(np.asarray(model.line_end_s).reshape(-1)[-1])
    if not np.isfinite(final_zero_rf_s):
        raise ValueError("A finite final command end is required to show return to zero.")
    tau = np.broadcast_to(np.asarray(model.current_response['tau_s'], float), (model.nc,))
    if not np.isfinite(tau).all() or np.any(tau <= 0):
        raise ValueError("The model needs positive finite channel time constants.")
    full = model.sample_waveform(currents_A)
    time_rf_s = np.asarray(full['time_s']).reshape(-1)
    command = np.asarray(full['commanded_A'])
    actual = np.asarray(full['actual_A'])
    jumps = np.abs(np.diff(np.column_stack((np.zeros(model.nc), command)), axis=1))
    if channel is None:
        selected, transition_index = np.unravel_index(np.argmax(jumps), jumps.shape)
    else:
        if isinstance(channel, (bool, np.bool_)) or channel not in mapping:
            raise ValueError("channel must be a physical output number present in channel_map.")
        selected = int(np.flatnonzero(mapping == channel)[0])
        transition_index = int(np.argmax(jumps[selected]))
    transition_rf_s = float(time_rf_s[transition_index])
    zoom_start, zoom_end = max(0., transition_rf_s - 5 * tau[selected]), transition_rf_s + 5 * tau[selected]
    zoom_times = np.unique(np.r_[np.linspace(zoom_start, zoom_end, 513), transition_rf_s])
    zoom = model.sample_waveform(currents_A, zoom_times)
    report = dict(origin=origin, origin_label=origin_label, origin_rf_s=origin_rf_s,
        pre_origin_samples_retained=bool(np.any(time_rf_s < origin_rf_s)),
        selected_physical_output=int(mapping[selected]), selected_model_channel=int(selected + 1),
        transition_time_from_origin_ms=(transition_rf_s - origin_rf_s) * 1000,
        largest_selected_command_change_A=float(jumps[selected].max()),
        selected_tau_us=float(tau[selected] * 1e6),
        final_zero_from_origin_ms=(final_zero_rf_s - origin_rf_s) * 1000,
        response_enabled=bool(model.current_response['enabled']),
        physical_current_source='Model prediction from supplied DAC commands; not current-sense telemetry',
        channel_map=mapping.tolist(), channel_polarity=polarity.tolist(),
        no_hardware_action_performed=True)
    data = dict(time_ms=(time_rf_s - origin_rf_s) * 1000,
        commanded_A=_map_outputs(command, mapping, polarity),
        actual_A=_map_outputs(actual, mapping, polarity),
        zoom_time_us=(zoom_times - transition_rf_s) * 1e6,
        zoom_commanded_A=np.asarray(zoom['commanded_A'])[selected] * polarity[selected],
        zoom_actual_A=np.asarray(zoom['actual_A'])[selected] * polarity[selected],
        transition_rf_s=transition_rf_s, report=report, csv=None)
    if csv_file is not None:
        if csv_origin_rf_s is None:
            raise ValueError("csv_origin_rf_s must state the actual CSV time origin relative to RF.")
        csv_origin = _finite_scalar(csv_origin_rf_s, 'csv_origin_rf_s')
        csv = read_waveform_csv(csv_file)
        csv_rf_s = csv['time_s'] + csv_origin
        if int(mapping[selected]) > csv['commanded_A'].shape[0]:
            raise ValueError("CSV has no column for the selected physical output.")
        csv.update(time_ms=(csv_rf_s - origin_rf_s) * 1000,
                   zoom_time_us=(csv_rf_s - transition_rf_s) * 1e6)
        data['csv'] = csv
        # Evaluate held CSV states at interval interiors, avoiding cancellation
        # at CSV timestamps obtained by subtracting/readding the RF origin.
        probe = np.r_[(csv_rf_s[:-1] + csv_rf_s[1:]) / 2, csv_rf_s[-1] + 1e-12]
        keep = probe >= 0
        difference = None
        if keep.any():
            model_command = model.sample_waveform(currents_A, probe[keep], response_enabled=False)['commanded_A']
            predicted = _map_outputs(model_command, mapping, polarity)
            output_count = max(predicted.shape[0], csv['commanded_A'].shape[0])
            predicted = np.pad(predicted, ((0, output_count - predicted.shape[0]), (0, 0)))
            stored = np.pad(csv['commanded_A'][:, keep],
                            ((0, output_count - csv['commanded_A'].shape[0]), (0, 0)))
            difference = float(np.max(np.abs(predicted - stored)))
        expected_events = np.unique(np.r_[model.line_start_s, model.line_end_s])
        distance = float(np.min(np.abs(expected_events[:, None] - csv_rf_s[None, :]), axis=1).max())
        report.update(csv_file=csv['file'], csv_origin_rf_s=csv_origin,
            csv_max_command_difference_A=difference,
            csv_commands_match_within_0p6mA=difference is not None and difference <= .0006,
            csv_max_model_event_distance_s=distance,
            csv_model_event_times_match_within_1ns=distance <= 1e-9,
            csv_final_row_is_zero=bool(np.all(csv['commanded_A'][:, -1] == 0)),
            csv_command_source='Values and timestamps read from the CSV; no added quantization')
    return data


def plot_acdc_waveform(model, currents_A, **options):
    """Return (figure, JSON-ready report); the notebook decides when to show it."""
    data = waveform_plot_data(model, currents_A, **options)
    report, csv = data['report'], data['csv']
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout='constrained')
    # When available, display the actual file commands rather than a second
    # copy of the intended model commands. The physical prediction is labelled.
    if csv is None:
        command_time, commands, command_label = data['time_ms'], data['commanded_A'], 'Commandes DAC du modèle'
    else:
        command_time, commands, command_label = csv['time_ms'], csv['commanded_A'], 'Commandes lues dans le CSV'
    for row, values in enumerate(commands):
        axes[0].step(command_time, values, where='post', color='gray', alpha=.22,
                     linewidth=.65, label=command_label if row == 0 else None)
    for row, values in enumerate(data['actual_A']):
        axes[0].plot(data['time_ms'], values, alpha=.65, linewidth=.8,
                     label='Courants physiques prédits' if row == 0 else None)
    axes[0].axvline(report['final_zero_from_origin_ms'], color='black', linestyle='--',
                    label='Retour commandé à zéro')
    time_label = f"Temps depuis {report['origin_label']} [ms]"
    if report['pre_origin_samples_retained']:
        time_label += ' — temps négatifs : avant le trigger'
    axes[0].set(xlabel=time_label, ylabel='Courant de sortie [A]',
                title='Forme d’onde ACDC complète — prédiction et commandes')
    axes[0].legend()
    axes[1].step(data['zoom_time_us'], data['zoom_commanded_A'], where='post',
                 color='gray', linestyle='--', label='Commande DAC du modèle')
    if csv is not None:
        axes[1].step(csv['zoom_time_us'], csv['commanded_A'][report['selected_physical_output'] - 1],
                     where='post', color='black', alpha=.8, label='Commande CSV')
    axes[1].plot(data['zoom_time_us'], data['zoom_actual_A'], label='Courant physique prédit')
    axes[1].set(xlabel='Temps depuis le changement sélectionné [µs]', ylabel='Courant de sortie [A]',
        xlim=(float(data['zoom_time_us'].min()), float(data['zoom_time_us'].max())),
        title=f"Sortie {report['selected_physical_output']} : changement à {report['transition_time_from_origin_ms']:.3f} ms depuis {report['origin_label']} ; τ={report['selected_tau_us']:.2f} µs")
    axes[1].legend()
    return fig, report


__all__ = ['read_waveform_csv', 'waveform_plot_data', 'plot_acdc_waveform']
