"""Small deterministic EPI fixture for MRzero tests and notebook examples.

Every image and calibration map here is synthetic.  The function neither reads
an acquisition nor writes files, and its calibration must never be exported to
scanner hardware.  Array axes follow the experiment readers: [phase Y, readout X].
"""

from __future__ import annotations

import numpy as np


DEFAULT_TAU_S = 133e-6 / np.log(100.0)


def _coil_basis(x_normalized, y_normalized, n_channels):
    """Create distinct smooth responses in Hz/A, including low-order shims."""
    x, y = x_normalized, y_normalized
    functions = [
        70.0 * np.ones_like(x),
        55.0 * x,
        55.0 * y,
        45.0 * (2.0 * x**2 - 1.0),
        45.0 * (2.0 * y**2 - 1.0),
        50.0 * x * y,
        45.0 * np.sin(np.pi * x),
        45.0 * np.sin(np.pi * y),
    ]
    # Localized responses resemble independently placed coils.  A fixed angular
    # sequence gives reproducible channels when a smaller channel count is used.
    for channel in range(max(0, n_channels - len(functions))):
        angle = channel * np.pi * (3.0 - np.sqrt(5.0))
        radius = 0.50 + 0.14 * (channel % 3)
        center_x, center_y = radius * np.cos(angle), radius * np.sin(angle)
        width = 0.30 + 0.025 * (channel % 4)
        response = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * width**2))
        functions.append(60.0 * response)
    return np.stack(functions[:n_channels], axis=-1)


def make_demo_experiment(matrix_size=16, n_channels=32,
                         current_response_enabled=True, tau_s=DEFAULT_TAU_S):
    """Return ``experiment``, ``calibration``, ``timing`` and inspection arrays.

    ``matrix_size`` is an even square matrix of at least 8 pixels.  A single RF
    excitation is centered at zero; one held current command per EPI line starts
    at 1 ms.  ADC times are stored in Cartesian order, so negative readouts have
    decreasing sample times.  ``line_index`` labels chronological states from 1.

    The signal convention is ``exp(-2j*pi*(kx*X + ky*Y + B0*time))`` times
    ``exp(1j*(RF_phase - ADC_phase))``.  The nonzero phase offsets exercise
    receiver/RF sign handling without changing the ideal magnitude image.

    The returned complex three-echo B0 images have four synthetic receivers,
    equally spaced echoes, and the same negative B0 phase convention.  They can
    be passed directly to ``b0_from_coil_images`` for independent loader tests.
    """
    if (isinstance(matrix_size, bool) or int(matrix_size) != matrix_size
            or matrix_size < 8 or matrix_size % 2):
        raise ValueError("matrix_size must be an even integer of at least 8.")
    if (isinstance(n_channels, bool) or int(n_channels) != n_channels
            or not 1 <= n_channels <= 64):
        raise ValueError("n_channels must be an integer between 1 and 64.")
    if not isinstance(current_response_enabled, (bool, np.bool_)):
        raise ValueError("current_response_enabled must be True or False.")
    matrix_size, n_channels = int(matrix_size), int(n_channels)
    tau = np.asarray(tau_s, dtype=float)
    if tau.ndim == 0:
        tau = np.full(n_channels, float(tau))
    if tau.shape != (n_channels,) or not np.isfinite(tau).all() or np.any(tau <= 0):
        raise ValueError("tau_s must be positive and finite: one scalar or one value per channel.")

    fov_m = 0.2
    pixel_size_m = fov_m / matrix_size
    axis_m = (np.arange(matrix_size) - matrix_size // 2) * pixel_size_m
    X_m, Y_m = np.meshgrid(axis_m, axis_m)
    x, y = X_m / (fov_m / 2), Y_m / (fov_m / 2)

    # Smooth, asymmetric object: a broad ellipse plus two off-center structures.
    magnitude = np.exp(-((x + 0.10)**2 / 0.58**2 + (y - 0.07)**2 / 0.70**2))
    magnitude += 0.42 * np.exp(-((x - 0.31)**2 + (y + 0.22)**2) / 0.16**2)
    magnitude -= 0.23 * np.exp(-((x + 0.24)**2 + (y - 0.24)**2) / 0.14**2)
    magnitude = np.maximum(magnitude, 0.0)
    magnitude /= magnitude.max()
    support = magnitude > 0.10
    m0 = np.where(support, magnitude, 0.0)

    b0_Hz = 65.0 + 48.0*x - 37.0*y + 42.0*(x**2 - y**2)
    b0_Hz += 22.0 * np.sin(np.pi*x) * np.sin(np.pi*y)
    coil_maps = _coil_basis(x, y, n_channels)

    dwell_s = 10e-6
    adc_delay_s = 30e-6
    echo_spacing_s = max(600e-6, matrix_size*dwell_s + 100e-6)
    starts = 1e-3 + np.arange(matrix_size) * echo_spacing_s
    ends = np.r_[starts[1:], starts[-1] + echo_spacing_s]
    polarity = np.where(np.arange(matrix_size) % 2 == 0, 1, -1)
    sample_times = adc_delay_s + (np.arange(matrix_size) + 0.5) * dwell_s
    time_s = starts[:, None] + sample_times[None, :]
    time_s[polarity < 0] = time_s[polarity < 0, ::-1]
    k_axis = (np.arange(matrix_size) - matrix_size // 2) / fov_m
    kx, ky = np.meshgrid(k_axis, k_axis)
    line_index = np.repeat(np.arange(1, matrix_size + 1)[:, None], matrix_size, axis=1)
    rf_phase_rad = 0.31
    adc_phase_rad = 0.17
    phase_offset = np.full((matrix_size, matrix_size), rf_phase_rad - adc_phase_rad)
    te_s = float(time_s[matrix_size // 2, matrix_size // 2])
    tr_s = float(ends[-1] + 20e-3)

    geometry = {
        "X_m": X_m, "Y_m": Y_m,
        "position_mm": np.array([100.0, -100.0, 0.0]),
        "spacing_mm": np.full(2, pixel_size_m * 1000),
        "orientation": np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0]),
        "slice_thickness_mm": 5.0,
    }
    acquisition = {
        "sequence_kind": "epi", "kx_mInv": kx, "ky_mInv": ky,
        "time_s": time_s, "rf_time_s": np.zeros_like(time_s),
        "phase_offset_rad": phase_offset, "line_index": line_index,
        "rf_phase_rad": rf_phase_rad, "adc_phase_rad": adc_phase_rad,
        "line_start_s": starts, "line_end_s": ends, "polarity": polarity,
        "dwell_s": dwell_s, "TE_s": te_s, "TR_s": tr_s,
        "FOV_m": np.full(2, fov_m), "matrix_size": np.full(2, matrix_size),
        "echo_spacing_s": echo_spacing_s,
    }
    timing = {
        "sequence_kind": "epi", "image_shape": [matrix_size, matrix_size],
        "matrix_size": [matrix_size, matrix_size],
        "state_start_s": starts.copy(), "state_end_s": ends.copy(),
        "state_edges_s": np.r_[starts, ends[-1]], "rf_center_s": 0.0,
        "rf_start_s": np.array([-50e-6]), "rf_end_s": np.array([50e-6]),
        "first_adc_delay_s": np.full(matrix_size, adc_delay_s),
        "first_adc_sample_center_delay_s": np.full(matrix_size, adc_delay_s + dwell_s/2),
        "sequence_duration_s": tr_s, "navigator_count": 0, "dummy_rf_count": 0,
        "FOV_m": np.full(2, fov_m), "readout_grid_offset_mInv": 0.0,
        "state_order": "Chronological lines; zero current before and after the imaging windows.",
        "synthetic": True,
    }
    provenance = {
        "synthetic": True,
        "description": "Deterministic synthetic EPI fixture; no measured acquisition or calibration.",
        "axis_order": "[phase Y, readout X]; coil maps [phase Y, readout X, channel]",
        "input_rotation_deg": 0,
        "roi_parameters": {"reference_threshold": 0.10},
        "hardware_export_allowed": False,
    }
    experiment = {
        "m0_proxy": m0, "geometry": geometry, "acquisition": acquisition,
        "field": {"b0Hz": np.where(support, b0_Hz, 0.0),
                  "measured_b0Hz": b0_Hz.copy(), "simulation_mask": support,
                  "mask": support.copy()},
        "relaxation": {"T2star_s": np.inf},
        "current_response": {"enabled": bool(current_response_enabled), "tau_s": tau},
        "images": {"gre_reference": magnitude.copy()},
        "provenance": provenance,
        "assumptions": ["Synthetic transverse amplitude; uniform simulated receive sensitivity.",
                        "One excitation; no relaxation decay or measured tissue contrast."],
        "optimization_ready": True,
    }
    calibration = {
        "acdc_maps": coil_maps, "X_m": X_m.copy(), "Y_m": Y_m.copy(),
        "units": "Hz/A", "spatial_valid_mask": np.ones_like(support),
        "registration_confirmed": True,
        "metadata": {"synthetic": True, "hardware_export_allowed": False,
                     "registration_note": "Analytical synthetic maps generated on the exact object grid.",
                     "source_channel_order": np.arange(1, n_channels + 1)},
    }
    reference = dict(geometry, magnitude=magnitude.copy(), valid_mask=np.ones_like(support),
                     TE_s=3e-3, TR_s=1.0, flip_deg=90.0, synthetic=True)

    echo_times_s = np.array([3e-3, 5e-3, 7e-3])
    receivers = []
    for receiver in range(4):
        angle = receiver * np.pi/2
        sensitivity = (0.9 + 0.15*(x*np.cos(angle) + y*np.sin(angle))) / 2
        receive_phase = 0.40*receiver + 0.30*x - 0.20*y
        receivers.append(sensitivity * np.exp(1j*receive_phase))
    receiver_maps = np.stack(receivers)
    b0_coil_images = (magnitude[None, None] * receiver_maps[None]
                      * np.exp(-2j*np.pi * echo_times_s[:, None, None, None] * b0_Hz[None, None]))
    return {
        "experiment": experiment, "calibration": calibration, "timing": timing,
        "gre": reference, "reference": reference, "reference_magnitude": magnitude,
        "b0_Hz": b0_Hz, "b0_on_reference": b0_Hz.copy(), "b0_valid": support.copy(),
        "support": support, "coil_maps": coil_maps,
        "b0_coil_images": b0_coil_images, "b0_echo_times_s": echo_times_s,
        "synthetic_receiver_maps": receiver_maps,
    }
