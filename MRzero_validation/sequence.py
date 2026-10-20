"""Cartesian EPI waveform translated from make_cartesian_epi_sequence.m.

Gradients are in T/m, time is in seconds, and k-space is in cycles/m. ADC
samples measure magnetization after their marked timestep. Event labels and
the nominal kx_index / ky_index arrays keep MATLAB's 1-based convention.

This translation preserves the original half-grid trajectory, transition
timing, and gradient design. It does not repair the original reconstruction.
"""

import numpy as np


def check_grad_limits(gradient, gmax, smax, dt, name="gradient"):
    """Raise if gradient amplitude or adjacent-step slew exceeds its limit."""
    gradient = np.asarray(gradient)
    if gradient.size == 0:
        return
    if np.max(np.abs(gradient)) > gmax * (1 + 1e-12):
        raise ValueError(f"{name} exceeds Gmax.")
    if gradient.size > 1 and np.max(np.abs(np.diff(gradient))) / dt > smax * (1 + 1e-12):
        raise ValueError(f"{name} exceeds Smax.")


def _slew_ramp(g0, g1, smax, dt):
    """Return ramp samples excluding the starting value and including the end."""
    n = int(np.ceil(abs(g1 - g0) / (smax * dt)))
    if n == 0:
        return np.empty(0)
    # Preserve MATLAB linspace arithmetic, including its symmetric zero tie.
    # NumPy linspace can choose the other central sample in an odd-length ramp.
    if g0 == -g1 and n > 1:
        ramp = np.arange(-n + 2, n + 1, 2, dtype=float) * (g1 / n)
    else:
        ramp = g0 + np.arange(1, n + 1, dtype=float) * (g1 - g0) / n
    ramp[-1] = g1
    return ramp


def _area_pulse(area, gmax, smax, dt):
    """Return the original triangular pulse with the requested signed area."""
    if area == 0:
        return np.zeros(1)
    n_slew = int(np.ceil(np.sqrt(abs(area) / (smax * dt * dt))))
    n_amplitude = int(np.ceil(abs(area) / (gmax * dt)))
    n = max(1, n_slew, n_amplitude)
    shape = np.r_[np.arange(n + 1), np.arange(n - 1, 0, -1), 0].astype(float)
    step = abs(area) / (shape.sum() * dt)
    gradient = np.sign(area) * shape * step
    check_grad_limits(gradient, gmax, smax, dt, "area pulse")
    return gradient


def _epi_transition(gx0, gx1, gy_area, gmax, smax, dt):
    """Center the phase-encoding blip around the ramp's first minimum |Gx|."""
    gx = _slew_ramp(gx0, gx1, smax, dt)
    gy_blip = _area_pulse(gy_area, gmax, smax, dt)
    zero_index = int(np.argmin(np.abs(gx)))
    gy_start = zero_index - len(gy_blip) // 2
    if gy_start < 0:
        prepend = -gy_start
        gx = np.r_[np.full(prepend, gx0), gx]
        zero_index += prepend
        gy_start = 0
    gy_end = gy_start + len(gy_blip)
    if gy_end > len(gx):
        gx = np.r_[gx, np.full(gy_end - len(gx), gx1)]
    gy = np.zeros_like(gx)
    gy[gy_start:gy_end] = gy_blip
    return gx, gy, zero_index


def make_cartesian_epi_sequence(params, display=False):
    """Build the original square EPI train from an English Python dictionary.

    Required fields: dt [s], matrix_size [pixels], FOV [m], Gmax [T/m], and
    Smax [T/m/s]. The waveform designer uses 42.57747892e6 Hz/T, exactly as
    the MATLAB source; params['gammaHz'] belongs to the forward model and
    may differ in saved files. Returned arrays are one-dimensional.

    Event 0 disables ACDC during the prephaser. Events 1..matrix_size select
    the corresponding current column. Each inter-line transition switches
    event at the first sample minimizing |Gx|, including the zero sample.
    """
    dt = float(params["dt"])
    nx = int(params["matrix_size"])
    fov = float(np.asarray(params["FOV"]).reshape(-1)[0])
    gmax, smax = float(params["Gmax"]), float(params["Smax"])
    if nx < 1 or nx != params["matrix_size"]:
        raise ValueError("matrix_size must be a positive integer.")
    if not all(np.isfinite(v) and v > 0 for v in (dt, fov, gmax, smax)):
        raise ValueError("dt, FOV, Gmax, and Smax must be finite and positive.")
    gamma_hz = 42.57747892e6
    dk = 1 / fov
    adc_step = int(np.ceil(dk / (gamma_hz * gmax) / dt))
    gread = dk / (gamma_hz * (adc_step * dt))
    gx0 = _slew_ramp(0, gread, smax, dt)
    first_adc_index = int(np.floor(adc_step / 2 + 0.5))  # MATLAB round, positive tie.

    kx_first = -(nx - 1) / 2 * dk
    ky_first = kx_first
    kx_pre_area = kx_first / gamma_hz - gx0.sum() * dt - gread * first_adc_index * dt
    gx_pre = _area_pulse(kx_pre_area, gmax, smax, dt)
    gy_pre = _area_pulse(ky_first / gamma_hz, gmax, smax, dt)
    n_pre = max(len(gx_pre), len(gy_pre))
    gx_pre = np.pad(gx_pre, (0, n_pre - len(gx_pre)))
    gy_pre = np.pad(gy_pre, (0, n_pre - len(gy_pre)))

    gx_parts = [gx_pre, gx0]
    gy_parts = [gy_pre, np.zeros_like(gx0)]
    adc_parts = [np.zeros(n_pre, dtype=bool), np.zeros(len(gx0), dtype=bool)]
    event_parts = [np.zeros(n_pre, dtype=int), np.ones(len(gx0), dtype=int)]
    kx_indices, ky_indices = [], []

    for line in range(1, nx + 1):
        polarity = (-1) ** (line - 1)
        gx_flat = np.full(nx * adc_step, polarity * gread)
        adc_flat = np.zeros(len(gx_flat), dtype=bool)
        adc_flat[np.arange(nx) * adc_step + first_adc_index - 1] = True
        gx_parts.append(gx_flat)
        gy_parts.append(np.zeros_like(gx_flat))
        adc_parts.append(adc_flat)
        event_parts.append(np.full(len(gx_flat), line, dtype=int))
        kx_indices.append(np.arange(1, nx + 1) if polarity > 0 else np.arange(nx, 0, -1))
        ky_indices.append(np.full(nx, line, dtype=int))

        if line < nx:
            gx_trans, gy_trans, zero_index = _epi_transition(
                polarity * gread, -polarity * gread, dk / gamma_hz, gmax, smax, dt
            )
            transition_events = np.full(len(gx_trans), line, dtype=int)
            transition_events[zero_index:] = line + 1
            gx_parts.append(gx_trans)
            gy_parts.append(gy_trans)
            adc_parts.append(np.zeros(len(gx_trans), dtype=bool))
            event_parts.append(transition_events)

    gx_end = _slew_ramp((-1) ** (nx - 1) * gread, 0, smax, dt)
    gx_parts.append(gx_end)
    gy_parts.append(np.zeros_like(gx_end))
    adc_parts.append(np.zeros(len(gx_end), dtype=bool))
    event_parts.append(np.full(len(gx_end), nx, dtype=int))

    gx, gy = np.concatenate(gx_parts), np.concatenate(gy_parts)
    adc = np.concatenate(adc_parts)
    kx, ky = gamma_hz * np.cumsum(gx) * dt, gamma_hz * np.cumsum(gy) * dt
    seq = dict(
        Gx=gx, Gy=gy, adc=adc, eventOfTime=np.concatenate(event_parts),
        kx_index=np.concatenate(kx_indices), ky_index=np.concatenate(ky_indices),
        dt=dt, dk=dk, adc_step=adc_step, Gread=gread,
        kx=kx, ky=ky, kx_adc=kx[adc], ky_adc=ky[adc],
    )
    check_grad_limits(gx, gmax, smax, dt, "Gx")
    check_grad_limits(gy, gmax, smax, dt, "Gy")
    if display:
        _display_sequence(seq)
    return seq


def audit_sequence(seq, matrix_size):
    """Summarize the preserved waveform and original rounded ADC grid mapping."""
    nx = int(matrix_size)
    dt = float(seq["dt"])
    steps = np.flatnonzero(seq["adc"])
    scaled = np.column_stack([seq["kx_adc"], seq["ky_adc"]]) / seq["dk"]
    bins = (np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)).astype(int) + nx // 2
    valid = np.all((bins >= 0) & (bins < nx), axis=1)
    unique_bins = np.unique(bins[valid], axis=0).shape[0]
    echo_spacing = float(np.median(np.diff(steps[::nx])) * dt) if nx > 1 else None
    return dict(
        timesteps=int(len(seq["Gx"])), adc_count=int(len(steps)),
        duration_s=float(len(seq["Gx"]) * dt), echo_spacing_s=echo_spacing,
        kept_adc=int(valid.sum()), rejected_adc=int((~valid).sum()),
        unique_bins=int(unique_bins), overwritten_adc=int(valid.sum() - unique_bins),
        max_gradient_T_per_m=float(max(np.max(np.abs(seq["Gx"])), np.max(np.abs(seq["Gy"])))),
        max_slew_T_per_m_per_s=float(max(np.max(np.abs(np.diff(seq["Gx"]))),
                                              np.max(np.abs(np.diff(seq["Gy"])))) / dt),
    )


def _display_sequence(seq):
    """Plot waveforms, slew rates, and the trajectory with English labels."""
    import matplotlib.pyplot as plt

    time_ms = np.arange(1, len(seq["Gx"]) + 1) * seq["dt"] * 1e3
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)
    for key in ("Gx", "Gy"):
        axes[0].plot(time_ms, seq[key] * 1e3, label=key)
        axes[1].plot(time_ms[:-1], np.diff(seq[key]) / seq["dt"], label=key)
    axes[0].plot(time_ms[seq["adc"]], seq["Gx"][seq["adc"]] * 1e3, "k.", ms=2, label="ADC")
    axes[0].set_ylabel("Gradient [mT/m]")
    axes[0].legend()
    axes[1].set(xlabel="Time after RF [ms]", ylabel="Slew rate [T/m/s]")
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.plot(seq["kx"], seq["ky"], "k-", lw=0.8)
    ax.set(xlabel="kx [cycles/m]", ylabel="ky [cycles/m]", title="Original EPI trajectory")
    ax.set_aspect("equal")
    plt.show()
