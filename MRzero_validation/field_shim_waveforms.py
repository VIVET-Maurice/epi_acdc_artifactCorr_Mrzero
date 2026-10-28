"""Conventional B0 shims using the pinned, unmodified ACDC_Optim solvers.

The measured B0 is constant during the EPI readout. The dynamic shim therefore
pre-emphasises the configured amplifier response to produce its cancellation
field over that window; it does not infer a time-varying B0 from the echoes.
Scanner export and continuous-time hardware checks are performed separately.
"""

from __future__ import annotations

import time

import numpy as np


UPSTREAM_COMMIT = "703af140b6cb05df45a05f8d49bc4417a8be9d56"
UPSTREAM_URL = "https://github.com/bughht/ACDC_Optim"


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compress_field_target(maps_Hz_per_A, target_Hz):
    """Reduce the spatial dimension without changing the current objective.

    For A = Q R and b = Q z + b_perp,
    ||A I - b||^2 = ||R I - z||^2 + ||b_perp||^2. This also holds independently
    at every time point after a common amplifier convolution. No spatial rank
    threshold is used: NumPy's reduced QR retains the entire represented span.
    """
    maps = np.asarray(maps_Hz_per_A, dtype=np.float64)
    target = np.asarray(target_Hz, dtype=np.float64)
    if (maps.ndim != 2 or min(maps.shape) < 1 or target.shape != (maps.shape[0],)
            or not np.isfinite(maps).all() or not np.isfinite(target).all()
            or not np.any(maps)):
        raise ValueError("Need finite nonzero maps [voxels, channels] and target [voxels].")
    q, r = np.linalg.qr(maps, mode="reduced")
    reduced_target = q.T @ target
    orthogonal = target - q @ reduced_target
    return r.T.copy(), reduced_target, {
        "method": "reduced QR without rank truncation; exact spatial objective",
        "original_spatial_points": int(maps.shape[0]),
        "solver_spatial_points": int(r.shape[0]),
        "target_squared_norm_Hz2": float(target @ target),
        "orthogonal_target_squared_norm_Hz2": float(orthogonal @ orthogonal),
    }


def amplifier_sirf(tau_s, raster_s):
    """Normalised exponential kernel from ACDC_Optim's dynamic example.

    The time constant comes from the model. As in the upstream example, the
    kernel is sampled at the command raster, truncated at ten time constants,
    and normalised to unit DC gain. Output sample k describes the end of
    command interval k, rather than the physical current at its left edge.
    """
    tau, step = float(tau_s), float(raster_s)
    if not np.isfinite([tau, step]).all() or tau <= 0 or step <= 0:
        raise ValueError("tau_s and raster_s must be positive and finite.")
    kernel_time = np.arange(0., 10. * tau, step)
    kernel = np.exp(-kernel_time / tau)
    kernel /= kernel.sum()
    return kernel


def _positive_scalar(value, name):
    if np.ndim(value) != 0 or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite scalar.")
    return result


def solve_field_shim_waveforms(model, *, total_current_limit_A=30.,
                             raster_s=10e-6, max_iterations=80,
                             static_ridge=1e-6, dynamic_ridge=1e-3,
                             min_update_interval_s=8e-6):
    """Compute static-QP and response-aware dynamic-FISTA B0 waveforms.

    Returns independent ``static`` and ``dynamic`` entries containing commands
    ``currents_A[channel, state]`` and RF-relative ``state_edges_rf_s``. The
    one-state static command and uniformly sampled dynamic command cover the
    same window as the EPI model. The input model is never modified.

    A zero channel bound disables that channel before fitting; the returned
    arrays retain every model channel with exact zeros in disabled rows.
    The upstream APIs have one shared active-channel bound and one shared
    SIRF; heterogeneous active limits or time constants are rejected rather
    than silently approximated. Commands retain a tiny strict-L1 margin.
    The dynamic objective includes the known orthogonal B0 residual in reported
    full-field NRMSE, although the solver operates only on the exact QR basis.
    """
    from _vendor.acdc_optim import (solve_shim_static_qp,
                                   solve_shim_waveform_fista_torch)
    from _vendor.acdc_optim.ACDC_optimization import _sirf_conv1d_numpy

    requested_raster = _positive_scalar(raster_s, "raster_s")
    minimum_raster = _positive_scalar(min_update_interval_s, "min_update_interval_s")
    total_limit = _positive_scalar(total_current_limit_A, "total_current_limit_A")
    if (isinstance(max_iterations, (bool, np.bool_)) or not np.isscalar(max_iterations)
            or not np.isfinite(max_iterations) or int(max_iterations) != max_iterations
            or max_iterations < 1):
        raise ValueError("max_iterations must be a positive integer.")
    ridge_static = _positive_scalar(static_ridge, "static_ridge")
    ridge_dynamic = _positive_scalar(dynamic_ridge, "dynamic_ridge")
    if getattr(model, "sequence_kind", "epi") != "epi":
        raise ValueError("Field shim waveforms currently require a single-RF EPI acquisition.")

    nc = int(model.nc)
    quality = np.asarray(model.quality_mask, dtype=bool)
    b0 = np.asarray(model.b0_numpy, dtype=np.float64)
    maps = np.asarray(model.maps_numpy, dtype=np.float64)
    if (quality.shape != b0.shape or maps.shape != b0.shape + (nc,)
            or not quality.any() or not np.isfinite(b0[quality]).all()
            or not np.isfinite(maps[quality]).all()):
        raise ValueError("Need finite B0 and center ACDC maps on a nonempty common quality mask.")
    raw_limits = _numpy(model.max_current_A)
    if np.iscomplexobj(raw_limits) or raw_limits.dtype.kind == "b":
        raise ValueError("Channel current limits must be real nonnegative finite values.")
    limits = raw_limits.astype(np.float64).reshape(-1)
    if limits.shape != (nc,) or not np.isfinite(limits).all() or np.any(limits < 0):
        raise ValueError("Channel current limits must be real nonnegative finite values.")
    active = limits > 0
    active_count = int(active.sum())
    active_limit = float(limits[active][0]) if active_count else 0.
    if not np.allclose(limits[active], active_limit, rtol=0, atol=1e-12):
        raise ValueError("The pinned ACDC_Optim solver requires identical positive channel current limits among active channels.")
    response = model.current_response
    if not response.get("enabled", False):
        raise ValueError("Dynamic field shimming requires the configured amplifier response to be enabled.")
    tau = np.asarray(response["tau_s"], dtype=np.float64).reshape(-1)
    if tau.size == 1:
        tau = np.repeat(tau, nc)
    if tau.shape != (nc,) or not np.isfinite(tau).all() or np.any(tau <= 0):
        raise ValueError("The pinned ACDC_Optim solver requires positive finite amplifier time constants.")
    active_tau = float(tau[active][0]) if active_count else float(tau[0])
    if not np.allclose(tau[active], active_tau, rtol=0, atol=1e-15):
        raise ValueError("The pinned ACDC_Optim solver requires a common amplifier time constant across channels.")
    starts = np.asarray(model.line_start_s, dtype=np.float64).reshape(-1)
    ends = np.asarray(model.line_end_s, dtype=np.float64).reshape(-1)
    if (starts.size < 1 or starts.shape != ends.shape or not np.isfinite(starts).all()
            or not np.isfinite(ends).all() or np.any(starts < 0)
            or np.any(ends <= starts) or np.any(np.diff(starts) <= 0)
            or np.any(ends[:-1] > starts[1:] + 1e-12)):
        raise ValueError("Need finite ordered EPI command windows.")
    start, end = float(starts[0]), float(ends[-1])
    duration = end - start
    count = max(1, int(np.ceil(duration / requested_raster - 1e-12)))
    step = duration / count
    if step + 1e-12 < minimum_raster:
        raise ValueError("The duration-preserving dynamic raster is below min_update_interval_s.")
    edges = np.linspace(start, end, count + 1)
    kernel = amplifier_sirf(active_tau, step)

    target = -b0[quality]
    active_maps = maps[quality][:, active]
    if active_count and np.any(active_maps):
        compressed_maps, compressed_target, compression_report = compress_field_target(active_maps, target)
    else:
        # No actuator can affect the target. Keep its entire energy in the
        # orthogonal residual, without calling a zero-dimensional vendor fit.
        compressed_maps = np.zeros((active_count, 0))
        compressed_target = np.zeros(0)
        compression_report = {
            "method": "no nonzero active coil maps; exact zero-current solution",
            "original_spatial_points": int(target.size), "solver_spatial_points": 0,
            "target_squared_norm_Hz2": float(target @ target),
            "orthogonal_target_squared_norm_Hz2": float(target @ target),
        }
    solver_total_limit = total_limit * (1. - 1e-9)
    target_norm2 = compression_report["target_squared_norm_Hz2"]
    orthogonal_norm2 = compression_report["orthogonal_target_squared_norm_Hz2"]
    device = str(model.device)

    def full_nrmse(reduced_residual, frames=1):
        residual2 = float(np.square(reduced_residual).sum()) + frames * orthogonal_norm2
        denominator2 = frames * target_norm2
        return float(np.sqrt(max(0., residual2) / denominator2)) if denominator2 else float(np.sqrt(max(0., residual2)))

    # A target orthogonal to all coil maps also has an exact zero-current
    # solution; upstream 0/0 diagnostics must not stop its export.
    if float(compressed_target @ compressed_target) == 0.:
        static_currents = np.zeros(active_count)
        dynamic_currents = np.zeros((count, active_count))
        static_solver_nrmse = dynamic_solver_nrmse = 0.
        dynamic_loss = 0.
        static_seconds = dynamic_seconds = 0.
    else:
        started = time.perf_counter()
        static_currents, static_solver_nrmse = solve_shim_static_qp(
            compressed_target, compressed_maps, amp_limit=active_limit,
            l1_limit=solver_total_limit, ridge_x=ridge_static, ridge_u=1e-8,
            verbose=False)
        static_seconds = time.perf_counter() - started
        # This has at most Nchannels spatial columns, rather than Nvoxels.
        compressed_timecourse = np.broadcast_to(compressed_target, (count, compressed_target.size)).copy()
        started = time.perf_counter()
        dynamic_currents, dynamic_solver_nrmse, dynamic_loss = solve_shim_waveform_fista_torch(
            compressed_timecourse, compressed_maps, sirf_kernel=kernel,
            amp_limit=active_limit, l1_limit=solver_total_limit,
            ridge_x=ridge_dynamic, max_iter=int(max_iterations), device=device,
            verbose=False, plot=False)
        dynamic_seconds = time.perf_counter() - started

    # Numerical QP roundoff must not cross a strict export L1 constraint.
    def retain_margin(currents):
        commands = np.asarray(currents, dtype=np.float64)
        if not active_count:
            return commands
        ratios = np.maximum(np.max(np.abs(commands), axis=-1) / active_limit,
                            np.sum(np.abs(commands), axis=-1) / solver_total_limit)
        return commands / np.maximum(1., ratios)[..., None]

    static_currents = retain_margin(np.asarray(static_currents)[None])[0]
    dynamic_currents = retain_margin(dynamic_currents)
    dynamic_actual = (_sirf_conv1d_numpy(dynamic_currents, kernel) if active_count
                      else np.zeros_like(dynamic_currents))
    static_residual = static_currents @ compressed_maps - compressed_target
    dynamic_residual = dynamic_actual @ compressed_maps - compressed_target
    common = {
        "target": "negative constant measured/model B0 on reliable center voxels",
        "amp_limit_A": active_limit, "total_current_limit_A": total_limit,
        "channel_limits_A": limits.tolist(),
        "active_model_channels_1based": (np.flatnonzero(active) + 1).tolist(),
        "disabled_model_channels_1based": (np.flatnonzero(~active) + 1).tolist(),
        "disabled_channels_excluded_before_fitting": True,
        "solver_skipped": bool(float(compressed_target @ compressed_target) == 0.),
        "solver_total_current_limit_A": solver_total_limit,
        "source_field_voxels": int(quality.sum()),
        "start_rf_s": start, "end_rf_s": end,
        "initial_current_assumption": "zero before first command",
    }
    static_report = dict(common, solver="ACDC_Optim.solve_shim_static_qp",
                         ridge_x=ridge_static, ridge_u=1e-8,
                         solver_reduced_nrmse=float(static_solver_nrmse),
                         full_field_nrmse=full_nrmse(static_residual),
                         field_evaluation="steady state; amplifier transient checked separately",
                         seconds=static_seconds)
    dynamic_report = dict(common, solver="ACDC_Optim.solve_shim_waveform_fista_torch",
                          ridge_x=ridge_dynamic, max_iterations=int(max_iterations),
                          device=device, requested_raster_s=requested_raster,
                          raster_s=step, min_update_interval_s=minimum_raster,
                          states=count, tau_s=active_tau,
                          sirf_samples=int(kernel.size), sirf_kernel=kernel.tolist(),
                          sirf_definition="exp(-k*dt/tau), truncated at 10*tau, normalised to unit DC gain",
                          sirf_sample_timing="current at the end of each command interval",
                          solver_reduced_nrmse=float(dynamic_solver_nrmse),
                          full_field_nrmse=full_nrmse(dynamic_residual, count),
                          solver_loss=float(dynamic_loss), seconds=dynamic_seconds)
    full_static = np.zeros((nc, 1))
    full_dynamic = np.zeros((nc, count))
    full_static[active, 0] = static_currents
    full_dynamic[active] = dynamic_currents.T
    return {
        "static": {"currents_A": full_static,
                   "state_edges_rf_s": np.array([start, end]), "report": static_report},
        "dynamic": {"currents_A": full_dynamic,
                    "state_edges_rf_s": edges, "report": dynamic_report},
        "report": {
            "upstream_url": UPSTREAM_URL, "upstream_commit": UPSTREAM_COMMIT,
            "upstream_source": "vendored unmodified, MIT licence in _vendor/acdc_optim/LICENSE",
            "compression": compression_report,
            "dynamic_interpretation": "constant B0 cancellation with amplifier-response pre-emphasis",
            "continuous_hardware_validation": "performed after solving, on exported command timing",
        },
    }
