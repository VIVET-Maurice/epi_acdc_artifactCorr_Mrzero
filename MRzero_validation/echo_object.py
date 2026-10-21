"""Estimate an effective object from native multi-echo GRE receiver images.

Fit on the native B0 grid BEFORE averaging voxels into an EPI parent voxel.
Differences between native B0 values can then be simulated explicitly by the
sub-spins without also fitting that coarse-voxel dephasing into a decay rate.
The fitted rate still includes unresolved dephasing inside a native voxel.
It is an effective R2*, not an intrinsic T2 measurement. The fitted amplitude
also contains GRE saturation and receive sensitivity; it is not proton density.
"""

from __future__ import annotations

import numpy as np


# Bit flags are retained per voxel so a zero-decay fallback is inspectable.
OUTSIDE_SUPPORT = 1
NO_NOISE_CALIBRATION = 2
LOW_SNR = 4
NONEXPONENTIAL = 8
TARGET_UNCERTAIN = 16
INVALID_DATA = 32


def _mask(value, shape, name):
    if value is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(value)
    if result.shape != shape or not np.isin(result, (0, 1)).all():
        raise ValueError(f"{name} must be a binary mask with image shape {shape}.")
    return result.astype(bool)


def _noise_covariance(coils, support, noise_mask, noise_std, covariance):
    """C = E[n n^H]; noise_std denotes one real/imaginary component's std."""
    channels = coils.shape[1]
    supplied = sum(value is not None for value in (noise_mask, noise_std, covariance))
    if supplied > 1:
        raise ValueError("Supply one of noise_mask, noise_std or noise_covariance.")
    report = {"calibrated": supplied > 0, "sample_count": 0,
              "proper_complex_Gaussian_assumption": True}
    if covariance is not None:
        C = np.asarray(covariance, dtype=complex)
        report["method"] = "Supplied complex receiver-noise covariance."
    elif noise_std is not None:
        sigma = np.asarray(noise_std, dtype=float)
        if sigma.size == 1:
            sigma = np.full(channels, sigma.item())
        if sigma.shape != (channels,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("noise_std must be positive scalar or one real/imaginary std per receiver.")
        C = np.diag(2 * sigma**2).astype(complex)
        report["method"] = "Supplied independent real/imaginary receiver-noise std."
    elif noise_mask is not None:
        background = _mask(noise_mask, coils.shape[-2:], "noise_mask")
        if np.any(background & support):
            raise ValueError("noise_mask must be outside support_mask and contain signal-free background.")
        samples = coils[:, :, background].transpose(1, 0, 2).reshape(channels, -1)
        samples = samples[:, np.isfinite(samples).all(axis=0)]
        if samples.shape[1] < max(32, 2 * channels):
            raise ValueError("Too few finite background samples to estimate receiver-noise covariance.")
        samples = samples - samples.mean(axis=1, keepdims=True)
        C = samples @ samples.conj().T / (samples.shape[1] - 1)
        report.update(method="Complex covariance across explicitly signal-free background pixels and echoes.",
                      sample_count=int(samples.shape[1]))
    else:
        C = np.zeros((channels, channels), dtype=complex)
        report["method"] = "Unavailable: raw fit is diagnostic; simulation uses explicit no-decay fallback."
    if (C.shape != (channels, channels) or not np.isfinite(C).all()
            or not np.allclose(C, C.conj().T, rtol=1e-8, atol=1e-12)):
        raise ValueError("noise_covariance must be a finite Hermitian receiver-by-receiver matrix.")
    eigenvalues = np.linalg.eigvalsh(C)
    if np.min(eigenvalues) < -1e-10 * max(float(np.max(np.abs(eigenvalues))), 1e-30):
        raise ValueError("noise_covariance must be positive semidefinite.")
    trace = float(np.trace(C).real)
    if supplied and trace <= 0:
        raise ValueError("Calibrated receiver-noise power must be positive.")
    report.update(receiver_count=channels, RSS_noise_power=trace,
                  covariance_eigenvalues=eigenvalues.tolist(),
                  spatial_noise_independence_not_estimated=True)
    return C, report


def _fit(amplitude, variance, times, calibrated, min_snr, max_chi2, target, max_log_std):
    """Weighted log-linear fit, vectorized over the final (voxel) axis."""
    x = (times - times[0])[:, None]
    usable = np.isfinite(amplitude).all(axis=0) & (amplitude > 0).all(axis=0)
    safe_amplitude = np.maximum(amplitude, np.finfo(float).tiny)
    log_signal = np.log(safe_amplitude)
    if calibrated:
        log_variance = variance / safe_amplitude**2
        weights = np.divide(1., log_variance, out=np.zeros_like(log_variance),
                            where=np.isfinite(log_variance) & (log_variance > 0))
        snr = np.divide(amplitude, np.sqrt(variance), out=np.zeros_like(amplitude),
                        where=np.isfinite(variance) & (variance > 0))
    else:
        weights = np.ones_like(amplitude)
        snr = np.full_like(amplitude, np.nan)
    weight_sum = weights.sum(axis=0)
    weight_sum = np.maximum(weight_sum, np.finfo(float).tiny)
    x_mean = (weights * x).sum(axis=0) / weight_sum
    y_mean = (weights * log_signal).sum(axis=0) / weight_sum
    information = (weights * (x - x_mean)**2).sum(axis=0)
    information = np.maximum(information, np.finfo(float).tiny)
    slope = (weights * (x - x_mean) * (log_signal - y_mean)).sum(axis=0) / information
    unbounded_rate = -slope
    rate = np.maximum(unbounded_rate, 0.)
    intercept = y_mean + rate * x_mean  # Refit amplitude at the R2*=0 boundary.
    residual = log_signal - (intercept - rate * x)
    reduced_chi2 = (weights * residual**2).sum(axis=0) / (len(times) - 2)
    # Known image noise supplies the variance. Residual model mismatch can
    # increase uncertainty, but a fortuitously exact three-point fit cannot
    # shrink it below that independently estimated noise floor.
    variance_scale = np.maximum(reduced_chi2, 1.)
    rate_std = np.sqrt(variance_scale / information)
    target_x = target - times[0]
    target_log_std = np.sqrt(variance_scale * (
        1. / weight_sum + (target_x - x_mean)**2 / information))
    anchor_log_std = np.sqrt(variance_scale * (1. / weight_sum + x_mean**2 / information))
    fit_valid = usable & (snr.min(axis=0) >= min_snr) & (reduced_chi2 <= max_chi2)
    reliable = fit_valid & (target_log_std <= max_log_std) & calibrated
    if not calibrated:
        rate_std[:] = np.nan
        target_log_std[:] = np.nan
        anchor_log_std[:] = np.nan
    # These are diagnostic values even where the fit fails quality checks.
    amplitude_anchor = np.exp(np.clip(intercept, -745, 709))
    amplitude_zero = np.exp(np.clip(intercept + rate * times[0], -745, 709))
    prediction = np.exp(np.clip(intercept - rate * target_x, -745, 709))
    return {"R2star_sInv": rate, "unbounded_R2star_sInv": unbounded_rate,
            "R2star_std_sInv": rate_std,
            "fitted_amplitude_at_reference": amplitude_anchor,
            "effective_amplitude_TE0": amplitude_zero,
            "fitted_amplitude_at_target": prediction,
            "amplitude_reference_log_std": anchor_log_std,
            "target_log_std": target_log_std,
            "reduced_chi2": reduced_chi2, "SNR": snr,
            "fit_valid_mask": fit_valid, "target_reliable_mask": reliable,
            "finite_positive_echo_mask": usable}


def fit_echo_object(coil_images, echo_times_s, *, support_mask=None,
                    noise_mask=None, noise_std=None, noise_covariance=None,
                    target_te_s=None, min_snr=5.,
                    max_target_log_std=np.log(2.) / 1.96,
                    max_reduced_chi2=9., pooled_mask=None,
                    pooled_effective_voxel_count=1.):
    """Fit native ``[echo, receiver, row, column]`` complex GRE images.

    RSS power is debiased using the supplied/estimated receiver-noise
    covariance. A high-SNR weighted log-linear fit imposes R2* >= 0. Its
    uncertainty is propagated to ``target_te_s`` (default: last measured TE).
    The default acceptance threshold is a factor of two for the approximate
    95% interval of the predicted amplitude. This is a configurable precision
    criterion, not a physical T2* prior. Three early echoes cannot establish
    that a monoexponential remains valid at a distant EPI TE.

    No rate is silently filled in: ``simulation_R2star_sInv`` is zero wherever
    the target prediction is unreliable, and its amplitude is the first-echo
    amplitude anchored at ``amplitude_reference_time_s``. Inspect the fallback
    bit flags before interpreting results. ``noise_mask`` must be explicitly
    signal-free; the complement of an arbitrary object ROI is not assumed to
    be background. Covariance and RSS uncertainty assume proper complex noise.

    ``pooled`` fits the ROI mean RSS echoes, with their mean variance divided
    by ``pooled_effective_voxel_count``. The conservative default is one
    effective voxel: spatial correlations have not been measured. Increase it
    only with a justified effective sample count. Pooled rate heterogeneity
    and GRE-to-EPI steady-state differences remain modelling assumptions.
    """
    coils = np.asarray(coil_images)
    times = np.asarray(echo_times_s, dtype=float)
    if (coils.ndim != 4 or coils.shape[0] < 3 or min(coils.shape[1:]) < 1
            or not np.iscomplexobj(coils) or times.shape != (coils.shape[0],)
            or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("Need >=3 complex receiver echoes and distinct increasing nonnegative TE_s.")
    for name, value in (("min_snr", min_snr), ("max_target_log_std", max_target_log_std),
                        ("max_reduced_chi2", max_reduced_chi2)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    target = float(times[-1] if target_te_s is None else target_te_s)
    if not np.isfinite(target) or target < 0:
        raise ValueError("target_te_s must be finite and nonnegative.")
    shape = coils.shape[-2:]
    support = _mask(support_mask, shape, "support_mask")
    C, noise_report = _noise_covariance(coils, support, noise_mask, noise_std, noise_covariance)
    calibrated = noise_report["calibrated"]
    finite = np.isfinite(coils).all(axis=(0, 1))
    safe_coils = np.where(np.isfinite(coils), coils, 0).astype(complex)
    power = np.sum(abs(safe_coils)**2, axis=1)
    noise_power = float(np.trace(C).real)
    amplitude = np.sqrt(np.maximum(power - noise_power, 0.))
    # For proper complex noise, Var(||s+n||²)=tr(C²)+2 s^H C s.
    # y^H C y contains one extra tr(C²); remove it in this estimate.
    trace_squared = float(np.sum(abs(C)**2))
    quadratic = np.einsum("ecyx,cd,edyx->eyx", safe_coils.conj(), C, safe_coils,
                          optimize=True).real
    power_variance = np.maximum(2 * quadratic - trace_squared, trace_squared)
    variance = np.divide(power_variance, 4 * amplitude**2,
                         out=np.full_like(amplitude, np.inf), where=amplitude > 0)
    fitted = _fit(amplitude.reshape(len(times), -1), variance.reshape(len(times), -1),
                  times, calibrated, min_snr, max_reduced_chi2, target, max_target_log_std)
    result = {key: value.reshape((len(times),) + shape) if key == "SNR"
              else value.reshape(shape) for key, value in fitted.items()}
    result["fit_valid_mask"] &= support & finite
    result["target_reliable_mask"] &= support & finite
    reliable = result["target_reliable_mask"]
    fallback = support & ~reliable
    reasons = np.zeros(shape, dtype=np.uint8)
    reasons[~support] |= OUTSIDE_SUPPORT
    if not calibrated:
        reasons[support] |= NO_NOISE_CALIBRATION
    else:
        reasons[support & (result["SNR"].min(axis=0) < min_snr)] |= LOW_SNR
        reasons[support & (result["reduced_chi2"] > max_reduced_chi2)] |= NONEXPONENTIAL
        reasons[support & (result["target_log_std"] > max_target_log_std)] |= TARGET_UNCERTAIN
    reasons[support & (~finite | ~result["finite_positive_echo_mask"])] |= INVALID_DATA
    first = np.where(support & finite, amplitude[0], 0.)
    simulation_rate = np.where(reliable, result["R2star_sInv"], 0.)
    simulation_amplitude = np.where(reliable, result["fitted_amplitude_at_reference"], first)
    rate = result["R2star_sInv"]
    result.update(echo_times_s=times.copy(), amplitude_reference_time_s=float(times[0]),
        target_te_s=target, first_echo_amplitude=first,
        raw_RSS=np.sqrt(power), debiased_RSS=amplitude, echo_amplitude_variance=variance,
        simulation_amplitude_at_reference=simulation_amplitude,
        simulation_R2star_sInv=simulation_rate, fallback_mask=fallback,
        fallback_reason_flags=reasons, noise_covariance=C,
        T2star_s=np.divide(1., rate, out=np.full(shape, np.inf), where=rate > 0),
        amplitude_at_target=np.where(reliable, result["fitted_amplitude_at_target"], np.nan))

    # Pool only independently usable native voxels, not invalid/zero echoes.
    pool = support & finite & result["finite_positive_echo_mask"]
    if pooled_mask is not None:
        pool &= _mask(pooled_mask, shape, "pooled_mask")
    if (not np.isscalar(pooled_effective_voxel_count)
            or not np.isfinite(pooled_effective_voxel_count) or pooled_effective_voxel_count < 1
            or pooled_effective_voxel_count > max(1, int(pool.sum()))):
        raise ValueError("pooled_effective_voxel_count must be between one and the pooled voxel count.")
    if pool.any():
        mean_amplitude = amplitude[:, pool].mean(axis=1)
        mean_variance = variance[:, pool].mean(axis=1) / pooled_effective_voxel_count
        pooled_fit = _fit(mean_amplitude[:, None], mean_variance[:, None], times,
                          calibrated, min_snr, max_reduced_chi2, target, max_target_log_std)
        pooled = {key: value[:, 0].tolist() if key == "SNR" else value[0].item()
                  for key, value in pooled_fit.items()}
        pooled.update(voxel_count=int(pool.sum()), effective_voxel_count=float(pooled_effective_voxel_count),
                      mean_RSS_echoes=mean_amplitude.tolist(),
                      assumption="Single exponential for ROI mean; heterogeneity and spatial-noise correlation are not inferred by this fit.")
    else:
        pooled = {"voxel_count": 0, "target_reliable_mask": False,
                  "reason": "No finite positive native echo trains in pooled ROI."}
    result["pooled"] = pooled
    result["report"] = {"native_image_shape": list(shape), "echo_times_s": times.tolist(),
        "echo_span_s": float(times[-1] - times[0]), "target_te_s": target,
        "target_distance_beyond_last_echo_s": max(0., target - float(times[-1])),
        "support_voxels": int(support.sum()), "fit_valid_voxels": int(result["fit_valid_mask"].sum()),
        "target_reliable_voxels": int(reliable.sum()), "fallback_voxels": int(fallback.sum()),
        "min_snr": float(min_snr), "max_reduced_chi2": float(max_reduced_chi2),
        "max_target_log_std": float(max_target_log_std),
        "noise": noise_report, "pooled": pooled,
        "amplitude_model": "Effective GRE steady-state, receive-weighted transverse amplitude; not PD/T1 corrected.",
        "rate_model": "Effective native-voxel R2*, including unresolved dephasing; not intrinsic T2.",
        "resolved_B0_policy": "Fit native echoes before parent aggregation; simulate B0 differences between native positions separately.",
        "fallback_policy": "First-echo amplitude at first TE, zero additional decay, explicitly flagged.",
        "target_interval_caveat": "Statistical log-linear uncertainty does not validate long-TE monoexponential extrapolation.",
        "reason_flag_values": {"outside_support": OUTSIDE_SUPPORT, "noise_uncalibrated": NO_NOISE_CALIBRATION,
            "low_snr": LOW_SNR, "nonexponential": NONEXPONENTIAL,
            "target_uncertain": TARGET_UNCERTAIN, "invalid_data": INVALID_DATA}}
    return result


__all__ = ["fit_echo_object"]
