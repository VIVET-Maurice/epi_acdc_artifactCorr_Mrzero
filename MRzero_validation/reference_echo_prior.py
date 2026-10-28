"""Combine registered GRE anatomy and calibrated complex-echo amplitudes.

A single GRE is not a proton-density or T1 measurement. Its measured contrast
is transferred to the multi-echo reference TE with the fitted echo envelope,
then only a global receive/sequence gain is estimated. Differences in TR,
flip angle, receiver profiles and resolution remain possible model errors.
The automatic and joint modes retain the native complex-echo model unless a
separately trained reference candidate predicts reserved measured echoes better.
Joint mode estimates a smooth, positive acquisition-contrast correction using
training echoes only. It never labels the measured GRE as a PD or T1 map.
"""
from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from resample_acdc_maps import resample_to_grid


def _envelope_kernel(result, times):
    """Complex signed kernel, including known and unresolved coherence."""
    times = np.asarray(times, float)
    delta = times[:, None, None] - result["amplitude_reference_time_s"]
    rate, field = result["residual_R2_sInv"], result["refined_b0_Hz"]
    kernel = np.exp((-rate[None] - 2j*np.pi*field[None]) * delta)
    gradient = np.asarray(result["known_inplane_gradient_Hz_per_pixel"])
    t = times[:, None, None]
    coherence = np.prod(np.sinc(t[..., None] * gradient[None]), axis=-1)
    width, kind = result["unresolved_width_Hz"], result["dispersion_kind"]
    coherence *= np.where(kind[None] == 1, np.sinc(t * width[None]),
        np.where(kind[None] == 2, np.exp(-2*np.pi**2*(t*width[None])**2), 1.))
    return kernel * coherence, np.abs(coherence)


def _reference_envelope(reference_on_native, reference_valid, result, reference_te,
                        minimum_coherence):
    kernel, coherence = _envelope_kernel(result, [reference_te])
    factor = np.abs(kernel[0])
    valid = (reference_valid & result["support_mask"]
             & (coherence[0] >= minimum_coherence) & (factor >= minimum_coherence)
             & np.isfinite(factor) & (reference_on_native > 0)
             & (result["amplitude_envelope_at_reference"] > 0))
    envelope = np.divide(reference_on_native, factor,
        out=np.zeros_like(factor), where=valid)
    return envelope, valid


def _joint_reference_envelope(reference_on_native, reference_valid, result,
                              reference_te, minimum_factor):
    """Transfer exponential contrast without undoing native-voxel sinc loss.

    Reference and multi-echo pixels need not have the same size. In particular,
    a measured 64x64 GRE cannot be inverted using the coherence of a fitted
    128x128 voxel. The smooth contrast field absorbs only the identifiable
    low-frequency acquisition difference; reserved echoes assess the residual
    effective-voxel approximation.
    """
    factor = np.exp(-np.asarray(result["residual_R2_sInv"])
                    * (reference_te - result["amplitude_reference_time_s"]))
    valid = (reference_valid & result["support_mask"] & np.isfinite(factor)
             & (factor >= minimum_factor) & (reference_on_native > 0)
             & np.isfinite(result["amplitude_envelope_at_reference"])
             & (result["amplitude_envelope_at_reference"] > 0))
    envelope = np.divide(reference_on_native, factor,
        out=np.zeros_like(factor), where=valid)
    return envelope, valid


def _smooth_contrast_field(reference, valid, result, X, Y, sigma_mm):
    """Masked Gaussian smoothing of a log ratio in physical millimetres.

    The geometric mean on measured coverage is fixed to one; the independent
    global gain subsequently calibrates receiver/acquisition intensity units.
    Invalid pixels never contribute and receive neutral field one. Gaussian
    mask normalization prevents a fictitious zero-contrast boundary.
    """
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    coordinates = np.stack((X, Y), axis=-1)
    if (X.ndim != 2 or min(X.shape) < 2 or Y.shape != X.shape
            or valid.shape != X.shape or not np.isfinite(coordinates).all()):
        raise ValueError("Joint contrast needs matching finite native physical grids.")
    row, column = coordinates[1, 0]-coordinates[0, 0], coordinates[0, 1]-coordinates[0, 0]
    rows, columns = np.indices(X.shape)
    regular = coordinates[0, 0] + rows[..., None]*row + columns[..., None]*column
    spacing_mm = 1000*np.array([np.linalg.norm(row), np.linalg.norm(column)])
    if (not np.isfinite(spacing_mm).all() or np.any(spacing_mm <= 0)
            or not np.allclose(coordinates, regular, rtol=0, atol=1e-10)
            or abs(np.dot(row, column)) > 1e-8*np.linalg.norm(row)*np.linalg.norm(column)):
        raise ValueError("Joint contrast needs a regular orthogonal native physical grid.")
    sigma_pixels = sigma_mm / spacing_mm
    logs = np.zeros(X.shape)
    logs[valid] = (np.log(result["amplitude_envelope_at_reference"][valid])
                   - np.log(reference[valid]))
    weight = gaussian_filter(valid.astype(float), sigma_pixels, mode="constant", cval=0.)
    numerator = gaussian_filter(logs, sigma_pixels, mode="constant", cval=0.)
    smooth = np.divide(numerator, weight, out=np.zeros_like(logs), where=weight > 0)
    centered = smooth[valid] - np.mean(smooth[valid])
    field = np.ones(X.shape)
    with np.errstate(over="ignore", under="ignore"):
        field[valid] = np.exp(centered)
    if not np.isfinite(field).all() or np.any(field <= 0):
        raise ValueError("The joint acquisition-contrast field is not finite and positive.")
    return field, {
        "contrast_sigma_mm": float(sigma_mm),
        "contrast_sigma_pixels_row_column": sigma_pixels.tolist(),
        "native_spacing_mm_row_column": spacing_mm.tolist(),
        "contrast_training_voxels": int(valid.sum()),
        "contrast_field_min": float(field[valid].min()),
        "contrast_field_max": float(field[valid].max()),
        "contrast_field_geometric_mean": float(np.exp(np.mean(np.log(field[valid])))),
        "contrast_field_policy": "Positive normalized masked-Gaussian log-ratio; unit geometric mean on common training coverage; global gain estimated separately.",
    }


def _global_gain(reference, valid, result, times, whitener):
    amplitude = np.asarray(result["amplitude_envelope_at_reference"])
    receivers = np.moveaxis(result["complex_receiver_amplitude"], 0, -1)
    whitened = receivers @ whitener.T
    kernel, _ = _envelope_kernel(result, times)
    # The complex variable-projection receiver amplitudes are the optimum at
    # fixed temporal parameters. These weights give the same global gain as
    # least squares against the training complex observations.
    weight = (np.sum(np.abs(whitened)**2, axis=-1)
              * np.sum(np.abs(kernel)**2, axis=0))
    weight = np.divide(weight, amplitude**2, out=np.zeros_like(weight), where=amplitude > 0)
    numerator = np.sum(weight[valid] * reference[valid] * amplitude[valid])
    denominator = np.sum(weight[valid] * reference[valid]**2)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Reference amplitude has no positive finite weighted energy.")
    return max(0., float(numerator / denominator))


def apply_reference_amplitude_prior(inputs, fit, *, source="auto", device="auto",
                                    require_cuda=False, validation_echo_count=6,
                                    candidate_weights=(0., .25, .5, 1.),
                                    min_relative_validation_improvement=.02,
                                    min_noise_sse_improvement=1.,
                                    minimum_coherence=.2, min_common_voxels=16,
                                    fit_options=None, contrast_sigma_mm=10.):
    """Return an independent fit with a selectable GRE amplitude prior.

    ``source='echo32'`` preserves the native fit. ``'auto'`` evaluates blend
    weights on the last six complex echoes after independently fitting only
    the preceding echoes; it changes amplitude only after the configured
    relative and noise-weighted SSE improvements. ``'reference'`` explicitly
    selects the full reference prior and still reports that validation.
    ``'joint'`` calibrates a positive smooth GRE-to-multi-echo contrast field
    from those training echoes before selecting the reference blend by the
    same reserved-echo gate. ``contrast_sigma_mm`` is its physical Gaussian
    standard deviation. The field is frozen before reserved-echo scoring and
    reused unchanged with the final all-echo fit. Joint mode transfers only
    exponential TE contrast; it never divides a lower-resolution GRE by a
    native multi-echo voxel's sinc or unresolved coherence.

    Registered reference X_m/Y_m are sampled on the native measured echo grid
    without spatial extrapolation. No support or B0 is invented; outside the
    common coverage or near a fitted coherence null the original amplitudes
    remain. Interpolating a 64x64 reference does not add measured 128x128 detail.
    Temporal parameters and receiver phases/directions always come from the
    complex echo acquisition. The selected blend is finally reapplied to its
    all-echo fit, retaining both its decay and dispersion losses.

    The reserved echoes select this optional prior; the score is internal
    model selection, not independent scanner validation or an uncertainty bound.
    """
    started = perf_counter()
    if source not in ("echo32", "auto", "reference", "joint"):
        raise ValueError("source must be 'echo32', 'auto', 'reference', or 'joint'.")
    sigma_value = np.asarray(contrast_sigma_mm)
    if (sigma_value.ndim != 0 or not np.issubdtype(sigma_value.dtype, np.number)
            or np.iscomplexobj(sigma_value) or isinstance(contrast_sigma_mm, (bool, np.bool_))
            or not np.isfinite(sigma_value) or sigma_value <= 0):
        raise ValueError("contrast_sigma_mm must be a finite positive scalar in millimetres.")
    if "multi_echo" not in fit:
        raise ValueError("Reference prior requires the complex multi-echo fit first.")
    for name, value in (("minimum_coherence", minimum_coherence),
                        ("min_relative_validation_improvement", min_relative_validation_improvement)):
        if not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be finite and in [0,1).")
    if minimum_coherence == 0:
        raise ValueError("minimum_coherence must be positive to guard coherence nulls.")
    if not np.isfinite(min_noise_sse_improvement) or min_noise_sse_improvement < 0:
        raise ValueError("min_noise_sse_improvement must be finite and nonnegative.")
    for name, value in (("min_common_voxels", min_common_voxels),
                        ("validation_echo_count", validation_echo_count)):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    weights = np.asarray(candidate_weights, float)
    if weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("candidate_weights must be finite values in [0,1].")
    weights = np.unique(np.r_[0., weights, 1.])
    updated = deepcopy(fit)
    report = {
        "format": "registered_GRE_amplitude_prior_v1", "requested_source": source,
        "selected_source": "echo32", "selected_reference_weight": 0., "applied": False,
        "common_valid_voxels": 0,
        "amplitude_policy": "Receive-weighted reference-TE envelope; GRE is not a PD or T1 map.",
        "contrast_policy": "Fitted decay/coherence transfer TE contrast; one positive global gain absorbs acquisition scale, not spatial T1/TR/flip differences.",
        "geometry_policy": "Registered physical coordinates; no extrapolation, registration, support expansion, or invented fine-resolution detail.",
        "resolution_limitation": "TE transfer is an effective native-voxel prior. Different GRE voxel averaging/point-spread functions are not inverted; unresolved dispersion has no identified spatial direction or resolution scaling.",
        "temporal_policy": "Keep complex-echo B0, residual decay and unresolved dispersion unchanged.",
        "validation_policy": "Train temporal parameters, complex receiver amplitudes and reference gain without reserved echoes; select prior with reserved complex noise-weighted SSE; final amplitude reapplied to all-echo fit.",
        "validation_limitation": "Internal selection in the same acquisition; not independent EPI validation or a physical uncertainty bound.",
    }
    if source == "joint":
        report.update(
            format="registered_GRE_joint_contrast_prior_v1",
            contrast_sigma_mm=float(contrast_sigma_mm),
            contrast_policy="Positive smooth effective GRE-to-multi-echo acquisition contrast fitted from training complex-echo amplitudes only, plus one global intensity gain; no PD/T1/B1 inference.",
            resolution_limitation="GRE is sampled onto the native multi-echo grid without an inverse point-spread model. Only exponential TE contrast is transferred; no native-voxel sinc/coherence correction is asserted for a differently sized GRE voxel. Reserved echoes gate this effective-voxel prior.",
            reference_TE_transfer="Exponential envelope only; known and unresolved coherence are never divided out of GRE in joint mode.",
            common_domain_policy="Measured GRE coverage, positive native training amplitude and finite stable exponential TE transfer; no cross-resolution sinc-null mask is asserted.",
            contrast_uses_reserved_echoes=False,
            contrast_field_refitted_after_selection=False,
            joint_reference_role="Registered anatomy and receive-weighted amplitude prior, calibrated jointly with native complex-echo amplitudes; independent of the measured GRE optimization target.")

    def finish(reason):
        report["decision"] = reason
        report["elapsed_s"] = perf_counter() - started
        updated["report"]["reference_amplitude_prior"] = report
        updated["multi_echo"]["report"]["reference_amplitude_prior"] = deepcopy(report)
        if "multi_echo" in updated["report"]:
            updated["report"]["multi_echo"].update(deepcopy(updated["multi_echo"]["report"]))
        return updated

    if source == "echo32":
        return finish("Native complex-echo amplitude explicitly selected.")
    gre = inputs.get("gre", {})
    if not all(key in gre for key in ("magnitude", "X_m", "Y_m", "TE_s")):
        if source == "reference":
            raise ValueError("Reference prior needs magnitude, registered X_m/Y_m and measured TE_s.")
        return finish("Reference geometry or measured TE is missing; retained native amplitude.")
    reference_te = np.asarray(gre["TE_s"])
    times = np.asarray(inputs["echo_times_s"], float)
    if reference_te.ndim != 0 or np.iscomplexobj(reference_te) or not np.isfinite(reference_te):
        raise ValueError("Reference TE_s must be one finite real scalar.")
    reference_te = float(reference_te)
    if not 0 <= reference_te <= times[-1]:
        if source == "reference":
            raise ValueError("Reference TE must lie between zero and the final measured echo; temporal extrapolation of reference contrast is disabled.")
        return finish("Reference TE lies outside the supported range; retained native amplitude.")
    if len(times) - validation_echo_count < 6:
        raise ValueError("At least six training echoes and one reserved echo are required.")
    reference = np.asarray(gre["magnitude"])
    if np.iscomplexobj(reference) or reference.ndim != 2 or np.any(reference[np.isfinite(reference)] < 0):
        raise ValueError("Reference magnitude must be a nonnegative real image.")
    mask = np.isfinite(reference) & np.asarray(gre.get("valid_mask", np.ones(reference.shape, bool)), bool)
    native = inputs["b0_raw"]
    resampled, coverage = resample_to_grid(reference, gre["X_m"], gre["Y_m"], mask,
                                          native["X_m"], native["Y_m"])
    full = updated["multi_echo"]
    report.update(reference_shape=list(reference.shape), native_echo_shape=list(full["support_mask"].shape),
        reference_TE_s=reference_te, amplitude_reference_time_s=float(full["amplitude_reference_time_s"]),
        reference_TR_s=float(gre["TR_s"]) if "TR_s" in gre else None,
        reference_flip_deg=float(gre["flip_deg"]) if "flip_deg" in gre else None,
        minimum_allowed_coherence=float(minimum_coherence), spatial_coverage_voxels=int(coverage.sum()),
        minimum_relative_validation_improvement=float(min_relative_validation_improvement),
        minimum_noise_SSE_improvement=float(min_noise_sse_improvement))
    # First fit only training echoes: using full-fit temporal parameters here
    # would leak the held-out observations into the prior selection score.
    from complex_echo_fit import fit_complex_echoes
    ntrain = len(times) - int(validation_echo_count)
    options = dict(fit_options or {})
    forbidden = {"noise_covariance", "initial_b0_Hz", "support_mask", "known_inplane_gradient_Hz_per_pixel",
                 "amplitude_reference_time_s", "device", "require_cuda", "validation_echo_count"}
    if forbidden.intersection(options):
        raise ValueError("fit_options cannot override training data, geometry, noise, device or echo holdout.")
    # Reuse the independently prepared early-echo seed, not the full-fit refined
    # frequency, so the same frequency-alias neighborhood is searched without
    # incorporating any reserved-echo estimate into training initialization.
    initial_b0 = full.get("initial_b0_Hz", inputs["field"]["b0Hz"])
    report["validation_initial_B0_source"] = ("stored early-echo seed used by the principal complex fit"
        if "initial_b0_Hz" in full else "early measured B0 field (legacy fit without stored seed)")
    train = fit_complex_echoes(np.asarray(native["coil_images"])[:ntrain], times[:ntrain],
        noise_covariance=fit["noise_covariance"], initial_b0_Hz=initial_b0,
        support_mask=full["support_mask"],
        known_inplane_gradient_Hz_per_pixel=full["known_inplane_gradient_Hz_per_pixel"],
        amplitude_reference_time_s=full["amplitude_reference_time_s"],
        device=device, require_cuda=require_cuda,
        validation_echo_count=min(int(validation_echo_count), max(1, ntrain - 3)), **options)
    reference_envelope = _joint_reference_envelope if source == "joint" else _reference_envelope
    prior_train, valid_train = reference_envelope(resampled, coverage, train, reference_te, minimum_coherence)
    # The validation domain must also be determined without reserved echoes.
    # A full-fit coherence mask would otherwise leak test observations through
    # domain selection even though its parameters are not used for prediction.
    common = valid_train
    report.update(training_echo_indices=list(range(ntrain)), validation_echo_indices=list(range(ntrain, len(times))),
        validation_temporal_parameters_refitted_without_reserved_echoes=True,
        train_fit_device=train["report"]["device"], validation_common_voxels=int(common.sum()),
        validation_domain_uses_only_training_parameters=True,
        excludes_reference_coherence_nulls=source != "joint")
    if common.sum() < min_common_voxels:
        if source == "reference":
            raise ValueError(f"Reference has only {int(common.sum())} common stable-coherence voxels; minimum is {min_common_voxels}.")
        return finish("Insufficient shared positive finite reference-envelope coverage; retained native amplitude." if source == "joint"
                      else "Insufficient shared, stable-coherence reference coverage; retained native amplitude.")
    if source == "joint":
        contrast, contrast_report = _smooth_contrast_field(prior_train, common, train,
            native["X_m"], native["Y_m"], float(contrast_sigma_mm))
        prior_train = prior_train * contrast
        full["reference_contrast_field"] = contrast
        full["reference_contrast_coverage_mask"] = common.copy()
        report.update(contrast_report)
    covariance = np.asarray(fit["noise_covariance"], complex)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitener = (eigenvectors / np.sqrt(np.maximum(eigenvalues, eigenvalues.max()*1e-10))[None]) @ eigenvectors.conj().T
    train_gain = _global_gain(prior_train, common, train, times[:ntrain], whitener)
    amplitude_train = train["amplitude_envelope_at_reference"]
    ratio_train = np.ones_like(amplitude_train)
    ratio_train[common] = train_gain * prior_train[common] / amplitude_train[common]
    test_kernel, _ = _envelope_kernel(train, times[ntrain:])
    selected_device = torch.device(train["report"]["device"])
    tensor = lambda a: torch.as_tensor(a, dtype=torch.complex128, device=selected_device)
    y = tensor(np.asarray(native["coil_images"])[ntrain:, :, common].transpose(0, 2, 1)) @ tensor(whitener).T
    prediction = tensor(test_kernel[:, common, None] * train["complex_receiver_amplitude"][:, common].T[None]) @ tensor(whitener).T
    ratios = torch.as_tensor(ratio_train[common], dtype=torch.float64, device=selected_device)
    scores = []
    for weight in weights:
        residual = y - prediction * (1 + float(weight)*(ratios - 1))[None, :, None]
        scores.append(float(residual.abs().square().sum().cpu()))
    baseline, best = scores[0], int(np.argmin(scores))
    gain_relative = (baseline - scores[best]) / max(baseline, np.finfo(float).tiny)
    improvement = baseline - scores[best]
    accepted = (weights[best] > 0 and gain_relative >= min_relative_validation_improvement
                and improvement >= min_noise_sse_improvement)
    chosen = 1. if source == "reference" else float(weights[best]) if accepted else 0.
    report.update(candidate_reference_weights=weights.tolist(), candidate_validation_whitened_complex_SSE=scores,
        validation_relative_SSE_improvement=float(gain_relative), validation_noise_SSE_improvement=float(improvement),
        validation_gate_passed=bool(accepted), validation_train_global_gain=train_gain,
        validation_best_weight=float(weights[best]), selected_reference_weight=chosen,
        selected_source=("joint" if source == "joint" else "reference") if chosen else "echo32", applied=bool(chosen),
        validation_uses_EPI=False, validation_coil_count=int(y.shape[-1]))
    if chosen == 0:
        return finish("Reference did not pass the reserved-echo improvement gate; retained native amplitude.")
    prior_full, valid_full = reference_envelope(resampled, coverage, full, reference_te, minimum_coherence)
    if source == "joint":
        prior_full = prior_full * contrast
    final_common = common & valid_full
    report.update(common_valid_voxels=int(final_common.sum()),
                  preserved_native_voxels=int(full["support_mask"].sum() - final_common.sum()),
                  final_domain_policy="Training-valid reference coverage intersected with all-echo positive finite envelope transfer only after prior selection." if source == "joint"
                    else "Training-valid reference coverage intersected with the all-echo coherence guard only after prior selection.")
    if final_common.sum() < min_common_voxels:
        report.update(applied=False, selected_reference_weight=0., selected_source="echo32")
        return finish("Selected reference has insufficient stable all-echo coverage; retained native amplitude.")
    final_gain = _global_gain(prior_full, final_common, full, times, whitener)
    original = np.asarray(full["amplitude_envelope_at_reference"])
    ratio = np.ones_like(original)
    ratio[final_common] = ((1 - chosen)
        + chosen * final_gain * prior_full[final_common] / original[final_common])
    full["amplitude_envelope_at_reference"] = original * ratio
    full["complex_receiver_amplitude"] = full["complex_receiver_amplitude"] * ratio[None]
    full["predicted_RSS"] = full["predicted_RSS"] * ratio[None]
    full["reference_amplitude_ratio"] = ratio
    full["reference_amplitude_coverage_mask"] = final_common
    # Keep the original residual diagnostics and recompute their counterparts
    # after imposing the prior. These remain descriptive residual checks.
    full["unconstrained_relative_complex_RMSE"] = full["relative_complex_RMSE"].copy()
    full["unconstrained_reduced_chi2"] = full["reduced_chi2"].copy()
    full["unconstrained_quality_mask"] = full["quality_mask"].copy()
    kernel, _ = _envelope_kernel(full, times)
    data = np.asarray(native["coil_images"])
    model = kernel[:, None] * full["complex_receiver_amplitude"][None]
    domain = np.asarray(full["support_mask"], bool)
    residual = (data[:, :, domain] - model[:, :, domain]).transpose(0, 2, 1) @ whitener.T
    energy = data[:, :, domain].transpose(0, 2, 1) @ whitener.T
    sse = np.sum(np.abs(residual)**2, axis=(0, 2))
    relative = np.sqrt(sse
                       / np.maximum(np.sum(np.abs(energy)**2, axis=(0, 2)), np.finfo(float).tiny))
    full["relative_complex_RMSE"][domain] = relative
    channels = data.shape[1]
    parameter_count = np.where(full["dispersion_kind"][domain] == 0, 2., 3.)
    full["reduced_chi2"][domain] = sse / (len(times)*channels - channels - parameter_count/2)
    full["quality_mask"] = (domain & (full["amplitude_envelope_at_reference"] > 0)
        & (full["reduced_chi2"] <= full["report"]["quality_reduced_chi2_limit"]))
    full["report"]["quality_voxel_count"] = int(full["quality_mask"].sum())
    report.update(final_global_gain=final_gain,
        final_relative_amplitude_change_RMS=float(np.sqrt(np.mean((ratio[final_common] - 1)**2))),
        residual_quality_policy="Unconstrained residual/quality maps retained separately; residual maps recomputed with the original comparable degrees of freedom after amplitude prior. No new uncertainty claim.")
    return finish("Explicit reference prior applied with validation diagnostics." if source == "reference"
                  else "Joint GRE anatomy and smooth training-only acquisition contrast passed reserved-echo prediction gate; frozen contrast reapplied to all-echo fit." if source == "joint"
                  else "Reference blend passed reserved-echo prediction gate and was reapplied to all-echo fit.")


__all__ = ["apply_reference_amplitude_prior"]
