"""Calibrate native temporal coherence from measured complex multi-echo data.

This adapter keeps noise/support estimation separate from the temporal model.
It does not infer a spatial direction for unresolved frequency dispersion.
"""
from copy import deepcopy

import numpy as np

from b0_regularization import regularize_b0_local_plane


def known_inplane_field(inputs, *, sigma_pixels=1., weighting="echo_power"):
    """Return the same measured local field/gradients used by the voxel model."""
    field, raw = inputs["field"], inputs["b0_raw"]
    weights = None
    if weighting == "echo_power":
        coils = np.asarray(raw["coil_images"])
        count = int(inputs.get("field_echo_count", coils.shape[0]))
        products = np.sum(coils[1:count] * coils[:count-1].conj(), axis=1)
        weights = np.mean(np.abs(products), axis=0)
    elif weighting != "uniform":
        raise ValueError("weighting must be uniform or echo_power.")
    X, Y = np.asarray(raw["X_m"]), np.asarray(raw["Y_m"])
    spacing = [np.hypot(X[1, 0]-X[0, 0], Y[1, 0]-Y[0, 0])*1000,
               np.hypot(X[0, 1]-X[0, 0], Y[0, 1]-Y[0, 0])*1000]
    return regularize_b0_local_plane(field["b0Hz"], field["mask"],
        sigma_pixels=sigma_pixels, spacing_mm=spacing, weights=weights)


def prepare_complex_echo_model(inputs, noise_support_fit, *, device="auto",
                               require_cuda=False, b0_sigma_pixels=1.,
                               b0_weighting="echo_power", **fit_options):
    """Replace zero-decay fallbacks by a fit to all measured complex echoes.

    Monoexponential, Gaussian-frequency and uniform-frequency models compete
    on reserved late echoes. The chosen model is refit to all echoes. Poor
    model agreement remains reported; it never silently restores zero decay.
    Amplitude is an envelope, excluding dephasing already simulated explicitly.
    The existing magnitude fit is retained as a diagnostic comparator.
    """
    from complex_echo_fit import fit_complex_echoes

    if not noise_support_fit["report"]["noise"]["calibrated"]:
        raise ValueError("Complex echo modelling requires calibrated receiver noise.")
    local = known_inplane_field(inputs, sigma_pixels=b0_sigma_pixels,
                               weighting=b0_weighting)
    mask = (np.asarray(noise_support_fit["simulation_support"], bool)
            & local["gradient_valid_mask"])
    if not mask.any():
        raise ValueError("No measured object pixels have identifiable in-plane field gradients.")
    result = fit_complex_echoes(inputs["b0_raw"]["coil_images"], inputs["echo_times_s"],
        noise_covariance=noise_support_fit["noise_covariance"],
        initial_b0_Hz=local["b0Hz"], support_mask=mask,
        known_inplane_gradient_Hz_per_pixel=local["gradient_Hz_per_pixel"],
        amplitude_reference_time_s=noise_support_fit["amplitude_reference_time_s"],
        device=device, require_cuda=require_cuda, **fit_options)
    # Reuse this early-echo seed when retraining a reference candidate. Its
    # frequency bounds must match the principal fit, without using the B0
    # refined on all echoes (which would leak reserved observations).
    result["initial_b0_Hz"] = np.asarray(local["b0Hz"]).copy()
    result["known_inplane_gradient_Hz_per_pixel"] = local["gradient_Hz_per_pixel"]
    result["known_inplane_gradient_valid_mask"] = local["gradient_valid_mask"]
    result["support_mask"] = mask
    result["known_inplane_field_report"] = local["report"]
    # Preserve the independently determined domain and its non-simulable mass.
    # Pixels without measured B0 are never assigned invented frequency values.
    result["amplitude_envelope_at_reference"] = np.where(mask,
        result["amplitude_envelope_at_reference"], noise_support_fit["first_echo_amplitude"])
    result["residual_R2_sInv"] = np.where(mask, result["residual_R2_sInv"], 0.)
    result["refined_b0_Hz"] = np.where(mask, result["refined_b0_Hz"], inputs["field"]["b0Hz"])
    result["unresolved_width_Hz"] = np.where(mask, result["unresolved_width_Hz"], 0.)
    result["dispersion_kind"] = np.where(mask, result["dispersion_kind"], 0).astype(np.int8)
    fit = deepcopy(noise_support_fit)
    fit["multi_echo"] = result
    fit["report"]["multi_echo"] = deepcopy(result["report"])
    fit["report"]["multi_echo"].update({
        "fitted_measured_pixels": int(mask.sum()),
        "uses_EPI_to_fit_object": False,
        "old_zero_decay_amplitude_fraction": float(np.sum(
            noise_support_fit["first_echo_amplitude"][mask]
            * (noise_support_fit["simulation_R2star_sInv"][mask] == 0))
            / np.sum(noise_support_fit["first_echo_amplitude"][mask])),
        "dispersion_spatial_assignment": "Unresolved; no through-plane direction or sign inferred.",
        "shim_coupling": "Fixed residual coherence shared by thickness children; no invented refocusing by ACDC.",
        "zero_B0_target": "Preserve unresolved coherence and residual decay; remove only represented macroscopic B0.",
        "extrapolation_to_EPI_TE_s": max(0., float(inputs["acquisition"]["TE_s"]
                                                   - inputs["echo_times_s"][-1])),
    })
    # Keep legacy per-voxel uncertainty/flags explicitly labelled as legacy.
    # They are not a certificate for the new out-of-range EPI extrapolation.
    fit["report"]["legacy_magnitude_fit"] = {
        "target_reliable_voxels": int(noise_support_fit["target_reliable_mask"][mask].sum()),
        "fallback_policy": "Legacy diagnostic only; complex mode does not use its zero-decay fallback."}
    return fit


def multiecho_comparison(inputs, fit):
    """Compare measured echo magnitudes, the legacy fallback and the new fit."""
    result = fit["multi_echo"]
    times = np.asarray(inputs["echo_times_s"])
    mask = result["support_mask"]
    first = fit["first_echo_amplitude"]
    measured = fit["debiased_RSS"]
    gradient = result["known_inplane_gradient_Hz_per_pixel"]
    local_factor = np.prod(np.sinc(times[:, None, None, None]
                                 * gradient[None]), axis=-1)
    legacy = (fit["simulation_amplitude_at_reference"][None]
              * np.exp(-(times-fit["amplitude_reference_time_s"])[:, None, None]
                       * fit["simulation_R2star_sInv"][None]) * np.abs(local_factor))
    predicted = result["predicted_RSS"]
    # This is a fixed descriptive mask, not a fit objective or segmentation edit.
    bright = mask & (fit["SNR"][0] >= 20) & (first > .1*np.max(first))
    index = min(14, len(times)-1)
    ratio = np.divide(measured[index], first, out=np.ones_like(first), where=first > 0)
    holes = bright & (ratio < .25)
    rows = []
    for label, region in (("measured_support", mask), ("early_bright", bright),
                          ("holes_at_echo15", holes)):
        if not region.any():
            continue
        denom = np.maximum(first[region], np.finfo(float).tiny)
        rows.append({"region": label, "pixels": int(region.sum()),
            "legacy_RSS_RMSE_over_first_echo": float(np.sqrt(np.mean(
                ((legacy[:, region]-measured[:, region])/denom[None])**2))),
            "complex_RSS_RMSE_over_first_echo": float(np.sqrt(np.mean(
                ((predicted[:, region]-measured[:, region])/denom[None])**2))),
            "legacy_echo15_ratio_median": float(np.median(legacy[index, region]/denom)),
            "complex_echo15_ratio_median": float(np.median(predicted[index, region]/denom)),
            "measured_echo15_ratio_median": float(np.median(measured[index, region]/denom)),
        })
    return {"rows": rows, "hole_mask": holes, "bright_mask": bright,
            "legacy_predicted_RSS": legacy,
            "comparison_policy": "Refit-to-all-echo descriptive comparison; reserved-echo errors are reported separately by the fitter."}


__all__ = ["known_inplane_field", "prepare_complex_echo_model", "multiecho_comparison"]
