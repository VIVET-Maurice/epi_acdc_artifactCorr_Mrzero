"""Complex, noise-weighted native-voxel echo fits on CPU or CUDA.

All measured echoes remain complex Gaussian observations, including signal
nulls.  Receiver amplitudes are eliminated analytically.  The candidate
envelopes are an exponential, an exponential with a uniform frequency
distribution, and an exponential with a Gaussian frequency distribution.
These unresolved distributions have NO assigned spatial direction and cannot
be interpreted as a measured through-plane gradient or a recoverable signal.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch


def _sinc_derivative(x):
    z = torch.pi * x
    small = z.abs() < .05
    z2 = z.square()
    polynomial = torch.pi * z * (-1 / 3 + z2 * (1 / 30 + z2 * (-1 / 840 + z2 / 45360)))
    safe = torch.where(small, torch.ones_like(z), z)
    regular = torch.pi * (z * torch.cos(z) - torch.sin(z)) / safe.square()
    return torch.where(small, polynomial, regular)


def _kernel(parameters, x, absolute_time, known_factor, kind):
    """Return h and dh/d(parameters), with dimensionless fitted parameters.

    Rate and frequency are multiplied by the echo span.  For a uniform
    distribution the third parameter is full width times the last TE; for a
    Gaussian it is 2*pi**2*(sigma*last_TE)**2, whose derivative is finite at 0.
    """
    rate, frequency = parameters[..., 0, None], parameters[..., 1, None]
    exponential = torch.exp(torch.complex(-rate * x, -2 * torch.pi * frequency * x))
    common = exponential * known_factor[:, None, :]
    derivatives = [-x * common, -2j * torch.pi * x * common]
    if kind == 0:
        return common, torch.stack(derivatives, dim=-1)
    shape = parameters[..., 2, None]
    if kind == 1:
        argument = shape * absolute_time
        factor = torch.sinc(argument)
        shape_derivative = _sinc_derivative(argument) * absolute_time
    elif kind == 2:
        factor = torch.exp(-shape * absolute_time.square())
        shape_derivative = -absolute_time.square() * factor
    else:
        raise ValueError("Unknown dispersion kind.")
    h = common * factor
    return h, torch.stack([derivatives[0] * factor, derivatives[1] * factor,
                          common * shape_derivative], dim=-1)


def _project(y, h):
    """Variable projection for independent whitened complex receivers."""
    denominator = h.abs().square().sum(dim=-1).clamp_min(torch.finfo(h.real.dtype).tiny)
    numerator = torch.einsum("vke,vec->vkc", h.conj(), y)
    amplitude = numerator / denominator[..., None]
    residual = y[:, None] - h[..., None] * amplitude[..., None, :]
    cost = residual.abs().square().sum(dim=(-2, -1))
    return amplitude, cost, denominator


def _normal_equations(y, h, dh, amplitude, denominator):
    """Reduced Gauss-Newton equations after eliminating receiver amplitudes.

    The projected derivatives eliminate the complex linear amplitude nuisance
    parameters.  The right hand side is the exact variable-projection gradient
    (minus one half of the derivative of residual sum of squares).
    """
    overlap = torch.einsum("vke,vkep->vkp", h.conj(), dh)
    projected = dh - h[..., None] * (overlap / denominator[..., None])[..., None, :]
    amplitude_power = amplitude.abs().square().sum(dim=-1)
    gram = torch.einsum("vkep,vkeq->vkpq", projected.conj(), projected).real
    gram = gram * amplitude_power[..., None, None]
    derivative_numerator = torch.einsum("vkep,vec->vkpc", dh.conj(), y)
    right = torch.einsum("vkpc,vkc->vkp", derivative_numerator, amplitude.conj()).real
    right = right - overlap.real * amplitude_power[..., None]
    return gram, right


def _solve(y, known_factor, x, absolute_time, kind, starts, limits, iterations):
    """Independent bounded LM iterations over voxels and multiple starts."""
    parameters = starts.clone()
    damping = torch.full(parameters.shape[:-1], .01, device=y.device, dtype=y.real.dtype)
    energy = y.abs().square().sum(dim=(-2, -1)).clamp_min(torch.finfo(y.real.dtype).tiny)
    h, dh = _kernel(parameters, x, absolute_time, known_factor, kind)
    amplitude, cost, denominator = _project(y, h)
    completed = 0
    for iteration in range(iterations):
        gram, right = _normal_equations(y, h, dh, amplitude, denominator)
        gram, right = gram / energy[:, None, None, None], right / energy[:, None, None]
        diagonal = gram.diagonal(dim1=-2, dim2=-1).clamp_min(1e-6)
        damped = gram + torch.diag_embed(damping[..., None] * diagonal)
        step = torch.linalg.solve(damped, right[..., None]).squeeze(-1)
        # A trust region in dimensionless units prevents one bad, almost
        # unidentified start from jumping across several frequency aliases.
        step = step.clamp(-1., 1.)
        candidate = parameters + step
        candidate[..., 0].clamp_(0., limits[0])
        candidate[..., 1].clamp_(-limits[1], limits[1])
        if kind:
            candidate[..., 2].clamp_(0., limits[2])
        trial_h, trial_dh = _kernel(candidate, x, absolute_time, known_factor, kind)
        trial_amplitude, trial_cost, trial_denominator = _project(y, trial_h)
        accepted = trial_cost < cost
        parameters = torch.where(accepted[..., None], candidate, parameters)
        h = torch.where(accepted[..., None], trial_h, h)
        dh = torch.where(accepted[..., None, None], trial_dh, dh)
        amplitude = torch.where(accepted[..., None], trial_amplitude, amplitude)
        denominator = torch.where(accepted, trial_denominator, denominator)
        improvement = torch.where(accepted, (cost - trial_cost) / energy[:, None], 0.)
        cost = torch.minimum(cost, trial_cost)
        damping = torch.where(accepted, damping / 3., damping * 10.).clamp(1e-8, 1e12)
        completed = iteration + 1
        if iteration >= 7 and iteration % 4 == 3:
            settled = ((right.abs().amax(dim=-1) < 1e-8)
                       | ((step.abs().amax(dim=-1) < 1e-6) & (improvement < 1e-9))
                       | (damping >= 1e11))
            if bool(settled.all()):
                break
    best = cost.argmin(dim=1)
    rows = torch.arange(y.shape[0], device=y.device)
    return parameters[rows, best], amplitude[rows, best], cost[rows, best], completed


def fit_complex_echoes(coil_images, echo_times_s, *, noise_covariance,
                      initial_b0_Hz, support_mask,
                      known_inplane_gradient_Hz_per_pixel=None,
                      amplitude_reference_time_s=None, device="auto", require_cuda=False,
                      batch_size=256, max_iterations=60, max_rate_sInv=250.,
                      max_width_Hz=None, max_frequency_offset_Hz=30.,
                      validation_echo_count=6, selection_min_relative_improvement=.02,
                      width_initializations_Hz=None, max_reduced_chi2=9.,
                      selection_min_noise_improvement=1.):
    """Fit every complex echo, choose its envelope on late held-out echoes.

    ``coil_images`` has axes [echo, receiver, row, column].  The known in-plane
    gradients, if supplied, are field changes across one FULL native row and
    column pixel, shaped [row, column, 2].  The fitted ``unresolved_width_Hz``
    is a FULL width for kind 1 (uniform/sinc), and a STANDARD DEVIATION for
    kind 2 (Gaussian).  Kind 0 is monoexponential.  None is assigned a spatial
    orientation; retain these unresolved losses in a zero-macroscopic-B0 target.

    The last ``validation_echo_count`` echoes select among models, using only
    amplitudes fitted on the preceding echoes.  Selected models are refitted
    on ALL echoes.  Poor agreement is flagged; it never substitutes R2=0.
    ``quality_mask`` is a diagnostic residual check, not physical uncertainty.
    The returned positive amplitude is the envelope at the explicit reference
    TE WITHOUT either known or unresolved sinc/Gaussian attenuation.
    """
    started = perf_counter()
    coils = np.asarray(coil_images)
    times = np.asarray(echo_times_s, float)
    if (coils.ndim != 4 or not np.iscomplexobj(coils) or min(coils.shape[1:]) < 1
            or coils.shape[0] < 6 or times.shape != (coils.shape[0],)
            or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("Need at least six complex receiver echoes with distinct increasing nonnegative TE_s.")
    ne, channels, ny, nx = coils.shape
    shape = (ny, nx)
    support = np.asarray(support_mask)
    if support.shape != shape or not np.isin(support, (0, 1)).all():
        raise ValueError("support_mask must be binary with the native image shape.")
    support = support.astype(bool)
    if not support.any() or not np.isfinite(coils[:, :, support]).all():
        raise ValueError("Supported complex echoes must be finite and support must be nonempty.")
    if np.iscomplexobj(initial_b0_Hz):
        raise ValueError("initial_b0_Hz must be real.")
    field = np.asarray(initial_b0_Hz, float)
    if field.ndim == 0:
        field = np.full(shape, field.item())
    if field.shape != shape or not np.isfinite(field[support]).all():
        raise ValueError("initial_b0_Hz must be finite on support and scalar or image shaped.")
    if np.iscomplexobj(known_inplane_gradient_Hz_per_pixel):
        raise ValueError("known_inplane_gradient_Hz_per_pixel must be real.")
    gradient = (np.zeros(shape + (2,)) if known_inplane_gradient_Hz_per_pixel is None
                else np.asarray(known_inplane_gradient_Hz_per_pixel, float))
    if gradient.shape != shape + (2,) or not np.isfinite(gradient[support]).all():
        raise ValueError("known_inplane_gradient_Hz_per_pixel must be finite [row,column,2] on support.")
    covariance = np.asarray(noise_covariance, complex)
    if (covariance.shape != (channels, channels) or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.conj().T, rtol=1e-8, atol=1e-12)):
        raise ValueError("noise_covariance must be a finite Hermitian receiver covariance.")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = float(eigenvalues.max())
    if maximum <= 0 or eigenvalues.min() < -1e-10 * maximum:
        raise ValueError("noise_covariance must be positive semidefinite with positive noise power.")
    floor = maximum * 1e-10
    values = np.maximum(eigenvalues, floor)
    whitening = (eigenvectors / np.sqrt(values)[None]) @ eigenvectors.conj().T
    unwhitening = (eigenvectors * np.sqrt(values)[None]) @ eigenvectors.conj().T
    span, last = float(times[-1] - times[0]), float(times[-1])
    anchor = float(times[0] if amplitude_reference_time_s is None else amplitude_reference_time_s)
    if not np.isfinite(anchor) or not 0 <= anchor <= times[-1]:
        raise ValueError("amplitude_reference_time_s must lie between zero and the last echo.")
    if max_width_Hz is None:
        max_width_Hz = min(250., .95 / float(np.min(np.diff(times))))
    for name, value in (("max_rate_sInv", max_rate_sInv), ("max_width_Hz", max_width_Hz),
                        ("max_frequency_offset_Hz", max_frequency_offset_Hz),
                        ("max_reduced_chi2", max_reduced_chi2)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    for name, value in (("batch_size", batch_size), ("max_iterations", max_iterations),
                        ("validation_echo_count", validation_echo_count)):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(selection_min_relative_improvement) or not 0 <= selection_min_relative_improvement < 1:
        raise ValueError("selection_min_relative_improvement must be in [0,1).")
    if not np.isfinite(selection_min_noise_improvement) or selection_min_noise_improvement < 0:
        raise ValueError("selection_min_noise_improvement must be finite and nonnegative.")
    validation_count = min(int(validation_echo_count), ne - 3)
    ntrain = ne - validation_count
    if width_initializations_Hz is None:
        widths = np.minimum(np.array([0., .5, 1., 1.5, 2., 3., 4., 6., 8., 12.]) / last, max_width_Hz)
    else:
        widths = np.asarray(width_initializations_Hz, float)
        if (widths.ndim != 1 or widths.size < 1 or not np.isfinite(widths).all()
                or np.any(widths < 0) or np.any(widths > max_width_Hz)):
            raise ValueError("width_initializations_Hz must contain finite widths within the width bound.")
    # The nonzero starts are necessary: sinc has exactly zero width derivative
    # at width=0.  A caller selecting only zero has requested an unusable search.
    if not np.any(widths > 0):
        raise ValueError("width_initializations_Hz must include a nonzero sinc width.")
    widths = np.unique(np.r_[0., widths])
    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if str(device) == "auto" else torch.device(device)
    if require_cuda and selected_device.type != "cuda":
        raise RuntimeError("Complex echo fitting requires a working CUDA device; the selected device is CPU.")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Complex echo fitting selected CUDA, but CUDA is unavailable.")
    dtype, complex_dtype = torch.float64, torch.complex128
    tensor = lambda data, dt=dtype: torch.as_tensor(data, device=selected_device, dtype=dt)
    white, unwhite = tensor(whitening, complex_dtype), tensor(unwhitening, complex_dtype)
    x, absolute = tensor((times - anchor) / span), tensor(times / last)
    indices = np.flatnonzero(support)
    nvox = len(indices)
    maps = {name: np.zeros(shape) for name in ("amplitude_envelope_at_reference", "residual_R2_sInv",
            "refined_b0_Hz", "unresolved_width_Hz", "reduced_chi2", "relative_complex_RMSE")}
    kinds = np.full(shape, -1, dtype=np.int8)
    predicted = np.zeros((ne,) + shape)
    receiver_amplitude = np.zeros((channels,) + shape, complex)
    validation_error = np.full((3,) + shape, np.nan)
    train_error = np.full((3,) + shape, np.nan)
    iteration_counts = []
    rate_limit, frequency_limit = max_rate_sInv * span, max_frequency_offset_Hz * span
    flattened = coils.reshape(ne, channels, -1)
    for first in range(0, nvox, int(batch_size)):
        selected = indices[first:first + int(batch_size)]
        count = len(selected)
        original_y = tensor(flattened[:, :, selected].transpose(2, 0, 1).copy(), complex_dtype)
        y = original_y @ white.T
        initial_field = tensor(field.ravel()[selected])
        y = y * torch.exp(2j * torch.pi * initial_field[:, None] * tensor(times - anchor)[None])[..., None]
        grad = tensor(gradient.reshape(-1, 2)[selected])
        known = torch.sinc(grad[:, 0, None] * tensor(times)[None]) * torch.sinc(grad[:, 1, None] * tensor(times)[None])
        # A short-TE estimate initializes, rather than constrains, the rate.
        power = y.abs().square().sum(dim=-1).sqrt()
        early = min(6, ntrain)
        early_time = tensor(times[:early] - times[0])
        log_amplitude = (power[:, :early] / known[:, :early].abs().clamp_min(.1)).clamp_min(1e-20).log()
        centered_time = early_time - early_time.mean()
        initial_rate = (-(log_amplitude * centered_time).sum(dim=-1)
                        / centered_time.square().sum()).clamp(0., max_rate_sInv) * span
        cross = (y[:, 1:early] * y[:, :early - 1].conj()).sum(dim=-1)
        cross = cross * torch.sign(known[:, 1:early] * known[:, :early - 1])
        estimate_frequency = (-torch.angle(cross.sum(dim=-1)) / (2 * torch.pi * float(np.mean(np.diff(times[:early]))))).clamp(-max_frequency_offset_Hz, max_frequency_offset_Hz) * span
        train_y, train_known = y[:, :ntrain], known[:, :ntrain]
        train_energy = train_y.abs().square().sum(dim=(-2, -1)).clamp_min(1e-30)
        test_energy = y[:, ntrain:].abs().square().sum(dim=(-2, -1)).clamp_min(1e-30)
        fitted_parameters, validation_costs = [], []
        for kind in range(3):
            starts_width = np.array([0.]) if kind == 0 else widths
            if kind == 2:
                # Gaussian sigma is usually much smaller than a uniform full
                # width.  Cover both scales without another unbounded search.
                starts_width = np.unique(np.r_[0., np.minimum(widths / np.sqrt(12.), max_width_Hz)])
            number = len(starts_width) * 2
            starts = torch.zeros((count, number, 2 if kind == 0 else 3), device=selected_device, dtype=dtype)
            starts[..., 0] = initial_rate[:, None]
            starts[:, ::2, 1] = estimate_frequency[:, None]
            if kind:
                dimensionless_width = starts_width * last
                if kind == 2:
                    dimensionless_width = 2 * np.pi**2 * dimensionless_width**2
                starts[..., 2] = tensor(np.repeat(dimensionless_width, 2))[None]
            shape_limit = max_width_Hz * last if kind == 1 else 2 * np.pi**2 * (max_width_Hz * last)**2
            parameters, amplitudes, costs, completed = _solve(train_y, train_known, x[:ntrain], absolute[:ntrain],
                kind, starts, (rate_limit, frequency_limit, shape_limit), int(max_iterations))
            iteration_counts.append(completed)
            test_h, _ = _kernel(parameters[:, None], x[ntrain:], absolute[ntrain:], known[:, ntrain:], kind)
            residual = y[:, ntrain:] - test_h[:, 0, :, None] * amplitudes[:, None]
            test_cost = residual.abs().square().sum(dim=(-2, -1))
            fitted_parameters.append(parameters)
            validation_costs.append(test_cost)
            train_error[kind].ravel()[selected] = torch.sqrt(costs / train_energy).detach().cpu().numpy()
            validation_error[kind].ravel()[selected] = torch.sqrt(test_cost / test_energy).detach().cpu().numpy()
        validation_costs = torch.stack(validation_costs, dim=-1)
        choice = torch.zeros(count, device=selected_device, dtype=torch.long)
        best_cost = validation_costs[:, 0]
        for kind in (1, 2):
            improves = ((validation_costs[:, kind] < best_cost * (1 - selection_min_relative_improvement))
                        & (best_cost - validation_costs[:, kind] > selection_min_noise_improvement))
            choice = torch.where(improves, kind, choice)
            best_cost = torch.where(improves, validation_costs[:, kind], best_cost)
        # Fit only the chosen model on the complete train.  No voxel loses its
        # estimated decay or dispersion because its residual check is poor.
        for kind in range(3):
            keep = choice == kind
            if not bool(keep.any()):
                continue
            destinations = selected[keep.detach().cpu().numpy()]
            shape_limit = max_width_Hz * last if kind == 1 else 2 * np.pi**2 * (max_width_Hz * last)**2
            parameters, amplitudes, costs, completed = _solve(y[keep], known[keep], x, absolute, kind,
                fitted_parameters[kind][keep, None], (rate_limit, frequency_limit, shape_limit), int(max_iterations))
            iteration_counts.append(completed)
            coil_amplitude = amplitudes @ unwhite.T
            envelope = coil_amplitude.abs().square().sum(dim=-1).sqrt()
            h, _ = _kernel(parameters[:, None], x, absolute, known[keep], kind)
            predictions = envelope[:, None] * h[:, 0].abs()
            maps["amplitude_envelope_at_reference"].ravel()[destinations] = envelope.detach().cpu().numpy()
            maps["residual_R2_sInv"].ravel()[destinations] = (parameters[:, 0] / span).detach().cpu().numpy()
            maps["refined_b0_Hz"].ravel()[destinations] = (initial_field[keep] + parameters[:, 1] / span).detach().cpu().numpy()
            if kind:
                width = parameters[:, 2] / last if kind == 1 else torch.sqrt(parameters[:, 2] / (2 * torch.pi**2)) / last
                maps["unresolved_width_Hz"].ravel()[destinations] = width.detach().cpu().numpy()
            degrees = ne * channels - channels - parameters.shape[-1] / 2
            maps["reduced_chi2"].ravel()[destinations] = (costs / degrees).detach().cpu().numpy()
            energy = y[keep].abs().square().sum(dim=(-2, -1)).clamp_min(1e-30)
            maps["relative_complex_RMSE"].ravel()[destinations] = torch.sqrt(costs / energy).detach().cpu().numpy()
            kinds.ravel()[destinations] = kind
            predicted.reshape(ne, -1)[:, destinations] = predictions.detach().cpu().numpy().T
            receiver_amplitude.reshape(channels, -1)[:, destinations] = coil_amplitude.detach().cpu().numpy().T
    quality = (support & np.isfinite(maps["reduced_chi2"])
               & (maps["reduced_chi2"] <= max_reduced_chi2)
               & (maps["amplitude_envelope_at_reference"] > 0))
    elapsed = perf_counter() - started
    report = {
        "format": "complex_multiecho_variable_projection_v1", "device": str(selected_device),
        "device_type": selected_device.type, "cuda_used": selected_device.type == "cuda",
        "gpu_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
        "dtype": str(dtype), "elapsed_s": elapsed, "fitted_voxel_count": nvox,
        "echo_count": ne, "training_echo_indices": list(range(ntrain)),
        "validation_echo_indices": list(range(ntrain, ne)), "refit_uses_all_echoes": True,
        "validation_amplitude_refitted": False,
        "selected_model_counts": {name: int(np.count_nonzero(kinds == code))
            for code, name in enumerate(("monoexponential", "uniform_frequency", "Gaussian_frequency"))},
        "dispersion_kind_codes": {"0": "monoexponential", "1": "uniform: full frequency width Hz",
                                  "2": "Gaussian: frequency standard deviation Hz"},
        "solver": "bounded batched variable-projection Levenberg-Marquardt; analytic reduced normal equations",
        "batch_size": int(batch_size), "max_iterations": int(max_iterations),
        "maximum_solver_iterations_used": max(iteration_counts, default=0),
        "selection_min_relative_SSE_improvement": float(selection_min_relative_improvement),
        "selection_min_whitened_noise_SSE_improvement": float(selection_min_noise_improvement),
        "width_initializations_Hz": widths.tolist(),
        "bounds": {"residual_R2_sInv": [0., float(max_rate_sInv)],
                   "unresolved_width_Hz": [0., float(max_width_Hz)],
                   "frequency_offset_Hz": [-float(max_frequency_offset_Hz), float(max_frequency_offset_Hz)]},
        "whitening_eigenvalue_floor": floor,
        "whitening_floored_eigenvalue_count": int(np.count_nonzero(eigenvalues < floor)),
        "quality_voxel_count": int(quality.sum()), "quality_reduced_chi2_limit": float(max_reduced_chi2),
        "quality_policy": "Residual diagnostic only; poor agreement never resets fitted decay or dispersion.",
        "amplitude_policy": "Positive receive-weighted envelope at reference TE excluding all known and unresolved dephasing factors.",
        "frequency_policy": "exp(-2*pi*i*B0*(TE-reference_TE)); initial field fixes the frequency alias neighborhood.",
        "unresolved_dispersion_policy": "Unlocalized frequency distribution; no inferred z direction, spatial gradient sign, or ACDC-recoverable fraction. Retain in zero-macroscopic-B0 target.",
        "uncertainty_policy": "No physical uncertainty or identifiability claim follows from the residual check or internal echo holdout.",
        "limitations": ["A small family of symmetric frequency distributions is an effective model, not an inferred frequency spectrum.",
            "Echo holdout checks temporal prediction within one acquisition; it is not independent scanner validation.",
            "Frequencies outside the fixed alias neighborhood and between-echo frequency structure are not identified.",
            "Receiver sensitivities are assumed constant through the echo train and unresolved distribution.",
            "Known in-plane dephasing uses a local linear-field, uniform-voxel approximation at each echo TE."],
    }
    return dict(maps, amplitude_reference_time_s=anchor, dispersion_kind=kinds,
        selected_sinc_mask=kinds == 1, selected_gaussian_mask=kinds == 2,
        quality_mask=quality, support_mask=support.copy(),
        predicted_RSS=predicted, complex_receiver_amplitude=receiver_amplitude,
        known_inplane_gradient_Hz_per_pixel=gradient.copy(),
        training_relative_complex_RMSE=train_error,
        validation_relative_complex_RMSE=validation_error, report=report)


__all__ = ["fit_complex_echoes"]
