"""Guarded calibration of two effective spiral gradient delays.

The existing native MRzero center-spin cache is reused without changing the
object, receive encoding, B0, relaxation, coherence or current timing. A
linearized complex-image fit proposes bounded delay steps. Nonlinear training
NRMSE selects the candidate before it is checked on reserved spatial subsets.
Reconstruction always uses the nominal trajectory.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares
import torch

from baseline_calibration import _jacobian_information, spatial_split
from experiment_voxel import normalized_sinc_and_derivative
from inplane_geometry import FrameReconstruction
from spiral_encoding import delayed_spiral_trajectory, reconstruction_acquisition


def _framed_trajectory(native_acquisition, delays_s, matrix, sequence_path):
    kx, ky = delayed_spiral_trajectory(native_acquisition, *delays_s,
                                      sequence_path=sequence_path)
    return (np.stack((kx, ky), axis=-1) @ matrix.T).reshape(-1, 2).T


@torch.no_grad()
def _signal_and_derivatives(model, target_k, derivative_k=None):
    """Sum cached points with a changed Fourier phase and exact box factors.

    ``target_k`` has shape [2, ADC]. ``derivative_k`` is optionally
    [parameter, 2, ADC], in cycles/metre per parameter unit. The native cache
    already contains amplitude, center B0, residual relaxation and coherence;
    it does not contain the two in-plane box integrals.
    """
    model._ensure_voxel_tensors()
    ensure_receive = getattr(model, "_ensure_receive_weights", None)
    if callable(ensure_receive):
        ensure_receive()
    weights = model._receive_weights
    if weights.ndim != 2 or weights.shape[1] != model._x.numel():
        raise ValueError("Spiral delay fitting requires receiver-by-active-point complex weights.")
    target = model.tensor(target_k)
    slopes = None if derivative_k is None else model.tensor(derivative_k)
    if (target.shape != (2, model.nsamples)
            or (slopes is not None and slopes.shape[1:] != target.shape)):
        raise ValueError("Delayed trajectories and their derivatives must match all model ADC samples.")
    output = np.empty((weights.shape[0], model.nsamples), np.complex128)
    derivative = (None if slopes is None else np.empty(
        (slopes.shape[0], weights.shape[0], model.nsamples), np.complex128))
    x, y = model._x.reshape(-1, 1), model._y.reshape(-1, 1)
    edges, gradient = model._voxel_edges, model._voxel_b0_gradient
    for first in range(0, model.nsamples, model.sample_chunk):
        stop = min(first + model.sample_chunk, model.nsamples)
        selected = slice(first, stop)
        # Reuse the native block boundaries, including the bounded CPU cache.
        cached = (model._base_cache[:, selected] if model._base_cache is not None
                  else model._make_base(selected, b0_enabled=True))
        for local_first in range(0, stop - first, 64):
            local_stop = min(local_first + 64, stop - first)
            adc = slice(first + local_first, first + local_stop)
            local = slice(local_first, local_stop)
            kx, ky = target[:, adc]
            delta = -2 * torch.pi * (x * (kx - model._kx[adc])[None]
                                     + y * (ky - model._ky[adc])[None])
            point = cached[:, local] * torch.exp(1j * delta)
            t = model._evolution[adc][None]
            q = [edges[axis, 0] * kx[None] + edges[axis, 1] * ky[None]
                 + gradient[:, axis, None] * t for axis in range(2)]
            if slopes is None:
                factor = torch.sinc(q[0]) * torch.sinc(q[1])
            else:
                f0, d0 = normalized_sinc_and_derivative(q[0])
                f1, d1 = normalized_sinc_and_derivative(q[1])
                factor = f0 * f1
            output[:, adc] = model.numpy(weights @ (point * factor))
            if slopes is not None:
                for parameter in range(slopes.shape[0]):
                    dx, dy = slopes[parameter, :, adc]
                    dphase = -2j * torch.pi * (x * dx[None] + y * dy[None])
                    dbox = ((edges[0, 0] * dx + edges[0, 1] * dy)[None] * d0 * f1
                            + (edges[1, 0] * dx + edges[1, 1] * dy)[None] * f0 * d1)
                    derivative[parameter, :, adc] = model.numpy(
                        weights @ (point * (dphase * factor + dbox)))
    shape = (weights.shape[0],) + tuple(model.sampling_shape)
    return output.reshape(shape), (None if derivative is None else
        derivative.reshape((derivative.shape[0],) + shape))


def _image_metrics(image, measured, masks):
    fit = masks["fit"]
    energy = float(np.sum(image[fit] ** 2))
    if energy <= 0:
        raise ValueError("Spiral delay calibration has no simulated training signal.")
    gain = float(np.sum(image[fit] * measured[fit]) / energy)
    scaled = gain * image
    scores = {}
    for name, mask in masks.items():
        norm = float(np.linalg.norm(measured[mask]))
        if norm <= 0:
            raise ValueError(f"Spiral delay calibration has no measured {name} signal.")
        scores[name] = {"pixels": int(mask.sum()), "NRMSE_percent":
            float(100 * np.linalg.norm(scaled[mask] - measured[mask]) / norm)}
    return {"gain_from_fit_pixels": gain, "metrics": scores}, scaled


def fit_spiral_gradient_delays(model, native_acquisition, sequence_path,
                               measured_image, frame_matrix, *,
                               max_delay_s=20e-6, difference_step_s=.1e-6,
                               max_proposal_steps=3):
    """Fit bounded proposals and accept only an exact held-out improvement.

    The supplied model must use nominal, zero-delay encoding. Delay parameters
    refer to native X/Y gradient axes before the model frame rotation. Positive
    delays lag the gradient waveform. With a sequence path the complete
    delayed Pulseq gradient moments are integrated; ``None`` explicitly uses
    sampled-trajectory interpolation for synthetic tests or diagnostics.
    At most three local proposals are used by default. Only exact training
    error selects among them; validation and test pixels are reserved for the
    final gate.
    """
    for value, name in ((max_delay_s, "max_delay_s"),
                        (difference_step_s, "difference_step_s")):
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                or not np.isrealobj(value) or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive finite seconds.")
    if difference_step_s >= max_delay_s:
        raise ValueError("The derivative step must be smaller than the delay bounds.")
    if (isinstance(max_proposal_steps, (bool, np.bool_))
            or int(max_proposal_steps) != max_proposal_steps or not 1 <= max_proposal_steps <= 10):
        raise ValueError("max_proposal_steps must be an integer between 1 and 10.")
    if np.iscomplexobj(frame_matrix):
        raise ValueError("The model frame matrix must be real orthogonal.")
    matrix = np.asarray(frame_matrix, dtype=float)
    if (matrix.shape != (2, 2) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(2), rtol=0, atol=1e-10)):
        raise ValueError("The model frame matrix must be orthogonal.")
    measured = np.asarray(measured_image)
    if (measured.ndim != 2 or np.iscomplexobj(measured)
            or not np.isfinite(measured).all() or np.any(measured < 0)):
        raise ValueError("Measured spiral RSS must be a finite nonnegative image.")
    nominal = reconstruction_acquisition(native_acquisition)
    k0 = _framed_trajectory(nominal, (0., 0.), matrix, sequence_path)
    stored = np.stack((model.numpy(model._kx), model.numpy(model._ky)))
    tolerance = 5e-5 if model.dtype == torch.float32 else 1e-8
    if stored.shape != k0.shape or not np.allclose(stored, k0, rtol=0, atol=tolerance):
        raise ValueError("Fit spiral gradient delays from a model with nominal zero-delay encoding.")
    framed_nominal = deepcopy(nominal)
    framed_nominal.update(kx_mInv=k0[0].reshape(model.sampling_shape),
        ky_mInv=k0[1].reshape(model.sampling_shape), encoding_origin_m=np.zeros(2))
    operator = FrameReconstruction(framed_nominal, "cpu", torch.float64, frame_matrix=matrix)
    if tuple(operator.image_size) != measured.shape:
        raise ValueError("Measured image shape differs from the fixed nominal reconstruction.")
    fit, validation, test = spatial_split(measured.shape, block_size=8, guard_pixels=1)
    masks = dict(fit=fit, validation=validation, test=test,
                 full_FOV=np.ones(measured.shape, dtype=bool))
    observed = measured[fit]
    norm = float(np.linalg.norm(observed))
    if norm <= 0:
        raise ValueError("The measured spiral has no training signal.")

    bounds_us = float(max_delay_s * 1e6)
    center_s, proposals = np.zeros(2), []
    best_training_cost = np.inf
    best = None
    selected_step = 0
    for step in range(int(max_proposal_steps)):
        center_k = _framed_trajectory(nominal, center_s, matrix, sequence_path)
        derivatives = []
        for axis in range(2):
            plus, minus = center_s.copy(), center_s.copy()
            plus[axis] += difference_step_s
            minus[axis] -= difference_step_s
            derivatives.append((_framed_trajectory(nominal, plus, matrix, sequence_path)
                - _framed_trajectory(nominal, minus, matrix, sequence_path))
                / (2 * difference_step_s * 1e6))
        samples, dsamples = _signal_and_derivatives(model, center_k, np.asarray(derivatives))
        image0 = operator.numpy_image(samples)
        dimage = np.stack([operator.numpy_image(data) for data in dsamples])
        if step == 0:
            baseline_rss = np.sqrt(np.sum(abs(image0)**2, axis=0))
        train0, train_d = image0[:, fit], dimage[:, :, fit]
        center_us = center_s * 1e6

        def evaluate(parameters):
            image = train0 + np.einsum("p,pcn->cn", parameters - center_us, train_d)
            magnitude = np.sqrt(np.sum(abs(image)**2, axis=0))
            slope = np.divide(np.sum((image.conj()[None] * train_d).real, axis=1),
                magnitude[None], out=np.zeros((2, len(magnitude))), where=magnitude[None] > 0)
            energy = float(magnitude @ magnitude)
            if energy <= 0:
                raise ValueError("The linearized spiral candidate has no training signal.")
            gain = float(magnitude @ observed / energy)
            dgain = (slope @ observed - 2 * gain * (slope @ magnitude)) / energy
            return ((gain * magnitude - observed) / norm,
                    (gain * slope + dgain[:, None] * magnitude).T / norm)

        solution = least_squares(lambda x: evaluate(x)[0], center_us,
            jac=lambda x: evaluate(x)[1], bounds=(-bounds_us, bounds_us),
            max_nfev=80, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        proposed_s = solution.x * 1e-6
        jacobian = evaluate(solution.x)[1]
        information = _jacobian_information(jacobian, np.full(2, bounds_us))
        hit_bound = bool(np.any(abs(solution.x) >= bounds_us * (1 - 1e-5)))
        actual_k = _framed_trajectory(nominal, proposed_s, matrix, sequence_path)
        exact_samples, _ = _signal_and_derivatives(model, actual_k)
        candidate_coils = operator.numpy_image(exact_samples)
        exact_rss = np.sqrt(np.sum(abs(candidate_coils)**2, axis=0))
        training_image = exact_rss[fit]
        exact_energy = float(training_image @ training_image)
        if exact_energy <= 0 or not np.isfinite(exact_energy):
            raise ValueError("The exact spiral candidate has no finite training signal.")
        exact_gain = float(training_image @ observed / exact_energy)
        exact_cost = float(100 * np.linalg.norm(exact_gain * training_image - observed) / norm)
        proposals.append({"step": step + 1,
            "parameters": dict(zip(("delay_x_s", "delay_y_s"), map(float, proposed_s))),
            "exact_training_NRMSE_percent": exact_cost,
            "training_gain": exact_gain, "bound_hit": hit_bound,
            "solver_success": bool(solution.success), "identifiability": information})
        if best is None:
            initial_image = baseline_rss[fit]
            initial_gain = float(initial_image @ observed / (initial_image @ initial_image))
            best_training_cost = float(100 * np.linalg.norm(
                initial_gain * initial_image - observed) / norm)
            initial_jacobian = evaluate(np.zeros(2))[1]
            initial_information = _jacobian_information(initial_jacobian, np.full(2, bounds_us))
            best = (np.zeros(2), baseline_rss, solution, initial_jacobian, initial_information, False)
        if exact_cost >= best_training_cost - 1e-6:
            break
        best_training_cost = exact_cost
        best = (proposed_s, exact_rss, solution, jacobian, information, hit_bound)
        selected_step = step + 1
        if (not solution.success or not information["Jacobian_identifiable"] or hit_bound
                or np.linalg.norm(proposed_s - center_s) < 1e-10):
            break
        center_s = proposed_s
    fitted_s, candidate_rss, solution, jacobian, information, hit_bound = best
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian))
    baseline_report, baseline_scaled = _image_metrics(baseline_rss, measured, masks)
    candidate_report, candidate_scaled = _image_metrics(candidate_rss, measured, masks)
    def improves(subset):
        before = baseline_report["metrics"][subset]["NRMSE_percent"]
        after = candidate_report["metrics"][subset]["NRMSE_percent"]
        return before - after > max(1e-6, before * 1e-4)
    accepted = bool(solution.success and information["Jacobian_identifiable"] and not hit_bound
                    and improves("validation") and improves("test"))
    if not solution.success:
        reason = "The bounded training fit did not converge."
    elif not information["Jacobian_identifiable"]:
        reason = "The training image Jacobian does not identify both gradient delays."
    elif hit_bound:
        reason = "A fitted delay reached its configured bound."
    elif selected_step == 0:
        reason = "No exact proposal improved training NRMSE."
    elif not improves("validation"):
        reason = "The exact candidate did not improve reserved validation pixels."
    elif not improves("test"):
        reason = "The exact candidate failed the untouched test-pixel gate."
    else:
        reason = "Accepted: the exact candidate improved reserved validation and test NRMSE."
    candidate_parameters = dict(zip(("delay_x_s", "delay_y_s"), map(float, fitted_s)))
    parameters = candidate_parameters if accepted else {"delay_x_s": 0., "delay_y_s": 0.}
    report = {"format": "guarded_spiral_gradient_delay_fit_v1", "accepted": accepted, "reason": reason,
        "parameters": parameters, "candidate_parameters": candidate_parameters,
        "bounds_seconds": [-float(max_delay_s), float(max_delay_s)],
        "difference_step_seconds": float(difference_step_s),
        "bound_hit": hit_bound, "solver_success": bool(solution.success),
        "training_Jacobian_rank": rank, "training_Jacobian_singular_values_per_us": singular_values.tolist(),
        "identifiability": information,
        "baseline": baseline_report, "candidate": candidate_report,
        "split": {"block_size": 8, "guard_pixels": 1,
            "scope": "Internal spatial subsets of one measured spiral; not independent acquisitions."},
        "proposal": "Bounded fits of complex-image first derivatives and a training-only scalar gain; choose lowest exact training NRMSE before accessing reserved subsets.",
        "max_proposal_steps": int(max_proposal_steps),
        "selected_proposal_step": selected_step, "proposals": proposals,
        "application_gate": "Full nonlinear finite-voxel signal recomputed; reserved validation and test NRMSE must both improve; reject bound hits or rank loss.",
        "minimum_NRMSE_improvement": {"absolute_percentage_points": 1e-6, "relative_fraction": 1e-4},
        "gradient_encoding": "Full Pulseq delayed-gradient integration including RF history." if sequence_path is not None
            else "Effective sampled-k interpolation and linear endpoint extrapolation.",
        "reconstruction": "Nominal trajectory fixed for baseline, derivative and candidate receiver images.",
        "fixed_object": ["support", "amplitude", "B0", "relaxation", "unresolved coherence", "receive weights"],
        "hardware_delay_independently_measured": False,
        "limitations": ["The fitted effective delay does not identify full gradient impulse response or eddy currents.",
            "A bounded local search does not certify that the nominal or fitted trajectory is physically correct.",
            "Native magnetization and temporal coherence remain subject to their measured-map assumptions."],
    }
    return dict(accepted=accepted, parameters=parameters, report=report,
        baseline_rss=baseline_rss, candidate_rss=candidate_rss,
        baseline_scaled=baseline_scaled, candidate_scaled=candidate_scaled)


__all__ = ["fit_spiral_gradient_delays"]
