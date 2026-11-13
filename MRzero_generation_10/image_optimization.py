"""Reference method: the image objective of generation 08, on the same phantom.

Generation 08 chooses the currents by comparing simulated images:

    J(x) = sum((|EPI(x)| - |target|)^2) / sum(|target|^2)
           + guidance_weight * mean((residual frequency / guidance_scale)^2)

The target is here the ideal image of the phantom (no off-resonance). The
second term is the "field guidance" of generation 08: the residual frequency
``B0 + B @ (mean current between two consecutive echoes)``.

Unlike the phase objective of generation 10, ``J`` needs one EPI simulation
and its gradient at every iteration, and it is not convex: the result depends
on the starting point. It is minimised as in generation 08, with projected
Adam, here with a fixed learning rate and a fixed number of iterations.
"""

import numpy as np
import torch

from acdc_hardware import project_onto_limits


def relative_image_error(image, target):
    """sum((|image| - |target|)^2) / sum(|target|^2), for NumPy arrays or tensors."""
    return ((abs(image) - abs(target))**2).sum() / (abs(target)**2).sum()


def image_error_and_gradient(simulator, currents, target_magnitude):
    """Image error and its gradient with respect to the currents (tensors).

    The image depends on all ADC samples, but building the computation graph
    of the whole simulation at once would need a lot of memory. Two passes
    give the same gradient:

    1. simulate k-space without graph, then differentiate the error with
       respect to k-space only (``sensitivity``);
    2. simulate again a few samples at a time, with graph, and differentiate
       ``Re(sum(conj(sensitivity) * signal))`` with respect to the currents.
    """
    with torch.no_grad():
        kspace = simulator.kspace(currents)
    kspace.requires_grad_(True)
    error = relative_image_error(simulator.reconstruct(kspace), target_magnitude)
    (sensitivity,) = torch.autograd.grad(error, kspace)
    sensitivity = sensitivity.reshape(-1)

    currents = currents.detach().requires_grad_(True)
    for samples in simulator.sample_chunks():
        signal = simulator.kspace_chunk(currents, samples)
        (sensitivity[samples].conj() * signal).real.sum().backward()
    return error.detach(), currents.grad


def optimize_image(simulator, hardware, coil_fields, phase_problem, initial_currents_A,
                   iterations=300, learning_rate_A=0.01,
                   guidance_weight=0.05, guidance_scale_Hz=100.0):
    """Minimise the generation-08 objective with projected Adam.

    ``phase_problem`` (voxel centres, equal weights) provides the small
    matrices of the field guidance. Returns the best commands found and
    the history of the image error and of the total objective.
    """
    def tensor(values):
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=simulator.device)

    def project(currents_A):
        return project_onto_limits(currents_A, hardware, coil_fields.maps_Hz_per_A,
                                   coil_fields.gradients_Hz_per_A)

    target = tensor(np.abs(simulator.ideal_image))

    # Field guidance: mean current between consecutive echoes = currents @ slopes.T
    echo_intervals = np.diff(phase_problem.echo_time_s)
    slopes = np.diff(phase_problem.current_integral_s, axis=0) / echo_intervals[:, None]

    def guidance_and_gradient(currents_A):
        residual_Hz = phase_problem.coil_factor @ currents_A @ slopes.T + phase_problem.field_factor[:, None]
        scaled = residual_Hz / guidance_scale_Hz
        penalty = np.sum(scaled**2) / len(echo_intervals)
        gradient = 2.0 * phase_problem.coil_factor.T @ scaled @ slopes / (guidance_scale_Hz * len(echo_intervals))
        return penalty, gradient

    currents = project(np.asarray(initial_currents_A, dtype=float))
    best_currents, best_objective = currents, np.inf
    first_moment, second_moment = np.zeros_like(currents), np.zeros_like(currents)
    history = {"image_error": [], "objective": []}

    for iteration in range(iterations + 1):
        error, gradient = image_error_and_gradient(simulator, tensor(currents), target)
        objective, gradient = float(error), gradient.cpu().numpy().astype(float)
        if guidance_weight > 0:
            penalty, penalty_gradient = guidance_and_gradient(currents)
            objective += guidance_weight * penalty
            gradient += guidance_weight * penalty_gradient
        history["image_error"].append(float(error))
        history["objective"].append(objective)
        if objective < best_objective:
            best_currents, best_objective = currents, objective
        if iteration == iterations:
            break

        # Adam step, then back inside the hardware limits.
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1 - 0.9**(iteration + 1))
        corrected_second = second_moment / (1 - 0.999**(iteration + 1))
        direction = corrected_first / (np.sqrt(corrected_second) + 1e-8)
        currents = project(currents - learning_rate_A * direction)

    return {"currents_A": best_currents,
            "image_error_history": np.asarray(history["image_error"]),
            "objective_history": np.asarray(history["objective"])}
