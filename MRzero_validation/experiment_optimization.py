"""Projected Adam for the experimental MRzero forward model.

The solver uses the native MRzero voxel magnetization from experiment_mrzero.py,
the exact current-response integrals and the analytic shim-phase adjoint.
Its periodic snapshots allow live images without a second forward pass.
"""

from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def optimize_mrzero_image(model, initial_currents_A=None, *, total_current_limit_A=30.0,
                   field_limit_Hz=500.0, max_iterations=150, learning_rate_A=0.01,
                   target_mse=None, patience=25, min_relative_improvement=1e-6,
                   callback=None, callback_every=1, use_cuda_graph=False):
    """Optimize native MRzero images and report selected evaluated candidates.

    ``callback(snapshot)`` runs at zero, every ``callback_every`` iterations,
    and the last evaluated iteration. Numeric history retains EVERY iteration.
    The snapshot contains the evaluated image, DAC commands and relative MSE
    from the SAME candidate, before the next Adam update. Arrays are copied
    so a plotting callback cannot modify the optimizer state. The returned
    solution contains the best feasible candidate, which may differ from
    the final evaluated candidate.

    Projection and stopping rules follow Experiment/acdc_gpu.py.

    The exact magnitude-image objective and analytic adjoint run on the GPU.
    Adam is a local optimizer: the returned feasible point can be useful after
    an iteration limit without implying mathematical convergence. History
    reports evaluated values and best values separately.
    """
    if (isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1
            or isinstance(patience, bool) or int(patience) != patience or patience < 1):
        raise ValueError("max_iterations and patience must be positive integers.")
    if not np.isfinite(learning_rate_A) or learning_rate_A <= 0:
        raise ValueError("learning_rate_A must be positive and finite.")
    if target_mse is not None and (not np.isfinite(target_mse) or target_mse <= 0):
        raise ValueError("target_mse must be positive or None.")
    if not np.isfinite(min_relative_improvement) or min_relative_improvement < 0:
        raise ValueError("min_relative_improvement must be finite and nonnegative.")
    if (isinstance(callback_every, bool) or int(callback_every) != callback_every
            or callback_every < 1):
        raise ValueError("callback_every must be a positive integer.")
    if not isinstance(use_cuda_graph, bool):
        raise ValueError("use_cuda_graph must be True or False.")
    from cuda_graph_helpers import GpuEvaluation
    evaluator = GpuEvaluation(model, total_current_limit_A, field_limit_Hz,
                              enabled=use_cuda_graph)
    if initial_currents_A is None:
        initial_currents_A = np.zeros((model.nc, model.nstates))
    requested = model._check_currents(initial_currents_A)
    requested_value = float(model._objective_tensor(requested, gradient=False)[0])
    currents = evaluator.project(requested)
    initial = currents.clone()
    best = currents.clone()
    best_value = np.inf
    moment, variance = torch.zeros_like(currents), torch.zeros_like(currents)
    learning_rate = float(learning_rate_A)
    history = {name: [] for name in ("iteration", "fval", "best_fval", "firstorderopt", "constrviolation", "learning_rate_A", "elapsed_s")}
    stale, reductions, adam_step = 0, 0, 0
    callback_seconds = 0.0
    status, converged = "iteration_limit", False
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for iteration in range(int(max_iterations) + 1):
        value_tensor, gradient, candidate_image, _ = evaluator.evaluate(currents)
        value = float(value_tensor)
        if not np.isfinite(value) or not bool(torch.isfinite(gradient).all()):
            status = "nonfinite_evaluation"
            break
        previous_best = best_value
        if value < best_value:
            best_value, best = value, currents.clone()
        significant = not np.isfinite(previous_best) or value < previous_best - min_relative_improvement * max(previous_best, 1e-12)
        stale = 0 if significant else stale + 1
        projected_step = currents - evaluator.project(currents - gradient)
        first_order = float(projected_step.abs().max())
        elapsed = time.perf_counter() - started
        row = {"iteration": iteration, "fval": value, "best_fval": best_value,
               "firstorderopt": first_order, "constrviolation": 0.0,
               "learning_rate_A": learning_rate, "elapsed_s": elapsed}
        for key, item in row.items():
            history[key].append(item)
        terminal = (iteration == max_iterations
                    or (target_mse is not None and best_value <= target_mse)
                    or first_order < (1e-7 if model.dtype == torch.float64 else 1e-6)
                    or (stale >= patience and reductions >= 3))
        if callback is not None and (iteration % callback_every == 0 or terminal):
            callback_started = time.perf_counter()
            snapshot = dict(row)
            snapshot.update(
                image=model.numpy(candidate_image).copy(),
                currents_A=model.numpy(currents).copy(),
                max_field_Hz=float((model.B_quality @ currents).abs().max()),
                max_channel_current_A=float(currents.abs().max()),
                max_total_current_A=float(currents.abs().sum(dim=0).max()),
                history={"iteration": np.asarray(history["iteration"]).copy(),
                         "fval": np.asarray(history["fval"]).copy()},
            )
            callback(snapshot)
            callback_seconds += time.perf_counter() - callback_started
        if target_mse is not None and best_value <= target_mse:
            status, converged = "target_mse", True
            break
        if first_order < (1e-7 if model.dtype == torch.float64 else 1e-6):
            # Radial field contraction is not an orthogonal projection, so
            # this stopping test does not certify constrained stationarity.
            status = "projected_step_small"
            break
        if iteration == max_iterations:
            break
        if stale >= patience:
            if reductions >= 3:
                status = "plateau"
                break
            learning_rate *= 0.5
            currents = best.clone()
            moment.zero_()
            variance.zero_()
            stale, reductions, adam_step = 0, reductions + 1, 0
            # Re-evaluate at the restored best point before its next update.
            continue
        moment.mul_(0.9).add_(gradient, alpha=0.1)
        variance.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
        adam_step += 1
        direction = (moment / (1 - 0.9**adam_step)) / ((variance / (1 - 0.999**adam_step)).sqrt() + 1e-8)
        currents = evaluator.project(currents - learning_rate * direction)
    if not np.isfinite(best_value):
        raise RuntimeError("No finite feasible objective evaluation was obtained.")
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    final_value, image, kdata = model._objective_tensor(best, gradient=False)
    commands = model.numpy(best)
    fields = model.maps_numpy @ commands
    limits = model.numpy(model.max_current_A)
    current_ok = bool(np.all(np.abs(commands) <= limits[:, None])
                      and np.all(np.abs(commands).sum(axis=0) < total_current_limit_A))
    field_ok = bool(np.max(np.abs(fields[model.quality_mask])) <= field_limit_Hz)
    initial_used = model.numpy(initial)
    diagnostics = model.response_diagnostics(commands, field_limit_Hz, total_current_limit_A)
    info = {"status": status, "exitflag": 1 if converged else 0,
            "fval": float(final_value), "converged": converged,
            "target_mse": target_mse,
            "target_reached": target_mse is not None and float(final_value) <= target_mse,
            "accepted": bool(current_ok and field_ok and np.isfinite(float(final_value))),
            "current_limits_met": current_ok, "field_limits_met": field_ok,
            "actual_adc_current_limits_met": diagnostics["actual_adc_channel_limits_met"] and diagnostics["actual_adc_current_sum_limits_met"],
            "actual_adc_field_limits_met": diagnostics["actual_adc_field_limits_met"],
            "iterations": len(history["iteration"]) - 1, "seconds": seconds,
            "callback_seconds": callback_seconds,
            "seconds_without_callback": max(0.0, seconds - callback_seconds),
            "callback_every": int(callback_every),
            "cuda_graph": evaluator.report,
            "initial_was_projected": not np.array_equal(initial_used, model.numpy(requested)),
            "algorithm": "projected Adam; box/L1 projection and radial field contraction"}
    return {"currents_A": commands, "initial_currents_A": initial_used,
            "requested_initial_currents_A": model.numpy(requested),
            "requested_initial_relative_mse": requested_value,
            "initial_relative_mse": history["fval"][0],
            "reference": model.reference, "image": model.numpy(image), "kdata": model.numpy(kdata),
            "generated_field_Hz": fields, "field_limit_Hz": field_limit_Hz,
            "current_response": model.current_response,
            "line_start_s": model.line_start_s.copy(), "line_end_s": model.line_end_s.copy(),
            "calibration_mode": "registered", "registration_confirmed": model.registration_confirmed,
            "history": {key: np.asarray(value) for key, value in history.items()},
            "optimization_info": info, "response_diagnostics": diagnostics,
            "runtime_info": model.runtime_info, "elapsed_seconds": seconds}
