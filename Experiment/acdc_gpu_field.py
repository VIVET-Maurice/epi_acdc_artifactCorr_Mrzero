"""Small GPU least-squares solvers for static and dynamic B0 shimming.

The measured target is constant in time. Compressing its spatial dimension to
``B.T @ B`` and ``B.T @ target`` avoids a huge [voxel, ADC] array at every step.
Only PyTorch performs the optimization; MATLAB and quadprog are not required.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from acdc_gpu import project_currents_tensor, select_device


def solve_field_gpu(
    basis_Hz_per_A,
    target_Hz,
    *,
    response=None,
    settings=None,
    device=None,
    dtype=None,
    max_current_A=3.0,
    total_current_limit_A=50.0,
    field_limit_Hz=None,
    max_iterations=600,
    tolerance=1e-6,
    ridge=0.0,
    initial_currents_A=None,
    history_every=10,
):
    """Fit a constant target field using GPU accelerated projected FISTA.

    ``basis_Hz_per_A`` has shape [voxel, channel]; ``target_Hz`` is [voxel].
    For a dynamic fit, ``response[channel, ADC, state]`` maps DAC commands to
    actual currents. Without it, there is one static command per channel.

    The objective is ``mean((B @ actual - target)**2) + ridge * mean(I**2)``.
    Thus ``ridge`` is expressed in Hz²/A²; it is NOT the old solver's weight
    after spectral scaling. Zero disables current regularization.

    ``settings`` accepts the same ``profile`` and ``precision`` options as
    :func:`acdc_gpu.select_device`. Explicit ``device`` / ``dtype`` override
    those choices, which also permits small CPU checks.

    Channel and total-current constraints use the shared box/L1 projection.
    If a nominal field limit is active, the shared helper additionally scales
    each command toward zero. This keeps commands feasible but is not the
    Euclidean projection onto all field constraints. Therefore an active
    field limit gives a feasible approximate fit, without a claim of global
    constrained optimality. Actual filtered-current limits must also be
    checked before exporting the waveform.

    Returns a dict with ``currents_A[channel, state]`` (static: ``[:, 0]``),
    ``history``, ``nrmse`` (None for a zero target), ``loss`` in Hz²,
    ``seconds``, ``iterations`` and ``converged``.
    """
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if not isinstance(history_every, int) or isinstance(history_every, bool) or history_every < 1:
        raise ValueError("history_every must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite.")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be nonnegative and finite.")
    if not np.isfinite(total_current_limit_A) or total_current_limit_A <= 0:
        raise ValueError("total_current_limit_A must be positive and finite.")
    if field_limit_Hz is not None and (not np.isfinite(field_limit_Hz) or field_limit_Hz <= 0):
        raise ValueError("field_limit_Hz must be positive and finite, or None.")

    settings = {} if settings is None else dict(settings)
    selected_device, selected_dtype, runtime_info = select_device(
        profile=settings.get("profile", "auto"), precision=settings.get("precision", "auto")
    )
    device = selected_device if device is None else torch.device(device)
    dtype = selected_dtype if dtype is None else dtype
    if isinstance(dtype, str):
        dtype = {"float32": torch.float32, "float64": torch.float64}.get(dtype)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Use torch.float32 or torch.float64 for the field fit.")

    def tensor(value):
        is_complex = value.is_complex() if isinstance(value, torch.Tensor) else np.iscomplexobj(value)
        if is_complex:
            raise ValueError("Field fitting requires real arrays.")
        return torch.as_tensor(value, device=device, dtype=dtype)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        B = tensor(basis_Hz_per_A)
        target = tensor(target_Hz)
        if B.ndim != 2 or B.shape[0] == 0 or B.shape[1] == 0:
            raise ValueError("basis_Hz_per_A must be a nonempty [voxel, channel] array.")
        if target.ndim != 1 or target.shape[0] != B.shape[0]:
            raise ValueError("target_Hz must be a [voxel] vector matching the basis.")
        if not bool(torch.isfinite(B).all()) or not bool(torch.isfinite(target).all()):
            raise ValueError("The basis and target must contain only finite values.")
        n_voxels, n_channels = B.shape
        limits = tensor(max_current_A)
        if limits.ndim > 1 or limits.numel() not in (1, n_channels):
            raise ValueError("max_current_A must be a scalar or one limit per channel.")
        limits = limits.reshape(-1).expand(n_channels)
        if not bool(torch.isfinite(limits).all()) or not bool((limits > 0).all()):
            raise ValueError("Per-channel limits must be positive and finite.")

        R = None if response is None else tensor(response)
        if R is not None:
            if R.ndim != 3 or R.shape[0] != n_channels or R.shape[1] == 0 or R.shape[2] == 0:
                raise ValueError("response must have shape [channel, ADC, state].")
            if not bool(torch.isfinite(R).all()):
                raise ValueError("The temporal response must contain only finite values.")
        n_times, n_states = (1, 1) if R is None else R.shape[1:]

        # Double precision is used only for these small statistics and metrics.
        # This avoids cancellation when a float32 fit is almost perfect.
        B64, target64 = B.double(), target.double()
        gram64 = (B64.T @ B64) / n_voxels
        rhs64 = (B64.T @ target64) / n_voxels
        target_power = float(torch.mean(target64.square()))
        del B64, target64
        gram, rhs = gram64.to(dtype), rhs64.to(dtype)

        field_basis = B if field_limit_Hz is not None else None
        if field_basis is not None:
            # Skip a large matrix multiply at every step if current constraints
            # alone already guarantee the nominal field bound everywhere.
            bound_box = torch.max(torch.abs(B) @ limits)
            bound_total = torch.max(torch.abs(B)) * total_current_limit_A
            if float(torch.minimum(bound_box, bound_total)) <= field_limit_Hz - 0.05:
                field_basis = None

        def project(I):
            return project_currents_tensor(
                I,
                max_current_A=limits,
                total_current_limit_A=total_current_limit_A,
                field_basis=field_basis,
                field_limit_Hz=float("inf") if field_limit_Hz is None else field_limit_Hz,
            )

        def actual_currents(I):
            if R is None:
                return I
            return torch.bmm(R, I.unsqueeze(-1)).squeeze(-1)

        def gradient(I):
            actual = actual_currents(I)
            residual_gradient = gram @ actual - rhs[:, None]
            if R is not None:
                residual_gradient = torch.bmm(R.transpose(1, 2), residual_gradient.unsqueeze(-1)).squeeze(-1)
            return (2.0 / n_times) * residual_gradient + (2.0 * ridge / I.numel()) * I

        def metrics(I):
            actual = actual_currents(I).double()
            field_mse = torch.mean(torch.sum(actual * (gram64 @ actual), dim=0)
                                   - 2 * torch.sum(rhs64[:, None] * actual, dim=0)) + target_power
            field_mse = float(torch.clamp(field_mse, min=0))
            loss = field_mse + ridge * float(I.double().square().mean())
            nrmse = None if target_power == 0 else math.sqrt(field_mse / target_power)
            return loss, nrmse, math.sqrt(field_mse)

        # A safe Lipschitz bound avoids expensive eigenproblems over all states.
        # ||R||₂² <= ||R||₁ ||R||∞, computed independently for every channel.
        largest_eigenvalue = float(torch.linalg.eigvalsh(gram).max().clamp(min=0))
        temporal_bound = 1.0
        if R is not None:
            # Physical first-order responses are nonnegative. Reuse their
            # storage instead of allocating another large [channel, ADC, state]
            # tensor; signed response operators remain supported as well.
            abs_R = R if float(R.amin()) >= 0 else torch.abs(R)
            norm_one = abs_R.sum(dim=1).amax(dim=1)
            norm_infinity = abs_R.sum(dim=2).amax(dim=1)
            temporal_bound = float((norm_one * norm_infinity).max())
            del abs_R
        lipschitz = 2 * largest_eigenvalue * temporal_bound / n_times + 2 * ridge / (n_channels * n_states)

        # The unconstrained static fit is a useful starting point for both fits.
        if initial_currents_A is None:
            matrix = gram64 + (ridge / n_channels) * torch.eye(n_channels, device=device, dtype=torch.float64)
            static_fit = (torch.linalg.pinv(matrix, hermitian=True) @ rhs64).to(dtype)
            I = project(static_fit[:, None].expand(n_channels, n_states).clone())
        else:
            I = tensor(initial_currents_A)
            if I.shape != (n_channels, n_states) or not bool(torch.isfinite(I).all()):
                raise ValueError("initial_currents_A must be finite [channel, state] commands.")
            I = project(I)
        loss, nrmse, rms = metrics(I)
        if loss > target_power:
            I = torch.zeros_like(I)
            loss, nrmse, rms = metrics(I)
        best_I, best_loss = I.clone(), loss
        history = [{"iteration": 0, "loss": loss, "nrmse": nrmse, "rms_Hz": rms}]
        extrapolated = I.clone()
        checkpoint_I = I.clone()
        checkpoint_loss = loss
        acceleration = 1.0
        step = 1.0 / lipschitz if lipschitz > 0 else 0.0
        converged = lipschitz == 0 or target_power == 0
        iterations = 0

        for iteration in range(1, max_iterations + 1):
            if converged:
                break
            previous = I
            I = project(extrapolated - step * gradient(extrapolated))
            next_acceleration = (1 + math.sqrt(1 + 4 * acceleration**2)) / 2
            extrapolated = I + ((acceleration - 1) / next_acceleration) * (I - previous)
            acceleration = next_acceleration
            iterations = iteration

            # Synchronize only at checkpoints, not after every GPU operation.
            if iteration % history_every and iteration != max_iterations:
                continue
            loss, nrmse, rms = metrics(I)
            if not math.isfinite(loss):
                raise RuntimeError("The field fit became nonfinite; use float64 or inspect the input scale.")
            history.append({"iteration": iteration, "loss": loss, "nrmse": nrmse, "rms_Hz": rms})
            if loss < best_loss:
                best_I, best_loss = I.clone(), loss
            increased = loss > best_loss + 1e-7 * max(1.0, target_power)
            if increased:
                # Monotone restart also protects the radial field safeguard.
                I = best_I.clone()
                extrapolated = I.clone()
                acceleration = 1.0
                if field_basis is not None:
                    step *= 0.5
            else:
                relative_step = float(torch.linalg.vector_norm(I - checkpoint_I)
                                      / (1 + torch.linalg.vector_norm(checkpoint_I)))
                relative_loss = abs(loss - checkpoint_loss) / max(1.0, abs(checkpoint_loss))
                converged = relative_step <= tolerance and relative_loss <= tolerance
            checkpoint_I = I.clone()
            checkpoint_loss = loss

        loss, nrmse, rms = metrics(best_I)
        currents = best_I.cpu().numpy().copy()
    synchronize()
    return {
        "currents_A": currents,
        "history": history,
        "nrmse": nrmse,
        "rms_Hz": rms,
        "loss": loss,
        "seconds": time.perf_counter() - started,
        "iterations": iterations,
        "converged": converged,
        "field_projection_is_radial": field_basis is not None,
        "runtime_info": dict(runtime_info, device=str(device), precision=str(dtype).split(".")[-1]),
    }
