"""GPU-capable PyTorch optimizers for the original ACDC shimming problem.

The conventional initializer minimizes the same regularized box-constrained
least-squares problem as ``initialize_acdc_currents.m``. The image optimizer
minimizes magnitude MSE with the same current bounds and linear field bounds
as ``epi_acdc_artCor.m``. No penalty term replaces either constraint.

Adam and optional L-BFGS use autograd through a differentiable MRzero adapter.
Each independent current state is radially mapped into the feasible polytope.
This map covers the original feasible set: every feasible state maps to itself.
The nonlinear image objective remains nonconvex, so no global optimum is claimed.
"""

from __future__ import annotations

import math
from time import perf_counter

import torch


def _device(device):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _basis(shim_maps, mask, device, dtype):
    maps = torch.as_tensor(shim_maps, device=device, dtype=dtype)
    if maps.ndim != 3 or maps.shape[-1] < 1 or not torch.isfinite(maps).all():
        raise ValueError("shim_maps must be finite with shape (Nx, Ny, Nc).")
    if mask is None:
        mask = torch.ones(maps.shape[:2], device=device, dtype=torch.bool)
    else:
        mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if mask.shape != maps.shape[:2] or not mask.any():
        raise ValueError("mask must match the image shape and select some voxels.")
    return maps, mask, maps[mask]


def initialize_acdc_currents(
    b0,
    shim_maps,
    current_limit_A,
    mask=None,
    device=None,
    *,
    max_steps=20000,
    tolerance=1e-9,
):
    """Solve the conventional static shim using accelerated projected gradient.

    All matrix operations stay on the selected device. The small quadratic
    problem is solved in float64, including on CUDA, with the MATLAB numerical
    regularization ``lambda = 1e-8 * trace(H) / Nc``. ``current_limit_A`` gives
    symmetric bounds. This initializer has no dynamic field-amplitude bound.

    Returns a dictionary with a channel vector ``currents_a``, residual maps,
    RMS values, a projected-gradient convergence diagnostic, and elapsed time.
    """
    if not math.isfinite(current_limit_A) or current_limit_A <= 0:
        raise ValueError("current_limit_A must be finite and positive.")
    if max_steps < 1 or tolerance <= 0:
        raise ValueError("max_steps and tolerance must be positive.")
    device = _device(device)
    started = perf_counter()
    maps, mask, a = _basis(shim_maps, mask, device, torch.float64)
    b0 = torch.as_tensor(b0, device=device, dtype=torch.float64)
    if b0.shape != maps.shape[:2] or not torch.isfinite(b0).all():
        raise ValueError("b0 must be finite and have the image shape.")
    channels = maps.shape[-1]
    hessian = 2 * (a.T @ a)
    linear = 2 * (a.T @ b0[mask])
    regularization = 1e-8 * hessian.trace() / channels
    if not torch.isfinite(regularization) or regularization <= 0:
        regularization = torch.tensor(1e-8, device=device, dtype=torch.float64)
    hessian = hessian + regularization * torch.eye(
        channels, device=device, dtype=torch.float64
    )

    # Scaling by the Lipschitz constant leaves the minimizer unchanged.
    lipschitz = torch.linalg.eigvalsh(hessian)[-1]
    h = hessian / lipschitz
    f = linear / lipschitz
    x = torch.linalg.solve(hessian, -linear).clamp(
        -current_limit_A, current_limit_A
    )
    y = x.clone()
    momentum = torch.ones((), device=device, dtype=torch.float64)
    converged = False
    for step in range(1, max_steps + 1):
        next_x = (y - (h @ y + f)).clamp(-current_limit_A, current_limit_A)
        next_momentum = (1 + (1 + 4 * momentum.square()).sqrt()) / 2
        next_y = next_x + (momentum - 1) / next_momentum * (next_x - x)
        # Adaptive restart prevents momentum from oscillating near a bound.
        restart = torch.dot(y - next_x, next_x - x) > 0
        y = torch.where(restart, next_x, next_y)
        momentum = torch.where(restart, torch.ones_like(momentum), next_momentum)
        x = next_x
        if step % 25 == 0 or step == max_steps:
            projected_gradient = x - (x - (h @ x + f)).clamp(
                -current_limit_A, current_limit_A
            )
            if projected_gradient.abs().max().item() <= tolerance:
                converged = True
                break

    shim_field = torch.einsum("xyc,c->xy", maps, x)
    residual = b0 + shim_field
    before = b0[mask].square().mean().sqrt().item()
    after = residual[mask].square().mean().sqrt().item()
    return {
        "currents_a": x.detach(),
        "shim_field_hz": shim_field.detach(),
        "residual_b0_hz": residual.detach(),
        "rms_before_hz": before,
        "rms_after_hz": after,
        "percent_improvement": 100 * (1 - after / before) if before else 0.0,
        "regularization": regularization.item(),
        "iterations": step,
        "converged": converged,
        "projected_gradient_max": projected_gradient.abs().max().item(),
        "elapsed_s": perf_counter() - started,
        "device": str(device),
        "method": "accelerated projected gradient for a box-constrained quadratic",
    }


def _line_states(shim_mode, n_lines, device):
    if shim_mode not in (1, 2, 3) or n_lines < 1:
        raise ValueError("shim_mode must be 1, 2, or 3 and n_lines positive.")
    lines = torch.arange(n_lines, device=device)
    if shim_mode == 1:
        return torch.zeros_like(lines), 1
    if shim_mode == 2:
        if n_lines < 2:
            raise ValueError("odd/even shimming needs at least two lines.")
        return lines % 2, 2
    return lines, n_lines


def _initial_states(initial_current, channels, line_states, states, device, dtype):
    initial = torch.as_tensor(initial_current, device=device, dtype=dtype)
    if not torch.isfinite(initial).all():
        raise ValueError("initial_current must be finite.")
    if initial.shape == (channels,):
        return initial[:, None].expand(channels, states).clone()
    if initial.shape == (channels, states):
        return initial.clone()
    if initial.shape == (channels, line_states.numel()):
        compact = torch.stack(
            [initial[:, line_states == state].mean(dim=1) for state in range(states)],
            dim=1,
        )
        if not torch.allclose(initial, compact[:, line_states]):
            raise ValueError("Initial line currents must respect the selected shim mode.")
        return compact
    raise ValueError("initial_current must have shape (Nc,), (Nc, Nstates), or (Nc, Nlines).")


def optimize_currents(
    forward_model,
    reference,
    initial_current,
    shim_maps,
    constraint_mask,
    field_limit_Hz,
    current_limit_A,
    shim_mode=3,
    n_lines=None,
    device=None,
    *,
    adam_steps=300,
    lbfgs_steps=100,
    learning_rate=0.05,
    log_every=25,
    callback=None,
    objective_mask=None,
):
    """Minimize magnitude image MSE with hard current and shim-field bounds.

    ``forward_model(currents)`` returns a differentiable complex Torch image;
    currents have shape (Nc, Nlines). Model attributes ``device`` and ``n_lines``
    are used when these arguments are omitted. Modes are 1=constant, 2=odd/even,
    and 3=per-line. All image pixels enter the original MATLAB objective unless
    ``objective_mask`` is explicitly provided.

    For each state z, I = z / max(1, max(abs(z))/Imax,
    max(abs(A @ z))/field_limit). Thus both constraints hold at every objective
    evaluation without a penalty or a block-diagonal constraint matrix. The
    constraints bound the ACDC field alone, not B0 + ACDC. The supplied mask is
    used exactly, including the original sparse constraint mask.

    Adam explores the nonconvex objective; optional L-BFGS refines the best
    state using analytic autograd derivatives and a strong-Wolfe line search.
    The loss is divided by mean(reference magnitude squared) for conditioning;
    the returned MSE and history are in the original image units. Float64
    current parameters and constraint calculations preserve boundary precision;
    the forward adapter may cast these currents to its faster float32 dtype.
    This is a GPU-capable solver, not a claim of universally fastest convergence.

    Returns best feasible ``currents_a``, compact ``state_currents_a``, history,
    and diagnostics. ``callback(record)`` runs at logging intervals.
    """
    if not math.isfinite(current_limit_A) or current_limit_A <= 0:
        raise ValueError("current_limit_A must be finite and positive.")
    if math.isnan(field_limit_Hz) or field_limit_Hz <= 0:
        raise ValueError("field_limit_Hz must be positive (infinity is allowed).")
    if adam_steps < 0 or lbfgs_steps < 0 or learning_rate <= 0 or log_every < 1:
        raise ValueError("Invalid optimizer iteration count, learning rate, or log interval.")
    device = _device(device or getattr(forward_model, "device", None))
    if n_lines is None:
        n_lines = getattr(forward_model, "n_lines", None)
    if n_lines is None:
        raise ValueError("n_lines or forward_model.n_lines must be supplied.")
    started = perf_counter()
    maps, _, a = _basis(shim_maps, constraint_mask, device, torch.float64)
    line_states, states = _line_states(shim_mode, n_lines, device)
    initial = _initial_states(
        initial_current, maps.shape[-1], line_states, states, device, torch.float64
    )
    reference = torch.as_tensor(reference, device=device).abs().to(torch.float64).detach()
    if not torch.isfinite(reference).all():
        raise ValueError("reference must be finite.")
    if objective_mask is None:
        objective_mask = torch.ones_like(reference, dtype=torch.bool)
    else:
        objective_mask = torch.as_tensor(objective_mask, device=device, dtype=torch.bool)
    if objective_mask.shape != reference.shape or not objective_mask.any():
        raise ValueError("objective_mask must match reference and select some pixels.")
    loss_scale = reference[objective_mask].square().mean().clamp_min(1e-12)

    def feasible(raw):
        box_ratio = raw.abs().amax(dim=0) / current_limit_A
        field_ratio = (a @ raw).abs().amax(dim=0) / field_limit_Hz
        ratio = torch.maximum(box_ratio, field_ratio).clamp_min(1.0)
        return raw / ratio[None, :]

    # The tiny margin accommodates float64 division/matmul rounding only.
    numerical_margin = 8 * torch.finfo(torch.float64).eps
    initial = feasible(initial) * (1 - numerical_margin)
    raw = torch.nn.Parameter(initial.clone())
    best_states = initial.clone()
    with torch.no_grad():
        image = forward_model(initial[:, line_states])
        if image.shape != reference.shape:
            raise ValueError("Forward image shape must match reference.")
        best_mse = (image.abs() - reference)[objective_mask].square().mean()
    if not torch.isfinite(best_mse):
        raise ValueError("The initial objective is not finite.")
    initial_mse = best_mse.item()
    history = []
    evaluations = 1

    def evaluate():
        nonlocal best_mse, best_states, evaluations
        currents = feasible(raw)
        image = forward_model(currents[:, line_states])
        mse = (image.abs() - reference)[objective_mask].square().mean()
        evaluations += 1
        with torch.no_grad():
            improved = mse < best_mse
            best_states = torch.where(improved, currents.detach(), best_states)
            best_mse = torch.where(improved, mse.detach(), best_mse)
        return mse / loss_scale, mse

    def record(step, stage, mse):
        item = {
            "step": step,
            "stage": stage,
            "mse": mse.item(),
            "best_mse": best_mse.item(),
            "elapsed_s": perf_counter() - started,
        }
        if not math.isfinite(item["mse"]):
            raise RuntimeError("Optimization produced a nonfinite objective.")
        history.append(item)
        if callback is not None:
            callback(item)

    record(0, "initial", best_mse)
    adam = None
    if adam_steps:
        adam = torch.optim.Adam([raw], lr=learning_rate, fused=device.type == "cuda")
    for step in range(1, adam_steps + 1):
        adam.zero_grad(set_to_none=True)
        loss, mse = evaluate()
        loss.backward()
        adam.step()
        if step % log_every == 0 or step == adam_steps:
            record(step, "adam", mse)
    # Evaluate the last Adam update, which has not yet entered the history.
    with torch.no_grad():
        _, mse = evaluate()
    if adam_steps:
        record(adam_steps, "adam-final", mse)

    if lbfgs_steps:
        with torch.no_grad():
            raw.copy_(best_states)
        lbfgs = torch.optim.LBFGS(
            [raw], lr=1.0, max_iter=1, max_eval=12,
            tolerance_grad=1e-7, tolerance_change=1e-9,
            history_size=20, line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            loss, _ = evaluate()
            loss.backward()
            return loss

        for step in range(1, lbfgs_steps + 1):
            lbfgs.step(closure)
            with torch.no_grad():
                _, mse = evaluate()
            if step % log_every == 0 or step == lbfgs_steps:
                record(step, "lbfgs", mse)

    # Repair only floating-point boundary errors, then compare against the
    # already feasible initializer so the returned result never worsens it.
    with torch.no_grad():
        solution = feasible(best_states) * (1 - numerical_margin)
        image = forward_model(solution[:, line_states])
        final_mse = (image.abs() - reference)[objective_mask].square().mean()
        if final_mse.item() > initial_mse:
            solution = initial
            final_mse = torch.as_tensor(initial_mse, device=device)
        max_current = solution.abs().max().item()
        max_field = (a @ solution).abs().max().item()
    current_violation = max(0.0, max_current - current_limit_A)
    field_violation = max(0.0, max_field - field_limit_Hz)
    feasible_result = current_violation <= 1e-10 and field_violation <= 1e-8
    if not feasible_result:
        raise RuntimeError("Returned current states failed the hard-constraint check.")
    if adam_steps and lbfgs_steps:
        method = "Adam + L-BFGS with hard-feasible radial parameterization"
    elif adam_steps:
        method = "Adam with hard-feasible radial parameterization"
    elif lbfgs_steps:
        method = "L-BFGS with hard-feasible radial parameterization"
    else:
        method = "feasible initialization only"
    return {
        "currents_a": solution[:, line_states].detach(),
        "state_currents_a": solution.detach(),
        "history": history,
        "diagnostics": {
            "initial_mse": initial_mse,
            "final_mse": final_mse.item(),
            "relative_improvement": 1 - final_mse.item() / initial_mse if initial_mse else 0.0,
            "max_current_a": max_current,
            "max_constraint_field_hz": max_field,
            "current_violation_a": current_violation,
            "field_violation_hz": field_violation,
            "feasible": feasible_result,
            "constraint_voxels": a.shape[0],
            "objective_voxels": int(objective_mask.sum().item()),
            "shim_mode": shim_mode,
            "independent_states": states,
            "forward_evaluations": evaluations + 1,
            "adam_steps": adam_steps,
            "lbfgs_steps": lbfgs_steps,
            "elapsed_s": perf_counter() - started,
            "device": str(device),
            "optimizer_dtype": "float64",
            "numerical_feasibility_margin": numerical_margin,
            "method": method,
        },
    }
