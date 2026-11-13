"""Generation 10: choose the ACDC currents by minimising the phase at every echo.

Idea
----
At the echo of EPI line ``n`` (time ``T_n`` after the RF pulse), the phase
accumulated by a point ``p`` of the object is

    phi_n(p) = 2 pi * B0(p) * T_n  +  2 pi * B(p) . J_n ,     J_n = sum_i H[n, i] * x_i

* ``B0(p)`` [Hz] is the off-resonance and ``B(p)`` [Hz/A] the field of the coils;
* ``x_i`` [A] is the vector of currents commanded during line ``i``;
* ``H[n, i]`` [s] is the time during which the current of line ``i`` has acted
  before the echo of line ``n``. ``H`` is (almost) lower triangular: every past
  line counts for one echo spacing and the current line for the part already
  played. ``J_n`` is therefore the running sum of the currents, in A s.

The objective is the squared phase, averaged over the object and summed over
the echoes:

    Psi(x) = sum_n ||phi_n||^2 ,        ||phi_n||^2 = mean over the object of phi_n^2

If ``Psi`` were zero, every line would be acquired with the same (zero) phase:
no distortion and no signal loss caused by B0.

``Psi`` is a quadratic function of the currents and the hardware limits form a
convex set, so the problem has no local minima. It is solved with an
accelerated projected gradient method (FISTA). No simulation is needed
during the optimisation.

Two options for the mean over the object
----------------------------------------
``intravoxel_weight``. A voxel is not a point: it has the size of a pixel in
the slice plane and the whole slice thickness. If the phase varies linearly
inside a voxel, with value ``phi_c`` at the centre and total changes
``d_x, d_y, d_z`` across it, the objective uses

    phi_c^2 + intravoxel_weight * (d_x^2 + d_y^2 + d_z^2) .

* ``0``: only the voxel centres count (the formula above, taken literally);
* ``1/12``: exact mean of ``phi^2`` over the volume of the voxel;
* larger values protect more against signal loss inside the voxels.

``voxel_weights``. One non-negative weight per voxel (default: all equal).
A weight of zero removes a voxel from the mean, for example where the signal
is lost whatever the currents.
"""

import numpy as np

from acdc_hardware import integrated_current_s, project_onto_limits

TWO_PI = 2.0 * np.pi


class PhaseProblem:
    """Phase of the object at every EPI echo, as a function of the currents."""

    def __init__(self, phantom, coil_fields, sequence, hardware,
                 intravoxel_weight=0.0, voxel_weights=None):
        support = phantom.support
        self.b0_Hz = phantom.B0_Hz[support]                 # [voxels]
        self.coil_maps = coil_fields.maps_Hz_per_A          # [voxels, channels]
        self.b0_gradient_Hz = phantom.B0_gradient_Hz[support]       # [voxels, 3]
        self.coil_gradients = coil_fields.gradients_Hz_per_A        # [voxels, 3, channels]
        self.echo_time_s = sequence.echo_time_s             # T_n, [echoes]
        # H[n, i]: seconds during which state i has acted before echo n.
        self.current_integral_s = integrated_current_s(
            sequence.echo_time_s, sequence.state_start_s, sequence.state_end_s,
            hardware.response_tau_s)

        # Weighted mean over the object: every row is scaled by sqrt(weight).
        weights = np.ones(len(self.b0_Hz)) if voxel_weights is None else np.asarray(voxel_weights, float)
        scale = np.sqrt(weights / weights.sum())

        # Rows of the least-squares system: one per voxel centre, plus three
        # per voxel for the phase change across it.
        field_rows = [scale * self.b0_Hz]
        coil_rows = [scale[:, None] * self.coil_maps]
        if intravoxel_weight > 0:
            across = scale * np.sqrt(intravoxel_weight)
            for axis in range(3):
                field_rows.append(across * self.b0_gradient_Hz[:, axis])
                coil_rows.append(across[:, None] * self.coil_gradients[:, axis])
        field_rows, coil_rows = np.concatenate(field_rows), np.concatenate(coil_rows)

        # Speed-up without approximation. The sum of squares of
        # (coil_rows @ J + field_rows * T) only depends on the small triangular
        # factor of [coil_rows, field_rows]: thousands of voxels are replaced
        # by (channels + 1) rows.
        triangle = np.linalg.qr(np.column_stack((coil_rows, field_rows)), mode="r")
        self.coil_factor = triangle[:, :-1]     # [channels + 1, channels]
        self.field_factor = triangle[:, -1]     # [channels + 1]

    def phase_rad(self, currents_A):
        """phi_n(p) at the voxel centres, shape [voxels, echoes], in radians."""
        running_sum = currents_A @ self.current_integral_s.T          # J_n, [channels, echoes]
        return TWO_PI * (np.outer(self.b0_Hz, self.echo_time_s) + self.coil_maps @ running_sum)

    def phase_across_voxel_rad(self, currents_A, axis):
        """Phase change across each voxel along one axis (0: x, 1: y, 2: z).

        Shape [voxels, echoes]. A change of 2 pi cancels the signal of the voxel.
        """
        running_sum = currents_A @ self.current_integral_s.T
        return TWO_PI * (np.outer(self.b0_gradient_Hz[:, axis], self.echo_time_s)
                         + self.coil_gradients[:, axis] @ running_sum)

    def residual_frequency_Hz(self, currents_A):
        """Mean frequency of every voxel between consecutive echoes, [voxels, echoes - 1].

        This is the phase gained from one echo to the next, divided by
        2 pi and by the echo spacing: what a field map would show.
        """
        return np.diff(self.phase_rad(currents_A), axis=1) / (TWO_PI * np.diff(self.echo_time_s))

    def mean_square_phase(self, currents_A):
        """||phi_n||^2 for every echo n, in rad^2 (with the options of this problem)."""
        compressed = (self.coil_factor @ currents_A @ self.current_integral_s.T
                      + np.outer(self.field_factor, self.echo_time_s))
        return TWO_PI**2 * np.sum(compressed**2, axis=0)

    def objective(self, currents_A):
        """Psi = sum over the echoes of ||phi_n||^2, in rad^2."""
        return float(self.mean_square_phase(currents_A).sum())


def minimize_least_squares(left, right, offset, project, initial,
                           max_iterations=3000, tolerance=1e-4, window=50):
    """Minimise ``||left @ X @ right.T + offset||^2`` over the feasible ``X``.

    ``project`` returns the feasible point closest to its argument.
    Accelerated projected gradient (FISTA) with a fixed, safe step size.
    Whenever the objective would increase, the momentum is dropped and a plain
    gradient step is taken instead, so the objective never increases.
    Stops when the objective has decreased by less than ``tolerance``
    (relative) over the last ``window`` iterations.

    Returns the solution and the objective value after every iteration.
    """
    def value(X):
        return float(np.sum((left @ X @ right.T + offset)**2))

    def gradient(X):
        return 2.0 * left.T @ (left @ X @ right.T + offset) @ right

    # 1 / (largest curvature of the objective).
    step = 0.5 / (np.linalg.norm(left, 2) * np.linalg.norm(right, 2))**2

    best = project(initial)
    best_value = value(best)
    history = [best_value]
    lookahead, momentum = best, 1.0
    for _ in range(max_iterations):
        candidate = project(lookahead - step * gradient(lookahead))
        candidate_value = value(candidate)
        if candidate_value > best_value:
            if lookahead is best:        # even a plain gradient step fails: finished
                break
            lookahead, momentum = best, 1.0
            continue
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        lookahead = candidate + (momentum - 1.0) / next_momentum * (candidate - best)
        best, best_value, momentum = candidate, candidate_value, next_momentum
        history.append(best_value)
        if len(history) > window and history[-window - 1] - best_value <= tolerance * history[-window - 1]:
            break
    return best, np.asarray(history)


def optimize_phase(problem, hardware, coil_fields, initial_currents_A=None, **solver_options):
    """Minimise Psi under the hardware limits.

    Returns a dictionary with the commands ``currents_A`` [channels, lines]
    and the value of Psi after every iteration.
    """
    n_states = problem.current_integral_s.shape[1]
    if initial_currents_A is None:
        initial_currents_A = np.zeros((hardware.n_channels, n_states))

    def project(currents):
        return project_onto_limits(currents, hardware, coil_fields.maps_Hz_per_A,
                                   coil_fields.gradients_Hz_per_A)

    currents, history = minimize_least_squares(
        left=problem.coil_factor, right=problem.current_integral_s,
        offset=np.outer(problem.field_factor, problem.echo_time_s),
        project=project, initial=initial_currents_A, **solver_options)
    return {"currents_A": currents, "objective_history": TWO_PI**2 * history}


def static_shim(problem, hardware, coil_fields, **solver_options):
    """Conventional shim, for comparison: one constant current per channel.

    Minimises the mean square of the residual *frequency* ``B0 + B @ I`` over
    the object (with the same options as ``problem``), under the same
    hardware limits. Returns the currents [channels].
    """
    def project(currents):
        return project_onto_limits(currents, hardware, coil_fields.maps_Hz_per_A,
                                   coil_fields.gradients_Hz_per_A)

    currents, _ = minimize_least_squares(
        left=problem.coil_factor, right=np.ones((1, 1)),
        offset=problem.field_factor[:, None], project=project,
        initial=np.zeros((hardware.n_channels, 1)), **solver_options)
    return currents[:, 0]
