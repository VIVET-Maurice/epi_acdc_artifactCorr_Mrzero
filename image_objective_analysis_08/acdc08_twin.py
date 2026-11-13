"""Independent re-implementation of the generation-08 EPI / ACDC signal model.

This file does NOT import any module of the pipeline and does not run MRzero.
It reads the arrays that notebook 08 saved in ``mrzero08_optimization.npz``
(object, B0, coil maps, gradients, trajectory, target) and rebuilds the same
forward model in closed form, so that every term can be switched on and off.

Signal model (identical to generation 08, see README.md for the derivation):

    s(t) = sum_p  A_p(t) * D_p(t) * exp(-2*pi*i * [k(t).r_p + Phi_p(t)])

    Phi_p(t) = b0_p * t + sum_c B_pc * J_c(t)              [cycles]
    J_c(t)   = integral_0^t  I_c(tau) dtau                 [A s]  (H @ commands)
    A_p(t)   = a_p * exp(-R_p * (t - t_ref)) * C_p(t)      amplitude, relaxation,
                                                           unresolved coherence
    D_p(t)   = prod_axis sinc(q_axis)                      box-voxel integral
    q_axis   = e_axis.k(t) + g0_p,axis * t + sum_c G_p,axis,c * J_c(t)   [cycles
               across the full voxel width along that axis]

Units: B0 [Hz], B [Hz/A], currents [A], time [s], k [cycles/m], r [m],
gradients [Hz per voxel width] and [Hz/A per voxel width]. Phase sign is
exp(-2*pi*i*cycles), as in ``acdc_gpu.ACDCModel._make_base``.

Array conventions: currents are ``[channel, line]`` = (32, 128); k-space and
images are ``[phase-encode row, readout column]`` = (128, 128). Row index is
the chronological EPI line. Voxel axes are 0 = phase encode (native row),
1 = readout (native column), 2 = slice normal.
"""

from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO / "MRzero_generations_08_09/results/mrzero08_20261112_135622_906979"
AXES = ("phase-encode", "readout", "slice")
DTYPE, CDTYPE = torch.float64, torch.complex128


def _step_integral(elapsed, tau):
    """Integral of the first-order response to a unit step, ``elapsed`` after it."""
    elapsed = np.maximum(elapsed, 0.0)
    return elapsed + tau * np.expm1(-elapsed / tau)


class Twin:
    """Closed-form generation-08 model built from one saved notebook run."""

    def __init__(self, run_dir=DEFAULT_RUN, threads=32):
        torch.set_num_threads(threads)
        run_dir = Path(run_dir)
        z = np.load(run_dir / "mrzero08_optimization.npz")
        self.run_dir = run_dir
        idx = z["parent_flat_index"]                      # voxel -> flat pixel of the 128x128 grid
        self.pixel_index = idx
        self.shape = z["kx_mInv"].shape                   # (lines, readout samples)
        self.n_lines, self.n_read = self.shape
        self.n_channels = z["currents_A"].shape[0]
        self.n_voxels = idx.size

        # --- object -------------------------------------------------------
        self.x = z["X_image_m"].ravel()[idx]              # voxel centres in the image frame [m]
        self.y = z["Y_image_m"].ravel()[idx]
        self.amplitude = z["subspin_amplitude_at_reference"][:, 0]   # at t_ref
        self.t_ref = float(z["echo_times_s"][0])          # first MEGRE echo (2.123 ms)
        self.rate = z["native_rate_sInv"].ravel()[idx]    # R2' / residual R2 [1/s]
        self.b0 = z["subspin_B0_Hz"][:, 0]                # [Hz]
        self.width = z["complex_echo_width_Hz"].ravel()[idx]        # unresolved dispersion [Hz]
        self.kind = z["complex_echo_dispersion_kind"].ravel()[idx]  # 0 none, 1 box, 2 Gaussian

        # --- ACDC ---------------------------------------------------------
        self.B = z["subspin_basis_Hz_per_A"][:, 0]                       # [voxel, channel] Hz/A
        self.g0 = z["voxel_b0_gradient_Hz_per_pixel"][:, 0]              # [voxel, axis] Hz / voxel width
        self.G = z["voxel_field_gradient_Hz_per_A_per_pixel"][:, 0]      # [voxel, axis, channel]
        self.corners = z["voxel_field_at_corners_Hz_per_A"][:, 0]        # [voxel, 8, channel]
        self.edges = z["voxel_edge_vectors_m"]                           # [axis, (x, y)] m
        self.max_current = z["channel_max_current_A"].astype(float)      # 0 = disabled output
        self.total_current = 30.0
        self.field_limit = 1000.0

        # --- trajectory and timing ---------------------------------------
        self.kx, self.ky = z["kx_mInv"].ravel(), z["ky_mInv"].ravel()   # cycles/m
        self.t = z["ADC_time_s"].ravel()                                # s after the RF centre
        self.TE = float(z["ADC_time_s"][self.n_lines // 2].mean())      # centre of the ky = 0 line
        line_centres = z["ADC_time_s"].mean(axis=1)
        self.esp = float(np.diff(line_centres).mean())
        self.line_centres = line_centres
        # One command per line, starting at its trigger; the next line ends it.
        self.line_start = line_centres[0] - 0.44e-3 + self.esp * np.arange(self.n_lines)
        self.line_end = self.line_start + self.esp
        self.tau = 133e-6 / np.log(100.0)                 # amplifier time constant
        a = self.t[:, None] - self.line_start[None]
        b = self.t[:, None] - self.line_end[None]
        self.H = _step_integral(a, self.tau) - _step_integral(b, self.tau)   # [sample, line] s

        # --- target of the notebook --------------------------------------
        self.reference = z["optimization_reference_image"]       # measured GRE / gain
        self.mask = z["optimization_reference_mask"]
        self.gain = float(z["optimization_fixed_gain"])
        self.saved = {name: z[name] for name in (
            "baseline_image", "uncalibrated_baseline_kdata", "ideal_image",
            "optimized_image", "optimized_kdata", "currents_A", "initial_currents_A",
            "relative_mse", "optimization_objective", "optimization_field_penalty",
            "optimization_residual_field_rms_Hz", "measured_image")}
        self.X_image, self.Y_image = z["X_image_m"], z["Y_image_m"]

        T = lambda v: torch.as_tensor(np.ascontiguousarray(v), dtype=DTYPE)
        self._t = {name: T(getattr(self, name)) for name in (
            "x", "y", "amplitude", "rate", "b0", "width", "B", "g0", "G", "kx", "ky", "t", "H")}
        self._kind = torch.as_tensor(self.kind.astype(np.int64))
        self._edges = T(self.edges)
        # Separable discrete Fourier reconstruction (normalised adjoint).
        ky_rows, kx_cols = z["ky_mInv"][:, 0], z["kx_mInv"][0]
        Yi, Xj = z["Y_image_m"][:, 0], z["X_image_m"][0]
        self._Ey = torch.as_tensor(np.exp(2j * np.pi * np.outer(Yi, ky_rows)), dtype=CDTYPE)
        self._Ex = torch.as_tensor(np.exp(2j * np.pi * np.outer(kx_cols, Xj)), dtype=CDTYPE)
        self._ref = T(self.reference)
        self._mask = T(self.mask.astype(float))
        self._ref_energy = float((self.mask * self.reference**2).sum())
        self._limits = T(self.max_current)
        self._corner_basis = T(self.corners.reshape(-1, self.n_channels))

    # ------------------------------------------------------------------
    # forward model
    # ------------------------------------------------------------------
    def signal(self, currents, *, b0=True, acdc_phase=True, acdc_gradient_axes=(0, 1, 2),
               b0_gradient=True, decay="evolving", voxel=True, chunk_lines=16,
               extra_gradient_Hz=None):
        """Return k-space ``[line, readout]`` for commands ``[channel, line]``.

        Switches (all True / default reproduce generation 08 exactly):
          b0                 B0 phase at the voxel centre.
          b0_gradient        B0 slope inside the voxel (in-plane only; z is assumed 0).
          acdc_phase         ACDC phase at the voxel centre.
          acdc_gradient_axes voxel axes along which the ACDC slope enters the sinc.
          decay              "evolving": amplitude follows relaxation and coherence loss;
                             "frozen": amplitude is held at its value at TE for all samples.
          voxel              False removes the three sinc factors (point spins).
          extra_gradient_Hz  optional [voxel, axis] additional static slope (diagnostic).
        """
        t = self._t
        currents = torch.as_tensor(currents, dtype=DTYPE)
        rows = []
        step = chunk_lines * self.n_read
        for first in range(0, self.t.size, step):
            sl = slice(first, first + step)
            time = t["t"][sl]
            J = currents @ t["H"][sl].T                                  # [channel, sample] A s
            cycles = t["x"][:, None] * t["kx"][sl] + t["y"][:, None] * t["ky"][sl]
            if b0:
                cycles = cycles + t["b0"][:, None] * time
            if acdc_phase:
                cycles = cycles + t["B"] @ J
            clock = time if decay == "evolving" else torch.full_like(time, self.TE)
            amplitude = t["amplitude"][:, None] * torch.exp(-t["rate"][:, None] * (clock - self.t_ref))
            q = t["width"][:, None] * clock
            coherence = torch.where(self._kind[:, None] == 1, torch.sinc(q),
                                    torch.where(self._kind[:, None] == 2,
                                                torch.exp(-2 * np.pi**2 * q**2), torch.ones_like(q)))
            amplitude = amplitude * coherence
            if voxel:
                for axis in range(3):
                    q = (self._edges[axis, 0] * t["kx"][sl] + self._edges[axis, 1] * t["ky"][sl])[None]
                    if b0 and b0_gradient:
                        q = q + t["g0"][:, axis, None] * time
                    if axis in acdc_gradient_axes:
                        q = q + t["G"][:, axis] @ J
                    if extra_gradient_Hz is not None:
                        q = q + torch.as_tensor(extra_gradient_Hz, dtype=DTYPE)[:, axis, None] * time
                    amplitude = amplitude * torch.sinc(q)
            rows.append((amplitude * torch.exp(-2j * np.pi * cycles)).sum(dim=0))
        return torch.cat(rows).reshape(self.shape)

    def image(self, kdata):
        """Normalised Fourier adjoint on the pixel grid of the notebook."""
        return self._Ey @ torch.as_tensor(kdata, dtype=CDTYPE) @ self._Ex / (self.n_lines * self.n_read)

    def simulate(self, currents, **switches):
        return self.image(self.signal(currents, **switches))

    # ------------------------------------------------------------------
    # objectives
    # ------------------------------------------------------------------
    def image_loss(self, image, reference=None):
        """J_image = sum_M (|image| - reference)^2 / sum_M reference^2.

        The default reference is the notebook target: measured GRE / fixed gain.
        """
        if reference is None:
            ref, energy = self._ref, self._ref_energy
        else:
            ref = torch.as_tensor(reference, dtype=DTYPE)
            energy = float((self._mask * ref**2).sum())
        return (self._mask * (torch.as_tensor(image).abs() - ref)**2).sum() / energy

    def mean_line_current(self, currents):
        """Actual mean current between successive line centres, [channel, 127]."""
        a = self.line_centres[:, None] - self.line_start[None]
        b = self.line_centres[:, None] - self.line_end[None]
        integral = _step_integral(a, self.tau) - _step_integral(b, self.tau)     # [centre, line]
        slopes = np.diff(integral, axis=0) / np.diff(self.line_centres)[:, None]
        return torch.as_tensor(currents, dtype=DTYPE) @ torch.as_tensor(slopes, dtype=DTYPE).T

    def field_penalty(self, currents, scale_Hz=100.0):
        """Field guidance of the notebook: mean((B0 + B @ mean current)^2) / scale^2."""
        residual = self._t["b0"][:, None] + self._t["B"] @ self.mean_line_current(currents)
        return (residual**2).mean() / scale_Hz**2

    def project(self, currents):
        """Box and total-current projection, then field contraction (as in acdc_gpu)."""
        currents = torch.as_tensor(currents, dtype=DTYPE)
        limits = self._limits[:, None]
        magnitude = currents.abs()
        clipped = torch.minimum(magnitude, limits)
        low = torch.zeros((1, currents.shape[1]), dtype=DTYPE)
        high = magnitude.amax(dim=0, keepdim=True)
        needs = clipped.sum(dim=0, keepdim=True) > self.total_current
        for _ in range(54):
            middle = (low + high) / 2
            over = torch.minimum((magnitude - middle).clamp(min=0), limits).sum(dim=0, keepdim=True) > self.total_current
            low, high = torch.where(over, middle, low), torch.where(over, high, middle)
        threshold = torch.where(needs, high, torch.zeros_like(high))
        projected = currents.sign() * torch.minimum((magnitude - threshold).clamp(min=0), limits)
        peak = (self._corner_basis @ projected).abs().amax(dim=0, keepdim=True)
        return projected * (self.field_limit / peak.clamp(min=1e-30)).clamp(max=1.0)

    # ------------------------------------------------------------------
    # per-voxel diagnostics at the centre of every EPI line
    # ------------------------------------------------------------------
    def amplitude_at(self, time):
        """Voxel amplitude A_p (relaxation and unresolved coherence) at ``time`` [s]."""
        q = self.width * time
        coherence = np.where(self.kind == 1, np.sinc(q),
                             np.where(self.kind == 2, np.exp(-2 * np.pi**2 * q**2), 1.0))
        return self.amplitude * np.exp(-self.rate * (time - self.t_ref)) * coherence

    def line_histories(self, currents):
        """Phase history of every voxel, sampled at the centre of every line.

        Returns ``phase`` (Phi_p, cycles), ``q_pe`` and ``q_slice`` (phase
        difference across the voxel along phase encode and along the slice
        normal, cycles), each ``[voxel, line]``. ``q_pe`` includes the
        encoding term; ``q_slice`` is ACDC only because the model assumes no
        through-slice B0 slope.
        """
        centres = self.line_centres
        H = (_step_integral(centres[:, None] - self.line_start[None], self.tau)
             - _step_integral(centres[:, None] - self.line_end[None], self.tau))
        J = np.asarray(currents, dtype=float) @ H.T                         # [channel, line] A s
        ky = self.ky.reshape(self.shape)[:, 0]
        return dict(
            phase=self.b0[:, None] * centres[None] + self.B @ J,
            q_pe=(self.edges[0, 1] * ky)[None] + self.g0[:, 0, None] * centres[None] + self.G[:, 0] @ J,
            q_slice=self.G[:, 2] @ J)

    def local_echo(self, currents):
        """Local echo of every voxel and what the image sees there.

        Neighbouring voxels along the phase-encode direction add in phase when
        ``q_pe = 0``; the first such instant is the local echo (line 64, t = TE,
        when there is no field). Returns per voxel: ``time`` [s] (nan if q_pe
        never crosses zero), ``crossings``, and ``shift_pixels``, the
        phase-encode displacement implied by the slope of the phase history
        over +-4 lines around the local echo (slope [Hz] * N * echo spacing).
        """
        history = self.line_histories(currents)
        q, centres = history["q_pe"], self.line_centres
        sign = np.sign(q)
        crossing = sign[:, :-1] * sign[:, 1:] <= 0
        count = crossing.sum(axis=1)
        first = np.argmax(crossing, axis=1)
        rows = np.arange(len(q))
        q0, q1 = q[rows, first], q[rows, np.minimum(first + 1, q.shape[1] - 1)]
        fraction = np.divide(q0, q0 - q1, out=np.zeros_like(q0), where=q0 != q1)
        time = np.where(count > 0, centres[first] + fraction * self.esp, np.nan)
        line = np.where(count > 0, np.clip(first, 4, self.n_lines - 5), self.n_lines // 2)
        slope = (history["phase"][rows, line + 4] - history["phase"][rows, line - 4]) / (8 * self.esp)
        return dict(time=time, crossings=count, shift_pixels=slope * self.n_lines * self.esp,
                    q_pe=q, q_slice=history["q_slice"])

    def summary(self, currents, perfect=None):
        """Scalar description of one waveform (all quantities are simulated)."""
        currents = np.asarray(currents, dtype=float)
        with torch.no_grad():
            if perfect is None:
                perfect = self.simulate(np.zeros_like(currents), b0=False).abs()
            image = self.simulate(currents).abs()
            result = {
                "J_vs_GRE": float(self.image_loss(image)),
                "J_vs_perfect": float(self.image_loss(image, perfect)),
                "signal_vs_perfect": float(image.sum() / perfect.sum()),
                "field_rms_Hz": 100 * float(self.field_penalty(currents))**0.5}
        echo = self.local_echo(currents)
        weight = self.amplitude_at(self.TE)
        w = weight / weight.sum()
        valid = np.isfinite(echo["time"])
        centre = self.n_lines // 2
        mean = currents.mean(axis=1, keepdims=True)
        result.update(
            mean_total_current_A=float(np.abs(currents).sum(axis=0).mean()),
            time_varying_fraction=float(np.linalg.norm(currents - mean) / max(np.linalg.norm(currents), 1e-30)),
            median_local_echo_ms=1e3 * float(np.nanmedian(echo["time"])),
            voxels_without_local_echo_percent=100 * float((~valid).mean()),
            amplitude_at_local_echo_vs_at_TE=float(
                np.where(valid, self.amplitude_at(np.where(valid, echo["time"], self.TE)), 0.0).sum() / weight.sum()),
            rms_shift_pixels=float(np.sqrt((w * echo["shift_pixels"]**2).sum())),
            rms_slice_dephasing_at_TE_cycles=float(np.sqrt((echo["q_slice"][:, centre]**2).mean())),
            voxels_slice_dephasing_above_half_cycle_percent=100 * float((np.abs(echo["q_slice"][:, centre]) > 0.5).mean()))
        return result

    def to_pixels(self, values, fill=np.nan):
        """Place one value per voxel on the 128 x 128 pixel grid."""
        grid = np.full(self.shape[0] * self.shape[1], fill, dtype=float)
        grid[self.pixel_index] = np.asarray(values, dtype=float)
        return grid.reshape(self.shape)
