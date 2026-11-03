"""Differentiable full-volume Cartesian EPI with a repeated partition waveform.

The native point-spin grid is (readout, phase, partition), centered on its RF
origin. One complete EPI train follows each partition RF; its DAC waveform is
repeated and physical currents initially vanish. Fields are supplied on the
actual 3D grid, never obtained by repeating a 2D map. A negative-sign 3D Fourier
signal includes B0 and the exact first-order integrated shim current at every
temporal interpolation node. Nodes are Chebyshev-Lobatto points within each
actual ADC window, not line centers. An exact-readout mode is also available.

Only X/Z encoding uses FFTs. Y encoding and uniform per-line readout offsets
remain explicit phases. Checkpointing bounds the saved autograd intermediates
to one line. Neither waveform projection nor hardware operation occurs here.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

_EXPERIMENT = Path(__file__).resolve().parent.parent / "Experiment"
if str(_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT))
from acdc_gpu import response_matrices


def _real(value, name, shape=None):
    array = np.asarray(value)
    if (np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number)
            or not np.isfinite(array).all() or (shape is not None and array.shape != shape)):
        raise ValueError(f"{name} must be finite real numeric data" + (f" with shape {shape}." if shape else "."))
    return array.astype(float, copy=False)


def _fft_centered(value, dims, *, inverse=False):
    transform = torch.fft.ifftn if inverse else torch.fft.fftn
    return torch.fft.fftshift(transform(torch.fft.ifftshift(value, dim=dims), dim=dims), dim=dims)


def _chebyshev_plan(times, count):
    """Barycentric interpolation at actual times; endpoints are included."""
    low, high = float(times.min()), float(times.max())
    normalized_nodes = -np.cos(np.pi * np.arange(count) / (count - 1))
    nodes = (low + high) / 2 + (high - low) / 2 * normalized_nodes
    samples = (2 * times - low - high) / (high - low)
    barycentric = (-1.) ** np.arange(count)
    barycentric[[0, -1]] *= .5
    difference = samples[:, None] - normalized_nodes[None, :]
    exact = np.abs(difference) <= 64 * np.finfo(float).eps
    weights = barycentric[None, :] / np.where(exact, 1., difference)
    weights /= weights.sum(axis=1, keepdims=True)
    for row in np.flatnonzero(exact.any(axis=1)):
        weights[row] = 0.
        weights[row, np.flatnonzero(exact[row])[0]] = 1.
    return nodes, weights


class VolumeEPIModel:
    """Full acquired 3D point-spin EPI model, with generic channel count.

    ``volume``: amplitude and b0_Hz (Nx,Ny,Nz), fieldmaps_Hz_per_A
    (Nx,Ny,Nz,Nc), FOV_m length three. Optional R2_sInv is a nonnegative
    scalar or 3D map, and amplitude_reference_time_s defaults to zero.
    ``acquisition``: actual kx_mInv/ky_mInv/time_s (Ny,Nx), RF-relative
    line_start_s/line_end_s, and optional kz_partition_mInv. Phase rows and
    kx samples must already be sorted ascending; times may decrease along a
    negative readout. ``line_index`` is unnecessary: exact response weights
    relate each time to the chronologically ordered command windows.

    ``forward`` returns complex image and k-space, both (Nx,Ny,Nz).
    Reconstruction is the centered nominal-grid inverse FFT, including the
    small actual trajectory offsets in the forward signal. The current
    integrals reset at each RF; a zero-current start per partition is assumed.
    This model has no inferred receive-coil phase, slice selection profile,
    subvoxel dispersion, or recoverable/nonrecoverable relaxation split.
    """

    def __init__(self, volume, acquisition, *, device="cpu", dtype=torch.float64,
                 interpolation_nodes=12, checkpoint_lines=True, current_response=None,
                 max_current_A=3.0):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64.")
        self.dtype = dtype
        self.complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.checkpoint_lines = bool(checkpoint_lines)
        amplitude = _real(volume["amplitude"], "amplitude")
        if amplitude.ndim != 3 or min(amplitude.shape) < 2 or np.any(amplitude < 0) or not np.any(amplitude > 0):
            raise ValueError("amplitude must be a nonnegative 3D grid with at least two voxels per axis and some signal.")
        self.shape = amplitude.shape
        self.nx, self.ny, self.nz = self.shape
        b0 = _real(volume["b0_Hz"], "b0_Hz", self.shape)
        maps = _real(volume["fieldmaps_Hz_per_A"], "fieldmaps_Hz_per_A")
        if maps.ndim != 4 or maps.shape[:3] != self.shape or maps.shape[-1] < 1:
            raise ValueError("fieldmaps_Hz_per_A must have shape (Nx,Ny,Nz,Nc).")
        self.nc = maps.shape[-1]
        self.FOV_m = _real(volume["FOV_m"], "FOV_m", (3,))
        if np.any(self.FOV_m <= 0):
            raise ValueError("FOV_m must be positive.")
        if "FOV_m" in acquisition and not np.allclose(self.FOV_m, acquisition["FOV_m"], rtol=0, atol=1e-6):
            raise ValueError("Object and acquired EPI FOV_m disagree.")
        self.axes_m = tuple((np.arange(n) - n // 2) * length / n
                            for n, length in zip(self.shape, self.FOV_m))
        kx = _real(acquisition["kx_mInv"], "kx_mInv", (self.ny, self.nx))
        ky = _real(acquisition["ky_mInv"], "ky_mInv", kx.shape)
        self.time_s = _real(acquisition["time_s"], "time_s", kx.shape).copy()
        if np.any(self.time_s < 0) or np.any(np.ptp(self.time_s, axis=1) <= 0):
            raise ValueError("ADC times must be RF-relative, nonnegative, and span each readout.")
        rf = np.asarray(acquisition.get("rf_time_s", 0.))
        if not np.isfinite(rf).all() or np.any(rf != 0):
            raise ValueError("A repeated EPI partition needs RF-relative times with rf_time_s=0.")
        self.nstates = self.ny
        self.line_start_s = _real(acquisition["line_start_s"], "line_start_s").reshape(-1).copy()
        self.line_end_s = _real(acquisition["line_end_s"], "line_end_s").reshape(-1).copy()
        if (self.line_start_s.shape != (self.nstates,) or self.line_end_s.shape != (self.nstates,)
                or np.any(self.line_start_s < 0) or np.any(np.diff(self.line_start_s) <= 0)
                or np.any(self.line_end_s <= self.line_start_s)
                or np.any(self.line_end_s[:-1] > self.line_start_s[1:])):
            raise ValueError("One finite, chronologically ordered nonoverlapping current window is required per phase row.")
        edges = np.unique(np.r_[self.line_start_s, self.line_end_s])
        for times in self.time_s:
            if np.any((edges > times.min() + 1e-12) & (edges < times.max() - 1e-12)):
                raise ValueError("A current transition inside an ADC window requires piecewise temporal interpolation.")
        labels = np.searchsorted(self.line_start_s, self.time_s.mean(axis=1), side="right")
        if (not np.array_equal(np.sort(labels), np.arange(1, self.nstates + 1))
                or np.any(self.time_s.min(axis=1) < self.line_start_s[labels - 1] - 1e-12)
                or np.any(self.time_s.max(axis=1) > self.line_end_s[labels - 1] + 1e-12)):
            raise ValueError("Every chronological current window must contain exactly one complete ADC readout.")
        self.line_index = np.broadcast_to(labels[:, None], self.time_s.shape).copy()
        if "line_index" in acquisition and not np.array_equal(self.line_index, acquisition["line_index"]):
            raise ValueError("Acquired line_index disagrees with RF-relative ADC/current-window chronology.")
        self.rf_time_s = np.zeros_like(self.time_s)
        self.sequence_kind = "epi3d"
        nominal_kx = (np.arange(self.nx) - self.nx // 2) / self.FOV_m[0]
        self.kx_offset_mInv = np.mean(kx - nominal_kx, axis=1)
        residual = np.max(np.abs(kx - nominal_kx - self.kx_offset_mInv[:, None]))
        tolerance = 1e-3  # Pulseq text gradient serialization; residual is reported.
        if residual > tolerance or np.any(np.diff(kx, axis=1) <= 0):
            raise ValueError("Readouts must have full uniform ascending Cartesian kx spacing; ramp sampling is unsupported.")
        self.ky_mInv = np.mean(ky, axis=1)
        nominal_ky = (np.arange(self.ny) - self.ny // 2) / self.FOV_m[1]
        if (np.max(np.abs(ky - self.ky_mInv[:, None])) > tolerance
                or np.any(np.diff(self.ky_mInv) <= 0)
                or np.max(np.abs(self.ky_mInv - nominal_ky)) > .02 / self.FOV_m[1]):
            raise ValueError("Phase rows must be a full ascending Cartesian ky grid, constant within each readout.")
        nominal_kz = (np.arange(self.nz) - self.nz // 2) / self.FOV_m[2]
        self.kz_mInv = _real(acquisition.get("kz_partition_mInv", nominal_kz), "kz_partition_mInv", (self.nz,))
        self.kz_offset_mInv = float(np.mean(self.kz_mInv - nominal_kz))
        if np.max(np.abs(self.kz_mInv - nominal_kz - self.kz_offset_mInv)) > tolerance:
            raise ValueError("Partitions must have full uniform ascending Cartesian kz spacing.")
        phase = _real(acquisition.get("phase_offset_rad", np.zeros(kx.shape)), "phase_offset_rad", kx.shape)
        self._phase = self.tensor(phase)
        self.TR_s = float(acquisition.get("TR_s", self.line_end_s[-1]))
        if not np.isfinite(self.TR_s) or self.TR_s < self.line_end_s[-1]:
            raise ValueError("TR_s must include the complete current waveform.")
        config = dict(current_response or {"enabled": True, "tau_s": 133e-6 / np.log(100.)})
        enabled = config.get("enabled", True)
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("current_response.enabled must be boolean.")
        tau = _real(config.get("tau_s", 133e-6 / np.log(100.)), "tau_s").reshape(-1)
        if tau.size == 1:
            tau = np.full(self.nc, tau.item())
        if tau.shape != (self.nc,) or np.any(tau <= 0):
            raise ValueError("tau_s must be positive, scalar or one per channel.")
        self.current_response = dict(enabled=bool(enabled), tau_s=tau.copy())
        limits = _real(max_current_A, "max_current_A").reshape(-1)
        limits = np.full(self.nc, limits.item()) if limits.size == 1 else limits
        if limits.shape != (self.nc,) or np.any(limits < 0):
            raise ValueError("max_current_A must be nonnegative, scalar or one per channel.")
        self.max_current_A = self.tensor(limits)
        self.amplitude_reference_time_s = float(volume.get("amplitude_reference_time_s", 0.))
        if not np.isfinite(self.amplitude_reference_time_s) or self.amplitude_reference_time_s < 0:
            raise ValueError("amplitude_reference_time_s must be finite and nonnegative.")
        rate = _real(volume.get("R2_sInv", 0.), "R2_sInv")
        if rate.size == 1:
            rate = np.full(self.shape, rate.item())
        if rate.shape != self.shape or np.any(rate < 0):
            raise ValueError("R2_sInv must be nonnegative, scalar or a 3D object map.")
        self._amplitude, self._b0, self._rate = map(self.tensor, (amplitude, b0, rate))
        self.B = self.tensor(maps.reshape(-1, self.nc))
        self.maps_numpy = maps.reshape(-1, self.nc)
        self.b0_numpy = b0.reshape(-1)
        self.amplitude_numpy = amplitude
        support = amplitude.reshape(-1) > 0
        self.B_quality = self.B if support.all() else self.B[torch.as_tensor(support, device=self.device)]
        self.b0_quality = self.tensor(b0.reshape(-1)[support])
        self.quality_mask = support
        self._x = self.tensor(self.axes_m[0])[:, None, None]
        self._y = self.tensor(self.axes_m[1])[None, :, None]
        self._z = self.tensor(self.axes_m[2])[None, None, :]
        self._target = self._roi = None
        self._gain = 1.
        self.interpolation_report = dict(
            spatial_shape=list(self.shape), channels=self.nc,
            forward="negative-sign 3D point-spin Fourier signal; exact filtered current integrals",
            partition_waveform="same commands repeated after each partition RF",
            current_start="zero before every partition; no periodic steady-state current approximation",
            reconstruction="centered nominal-grid 3D inverse FFT",
            native_axes=["readout", "phase", "partition"],
            temporal_interpolation="Chebyshev-Lobatto within actual ADC readout windows",
            checkpoint_lines=self.checkpoint_lines, actual_kx_offset_preserved=True,
            actual_ky_used=True, actual_kz_offset_preserved=True,
            kx_uniform_grid_residual_mInv=float(residual), grid_tolerance_mInv=tolerance,
            accuracy_policy="Compare nodes/refined nodes on actual candidates; a fixed rank is not a global accuracy guarantee.")
        gap = self.TR_s - self.line_end_s[-1]
        self.interpolation_report["current_tail_fraction_bound_at_next_RF"] = float(np.exp(-gap / tau.max())) if enabled else 0.
        self._worst_frequency_Hz = float(np.max(np.abs(b0) + np.sum(np.abs(maps) * limits, axis=-1)))
        self.set_interpolation_nodes(interpolation_nodes)

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    @staticmethod
    def numpy(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    def _check_currents(self, currents):
        is_complex = currents.is_complex() if isinstance(currents, torch.Tensor) else np.iscomplexobj(currents)
        if is_complex:
            raise ValueError("currents must be real [channel,state] commands.")
        result = self.tensor(currents)
        if result.shape != (self.nc, self.nstates) or not bool(torch.isfinite(result).all()):
            raise ValueError(f"currents must be finite with shape {(self.nc, self.nstates)}.")
        return result

    def _integral_weights(self, times):
        if self.current_response["enabled"]:
            tau = self.current_response["tau_s"]
            return response_matrices(self.line_start_s, self.line_end_s, times, tau)[0]
        weights = np.maximum(times[:, None] - self.line_start_s, 0.) - np.maximum(times[:, None] - self.line_end_s, 0.)
        return np.broadcast_to(weights, (self.nc,) + weights.shape).copy()

    def set_interpolation_nodes(self, count, *, warn=True):
        """Use one fixed temporal rank, or 'exact' for every actual ADC time."""
        exact = count == "exact" if isinstance(count, str) else False
        if not exact and (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 2):
            raise ValueError("interpolation_nodes must be an integer >=2 or 'exact'.")
        count = self.nx if exact or count >= self.nx else int(count)
        self.exact_readout = exact or count == self.nx
        self.interpolation_nodes = count
        plans = [(times.copy(), np.eye(self.nx)) if self.exact_readout else _chebyshev_plan(times, count)
                 for times in self.time_s]
        nodes = np.stack([plan[0] for plan in plans])
        weights = np.stack([plan[1] for plan in plans])
        integrals = self._integral_weights(nodes.reshape(-1)).reshape(self.nc, self.ny, count, self.nstates)
        self._nodes = self.tensor(nodes)
        self._weights = self.tensor(weights)
        self._integrals = self.tensor(integrals)
        half_phase = float(np.pi * self._worst_frequency_Hz * np.ptp(self.time_s, axis=1).max())
        inadequate = not self.exact_readout and half_phase >= count / 2
        self.interpolation_report.update(nodes_per_line=count, exact_readout_times=self.exact_readout,
            worst_case_half_phase_excursion_rad=half_phase,
            rank_warning=bool(inadequate),
            rank_warning_basis="Conservative declared-current field bound; not an interpolation error certificate.")
        if inadequate and warn:
            warnings.warn("Temporal interpolation rank may be insufficient for the declared field/current bounds. "
                          "Run adapt_interpolation/convergence_check on actual candidates or use interpolation_nodes='exact'.",
                          RuntimeWarning, stacklevel=2)

    def _line_forward(self, currents, row):
        integrated = torch.einsum("cns,cs->cn", self._integrals[:, row], currents)
        spatial = self.ky_mInv[row] * self._y + self.kx_offset_mInv[row] * self._x + self.kz_offset_mInv * self._z
        result = None
        for node in range(self.interpolation_nodes):
            t = self._nodes[row, node]
            cycles = self._b0 * t + (self.B @ integrated[:, node]).reshape(self.shape) + spatial
            spins = self._amplitude * torch.exp(-self._rate * (t - self.amplitude_reference_time_s)) * torch.exp(-2j * np.pi * cycles)
            encoded = _fft_centered(spins.sum(dim=1), (0, 1))
            term = encoded * self._weights[row, :, node, None]
            result = term if result is None else result + term
        return result * torch.exp(1j * self._phase[row, :, None])

    def forward(self, currents):
        """Return differentiable (complex imageXYZ, complex kdataXYZ)."""
        currents = self._check_currents(currents)
        lines = []
        for row in range(self.ny):
            if self.checkpoint_lines and currents.requires_grad and torch.is_grad_enabled():
                line = checkpoint(lambda c, r=row: self._line_forward(c, r), currents, use_reentrant=False)
            else:
                line = self._line_forward(currents, row)
            lines.append(line)
        kdata = torch.stack(lines, dim=1)
        image = _fft_centered(kdata, (0, 1, 2), inverse=True)
        return image, kdata

    def set_target(self, reference_magnitude, roi_mask=None, *, gain=1.0):
        reference = _real(reference_magnitude, "reference_magnitude", self.shape)
        if np.any(reference < 0):
            raise ValueError("reference_magnitude must be nonnegative.")
        mask = np.ones(self.shape, bool) if roi_mask is None else np.asarray(roi_mask)
        if mask.shape != self.shape or not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError("roi_mask must be a nonempty binary 3D mask matching the acquired volume.")
        if not np.isfinite(gain) or gain <= 0 or np.sum(reference[mask.astype(bool)] ** 2) <= 0:
            raise ValueError("gain must be positive and the reference must have nonzero ROI energy.")
        self._target, self._roi = self.tensor(reference), torch.as_tensor(mask.astype(bool), device=self.device)
        self._gain = float(gain)
        self._target_energy = self._target[self._roi].square().sum()

    def _image_loss(self, image):
        if self._target is None:
            raise RuntimeError("Call set_target with the measured 3D reference and a fixed gain first.")
        return (self._gain * image.abs()[self._roi] - self._target[self._roi]).square().sum() / self._target_energy

    def torch_objective(self, currents):
        """Scalar 3D magnitude relative MSE for a native PyTorch optimizer."""
        return self._image_loss(self.forward(currents)[0])

    def objective_and_grad(self, currents):
        """Return (float loss, NumPy gradient, NumPy complex image, NumPy kdata)."""
        with torch.enable_grad():
            commands = self._check_currents(currents).detach().clone().requires_grad_(True)
            image, kdata = self.forward(commands)
            loss = self._image_loss(image)
            gradient, = torch.autograd.grad(loss, commands)
        return float(loss.detach()), self.numpy(gradient), self.numpy(image), self.numpy(kdata)

    @torch.no_grad()
    def convergence_check(self, currents, *, refined_nodes=None):
        """Compare two temporal ranks on actual commands; restore the chosen rank."""
        original = "exact" if self.exact_readout else self.interpolation_nodes
        refined = (min(self.nx, 2 * self.interpolation_nodes) if refined_nodes is None else refined_nodes)
        image, kdata = self.forward(currents)
        try:
            self.set_interpolation_nodes(refined, warn=False)
            refined_image, refined_kdata = self.forward(currents)
            denominator = torch.linalg.vector_norm(refined_kdata).clamp_min(torch.finfo(self.dtype).tiny)
            error = torch.linalg.vector_norm(kdata - refined_kdata) / denominator
            magnitude_error = torch.linalg.vector_norm(image.abs() - refined_image.abs()) / torch.linalg.vector_norm(refined_image.abs()).clamp_min(torch.finfo(self.dtype).tiny)
            return dict(nodes=original, refined_nodes="exact" if self.exact_readout else self.interpolation_nodes,
                        relative_complex_kdata_error=float(error), relative_magnitude_image_error=float(magnitude_error),
                        interpretation="A posteriori comparison on these commands only; not an error bound for future optimizer candidates.")
        finally:
            self.set_interpolation_nodes(original, warn=False)

    @torch.no_grad()
    def adapt_interpolation(self, currents, *, tolerance=1e-5, maximum_nodes=None):
        """Double ranks against actual-command convergence, falling back to exact.

        The retained rank is the finer of the converged pair. Revalidate the
        final optimizer candidate: its phase evolution can differ from the
        initialization used to choose this fixed differentiable plan.
        """
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Interpolation tolerance must be positive and finite.")
        maximum = self.nx if maximum_nodes is None else min(self.nx, int(maximum_nodes))
        if maximum < self.interpolation_nodes:
            raise ValueError("maximum_nodes cannot be smaller than the current rank.")
        checks = []
        while not self.exact_readout:
            refined = min(maximum, 2 * self.interpolation_nodes)
            if refined == self.interpolation_nodes:
                break
            check = self.convergence_check(currents, refined_nodes=refined)
            checks.append(check)
            self.set_interpolation_nodes(refined, warn=False)
            if check["relative_complex_kdata_error"] <= tolerance:
                break
        passed = self.exact_readout or bool(checks and checks[-1]["relative_complex_kdata_error"] <= tolerance)
        result = dict(tolerance=tolerance, passed=passed, selected_nodes="exact" if self.exact_readout else self.interpolation_nodes,
                      checks=checks, future_candidates_require_revalidation=True)
        self.interpolation_report["last_adaptation"] = result
        if not passed:
            warnings.warn("Temporal convergence was not reached before maximum_nodes. Increase the cap or use exact readout times.", RuntimeWarning, stacklevel=2)
        return result


__all__ = ["VolumeEPIModel"]
