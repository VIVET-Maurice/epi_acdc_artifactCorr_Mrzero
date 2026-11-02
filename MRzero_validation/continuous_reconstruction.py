"""Fourier reconstruction and its Hermitian adjoint at actual ADC coordinates.

The reconstruction is the normalized Fourier adjoint, not a least-squares
inverse for arbitrary nonuniform sampling. Its output grid has the requested
image matrix and FOV, with the FFT origin at ``n // 2``. Every ADC is retained.
Uniform Cartesian grids use an FFT and an exact offset phase. A trajectory
with constant ky within each row uses two smaller Fourier operators. Other
trajectories use bounded sample blocks without storing a full dense matrix.
"""

from __future__ import annotations

import numpy as np
import torch


class ContinuousReconstruction:
    """Reconstruct trailing 2D sample axes, with optional leading batches.

    ``acquisition`` supplies 2D ``kx_mInv`` / ``ky_mInv`` arrays in cycles/m
    and ``FOV_m=[readout_FOV, phase_FOV]`` in metres. Optional
    ``encoding_origin_m=[X, Y]`` locates the receiver's image origin in the
    same physical XY frame. Output coordinates include that translation;
    Fourier kernels use coordinates relative to it, as does scanner encoding.
    ``dtype`` may be a Torch
    float or complex dtype. ``image`` and ``adjoint`` return complex Torch
    tensors; ``numpy_image`` is a convenience for measured receiver batches.
    A Twix ``[phase, receiver, readout]`` array must first move its receiver
    axis to the front. ``image_shape=[rows, columns]`` may differ from the
    sample storage shape for spiral data. Optional nonnegative
    ``density_compensation_weights`` are fixed sample weights: both this
    operator and its exact Hermitian adjoint include them, normalized by
    their sum. They never depend on currents or the reconstructed image.

    ``reconstruction_method='direct'`` (default) retains exact Fourier sums.
    ``'nufft'`` selects the optional installed ``torchkbnufft`` approximation
    and its matched adjoint for non-Cartesian trajectories. Kernels are cached
    only below a fixed 64 MiB budget; no dense sample/pixel matrix is stored.
    """

    _CACHE_BYTES = 64 * 1024**2

    def __init__(self, acquisition, device, dtype, sample_chunk=256):
        self.device = torch.device(device)
        if dtype in (torch.float32, torch.complex64):
            self.dtype, self.complex_dtype = torch.float32, torch.complex64
        elif dtype in (torch.float64, torch.complex128):
            self.dtype, self.complex_dtype = torch.float64, torch.complex128
        else:
            raise ValueError("dtype must be float32, float64, complex64 or complex128.")
        if (isinstance(sample_chunk, bool) or int(sample_chunk) != sample_chunk
                or sample_chunk < 1):
            raise ValueError("sample_chunk must be a positive integer.")
        self.sample_chunk = int(sample_chunk)
        if (np.iscomplexobj(acquisition["kx_mInv"])
                or np.iscomplexobj(acquisition["ky_mInv"])):
            raise ValueError("kx_mInv and ky_mInv must be real coordinates.")
        kx = np.asarray(acquisition["kx_mInv"], dtype=np.float64)
        ky = np.asarray(acquisition["ky_mInv"], dtype=np.float64)
        if (kx.ndim != 2 or min(kx.shape) < 1 or ky.shape != kx.shape
                or not np.isfinite(kx).all() or not np.isfinite(ky).all()):
            raise ValueError("kx_mInv and ky_mInv need matching finite nonempty 2D arrays.")
        if np.iscomplexobj(acquisition["FOV_m"]):
            raise ValueError("FOV_m must be real.")
        fov = np.asarray(acquisition["FOV_m"], dtype=np.float64)
        if fov.shape != (2,) or not np.isfinite(fov).all() or np.any(fov <= 0):
            raise ValueError("FOV_m must contain positive readout and phase FOVs in metres.")
        origin = acquisition.get("encoding_origin_m", (0., 0.))
        if np.iscomplexobj(origin):
            raise ValueError("encoding_origin_m must contain two finite real coordinates.")
        origin = np.asarray(origin, dtype=np.float64)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("encoding_origin_m must contain two finite real coordinates.")
        self.encoding_origin_m = origin.copy()
        self.sampling_shape = kx.shape
        image_shape = np.asarray(acquisition.get("image_shape", self.sampling_shape))
        if (image_shape.shape != (2,) or np.iscomplexobj(image_shape)
                or not np.isfinite(image_shape).all() or np.any(image_shape < 1)
                or np.any(image_shape != np.floor(image_shape))):
            raise ValueError("image_shape must contain two positive integers.")
        self.image_size = tuple(int(value) for value in image_shape)
        self.nsamples = int(kx.size)
        weights = acquisition.get("density_compensation_weights", np.ones(self.sampling_shape))
        if np.iscomplexobj(weights):
            raise ValueError("density_compensation_weights must be real.")
        weights = np.asarray(weights, dtype=np.float64)
        if (weights.shape != self.sampling_shape or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.any(weights > 0)):
            raise ValueError("density_compensation_weights must be finite, nonnegative, sample-shaped and nonzero.")
        self._weights = self._real(weights)
        self._weight_sum = float(weights.sum())
        if not np.isfinite(self._weight_sum):
            raise ValueError("The density-compensation weight sum must be finite.")
        self._unit_weights = bool(np.all(weights == weights.flat[0]))
        requested_method = str(acquisition.get("reconstruction_method", "direct")).lower()
        if requested_method not in {"direct", "nufft"}:
            raise ValueError("reconstruction_method must be direct or nufft.")
        ny, nx = self.image_size
        x = (np.arange(nx) - nx // 2) * fov[0] / nx
        y = (np.arange(ny) - ny // 2) * fov[1] / ny
        local_X, local_Y = np.meshgrid(x, y)
        self.X_m, self.Y_m = local_X + origin[0], local_Y + origin[1]
        self._x = self._real(x)
        self._y = self._real(y)
        self._output_x = self._real(local_X.reshape(-1))
        self._output_y = self._real(local_Y.reshape(-1))
        self._kx = self._real(kx)
        self._ky = self._real(ky)
        nominal_x = (np.arange(nx) - nx // 2) / fov[0]
        nominal_y = (np.arange(ny) - ny // 2) / fov[1]
        matching_shapes = self.sampling_shape == self.image_size
        offset_x = offset_y = 0.0
        grid_error = np.inf
        if matching_shapes:
            offset_x = float(np.median(kx - nominal_x[None, :]))
            offset_y = float(np.median(ky - nominal_y[:, None]))
            grid_error = max(float(np.max(np.abs(kx - nominal_x[None, :] - offset_x))),
                             float(np.max(np.abs(ky - nominal_y[:, None] - offset_y))))
        # Only roundoff at the input's float64 precision is accepted. This is
        # intentionally far tighter than the sequence reader's grid tolerance.
        tolerance = 8 * np.finfo(np.float64).eps * max(
            1.0, float(np.max(np.abs(kx))), float(np.max(np.abs(ky))))
        row_error = float(np.max(np.abs(ky - ky[:, :1])))
        self._offset_phase = self._row_x = self._row_y = None
        cached_bytes = 0
        complex_bytes = torch.empty((), dtype=self.complex_dtype).element_size()
        if matching_shapes and grid_error <= tolerance:
            self.method = "cartesian_fft"
            phase = 2 * np.pi * (offset_x * local_X + offset_y * local_Y)
            self._offset_phase = torch.as_tensor(
                np.exp(1j * phase), device=self.device, dtype=self.complex_dtype)
            cached_bytes = self.nsamples * complex_bytes
        elif matching_shapes and row_error <= tolerance:
            self.method = "row_separable"
            self._row_y = self._phase(self._ky[:, :1] * self._y[None, :]) / ny
            cached_bytes = ny * ny * complex_bytes
            if ny * nx * nx * complex_bytes + cached_bytes <= self._CACHE_BYTES:
                self._row_x = self._row_kernel(0, ny)
                cached_bytes += ny * nx * nx * complex_bytes
        else:
            self.method = "direct_chunked"
            if requested_method == "nufft":
                self._prepare_nufft(fov)
                self.method = "kaiser_bessel_nufft"
        self.report = {
            "method": self.method,
            "operator": "normalized Fourier adjoint at actual ADC coordinates",
            "normalization": "1 / sum of fixed density-compensation weights",
            "sampling_shape": list(self.sampling_shape),
            "image_shape": list(self.image_size),
            "density_compensation": "uniform" if self._unit_weights else "fixed supplied sample weights",
            "density_compensation_weight_sum": self._weight_sum,
            "requested_method": requested_method,
            "nufft_interpolation_points": 6 if self.method == "kaiser_bessel_nufft" else None,
            "FOV_m": fov.tolist(),
            "encoding_origin_m": origin.tolist(),
            "spatial_encoding": "Fourier phase uses physical XY minus encoding_origin_m",
            "sample_chunk": self.sample_chunk,
            "cached_kernel_MiB": cached_bytes / 1024**2,
            "kernel_cache_limit_MiB": self._CACHE_BYTES / 1024**2,
            "cartesian_roundoff_tolerance_mInv": tolerance,
            "cartesian_residual_mInv": grid_error if np.isfinite(grid_error) else None,
            "within_row_ky_residual_mInv": row_error,
            "readout_offset_mInv": offset_x,
            "phase_offset_mInv": offset_y,
            "rounded_or_discarded_samples": 0,
        }

    def _real(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def _phase(self, cycles):
        angles = 2 * torch.pi * cycles
        return torch.polar(torch.ones_like(angles), angles)

    def _input(self, value, shape):
        result = torch.as_tensor(value, device=self.device, dtype=self.complex_dtype)
        if result.ndim < 2 or tuple(result.shape[-2:]) != shape:
            raise ValueError(f"Expected trailing image axes {shape} for this sample/image input.")
        return result

    def _prepare_nufft(self, fov):
        """Prepare a matched NUFFT pair using radians per image-grid pixel."""
        try:
            from torchkbnufft import KbNufft, KbNufftAdjoint
        except ImportError as error:
            raise ImportError("reconstruction_method='nufft' requires torchkbnufft; use 'direct' for exact sums.") from error
        ny, nx = self.image_size
        # NUFFT spatial dimensions follow [Y, X], whereas Pulseq gives [kx, ky].
        self._omega = 2 * torch.pi * torch.stack((
            self._ky.reshape(-1) * (fov[1] / ny),
            self._kx.reshape(-1) * (fov[0] / nx)))
        options = dict(im_size=self.image_size, n_shift=(ny // 2, nx // 2),
                       numpoints=6, table_oversamp=4096, dtype=self.dtype, device=self.device)
        self._nufft_forward = KbNufft(**options)
        self._nufft_adjoint = KbNufftAdjoint(**options)

    def _row_kernel(self, first, stop):
        return self._phase(self._kx[first:stop, :, None] * self._x[None, None, :]) / self.image_size[1]

    def _direct_kernel(self, first, stop):
        cycles = (self._kx.reshape(-1)[first:stop, None] * self._output_x[None, :]
                  + self._ky.reshape(-1)[first:stop, None] * self._output_y[None, :])
        return self._phase(cycles) / self.nsamples

    def image(self, kdata):
        """Apply the normalized Fourier adjoint to all supplied ADC samples."""
        values = self._input(kdata, self.sampling_shape)
        # Existing fast Cartesian paths normalize by nsamples. Apply the
        # relative weights once here; the conjugate operator does the same.
        values = values * (self._weights * (self.nsamples / self._weight_sum))
        if self.method == "cartesian_fft":
            result = torch.fft.fftshift(torch.fft.ifft2(
                torch.fft.ifftshift(values, dim=(-2, -1))), dim=(-2, -1))
            return result * self._offset_phase
        if self.method == "row_separable":
            if self._row_x is not None:
                rows = (values.unsqueeze(-2) @ self._row_x).squeeze(-2)
            else:
                rows = []
                row_chunk = max(1, self.sample_chunk // self.image_size[1])
                for first in range(0, self.image_size[0], row_chunk):
                    stop = min(first + row_chunk, self.image_size[0])
                    rows.append((values[..., first:stop, :].unsqueeze(-2)
                                 @ self._row_kernel(first, stop)).squeeze(-2))
                rows = torch.cat(rows, dim=-2)
            return self._row_y.T @ rows
        flat = values.flatten(-2)
        if self.method == "kaiser_bessel_nufft":
            batches = flat.reshape(-1, 1, self.nsamples)
            result = self._nufft_adjoint(batches, self._omega) / self.nsamples
            return result.reshape(values.shape[:-2] + self.image_size)
        result = torch.zeros(values.shape[:-2] + (int(np.prod(self.image_size)),),
                             device=self.device, dtype=self.complex_dtype)
        for first in range(0, self.nsamples, self.sample_chunk):
            stop = min(first + self.sample_chunk, self.nsamples)
            result = result + flat[..., first:stop] @ self._direct_kernel(first, stop)
        return result.reshape(values.shape[:-2] + self.image_size)

    def adjoint(self, image):
        """Apply the true Hermitian adjoint of ``image``, including its scale."""
        values = self._input(image, self.image_size)
        return self._unweighted_adjoint(values) * (self._weights * (self.nsamples / self._weight_sum))

    def _unweighted_adjoint(self, values):
        """Adjoint before the real diagonal density-compensation operator."""
        if self.method == "cartesian_fft":
            return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(
                values * self._offset_phase.conj(), dim=(-2, -1))), dim=(-2, -1)) / self.nsamples
        if self.method == "row_separable":
            rows = self._row_y.conj() @ values
            if self._row_x is not None:
                return (rows.unsqueeze(-2) @ self._row_x.conj().transpose(-2, -1)).squeeze(-2)
            samples = []
            row_chunk = max(1, self.sample_chunk // self.image_size[1])
            for first in range(0, self.image_size[0], row_chunk):
                stop = min(first + row_chunk, self.image_size[0])
                samples.append((rows[..., first:stop, :].unsqueeze(-2)
                                @ self._row_kernel(first, stop).conj().transpose(-2, -1)).squeeze(-2))
            return torch.cat(samples, dim=-2)
        flat = values.flatten(-2)
        if self.method == "kaiser_bessel_nufft":
            batches = values.reshape((-1, 1) + self.image_size)
            result = self._nufft_forward(batches, self._omega) / self.nsamples
            return result.reshape(values.shape[:-2] + self.sampling_shape)
        samples = []
        for first in range(0, self.nsamples, self.sample_chunk):
            stop = min(first + self.sample_chunk, self.nsamples)
            samples.append(flat @ self._direct_kernel(first, stop).conj().T)
        return torch.cat(samples, dim=-1).reshape(values.shape[:-2] + self.sampling_shape)

    def numpy_image(self, kdata):
        """Reconstruct NumPy data with optional leading receiver/batch axes."""
        return self.image(kdata).detach().cpu().numpy()


def compute_spiral_density_weights(acquisition, *, iterations=10, device="cpu"):
    """Return fixed Pipe-Menon spiral weights and an auditable settings report.

    This preparation is performed once before reconstructing measured or
    simulated data. It uses only the nominal trajectory and output grid;
    no signal amplitude, B0, shim current or optimization target enters it.
    The same weights must remain in every copy of the acquisition metadata.
    Mean-one normalization preserves a convenient numerical signal scale.
    """
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    try:
        from torchkbnufft import calc_density_compensation_function
    except ImportError as error:
        raise ImportError("Spiral Pipe-Menon density compensation requires torchkbnufft.") from error
    # Validate the geometry through the exact operator without recursively
    # constructing NUFFT interpolation tables or inheriting existing weights.
    geometry = dict(acquisition)
    geometry["reconstruction_method"] = "direct"
    geometry.pop("density_compensation_weights", None)
    operator = ContinuousReconstruction(geometry, device, torch.float64)
    ny, nx = operator.image_size
    fov = np.asarray(acquisition["FOV_m"], dtype=np.float64)
    omega = 2 * torch.pi * torch.stack((
        operator._ky.reshape(-1) * (fov[1] / ny),
        operator._kx.reshape(-1) * (fov[0] / nx)))
    with torch.no_grad():
        raw = calc_density_compensation_function(
            omega, operator.image_size, num_iterations=int(iterations),
            numpoints=6, n_shift=(ny // 2, nx // 2), table_oversamp=4096)
    weights = raw.real.detach().cpu().numpy().reshape(operator.sampling_shape)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Pipe-Menon density compensation produced invalid sample weights.")
    weights = weights / weights.mean()
    report = {
        "method": "Pipe-Menon fixed trajectory density compensation",
        "iterations": int(iterations), "interpolation_points": 6,
        "table_oversampling": 4096, "grid_oversampling": 2,
        "normalization": "mean sample weight equals one",
        "minimum_weight": float(weights.min()), "maximum_weight": float(weights.max()),
        "sample_count": operator.nsamples, "image_shape": list(operator.image_size),
        "depends_on_signal_or_currents": False,
        "application": "identical real diagonal weights in reconstruction and its Hermitian adjoint",
    }
    return weights, report


__all__ = ["ContinuousReconstruction", "compute_spiral_density_weights"]
