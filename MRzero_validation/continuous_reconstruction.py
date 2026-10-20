"""Fourier reconstruction and its Hermitian adjoint at actual ADC coordinates.

The reconstruction is the normalized Fourier adjoint, not a least-squares
inverse for arbitrary nonuniform sampling. Its output grid has the acquired
matrix and FOV, with the FFT origin at ``n // 2``. Every ADC is retained.
Uniform Cartesian grids use an FFT and an exact offset phase. A trajectory
with constant ky within each row uses two smaller Fourier operators. Other
trajectories use bounded sample blocks without storing a full dense matrix.
"""

from __future__ import annotations

import numpy as np
import torch


class ContinuousReconstruction:
    """Reconstruct trailing ``[phase, readout]`` axes, with optional batches.

    ``acquisition`` supplies 2D ``kx_mInv`` / ``ky_mInv`` arrays in cycles/m
    and ``FOV_m=[readout_FOV, phase_FOV]`` in metres. ``dtype`` may be a Torch
    float or complex dtype. ``image`` and ``adjoint`` return complex Torch
    tensors; ``numpy_image`` is a convenience for measured receiver batches.
    A Twix ``[phase, receiver, readout]`` array must first move its receiver
    axis to the front. Kernels are cached only below a fixed 64 MiB budget.
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
        self.image_size = kx.shape
        self.nsamples = int(kx.size)
        ny, nx = self.image_size
        x = (np.arange(nx) - nx // 2) * fov[0] / nx
        y = (np.arange(ny) - ny // 2) * fov[1] / ny
        self.X_m, self.Y_m = np.meshgrid(x, y)
        self._x = self._real(x)
        self._y = self._real(y)
        self._output_x = self._real(self.X_m.reshape(-1))
        self._output_y = self._real(self.Y_m.reshape(-1))
        self._kx = self._real(kx)
        self._ky = self._real(ky)
        nominal_x = (np.arange(nx) - nx // 2) / fov[0]
        nominal_y = (np.arange(ny) - ny // 2) / fov[1]
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
        if grid_error <= tolerance:
            self.method = "cartesian_fft"
            phase = 2 * np.pi * (offset_x * self.X_m + offset_y * self.Y_m)
            self._offset_phase = torch.as_tensor(
                np.exp(1j * phase), device=self.device, dtype=self.complex_dtype)
            cached_bytes = self.nsamples * complex_bytes
        elif row_error <= tolerance:
            self.method = "row_separable"
            self._row_y = self._phase(self._ky[:, :1] * self._y[None, :]) / ny
            cached_bytes = ny * ny * complex_bytes
            if ny * nx * nx * complex_bytes + cached_bytes <= self._CACHE_BYTES:
                self._row_x = self._row_kernel(0, ny)
                cached_bytes += ny * nx * nx * complex_bytes
        else:
            self.method = "direct_chunked"
        self.report = {
            "method": self.method,
            "operator": "normalized Fourier adjoint at actual ADC coordinates",
            "normalization": "1 / number of ADC samples",
            "image_shape": list(self.image_size),
            "FOV_m": fov.tolist(),
            "sample_chunk": self.sample_chunk,
            "cached_kernel_MiB": cached_bytes / 1024**2,
            "kernel_cache_limit_MiB": self._CACHE_BYTES / 1024**2,
            "cartesian_roundoff_tolerance_mInv": tolerance,
            "cartesian_residual_mInv": grid_error,
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

    def _input(self, value):
        result = torch.as_tensor(value, device=self.device, dtype=self.complex_dtype)
        if result.ndim < 2 or tuple(result.shape[-2:]) != self.image_size:
            raise ValueError(f"Expected trailing image axes {self.image_size}.")
        return result

    def _row_kernel(self, first, stop):
        return self._phase(self._kx[first:stop, :, None] * self._x[None, None, :]) / self.image_size[1]

    def _direct_kernel(self, first, stop):
        cycles = (self._kx.reshape(-1)[first:stop, None] * self._output_x[None, :]
                  + self._ky.reshape(-1)[first:stop, None] * self._output_y[None, :])
        return self._phase(cycles) / self.nsamples

    def image(self, kdata):
        """Apply the normalized Fourier adjoint to all supplied ADC samples."""
        values = self._input(kdata)
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
        result = torch.zeros_like(flat)
        for first in range(0, self.nsamples, self.sample_chunk):
            stop = min(first + self.sample_chunk, self.nsamples)
            result = result + flat[..., first:stop] @ self._direct_kernel(first, stop)
        return result.reshape(values.shape)

    def adjoint(self, image):
        """Apply the true Hermitian adjoint of ``image``, including its scale."""
        values = self._input(image)
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
        samples = []
        for first in range(0, self.nsamples, self.sample_chunk):
            stop = min(first + self.sample_chunk, self.nsamples)
            samples.append(flat @ self._direct_kernel(first, stop).conj().T)
        return torch.cat(samples, dim=-1).reshape(values.shape)

    def numpy_image(self, kdata):
        """Reconstruct NumPy data with optional leading receiver/batch axes."""
        return self.image(kdata).detach().cpu().numpy()


__all__ = ["ContinuousReconstruction"]
