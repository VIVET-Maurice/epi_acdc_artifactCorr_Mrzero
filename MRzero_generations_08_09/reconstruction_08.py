"""DFT encoding matrices and least-squares reconstruction for generation 08 V2.

The measured trajectory is in cycles/metre, so A[i,j] = exp(-2*pi*i*k[i].r[j]).
Reconstruction minimizes ||sqrt(W)(A m - s)||^2 / N_ADC + lambda*||m||^2.
SVD solves the inverse; no FFT, explicit normal-matrix inverse, or current-dependent
reconstruction is used. Large EPI grids use exact row/phase matrix factors.
"""

from __future__ import annotations

import numpy as np
import torch

from inplane_geometry import FrameReconstruction, rotate_acquisition_k


class MatrixDFTReconstruction:
    """A fixed linear DFT inverse and its exact Hermitian adjoint.

    ``regularization=0`` uses the Moore-Penrose inverse, with singular values
    below ``rcond * largest_singular_value`` discarded. Positive regularization
    is Tikhonov: (A.H W A + lambda*N_ADC*I)^-1 A.H W. W has mean one.

    Small general trajectories use the full encoding matrix. For large square
    EPI grids, s[r,:] = X[r] @ (Y @ m)[r,:] exactly. We cache small SVD factors
    instead of a dense N_ADC-by-N_voxel matrix. The unregularized factor solve
    requires full rank; global Tikhonov additionally requires orthogonal phase
    encoding. Unsupported large geometries fail explicitly.
    """

    def __init__(self, acquisition, device, dtype, sample_chunk=256, *,
                 regularization=0., rcond=1e-10, dense_limit_MiB=64.):
        from continuous_reconstruction import ContinuousReconstruction

        for name, value in (("regularization", regularization), ("rcond", rcond),
                            ("dense_limit_MiB", dense_limit_MiB)):
            if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                    or np.iscomplexobj(value) or not np.isreal(value) or not np.isfinite(value)):
                raise ValueError(f"{name} must be a finite real scalar.")
        if regularization < 0 or not 0 < rcond < 1 or dense_limit_MiB < 0:
            raise ValueError("Require regularization >= 0, 0 < rcond < 1, and dense_limit_MiB >= 0.")
        geometry = ContinuousReconstruction(dict(acquisition,
            reconstruction_method="direct", _skip_matrix_dft=True), device, dtype, sample_chunk)
        for name in ("device", "dtype", "complex_dtype", "image_size", "nsamples",
                     "sampling_shape", "X_m", "Y_m", "encoding_origin_m"):
            setattr(self, name, getattr(geometry, name))
        self.regularization, self.rcond = float(regularization), float(rcond)
        self.method = "matrix_dft"
        self.npixels = int(np.prod(self.image_size))
        self._dense_A = self._dense_R = None
        self._X = self._Y = self._readout_R = self._phase_R = None
        kx, ky = (np.asarray(acquisition[key], dtype=float) for key in ("kx_mInv", "ky_mInv"))
        x = geometry.X_m - geometry.encoding_origin_m[0]
        y = geometry.Y_m - geometry.encoding_origin_m[1]
        weights = np.asarray(acquisition.get("density_compensation_weights", np.ones(kx.shape)), float)
        weights = weights / weights.mean()
        dense_MiB = self.nsamples * self.npixels * 16 / 1024**2
        alpha = self.regularization * self.nsamples
        # Factorization is deliberately float64/complex128 on CPU, even when
        # repeated reconstruction runs in float32 on CUDA. Rank diagnostics
        # must not depend on the native float32 simulation precision.
        if dense_MiB <= dense_limit_MiB:
            A = np.exp(-2j * np.pi * (kx.reshape(-1, 1) * x.reshape(1, -1)
                                      + ky.reshape(-1, 1) * y.reshape(1, -1)))
            root_weight = np.sqrt(weights.reshape(-1))
            U, singular, Vh = np.linalg.svd(root_weight[:, None] * A, full_matrices=False)
            keep = singular > self.rcond * singular[0]
            factors = self._inverse_weights(singular, alpha, keep)
            R = (Vh.conj().T * factors) @ U.conj().T * root_weight[None, :]
            self._dense_A, self._dense_R = self._complex(A), self._complex(R)
            rank = int(keep.sum())
            condition = float(singular[0] / singular[-1]) if rank == min(A.shape) else np.inf
            self.singular_values = singular.copy()
            path = "dense SVD"
            stored = A.nbytes + R.nbytes
            rank_policy = "global Moore-Penrose cutoff" if not alpha else "global Tikhonov SVD filter"
            phase_error = None
        else:
            ny, nx = self.image_size
            scale = max(1., float(np.abs(ky).max()))
            row_tolerance = 32 * np.finfo(float).eps * scale
            if (self.sampling_shape != self.image_size
                    or np.max(np.abs(ky - ky[:, :1])) > row_tolerance
                    or not np.all(weights == weights.flat[0])):
                raise ValueError("Large matrix DFT requires a square EPI grid with constant ky per row and uniform weights. "
                                 "Use a smaller image or increase dense_limit_MiB for a general dense SVD.")
            X = np.exp(-2j * np.pi * kx[:, :, None] * x[0][None, None, :])
            Y = np.exp(-2j * np.pi * ky[:, :1] * y[:, 0][None, :])
            Uy, sy, Vhy = np.linalg.svd(Y, full_matrices=False)
            Ux, sx, Vhx = np.linalg.svd(X, full_matrices=False)
            phase_error = float(np.max(np.abs(Y.conj().T @ Y / ny - np.eye(ny))))
            phase_unitary = phase_error <= 1e-10
            condition = (float((sy.max() / sy.min()) * (sx.max() / sx.min()))
                         if sy.min() > 0 and sx.min() > 0 else np.inf)
            full_rank = (sy.min() > self.rcond * sy.max()
                         and sx.min() > self.rcond * sx.max() and condition < 1 / self.rcond)
            if alpha == 0 and not full_rank:
                raise ValueError("Large factorized DFT is rank-deficient or exceeds the global rcond condition bound. "
                                 "Use a smaller dense SVD or regularize an orthogonal phase-encoding grid.")
            if alpha > 0 and not phase_unitary:
                raise ValueError("Global Tikhonov on a large EPI grid requires orthogonal phase encoding. "
                                 "Set DFT_REGULARIZATION=0 for this trajectory or use a small dense SVD.")
            if alpha:
                # ||Y m||^2 = Ny*||m||^2, hence each readout solve uses
                # alpha/Ny = lambda*Nx. This is the global 2D ridge solution.
                px = sx / (sx**2 + alpha / ny)
                Ry = Y.conj().T / ny
                rank_policy = "global Tikhonov; exact orthogonal phase factorization"
            else:
                px = 1 / sx
                Ry = (Vhy.conj().T / sy) @ Uy.conj().T
                rank_policy = "full-rank global inverse; condition bound satisfies global rcond"
            Rx = (Vhx.conj().transpose(0, 2, 1) * px[:, None, :]) @ Ux.conj().transpose(0, 2, 1)
            self._X, self._Y = self._complex(X), self._complex(Y)
            self._readout_R, self._phase_R = self._complex(Rx), self._complex(Ry)
            if phase_unitary:
                self.singular_values = np.sort((np.sqrt(ny) * sx).reshape(-1))[::-1].copy()
                rank = int(np.count_nonzero(self.singular_values > self.rcond * self.singular_values[0]))
                condition = float(self.singular_values[0] / self.singular_values[-1]) if rank == self.npixels else np.inf
            else:
                self.singular_values = None
                rank = self.npixels
            path = "factorized readout and phase SVD"
            stored = X.nbytes + Y.nbytes + Rx.nbytes + Ry.nbytes
        self.report = dict(geometry.report)
        # Geometry validation may briefly allocate an accelerated Fourier
        # kernel. Those kernels are not part of this matrix solver's cache.
        self.report.pop("cached_kernel_MiB", None)
        self.report.pop("kernel_cache_limit_MiB", None)
        matrices = (self._dense_A, self._dense_R, self._X, self._Y,
                    self._readout_R, self._phase_R)
        cached_bytes = sum(value.numel() * value.element_size()
                           for value in matrices if value is not None)
        self.report.update(method=self.method, evaluation=path,
            operator="DFT matrix least-squares inverse with its matched Hermitian adjoint",
            encoding_equation="s = A m; A[i,j] = exp(-2*pi*i*k[i].r[j])",
            inverse_equation="R = (A.H W A + lambda*N_ADC*I)^-1 A.H W; lambda=0 uses A+",
            regularization=self.regularization, raw_normal_matrix_regularization=alpha,
            objective="||sqrt(W)(A m - s)||^2/N_ADC + lambda*||m||^2",
            rcond=self.rcond, rank=rank, rank_policy=rank_policy,
            condition_number=condition if np.isfinite(condition) else None,
            condition_number_kind="exact" if self.singular_values is not None else "upper bound",
            phase_orthogonality_error=phase_error,
            encoding_matrix_shape=[self.nsamples, self.npixels],
            hypothetical_dense_matrix_MiB=dense_MiB,
            cached_matrix_MiB=cached_bytes / 1024**2,
            factorization_matrix_MiB=stored / 1024**2,
            requested_method="matrix_dft",
            factorization_precision="complex128 on CPU; cached operators cast to execution dtype",
            normalization="unnormalized forward DFT; inverse scale follows the matrix solve",
            FFT_used=False, rounded_or_discarded_samples=0)

    @staticmethod
    def _inverse_weights(singular, alpha, keep):
        if alpha:
            return singular / (singular**2 + alpha)
        return np.divide(1., singular, out=np.zeros_like(singular), where=keep)

    def _complex(self, values):
        return torch.as_tensor(values.copy(), device=self.device, dtype=self.complex_dtype)

    def _input(self, value, trailing_shape):
        result = torch.as_tensor(value, device=self.device, dtype=self.complex_dtype)
        if result.ndim < 2 or tuple(result.shape[-2:]) != tuple(trailing_shape):
            raise ValueError(f"Expected trailing axes {tuple(trailing_shape)}.")
        return result

    def encode(self, image):
        """Evaluate s=A m with the negative-sign encoding matrix."""
        values = self._input(image, self.image_size)
        if self._dense_A is not None:
            return (values.flatten(-2) @ self._dense_A.T).reshape(values.shape[:-2] + self.sampling_shape)
        rows = self._Y @ values
        return torch.einsum("rcx,...rx->...rc", self._X, rows)

    def image(self, kdata):
        """Evaluate m=R s, where R is the fixed least-squares inverse."""
        values = self._input(kdata, self.sampling_shape)
        if self._dense_R is not None:
            return (values.flatten(-2) @ self._dense_R.T).reshape(values.shape[:-2] + self.image_size)
        rows = torch.einsum("rxc,...rc->...rx", self._readout_R, values)
        return self._phase_R @ rows

    def adjoint(self, image):
        """Apply R.H, the operator needed by the current-optimization gradient."""
        values = self._input(image, self.image_size)
        if self._dense_R is not None:
            return (values.flatten(-2) @ self._dense_R.conj()).reshape(values.shape[:-2] + self.sampling_shape)
        rows = self._phase_R.conj().T @ values
        return torch.einsum("rxc,...rx->...rc", self._readout_R.conj(), rows)

    def numpy_image(self, kdata):
        return self.image(kdata).detach().cpu().numpy()


class FrameDFTReconstruction(FrameReconstruction):
    """Matrix DFT inverse in the same joint spin/k/output frame as 08."""

    def __init__(self, acquisition, device, dtype, frame_angle_deg=0.,
                 sample_chunk=256, frame_matrix=None, *, regularization=None,
                 rcond=1e-10, dense_limit_MiB=64.):
        regularization = acquisition.get("dft_regularization", 0.) if regularization is None else regularization
        geometry = dict(acquisition, _skip_matrix_dft=True, reconstruction_method="direct")
        super().__init__(geometry, device, dtype, frame_angle_deg, sample_chunk, frame_matrix)
        frame_report = self.report
        native_acquisition = rotate_acquisition_k(acquisition, matrix=self.frame_matrix.T)
        self.base = MatrixDFTReconstruction(native_acquisition, device, dtype, sample_chunk,
            regularization=regularization, rcond=rcond, dense_limit_MiB=dense_limit_MiB)
        self.report = dict(frame_report, **self.base.report)
        self.report.update(operator="DFT matrix least-squares inverse on the joint spin/k/output frame",
            frame_matrix=self.frame_matrix.tolist(), encoding_origin_m=self.encoding_origin_m.tolist())


__all__ = ["MatrixDFTReconstruction", "FrameDFTReconstruction"]
