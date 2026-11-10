"""Independent matrix references for generation 08 V2 least squares."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent / 'MRzero_generations_08_09'
sys.path.insert(0, str(ROOT))
import reconstruction_08
import continuous_reconstruction
from inplane_geometry import inplane_matrix, rotate_acquisition_k

MatrixDFTReconstruction = reconstruction_08.MatrixDFTReconstruction

torch.set_num_threads(2)


def acquisition(ny=3, nx=4, *, arbitrary=False, irregular_x=False,
                irregular_y=False, origin=False):
    rng = np.random.default_rng(394)
    fov = np.array([.177, .231])
    kx, ky = np.meshgrid((np.arange(nx) - nx // 2) / fov[0],
                         (np.arange(ny) - ny // 2) / fov[1])
    if arbitrary:
        kx += rng.normal(size=(ny, nx)) * .18 / fov[0]
        ky += rng.normal(size=(ny, nx)) * .17 / fov[1]
    else:
        kx += rng.normal(size=(ny, 1)) * .14 / fov[0]
        if irregular_x:
            kx += rng.normal(size=(ny, nx)) * .11 / fov[0]
        if irregular_y:
            ky += rng.normal(size=(ny, 1)) * .09 / fov[1]
    result = dict(kx_mInv=kx, ky_mInv=ky, FOV_m=fov)
    if origin:
        result['encoding_origin_m'] = np.array([.032, -.019])
    return result


def dense_encoding(acq):
    ny, nx = acq.get('image_shape', np.asarray(acq['kx_mInv']).shape)
    fov = acq['FOV_m']
    x = (np.arange(nx) - nx // 2) * fov[0] / nx
    y = (np.arange(ny) - ny // 2) * fov[1] / ny
    X, Y = np.meshgrid(x, y)
    return np.exp(-2j * np.pi * (
        np.asarray(acq['kx_mInv']).ravel()[:, None] * X.ravel()[None, :]
        + np.asarray(acq['ky_mInv']).ravel()[:, None] * Y.ravel()[None, :]))


def dense_reconstructor(A, regularization=0., rcond=1e-10):
    if regularization == 0.:
        return np.linalg.pinv(A, rcond=rcond)
    return np.linalg.solve(A.conj().T @ A + regularization * len(A) * np.eye(A.shape[1]), A.conj().T)


class MatrixDFTTests(unittest.TestCase):
    def test_reader_matrix_switch_preserves_tikhonov_and_adjoint(self):
        acq = dict(acquisition(3, 4, arbitrary=True), dft_regularization=.02)
        A = dense_encoding(acq)
        R = dense_reconstructor(A, .02)
        signal = np.arange(12).reshape(3, 4) + 1j
        with patch.object(continuous_reconstruction, 'MATRIX_DFT_ENABLED', True):
            operator = continuous_reconstruction.ContinuousReconstruction(acq, 'cpu', torch.float64)
            actual = operator.numpy_image(signal)
            np.testing.assert_allclose(actual.ravel(), R @ signal.ravel(), rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(operator.adjoint(actual).numpy().ravel(),
                                       R.conj().T @ actual.ravel(), rtol=1e-11, atol=1e-11)
            self.assertEqual(operator.report['regularization'], .02)

    def test_cache_report_uses_execution_precision(self):
        acq = acquisition(3, 4)
        single = MatrixDFTReconstruction(acq, 'cpu', torch.float32)
        double = MatrixDFTReconstruction(acq, 'cpu', torch.float64)
        self.assertEqual(double.report['cached_matrix_MiB'], 2 * single.report['cached_matrix_MiB'])
        self.assertNotIn('cached_kernel_MiB', single.report)
        self.assertNotIn('kernel_cache_limit_MiB', single.report)

    def check_operator(self, acq, regularization=0., *, factored=False):
        kwargs = dict(regularization=regularization, rcond=1e-10)
        if factored:
            kwargs['dense_limit_MiB'] = 0.
        op = MatrixDFTReconstruction(acq, 'cpu', torch.float64, **kwargs)
        A = dense_encoding(acq)
        R = dense_reconstructor(A, regularization)
        rng = np.random.default_rng(395)
        ny, nx = acq.get('image_shape', acq['kx_mInv'].shape)
        samples = np.asarray(acq['kx_mInv']).shape
        m = rng.normal(size=(2, 3, ny, nx)) + 1j * rng.normal(size=(2, 3, ny, nx))
        s = rng.normal(size=(2, 3) + samples) + 1j * rng.normal(size=(2, 3) + samples)
        encoded = op.encode(torch.as_tensor(m)).numpy()
        actual = op.image(torch.as_tensor(s)).numpy()
        adjoint = op.adjoint(torch.as_tensor(m)).numpy()
        expected_encoded = np.einsum('ij,...j->...i', A, m.reshape(2, 3, -1)).reshape((2, 3) + samples)
        expected = np.einsum('ij,...j->...i', R, s.reshape(2, 3, -1)).reshape(2, 3, ny, nx)
        expected_adjoint = np.einsum('ji,...j->...i', R.conj(), m.reshape(2, 3, -1)).reshape((2, 3) + samples)
        np.testing.assert_allclose(encoded, expected_encoded, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(adjoint, expected_adjoint, rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(np.vdot(actual, m), np.vdot(s, adjoint), rtol=2e-11, atol=2e-11)
        self.assertIsInstance(op.report, dict)
        return op, A, R

    def test_dense_arbitrary_complex_encoding_inverse_and_adjoint(self):
        acq = acquisition(3, 3, arbitrary=True, origin=True)
        self.check_operator(acq)

    def test_dense_tikhonov_matches_normal_equations(self):
        self.check_operator(acquisition(3, 4, arbitrary=True), regularization=.017)

    def test_dense_rank_deficient_matches_moore_penrose(self):
        acq = acquisition(3, 3, arbitrary=True)
        for name in ('kx_mInv', 'ky_mInv'):
            acq[name].ravel()[1] = acq[name].ravel()[0]
        _, A, R = self.check_operator(acq)
        self.assertEqual(np.linalg.matrix_rank(A), 8)
        np.testing.assert_allclose(A @ R @ A, A, rtol=2e-10, atol=2e-10)

    def test_dense_rectangular_image_grid(self):
        acq = acquisition(4, 4, arbitrary=True)
        acq['image_shape'] = (3, 3)
        self.check_operator(acq)

    def test_factored_arbitrary_readout_nonunitary_phase_inverse(self):
        self.check_operator(acquisition(7, 6, irregular_x=True, irregular_y=True), factored=True)

    def test_factored_tikhonov_uniform_phase_irregular_readout(self):
        self.check_operator(acquisition(7, 6, irregular_x=True), regularization=.023, factored=True)

    def test_factored_tikhonov_uniform_readout_nonunitary_phase(self):
        with self.assertRaisesRegex(ValueError, 'orthogonal phase'):
            self.check_operator(acquisition(7, 6, irregular_y=True), regularization=.023, factored=True)

    def test_large_rank_deficiency_is_rejected(self):
        acq = acquisition(7, 6, irregular_x=True)
        acq['kx_mInv'][0, 1] = acq['kx_mInv'][0, 0]
        with self.assertRaisesRegex(ValueError, 'rank-deficient|condition bound'):
            MatrixDFTReconstruction(acq, 'cpu', torch.float64, dense_limit_MiB=0.)

    def test_global_cutoff_condition_bound_is_preserved(self):
        acq = acquisition(4, 4)
        # Each factor is invertible above its own cutoff, but the combined
        # encoding matrix has a mode below the global cutoff.
        fov = acq['FOV_m']
        base = np.array([0., 0.02, 1., 2.])
        acq['kx_mInv'] = np.broadcast_to(base[None, :] / fov[0], (4, 4)).copy()
        acq['ky_mInv'] = np.broadcast_to(base[:, None] / fov[1], (4, 4)).copy()
        A = dense_encoding(acq)
        X = np.exp(-2j * np.pi * acq['kx_mInv'][0, :, None]
                   * ((np.arange(4) - 2) * fov[0] / 4)[None, :])
        singular = np.linalg.svd(X, compute_uv=False)
        ratio = singular[-1] / singular[0]
        self.assertGreater(ratio, .001)
        self.assertLess(np.linalg.svd(A, compute_uv=False)[-1] / np.linalg.svd(A, compute_uv=False)[0], .001)
        with self.assertRaisesRegex(ValueError, 'condition bound'):
            MatrixDFTReconstruction(acq, 'cpu', torch.float64, rcond=.001, dense_limit_MiB=0.)

    def test_encoding_reconstruction_and_adjoint_never_call_fft(self):
        op = MatrixDFTReconstruction(acquisition(4, 5, arbitrary=True), 'cpu', torch.float64)
        value = torch.ones((4, 5), dtype=torch.complex128)
        with patch.object(torch.fft, 'fft2', side_effect=AssertionError('FFT forbidden')), \
                patch.object(torch.fft, 'ifft2', side_effect=AssertionError('IFFT forbidden')), \
                patch.object(np.fft, 'fftn', side_effect=AssertionError('FFT forbidden')), \
                patch.object(np.fft, 'ifftn', side_effect=AssertionError('IFFT forbidden')):
            op.encode(value)
            op.image(value)
            op.adjoint(value)

    def test_asymmetric_complex_point_fixes_encoding_sign(self):
        acq = acquisition(3, 5, origin=True)
        op = MatrixDFTReconstruction(acq, 'cpu', torch.float64)
        image = np.zeros((3, 5), complex)
        image[0, 4] = 1.2 + .9j
        expected = (dense_encoding(acq) @ image.ravel()).reshape(3, 5)
        actual = op.encode(torch.as_tensor(image)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
        restored = op.image(torch.as_tensor(actual)).numpy()
        np.testing.assert_allclose(restored, image, rtol=2e-11, atol=2e-11)

    def test_frame_rotation_and_reflection_preserve_matrix_solution(self):
        acq = acquisition(5, 4, irregular_x=True, irregular_y=True)
        Q = inplane_matrix(37., True, False)
        rotated = rotate_acquisition_k(acq, matrix=Q)
        base = MatrixDFTReconstruction(acq, 'cpu', torch.float64)
        framed = reconstruction_08.FrameDFTReconstruction(rotated, 'cpu', torch.float64, frame_matrix=Q)
        rng = np.random.default_rng(396)
        signal = torch.as_tensor(rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4)))
        torch.testing.assert_close(framed.image(signal), base.image(signal), rtol=2e-10, atol=2e-11)
        torch.testing.assert_close(framed.adjoint(signal), base.adjoint(signal), rtol=2e-10, atol=2e-11)


if __name__ == '__main__':
    unittest.main(verbosity=2)
