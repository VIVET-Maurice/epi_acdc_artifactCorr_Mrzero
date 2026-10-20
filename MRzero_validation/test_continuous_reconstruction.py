"""Independent Fourier sums and adjoint checks for continuous reconstruction."""

import unittest

import numpy as np
import torch

from continuous_reconstruction import ContinuousReconstruction


def acquisition(shape=(5, 6), offset=(0.0, 0.0), kind="cartesian"):
    ny, nx = shape
    fov = np.asarray([.21, .17])
    kx, ky = np.meshgrid((np.arange(nx) - nx // 2) / fov[0] + offset[0],
                         (np.arange(ny) - ny // 2) / fov[1] + offset[1])
    if kind in {"row", "general"}:
        rng = np.random.default_rng(29)
        kx += rng.normal(size=shape) * .02
        ky += rng.normal(size=(ny, 1)) * .01
    if kind == "general":
        ky += np.random.default_rng(30).normal(size=shape) * .03
    return {"kx_mInv": kx, "ky_mInv": ky, "FOV_m": fov}


def independent_sum(data, acq):
    """Scalar pixel/sample loops, independent of the Torch factorizations."""
    ny, nx = data.shape[-2:]
    result = np.zeros_like(data, dtype=np.complex128)
    for row in range(ny):
        y = (row - ny // 2) * acq["FOV_m"][1] / ny
        for col in range(nx):
            x = (col - nx // 2) * acq["FOV_m"][0] / nx
            for source_row in range(ny):
                for source_col in range(nx):
                    cycles = (acq["kx_mInv"][source_row, source_col] * x
                              + acq["ky_mInv"][source_row, source_col] * y)
                    result[..., row, col] += (data[..., source_row, source_col]
                                             * np.exp(2j * np.pi * cycles) / (nx * ny))
    return result


class ContinuousReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(15)

    def complex_values(self, shape):
        return self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)

    def model(self, acq, **kwargs):
        return ContinuousReconstruction(acq, "cpu", torch.float64, **kwargs)

    def test_cartesian_fft_parity_with_odd_and_nonsquare_batches(self):
        for shape in [(5, 6), (4, 7), (6, 8), (1, 5)]:
            with self.subTest(shape=shape):
                model = self.model(acquisition(shape))
                data = self.complex_values((2,) + shape)
                expected = np.fft.fftshift(np.fft.ifft2(
                    np.fft.ifftshift(data, axes=(-2, -1))), axes=(-2, -1))
                self.assertEqual(model.method, "cartesian_fft")
                np.testing.assert_allclose(model.numpy_image(data), expected, rtol=2e-14, atol=2e-14)

    def test_uniform_offset_has_correct_complex_phase_and_same_magnitude(self):
        acq = acquisition(offset=(.5 / .21, -.25 / .17))
        model = self.model(acq)
        data = self.complex_values(model.image_size)
        baseline = self.model(acquisition()).numpy_image(data)
        phase = np.exp(2j * np.pi * (.5 / .21 * model.X_m - .25 / .17 * model.Y_m))
        actual = model.numpy_image(data)
        self.assertEqual(model.method, "cartesian_fft")
        np.testing.assert_allclose(actual, baseline * phase, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(actual, independent_sum(data, acq), rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(np.abs(actual), np.abs(baseline), rtol=2e-14, atol=2e-14)

    def test_off_grid_branches_match_independent_scalar_sum(self):
        for kind, method in [("row", "row_separable"), ("general", "direct_chunked")]:
            acq = acquisition(kind=kind)
            data = self.complex_values((2, 3) + acq["kx_mInv"].shape)
            model = self.model(acq, sample_chunk=7)
            self.assertEqual(model.method, method)
            np.testing.assert_allclose(model.numpy_image(data), independent_sum(data, acq),
                                       rtol=3e-14, atol=3e-14)

    def test_hermitian_inner_product_for_all_branches_and_batches(self):
        for kind in ["cartesian", "row", "general"]:
            with self.subTest(kind=kind):
                model = self.model(acquisition(kind=kind), sample_chunk=7)
                samples = torch.as_tensor(self.complex_values((2,) + model.image_size))
                image = torch.as_tensor(self.complex_values(samples.shape))
                left = torch.vdot(model.image(samples).flatten(), image.flatten())
                right = torch.vdot(samples.flatten(), model.adjoint(image).flatten())
                torch.testing.assert_close(left, right, rtol=2e-13, atol=2e-13)

    def test_direct_chunk_sizes_and_uncached_row_kernels_agree(self):
        acq = acquisition(kind="general")
        data = self.complex_values(acq["kx_mInv"].shape)
        small, full = self.model(acq, sample_chunk=1), self.model(acq, sample_chunk=data.size)
        np.testing.assert_allclose(small.numpy_image(data), full.numpy_image(data), atol=2e-14)
        torch.testing.assert_close(small.adjoint(data), full.adjoint(data), rtol=2e-14, atol=2e-14)
        acq = acquisition(kind="row")
        cached, uncached = self.model(acq), self.model(acq, sample_chunk=7)
        uncached._row_x = None
        torch.testing.assert_close(cached.image(data), uncached.image(data), rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(cached.adjoint(data), uncached.adjoint(data), rtol=2e-14, atol=2e-14)

    def test_small_off_grid_deviations_are_not_rounded_to_fft(self):
        acq = acquisition()
        acq["kx_mInv"][2, 3] += 1e-8
        model = self.model(acq)
        self.assertEqual(model.method, "row_separable")
        self.assertEqual(model.report["rounded_or_discarded_samples"], 0)

    def test_float32_and_complex_dtype_conversion(self):
        for dtype in [torch.float32, torch.complex64, torch.float64, torch.complex128]:
            model = ContinuousReconstruction(acquisition(kind="row"), "cpu", dtype)
            data = self.complex_values(model.image_size)
            expected_dtype = torch.complex64 if dtype in (torch.float32, torch.complex64) else torch.complex128
            self.assertEqual(model.image(data).dtype, expected_dtype)
            np.testing.assert_allclose(model.numpy_image(data), independent_sum(data, acquisition(kind="row")),
                                       rtol=2e-5, atol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_matches_cpu_for_all_branches(self):
        for kind in ["cartesian", "row", "general"]:
            with self.subTest(kind=kind):
                acq = acquisition(kind=kind)
                cpu = self.model(acq, sample_chunk=7)
                gpu = ContinuousReconstruction(acq, "cuda", torch.float32, sample_chunk=7)
                data = self.complex_values((2,) + cpu.image_size)
                np.testing.assert_allclose(gpu.numpy_image(data), cpu.numpy_image(data), rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(gpu.adjoint(data).cpu().numpy(), cpu.adjoint(data).numpy(),
                                           rtol=2e-5, atol=2e-6)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample_chunk"):
            self.model(acquisition(), sample_chunk=0)
        acq = acquisition()
        acq["FOV_m"][0] = 0
        with self.assertRaisesRegex(ValueError, "FOV_m"):
            self.model(acq)
        acq = acquisition()
        acq["kx_mInv"] = acq["kx_mInv"].astype(complex) + 1j
        with self.assertRaisesRegex(ValueError, "real coordinates"):
            self.model(acq)
        with self.assertRaisesRegex(ValueError, "trailing image axes"):
            self.model(acquisition()).image(np.zeros((6, 5)))


if __name__ == "__main__":
    unittest.main()
