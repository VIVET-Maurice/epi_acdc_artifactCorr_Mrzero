"""Independent phasor, relaxation-anchor and receiver tests of native twins."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from epi_twin import NativeEPITwin
from native_resummation import RESUMMATION_ROUNDOFF, resum_native_points
from twin_geometry import fft_origin_reverse


def fixture(multiple_rf=False):
    X, Y = np.meshgrid(np.asarray([-.023, .011, .034]), np.asarray([-.016, .019]))
    amplitude = np.asarray([[.3, .7, .5], [.6, .2, .4]])
    b0 = np.asarray([[31., -17., 9.], [43., -6., 22.]])
    rate = np.asarray([[9., 13., 17.], [5., 21., 11.]])
    kx, ky = np.meshgrid(np.asarray([-10., 0., 10.]), np.asarray([-5., 0.]))
    times = np.asarray([[.0011, .0012, .0013], [.0023, .0022, .0021]])
    rf = np.zeros_like(times)
    if multiple_rf:
        times[1] += .01
        rf[1] = .01
    acq = {"kx_mInv": kx, "ky_mInv": ky, "time_s": times, "rf_time_s": rf,
           "phase_offset_rad": np.asarray([[.1, -.2, .3], [-.1, .4, -.3]]),
           "FOV_m": np.asarray([.1, .2])}
    return acq, X, Y, amplitude, b0, np.ones(X.shape, bool), rate


def cancellation_twin():
    """A uniform object has nearly cancelling peripheral Cartesian samples."""
    positions = (np.arange(32) - 16) * .2 / 32
    X, Y = np.meshgrid(positions, positions)
    kx = np.asarray([[10., 50., 100., 150.]])
    acq = {"kx_mInv": kx, "ky_mInv": np.full_like(kx, 50.),
           "time_s": np.asarray([[.01, .0101, .0102, .0103]]),
           "phase_offset_rad": np.asarray([[0., .3, .6, .9]]), "FOV_m": np.asarray([.2, .2])}
    return NativeEPITwin(acq, X, Y, np.ones_like(X), np.zeros_like(X), np.ones_like(X, bool))


def direct_oracle(acq, X, Y, amplitude, b0, support, rate, anchor, receive=None, b0_enabled=True):
    receive = np.ones((1,) + X.shape, complex) if receive is None else receive
    signal = np.zeros((len(receive),) + acq["kx_mInv"].shape, complex)
    for index in np.ndindex(acq["kx_mInv"].shape):
        elapsed = acq["time_s"][index] - acq["rf_time_s"][index]
        cycles = X * acq["kx_mInv"][index] + Y * acq["ky_mInv"][index]
        if b0_enabled:
            cycles += b0 * elapsed
        point = (amplitude * np.exp(-rate * (elapsed - anchor))
                 * np.exp(-2j * np.pi * cycles + 1j * acq["phase_offset_rad"][index]))
        for receiver in range(len(receive)):
            signal[(receiver,) + index] = np.sum(point[support] * receive[receiver][support])
    ny, nx = acq["kx_mInv"].shape
    outX, outY = np.meshgrid((np.arange(nx) - nx // 2) * acq["FOV_m"][0] / nx,
                            (np.arange(ny) - ny // 2) * acq["FOV_m"][1] / ny)
    images = np.zeros(signal.shape, complex)
    for index in np.ndindex(outX.shape):
        kernel = np.exp(2j * np.pi * (acq["kx_mInv"] * outX[index] + acq["ky_mInv"] * outY[index])) / (ny * nx)
        images[(slice(None),) + index] = np.sum(signal * kernel[None], axis=(1, 2))
    return signal, images, np.sqrt(np.sum(np.abs(images)**2, axis=0))


class NativeEPITwinTests(unittest.TestCase):
    def test_native_cancelling_spins_use_accurate_sum_and_bounded_roundoff(self):
        twin = cancellation_twin()
        execute = twin._mr0.execute_graph
        captured = {}

        def capture(*args, **kwargs):
            signal, pathways = execute(*args, **kwargs)
            captured["points"] = pathways[0][0].cpu().numpy()
            return signal, pathways

        with patch.object(twin._mr0, "execute_graph", side_effect=capture):
            result = twin.simulate()
        expected = captured["points"].sum(axis=1, dtype=np.complex128).astype(np.complex64)
        np.testing.assert_array_equal(result["kdata"].reshape(-1), expected)
        report = result["report"]
        # The former relative-only gate fails on this real native simulation.
        self.assertGreater(report["native_resummation_relative_error"], 5e-5)
        self.assertLess(report["native_resummation_mass_relative_error"], RESUMMATION_ROUNDOFF)
        self.assertLessEqual(report["native_resummation_max_tolerance_ratio"], 1.)

    def test_corrupt_native_signal_is_rejected_even_when_spins_cancel(self):
        twin = cancellation_twin()
        execute = twin._mr0.execute_graph

        def corrupt(*args, **kwargs):
            signal, pathways = execute(*args, **kwargs)
            signal = signal.clone()
            signal[0, 0] += 1e-4 * pathways[0][0][0].abs().sum()
            return signal, pathways

        with patch.object(twin._mr0, "execute_graph", side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, "point resummation failed.*tolerance ratio"):
                twin.simulate()

    def test_nonfinite_native_signal_and_points_are_rejected(self):
        for target in ["signal", "points"]:
            with self.subTest(target=target):
                twin = cancellation_twin()
                execute = twin._mr0.execute_graph

                def corrupt(*args, **kwargs):
                    signal, pathways = execute(*args, **kwargs)
                    if target == "signal":
                        signal[0, 0] = complex(np.nan, 0)
                    else:
                        pathways[0][0][0, 0] = complex(np.inf, 0)
                    return signal, pathways

                with patch.object(twin._mr0, "execute_graph", side_effect=corrupt):
                    with self.assertRaisesRegex(RuntimeError, "non-finite"):
                        twin.simulate()

    def test_dark_adc_error_is_not_hidden_by_a_bright_adc(self):
        points = torch.tensor([[1., -1., 0.], [1e6, 1e6, 0.]], dtype=torch.complex64)
        native = points.sum(dim=1)
        native[0] = .001
        # A block-relative check would miss this error next to the bright ADC.
        self.assertLess(float(torch.linalg.vector_norm(points.sum(dim=1) - native)
                              / torch.linalg.vector_norm(native)), 5e-5)
        with self.assertRaisesRegex(RuntimeError, "tolerance ratio"):
            resum_native_points(points, native)

    def test_zero_points_do_not_admit_an_unscaled_absolute_error(self):
        points = torch.zeros((2, 3), dtype=torch.complex64)
        result, diagnostics = resum_native_points(points, torch.zeros(2, dtype=torch.complex64))
        self.assertEqual(diagnostics["max_tolerance_ratio"], 0.)
        self.assertTrue(bool(torch.all(result == 0)))
        with self.assertRaisesRegex(RuntimeError, "point resummation failed"):
            resum_native_points(points, torch.full((2,), 1e-20, dtype=torch.complex64))

    def test_native_negative_bloch_phase_and_nonzero_relaxation_anchor(self):
        acq, X, Y, amplitude, b0, support, rate = fixture()
        anchor = .004
        twin = NativeEPITwin(acq, X, Y, amplitude, b0, support, rate, anchor, sample_chunk=4)
        result = twin.simulate()
        signal, coils, rss = direct_oracle(acq, X, Y, amplitude, b0, support, rate, anchor)
        np.testing.assert_allclose(result["kdata"], signal, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(result["coil_images"], coils, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(result["image"], rss, rtol=2e-5, atol=2e-6)
        self.assertLess(result["report"]["native_resummation_relative_error"], 5e-5)
        self.assertGreater(result["report"]["native_blocks_simulated"], 0)
        self.assertFalse(result["report"]["point_magnetization_cache"])

    def test_zero_b0_twin_preserves_object_relaxation_and_acquisition(self):
        arguments = fixture()
        twin = NativeEPITwin(*arguments[:6], rate_sInv=arguments[6], amplitude_reference_time_s=.003)
        zero = twin.simulate(False)
        expected = direct_oracle(*arguments[:6], arguments[6], .003, b0_enabled=False)
        np.testing.assert_allclose(zero["kdata"], expected[0], rtol=2e-5, atol=2e-6)
        self.assertFalse(zero["report"]["B0_enabled"])
        self.assertGreater(np.linalg.norm(twin.simulate()["image"] - zero["image"]), .001)

    def test_rf_histories_reset_transverse_evolution_and_chunk_order(self):
        arguments = fixture(multiple_rf=True)
        twin = NativeEPITwin(*arguments[:6], rate_sInv=arguments[6], sample_chunk=4)
        actual = twin.simulate()
        expected = direct_oracle(*arguments[:6], arguments[6], 0.)
        np.testing.assert_allclose(actual["kdata"], expected[0], rtol=2e-5, atol=2e-6)

    def test_receiver_weights_are_applied_once_before_reconstruction_and_rss(self):
        arguments = fixture()
        receive = np.stack([np.ones(arguments[1].shape) / np.sqrt(2),
                            np.exp(1j * arguments[1] * 20) / np.sqrt(2)])
        twin = NativeEPITwin(*arguments[:6], rate_sInv=arguments[6], receive_weights=receive)
        result = twin.simulate()
        expected = direct_oracle(*arguments[:6], arguments[6], 0., receive)
        np.testing.assert_allclose(result["kdata"], expected[0], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(result["coil_images"], expected[1], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(result["image"], expected[2], rtol=2e-5, atol=2e-6)
        with self.assertRaisesRegex(ValueError, "unit RSS"):
            NativeEPITwin(*arguments[:6], receive_weights=receive * 2)

    def test_zero_k_amplitude_weights_are_not_multiplied_by_pixel_count(self):
        acq, X, Y, amplitude, b0, support, _ = fixture()
        for key in ("kx_mInv", "ky_mInv", "phase_offset_rad"):
            acq[key][:] = 0
        result = NativeEPITwin(acq, X, Y, amplitude, b0, support).simulate(False)
        np.testing.assert_allclose(result["kdata"], amplitude.sum(), rtol=2e-6, atol=2e-6)

    def test_adc_chunk_sizes_retain_complex_signal(self):
        arguments = fixture()
        results = [NativeEPITwin(*arguments[:6], rate_sInv=arguments[6], sample_chunk=chunk).simulate()
                   for chunk in [1, 4, 6]]
        for result in results[1:]:
            np.testing.assert_allclose(result["kdata"], results[0]["kdata"], rtol=3e-5, atol=3e-6)

    def test_joint_object_k_frame_reversal_preserves_signal_and_rotates_rss_about_fft_origin(self):
        for offset in [(0., 0.), (.7, -.4)]:
            acq, X, Y, amplitude, b0, support, rate = fixture()
            acq["kx_mInv"] += offset[0]
            acq["ky_mInv"] += offset[1]
            original = NativeEPITwin(acq, X, Y, amplitude, b0, support, rate_sInv=rate).simulate()
            reversed_acq = deepcopy(acq)
            reversed_acq["kx_mInv"] *= -1
            reversed_acq["ky_mInv"] *= -1
            reversed_frame = NativeEPITwin(reversed_acq, -X, -Y, amplitude, b0, support,
                                           rate_sInv=rate).simulate()
            np.testing.assert_allclose(reversed_frame["kdata"], original["kdata"], rtol=2e-6, atol=2e-6)
            # A uniform continuous-k offset contributes a unit-magnitude
            # boundary phase; RSS still follows the periodic-origin reversal.
            np.testing.assert_allclose(reversed_frame["image"], fft_origin_reverse(original["image"]),
                                       rtol=3e-5, atol=3e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_matches_cpu_native_points_and_continuous_reconstruction(self):
        arguments = fixture()
        cpu = NativeEPITwin(*arguments[:6], rate_sInv=arguments[6], sample_chunk=4).simulate()
        gpu = NativeEPITwin(*deepcopy(arguments[:6]), rate_sInv=arguments[6], sample_chunk=4, device="cuda").simulate()
        np.testing.assert_allclose(gpu["kdata"], cpu["kdata"], rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(gpu["image"], cpu["image"], rtol=3e-5, atol=3e-6)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
