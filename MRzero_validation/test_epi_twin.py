"""Independent phasor, relaxation-anchor and receiver tests of native twins."""

from copy import deepcopy
import unittest

import numpy as np
import torch

from epi_twin import NativeEPITwin
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
