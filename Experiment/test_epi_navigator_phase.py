"""Regressions for three-navigator EPI correction with temporal B0 phase."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from prepare_acdc_inputs import centered_fft, estimate_epi_navigator_phase, read_scan
import test_prepare_acdc_inputs as preparation_tests


def navigator_echoes(amplitude, odd_even_phase, temporal_phase, common_phase=0.0):
    """Generate equally spaced +,-,+ echoes in readout image space.

    ``temporal_phase`` is the phase accumulated in one echo spacing. Only
    the middle (negative-polarity) echo has odd/even phase; temporal phase
    evolves symmetrically between the outer echoes. Tests specify the
    correction branch explicitly when the odd/even phase is near pi.
    """
    amplitude = np.asarray(amplitude)
    return np.stack([
        amplitude * np.exp(1j * (common_phase - temporal_phase)),
        amplitude * np.exp(1j * (common_phase + odd_even_phase)),
        amplitude * np.exp(1j * (common_phase + temporal_phase)),
    ])


class NavigatorPhaseTests(unittest.TestCase):
    def assert_same_phase(self, actual, expected, atol=1e-11):
        # A common multiple of 2*pi does not change the applied correction.
        np.testing.assert_allclose(np.exp(1j * actual), np.exp(1j * expected), atol=atol)

    def test_affine_error_is_recovered_for_each_receiver(self):
        x = np.linspace(-1, 1, 64)
        phase = np.array([.23, -.42, 1.1])[:, None] + np.array([.51, -.37, .22])[:, None] * x
        amplitude = np.array([1., 2., .7])[:, None] * np.ones((3, 64))
        common = np.array([.1, -.8, 2.3])[:, None] + .3 * x
        correction, diagnostics = estimate_epi_navigator_phase(
            navigator_echoes(amplitude, phase, .0, common))
        self.assertEqual(correction.shape, (3, 64))
        self.assertIsInstance(diagnostics, dict)
        self.assert_same_phase(correction, phase)

    def test_temporal_phase_is_removed_near_complex_echo_cancellation(self):
        x = np.linspace(-1, 1, 64)
        phase = .35 + .48 * x
        temporal = np.array([1.53, -1.53])[:, None]
        nav = navigator_echoes(np.ones((2, 64)), phase, temporal, .8 * x)
        retention = np.abs(nav[0] + nav[2]) / (np.abs(nav[0]) + np.abs(nav[2]))
        self.assertLess(float(retention.max()), .041)
        correction, _ = estimate_epi_navigator_phase(nav)
        self.assert_same_phase(correction, np.broadcast_to(phase, correction.shape))

    def test_temporal_phase_can_cross_the_complex_average_sign_change(self):
        x = np.linspace(-1, 1, 64)
        phase = .1 + .2 * x
        temporal = 1.57 + .16 * np.sin(2 * np.pi * x)
        # The average of the positive navigators changes sign within the
        # object. The middle-to-outer phase differences remain continuous.
        nav = navigator_echoes(np.ones((1, 64)), phase, temporal)
        self.assertLess(float(np.cos(temporal).min()), 0)
        self.assertGreater(float(np.cos(temporal).max()), 0)
        correction, _ = estimate_epi_navigator_phase(nav)
        self.assert_same_phase(correction, phase[None, :])

    def test_explicit_phase_branch_recovers_a_pi_shifted_error(self):
        x = np.linspace(-1, 1, 64)
        phase = np.array([3.0, -3.0])[:, None] + .32 * x
        nav = navigator_echoes(np.ones((2, 64)), phase, 1.45, -.4 * x)
        # Three symmetric navigators determine the odd/even phase only
        # modulo pi when the temporal phase is unknown. Select the other
        # branch explicitly instead of claiming it can be inferred.
        correction, _ = estimate_epi_navigator_phase(nav, phase_branch=1)
        self.assert_same_phase(correction, phase)

    def test_low_signal_background_does_not_bias_the_fit(self):
        rng = np.random.default_rng(784)
        x = np.linspace(-1, 1, 64)
        phase = .37 + .55 * x
        support = np.zeros(64, bool)
        support[18:47] = True
        nav = .09 * np.exp(1j * rng.uniform(-np.pi, np.pi, (3, 2, 64)))
        signal = navigator_echoes(np.ones((2, 64)), phase, 1.53, .2 * x)
        nav[:, :, support] = signal[:, :, support]
        correction, diagnostics = estimate_epi_navigator_phase(nav, signal_fraction=.10)
        self.assert_same_phase(correction, np.broadcast_to(phase, correction.shape))
        np.testing.assert_array_equal(diagnostics["fit_mask"], np.broadcast_to(support, (2, 64)))
        np.testing.assert_array_equal(diagnostics["fitted_channels"], [True, True])
        self.assertTrue(np.isnan(diagnostics["phase_estimate_rad"][:, ~support]).all())

    def test_disconnected_signal_islands_are_not_unwrapped_together(self):
        x = np.linspace(-1, 1, 64)
        support = np.zeros(64, bool)
        support[20:49] = True
        phase = -.21 + .35 * x
        nav = np.zeros((3, 1, 64), complex)
        signal = navigator_echoes(np.ones((1, 64)), phase, 1.53)
        nav[:, :, support] = signal[:, :, support]
        # The smaller island is above the reliability threshold and even
        # brighter, but supplies less total signal and a different phase.
        island = navigator_echoes(np.full((1, 64), 1.4), 1.2 - .8 * x, 1.53)
        nav[:, :, 3:8] = island[:, :, 3:8]
        correction, diagnostics = estimate_epi_navigator_phase(nav)
        self.assert_same_phase(correction, phase[None, :])
        np.testing.assert_array_equal(diagnostics["fit_mask"], support[None, :])

    def test_insufficient_signal_is_reported_and_leaves_the_phase_unchanged(self):
        nav = np.zeros((3, 2, 64), complex)
        nav[:, 1, 10:13] = navigator_echoes(np.ones((1, 3)), .4, .2)[:, 0]
        correction, diagnostics = estimate_epi_navigator_phase(nav)
        np.testing.assert_array_equal(correction, np.zeros((2, 64)))
        np.testing.assert_array_equal(diagnostics["fitted_channels"], [False, False])

    def test_corrected_epi_recovers_the_object_without_a_half_fov_copy(self):
        ny = nx = 64
        row, col = np.indices((ny, nx))
        phantom = (((row - 21) / 8) ** 2 + ((col - 32) / 11) ** 2 <= 1).astype(float)
        coil_images = phantom[:, None, :] * np.array([1., .7 * np.exp(.4j)])[None, :, None]
        clean_kspace = centered_fft(coil_images, (0, 2))
        x = np.linspace(-1, 1, nx)
        phase = 1.1 + .6 * x
        negative = np.arange(ny) % 2 == 1
        measured_hybrid = centered_fft(clean_kspace, (2,), inverse=True)
        measured_hybrid[negative] *= np.exp(1j * phase)[None, None, :]
        uncorrected = centered_fft(measured_hybrid, (0,), inverse=True)
        nav_amplitude = np.sum(np.abs(coil_images), axis=0)
        nav = navigator_echoes(nav_amplitude, phase, 1.50, .25 * x)
        correction, _ = estimate_epi_navigator_phase(nav)
        measured_hybrid[negative] *= np.exp(-1j * correction)[None, :, :]
        corrected = centered_fft(measured_hybrid, (0,), inverse=True)
        np.testing.assert_allclose(corrected, coil_images, atol=1e-11)
        ghost_region = np.roll(phantom.astype(bool), ny // 2, axis=0)
        original_ghost_energy = np.sum(np.sum(np.abs(uncorrected) ** 2, axis=1)[ghost_region])
        corrected_ghost_energy = np.sum(np.sum(np.abs(corrected) ** 2, axis=1)[ghost_region])
        self.assertGreater(original_ghost_energy, 1)
        self.assertLess(corrected_ghost_energy, 1e-20)

    def test_read_scan_requires_equally_spaced_navigator_echoes(self):
        twix, expected = preparation_tests.GenericTwixTests().fixture()
        nav = navigator_echoes(np.ones((3, 8)), .0, .2)
        nav_kspace = centered_fft(nav, (2,))
        navigators = []
        for index in range(3):
            block = copy.copy(twix["mdb"][0])
            block.data = nav_kspace[index, :, ::-1] if index == 1 else nav_kspace[index]
            block.is_image_scan = lambda: False
            block.is_flag_set = lambda flag, index=index: (
                flag == "PHASCOR" or (flag == "REFLECT" and index == 1))
            navigators.append(block)
        twix["mdb"] = navigators + twix["mdb"]
        acquisition = {"kx_mInv": np.zeros(expected.shape),
                       "navigator_echo_center_s": [.010, .012, .014]}
        with patch("twixtools.read_twix", return_value=[twix]):
            reconstructed, metadata = read_scan(Path("synthetic.dat"), acquisition)
            np.testing.assert_allclose(reconstructed, expected, atol=1e-12)
            self.assertEqual(metadata["navigator_count"], 3)
            acquisition["navigator_echo_center_s"] = [.010, .012, .0143]
            with self.assertRaisesRegex(ValueError, "equally spaced echoes"):
                read_scan(Path("synthetic.dat"), acquisition)


if __name__ == "__main__":
    unittest.main()
