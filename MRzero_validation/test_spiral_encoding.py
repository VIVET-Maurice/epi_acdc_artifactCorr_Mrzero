"""Effective gradient delays preserve ADC timing and nominal reconstruction."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Experiment"))

from spiral_encoding import (
    apply_spiral_encoding, calibration_acquisition, delayed_spiral_trajectory,
    reconstruction_acquisition,
)


SEQUENCE = ROOT / "acquisition_032_(07)/sequences/gre_spiral_128x128_fov200mm_nav3_cimax_turn_triggers_v1.seq"


def fixture(matrix=None):
    matrix = np.eye(2) if matrix is None else np.asarray(matrix)
    time = np.linspace(.005, .025, 121)[None]
    angle = 2 * np.pi * (time - time[0, 0]) / .004
    native = dict(sequence_kind="spiral", kx_mInv=1000 * time * np.cos(angle),
        ky_mInv=1000 * time * np.sin(angle), time_s=time, rf_time_s=np.zeros_like(time),
        phase_offset_rad=np.full(time.shape, .12), line_index=np.repeat(np.arange(1, 7), [21, 20, 20, 20, 20, 20])[None],
        line_start_s=np.arange(6)[:, None] * .004 + .004,
        line_end_s=np.arange(1, 7)[:, None] * .004 + .004,
        density_compensation_weights=np.linspace(0, 1, time.size)[None],
        FOV_m=[.2, .2], image_shape=[32, 32])
    framed = deepcopy(native)
    k = np.stack((native["kx_mInv"], native["ky_mInv"]), axis=-1) @ matrix.T
    framed.update(kx_mInv=k[..., 0], ky_mInv=k[..., 1], combined_frame_matrix=matrix.copy())
    return {"expanded_experiment": {"acquisition": framed, "m0_proxy": np.array([1., 2.])},
        "acquisition_native": native, "combined_frame_matrix": matrix.copy(),
        "weights": np.array([[.5, .5]]), "report": {"original": True}}


class SpiralEncodingTests(unittest.TestCase):
    def test_positive_delay_is_gradient_lag_and_point_phase_has_correct_sign(self):
        native = fixture()["acquisition_native"]
        t = native["time_s"]
        native["kx_mInv"], native["ky_mInv"] = 3000 * t + 7, -2000 * t + 4
        dx, dy = 8e-6, -3e-6
        kx, ky = delayed_spiral_trajectory(native, dx, dy)
        np.testing.assert_allclose(kx, 3000 * (t - dx) + 7, atol=2e-14)
        np.testing.assert_allclose(ky, -2000 * (t - dy) + 4, atol=2e-14)
        x, y = .023, -.017
        original = np.exp(-2j * np.pi * (native["kx_mInv"] * x + native["ky_mInv"] * y))
        delayed = np.exp(-2j * np.pi * (kx * x + ky * y))
        np.testing.assert_allclose(delayed / original,
            np.exp(2j * np.pi * (3000 * dx * x - 2000 * dy * y)), atol=3e-15)

    def test_zero_delay_is_exact_identity_and_does_not_mutate_inputs(self):
        prepared = fixture(np.array([[.6, -.8], [-.8, -.6]]))
        original = deepcopy(prepared)
        result = apply_spiral_encoding(prepared, {})
        for key in ("kx_mInv", "ky_mInv", "time_s", "line_index", "line_start_s", "line_end_s"):
            np.testing.assert_array_equal(result["expanded_experiment"]["acquisition"][key],
                original["expanded_experiment"]["acquisition"][key])
            np.testing.assert_array_equal(prepared["expanded_experiment"]["acquisition"][key],
                original["expanded_experiment"]["acquisition"][key])
        self.assertNotIn("nominal_kx_mInv", prepared["expanded_experiment"]["acquisition"])

    def test_native_axis_delays_are_covariant_under_rotation_and_reflection(self):
        matrix = np.array([[.6, -.8], [-.8, -.6]])
        native_prepared, framed_prepared = fixture(), fixture(matrix)
        delays = dict(delay_x_s=7e-6, delay_y_s=-2e-6)
        native_result = apply_spiral_encoding(native_prepared, delays)
        framed_result = apply_spiral_encoding(framed_prepared, delays)
        actual_native = native_result["expanded_experiment"]["acquisition"]
        actual_framed = framed_result["expanded_experiment"]["acquisition"]
        expected = np.stack((actual_native["kx_mInv"], actual_native["ky_mInv"]), axis=-1) @ matrix.T
        np.testing.assert_allclose(actual_framed["kx_mInv"], expected[..., 0], atol=1e-14)
        np.testing.assert_allclose(actual_framed["ky_mInv"], expected[..., 1], atol=1e-14)
        native_point = np.array([.017, -.023])
        framed_point = matrix @ native_point
        np.testing.assert_allclose(actual_framed["kx_mInv"] * framed_point[0] + actual_framed["ky_mInv"] * framed_point[1],
            actual_native["kx_mInv"] * native_point[0] + actual_native["ky_mInv"] * native_point[1], atol=2e-16)
        calibration = calibration_acquisition(framed_prepared["acquisition_native"], actual_framed, matrix)
        np.testing.assert_allclose(calibration["kx_mInv"], actual_native["kx_mInv"], atol=1e-14)
        np.testing.assert_allclose(calibration["ky_mInv"], actual_native["ky_mInv"], atol=1e-14)

    def test_nominal_reconstruction_and_current_timing_are_frozen(self):
        prepared = fixture()
        result = apply_spiral_encoding(prepared, dict(delay_x_s=10e-6, delay_y_s=-6e-6))
        forward = result["expanded_experiment"]["acquisition"]
        original = prepared["expanded_experiment"]["acquisition"]
        self.assertGreater(np.max(abs(forward["kx_mInv"] - original["kx_mInv"])), 0)
        reconstructed = reconstruction_acquisition(forward)
        for key in ("kx_mInv", "ky_mInv", "time_s", "rf_time_s", "phase_offset_rad",
                    "line_index", "line_start_s", "line_end_s", "density_compensation_weights"):
            np.testing.assert_array_equal(reconstructed[key], original[key])
        # Repeated application starts from nominal, never from an already delayed path.
        repeated = apply_spiral_encoding(result, {})
        np.testing.assert_array_equal(repeated["expanded_experiment"]["acquisition"]["kx_mInv"], original["kx_mInv"])
        np.testing.assert_array_equal(result["expanded_experiment"]["m0_proxy"], prepared["expanded_experiment"]["m0_proxy"])
        np.testing.assert_array_equal(result["weights"], prepared["weights"])
        report = result["report"]["spiral_encoding"]
        self.assertFalse(report["adc_times_changed"])
        self.assertFalse(report["turn_state_indices_changed"])
        self.assertFalse(report["current_integration_windows_changed"])

    def test_invalid_delay_trajectory_and_frame_are_rejected(self):
        prepared = fixture()
        for parameters in ({"delay_x_s": np.nan}, {"delay_x_s": True}, {"delay_z_s": 0.}):
            with self.assertRaises(ValueError):
                apply_spiral_encoding(prepared, parameters)
        invalid = deepcopy(prepared["acquisition_native"])
        invalid["time_s"][0, 3] = invalid["time_s"][0, 2]
        with self.assertRaises(ValueError):
            delayed_spiral_trajectory(invalid, 0., 0.)
        invalid = deepcopy(prepared)
        invalid["combined_frame_matrix"] = np.ones((2, 2))
        with self.assertRaises(ValueError):
            apply_spiral_encoding(invalid, {})
        invalid = deepcopy(prepared)
        invalid["expanded_experiment"]["acquisition"]["nominal_kx_mInv"] = invalid["acquisition_native"]["kx_mInv"]
        with self.assertRaises(ValueError):
            apply_spiral_encoding(invalid, {})

    @unittest.skipUnless(SEQUENCE.is_file(), "Actual spiral sequence is unavailable")
    def test_actual_pulseq_delay_uses_full_gradient_history_with_fixed_adc_times(self):
        import pypulseq as pp
        from read_spiral_sequence import read_spiral_sequence

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            native, timing = read_spiral_sequence(SEQUENCE)
            native["sequence_sha256"] = timing["sequence_sha256"]
            dx, dy = 5.9e-6, -.33e-6
            kx, ky = delayed_spiral_trajectory(native, dx, dy, sequence_path=SEQUENCE)
            sequence = pp.Sequence()
            sequence.read(str(SEQUENCE))
            expected = sequence.calculate_kspace(trajectory_delay=[-dx, -dy, 0.])[0][:, -kx.size:]
        np.testing.assert_array_equal(kx.ravel(), expected[0])
        np.testing.assert_array_equal(ky.ravel(), expected[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            zero_x, zero_y = delayed_spiral_trajectory(native, 0., 0., sequence_path=SEQUENCE)
        np.testing.assert_array_equal(zero_x, native["kx_mInv"])
        np.testing.assert_array_equal(zero_y, native["ky_mInv"])
        mismatched = deepcopy(native)
        mismatched["sequence_sha256"] = "0" * 64
        with warnings.catch_warnings(), self.assertRaises(ValueError):
            warnings.simplefilter("ignore", UserWarning)
            delayed_spiral_trajectory(mismatched, dx, dy, sequence_path=SEQUENCE)


if __name__ == "__main__":
    unittest.main()
