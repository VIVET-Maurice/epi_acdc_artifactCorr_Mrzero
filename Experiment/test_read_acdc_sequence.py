"""Synthetic Pulseq fixtures and RF-to-ADC timing checks for the sequence reader."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import loadmat

from read_acdc_sequence import read_sequence


ROOT = Path(__file__).resolve().parent


def make_gre_sequence(path, nx=8, ny=6, fov=.2, te=.006, tr=.015, dummies=0,
                      *, phase_order=None, half_step=False, echo_count=1,
                      adc_phase_difference=0, dwell=20e-6):
    """Write a spoiled, constant-TR test GRE; no fixture represents measured data.

    FOV is metres (scalar or [readout, phase]); TE/TR are seconds. ``ny`` is
    the number of phase lines, independent of ``nx`` ADC samples. A spoiler
    follows every readout. Leading dummy TRs have the same RF/gradients but
    no ADC. This helper is shared with folder-pipeline integration tests.
    """
    import pypulseq as pp

    fov = np.broadcast_to(np.asarray(fov, dtype=float), (2,))
    order = np.arange(ny) if phase_order is None else np.asarray(phase_order)
    if sorted(order.tolist()) != list(range(ny)):
        raise ValueError("phase_order must permute 0..ny-1.")
    sequence = pp.Sequence()
    rf_duration = .0004
    rf_center = rf_duration / 2
    readout = pp.make_trapezoid("x", flat_area=nx / fov[0], flat_time=nx * dwell,
                               rise_time=.0001, fall_time=.0001)
    offset = 0 if half_step else -.5 / fov[0]
    pre_readout = pp.make_trapezoid("x", area=-readout.area / 2 + offset, duration=.001)
    spoiler = pp.make_trapezoid("x", area=2 * nx / fov[0], duration=.001)
    # Integer-grid kx=0 is a sample center; the half-step echo lies between samples.
    readout_echo = readout.rise_time + (nx / 2 + (0 if half_step else .5)) * dwell
    delay = rf_center + te - readout_echo - rf_duration - .001
    if delay < 0:
        raise ValueError("Fixture TE is too short for RF and prephasers.")
    for repetition in range(dummies + ny):
        phase_row = 0 if repetition < dummies else int(order[repetition - dummies])
        rf_phase = (repetition * repetition * .117) % (2 * np.pi)
        rf = pp.make_block_pulse(np.deg2rad(15), duration=rf_duration, use="excitation",
                                phase_offset=rf_phase)
        gy = pp.make_trapezoid("y", area=(phase_row - ny // 2) / fov[1], duration=.001)
        sequence.add_block(rf)
        sequence.add_block(pre_readout, gy)
        sequence.add_block(pp.make_delay(delay))
        for echo in range(echo_count):
            if repetition < dummies:
                sequence.add_block(readout)
            else:
                adc = pp.make_adc(nx, dwell=dwell, delay=readout.rise_time,
                                  phase_offset=rf_phase + adc_phase_difference)
                sequence.add_block(readout, adc)
        sequence.add_block(spoiler)
        elapsed = rf_duration + .001 + delay + echo_count * pp.calc_duration(readout) + .001
        if tr <= elapsed:
            raise ValueError("Fixture TR is too short.")
        sequence.add_block(pp.make_delay(tr - elapsed))
    sequence.set_definition("Name", "synthetic_spoiled_gre_fixture")
    sequence.set_definition("FOV", [*fov, .005])
    sequence.set_definition("Matrix", [nx, ny])
    sequence.set_definition("TE", te)
    sequence.set_definition("TR", tr)
    sequence.set_definition("NavigatorEchoes", 0)
    sequence.write(str(path))
    return Path(path)


class SequenceReaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)

    def fixture(self, **kwargs):
        return make_gre_sequence(self.folder / "gre.seq", **kwargs)

    def test_rectangular_gre_uses_each_lines_own_rf(self):
        acquisition, report = read_sequence(self.fixture(nx=8, ny=6))
        self.assertEqual(report["sequence_kind"], "gre")
        self.assertEqual(report["image_shape"], [6, 8])
        np.testing.assert_array_equal(acquisition["matrix_size"], [8, 6])
        np.testing.assert_allclose(acquisition["time_s"][:, 4] - acquisition["rf_time_s"][:, 4], .006)
        np.testing.assert_allclose(acquisition["rf_time_s"][:, 0], np.arange(6) * .015, atol=1e-12)
        np.testing.assert_allclose(acquisition["phase_offset_rad"], 0, atol=1e-10)
        self.assertTrue(np.all(np.asarray(report["state_end_s"][:-1]) < report["state_start_s"][1:]))
        np.testing.assert_allclose(report["state_start_s"], report["rf_end_s"])

    def test_dummies_do_not_create_current_states(self):
        acquisition, report = read_sequence(self.fixture(dummies=3))
        self.assertEqual(report["dummy_rf_count"], 3)
        self.assertEqual(len(report["rf_start_s"]), 9)
        self.assertEqual(len(report["state_start_s"]), 6)
        self.assertAlmostEqual(acquisition["rf_time_s"][0, 0], 3 * .015)
        self.assertAlmostEqual(report["sequence_duration_s"], 9 * .015)

    def test_cartesian_sort_keeps_the_chronological_current_mapping(self):
        order = [3, 2, 4, 1, 5, 0]
        acquisition, report = read_sequence(self.fixture(phase_order=order))
        np.testing.assert_array_equal(acquisition["line_index"][:, 0], np.argsort(order) + 1)
        np.testing.assert_allclose(acquisition["rf_time_s"][:, 0], np.argsort(order) * .015)
        self.assertTrue(np.all(np.diff(report["state_start_s"]) > 0))

    def test_standard_half_step_readout_uses_interpolated_echo_center(self):
        acquisition, report = read_sequence(self.fixture(half_step=True))
        # Pulseq's text serialization rounds gradient amplitudes.
        np.testing.assert_allclose(acquisition["kx_mInv"][0], (np.arange(8) - 3.5) / .2, atol=1e-3)
        self.assertAlmostEqual(report["readout_grid_offset_mInv"], 2.5)
        self.assertAlmostEqual(acquisition["TE_s"], .006)
        self.assertAlmostEqual(acquisition["time_s"][0, 4], .00601)

    def test_adc_phase_is_retained_in_forward_model_convention(self):
        acquisition, _ = read_sequence(self.fixture(adc_phase_difference=.3))
        np.testing.assert_allclose(acquisition["phase_offset_rad"], -.3, atol=1e-9)

    def test_multiple_echoes_per_rf_rejected(self):
        with self.assertRaisesRegex(ValueError, "Multi-echo"):
            read_sequence(self.fixture(echo_count=2))

    def test_odd_matrix_rejected(self):
        with self.assertRaisesRegex(ValueError, "even"):
            read_sequence(self.fixture(ny=5))

    def test_header_matrix_cannot_override_acquired_trajectory(self):
        import pypulseq as pp
        path = self.fixture()
        sequence = pp.Sequence()
        sequence.read(str(path))
        sequence.set_definition("Matrix", [8, 12])
        sequence.write(str(path))
        with self.assertRaisesRegex(ValueError, "Matrix disagrees"):
            read_sequence(path)

    def test_variable_rf_tr_rejected(self):
        import pypulseq as pp
        path = self.fixture()
        sequence = pp.Sequence()
        sequence.read(str(path))
        sequence.set_definition("TR", .02)
        sequence.write(str(path))
        with self.assertRaisesRegex(ValueError, "constant TR"):
            read_sequence(path)

    def test_requested_matrix_sizes_with_different_acquisition_parameters(self):
        cases = [(32, [.16, .24], .008, .024, 20e-6),
                 (64, [.24, .30], .012, .030, 40e-6)]
        for size, fov, te, tr, dwell in cases:
            with self.subTest(matrix=size):
                acquisition, report = read_sequence(self.fixture(
                    nx=size, ny=size, fov=fov, te=te, tr=tr, dwell=dwell))
                self.assertEqual(report["image_shape"], [size, size])
                np.testing.assert_allclose(acquisition["FOV_m"], fov, rtol=1e-5)
                self.assertAlmostEqual(acquisition["TE_s"], te)
                self.assertAlmostEqual(acquisition["TR_s"], tr)
                self.assertAlmostEqual(acquisition["dwell_s"], dwell)
                np.testing.assert_allclose(np.diff(acquisition["rf_time_s"][:, 0]), tr)

    def test_changing_rf_flip_angle_rejected(self):
        import pypulseq as pp
        path = self.fixture()
        sequence = pp.Sequence()
        sequence.read(str(path))
        rf_blocks = [number for number in sequence.block_events if sequence.get_block(number).rf]
        old = sequence.get_block(rf_blocks[1]).rf
        sequence.set_block(rf_blocks[1], pp.make_block_pulse(
            .1, duration=.0004, use="excitation", phase_offset=old.phase_offset))
        sequence.write(str(path))
        with self.assertRaisesRegex(ValueError, "same RF waveform"):
            read_sequence(path)

    def test_changing_rf_slice_frequency_rejected(self):
        import pypulseq as pp
        path = self.fixture()
        sequence = pp.Sequence()
        sequence.read(str(path))
        rf_blocks = [number for number in sequence.block_events if sequence.get_block(number).rf]
        old = sequence.get_block(rf_blocks[1]).rf
        sequence.set_block(rf_blocks[1], pp.make_block_pulse(
            np.deg2rad(15), duration=.0004, use="excitation", phase_offset=old.phase_offset,
            freq_offset=200))
        sequence.write(str(path))
        with self.assertRaisesRegex(ValueError, "RF frequency"):
            read_sequence(path)

    def test_measured_64_gre_sequence_keeps_native_matrix_and_timing(self):
        path = ROOT.parent / "Mrzero" / "sequence_simulation" / "gre_64x64_fov200mm_tr1s.seq"
        acquisition, report = read_sequence(path)
        self.assertEqual(report["sequence_kind"], "gre")
        self.assertEqual(report["image_shape"], [64, 64])
        np.testing.assert_array_equal(acquisition["matrix_size"], [64, 64])
        for key in ("kx_mInv", "ky_mInv", "time_s", "rf_time_s", "line_index"):
            self.assertEqual(acquisition[key].shape, (64, 64))
        self.assertEqual(len(report["state_start_s"]), 64)
        self.assertEqual(report["navigator_count"], 0)
        np.testing.assert_array_equal(acquisition["polarity"], np.ones(64))
        np.testing.assert_array_equal(acquisition["line_index"][:, 0], np.arange(1, 65))
        np.testing.assert_allclose(acquisition["FOV_m"], [.2, .2], rtol=1e-4)
        self.assertAlmostEqual(acquisition["TE_s"], .00407825, places=10)
        self.assertAlmostEqual(acquisition["TR_s"], 1.0, places=10)
        self.assertAlmostEqual(acquisition["dwell_s"], 78.1e-6, places=12)
        np.testing.assert_allclose(np.diff(acquisition["rf_time_s"][:, 0]), 1.0, atol=1e-10)
        self.assertTrue(np.all(np.diff(report["state_start_s"]) > 0))
        self.assertTrue(np.all(np.asarray(report["state_end_s"][:-1]) < report["state_start_s"][1:]))

    def test_measured_64_epi_sequence_keeps_navigators_and_current_states(self):
        path = ROOT.parent / "Mrzero" / "sequence_simulation" / (
            "gre_epi_64x64_fov200mm_nav3_rise1ms_ext1_zero_crossings_v3.seq")
        acquisition, report = read_sequence(path)
        self.assertEqual(report["sequence_kind"], "epi")
        self.assertEqual(report["image_shape"], [64, 64])
        np.testing.assert_array_equal(acquisition["matrix_size"], [64, 64])
        for key in ("kx_mInv", "ky_mInv", "time_s", "rf_time_s", "line_index"):
            self.assertEqual(acquisition[key].shape, (64, 64))
        self.assertEqual(report["navigator_count"], 3)
        self.assertEqual(len(report["state_start_s"]), 64)
        self.assertEqual(len(report["state_edges_s"]), 65)
        np.testing.assert_array_equal(acquisition["line_index"][:, 0], np.arange(1, 65))
        np.testing.assert_array_equal(acquisition["polarity"], np.tile([-1, 1], 32))
        np.testing.assert_array_equal(acquisition["rf_time_s"], 0)
        np.testing.assert_allclose(acquisition["FOV_m"], [.2, .2], rtol=1e-4)
        self.assertAlmostEqual(acquisition["TE_s"], .0984395, places=10)
        self.assertAlmostEqual(acquisition["TR_s"], .18594, places=10)
        self.assertAlmostEqual(acquisition["dwell_s"], 11e-6, places=12)
        self.assertTrue(np.all(np.diff(report["state_start_s"]) > 0))
        np.testing.assert_allclose(report["state_end_s"][:-1], report["state_start_s"][1:], atol=1e-12)

    def test_phase_gradient_during_readout_is_rejected_as_non_cartesian(self):
        import pypulseq as pp
        path = self.fixture()
        sequence = pp.Sequence()
        sequence.read(str(path))
        number = next(number for number in sequence.block_events if sequence.get_block(number).adc)
        block = sequence.get_block(number)
        phase_gradient = pp.make_trapezoid(
            "y", flat_area=1 / .2, flat_time=block.gx.flat_time,
            rise_time=block.gx.rise_time, fall_time=block.gx.fall_time)
        sequence.set_block(number, block.gx, phase_gradient, block.adc)
        sequence.write(str(path))
        with self.assertRaisesRegex(ValueError, "Cartesian grid"):
            read_sequence(path)

    def test_missing_phase_position_is_rejected_as_undersampling(self):
        import pypulseq as pp
        path = self.fixture()
        sequence = pp.Sequence()
        sequence.read(str(path))
        phase_blocks = [number for number in sequence.block_events if sequence.get_block(number).gy]
        number = phase_blocks[-1]
        block = sequence.get_block(number)
        # Acquire ky=3/FOV instead of 2/FOV: one Cartesian position is missing.
        skipped_phase = pp.make_trapezoid("y", area=3 / .2, duration=.001)
        sequence.set_block(number, block.gx, skipped_phase)
        sequence.write(str(path))
        with self.assertRaisesRegex(ValueError, "Cartesian grid"):
            read_sequence(path)

    def test_historical_epi_trajectory_and_timing_unchanged(self):
        path = ROOT / "sequence" / "gre_epi_32x32_fov200mm_nav3_mintr.seq"
        prepared = ROOT / "prepared" / "experiment_input.mat"
        if not path.is_file() or not prepared.is_file():
            self.skipTest("Historical measured EPI fixture is unavailable.")
        acquisition, timing = read_sequence(path)
        historical = loadmat(prepared, simplify_cells=True)["experiment"]["acquisition"]
        for key in ("kx_mInv", "ky_mInv", "time_s", "line_index", "line_start_s"):
            np.testing.assert_allclose(np.asarray(acquisition[key]).squeeze(), historical[key], atol=1e-12)
        self.assertEqual(timing["sequence_kind"], "epi")
        self.assertEqual(timing["navigator_count"], 3)
        np.testing.assert_allclose(acquisition["rf_time_s"], 0)
        np.testing.assert_allclose(timing["first_adc_delay_s"], 94e-6, atol=1e-12)

    def test_epi_matrix_is_inferred_from_events(self):
        import pypulseq as pp
        sequence = pp.Sequence()
        nx, ny, fov, dwell = 8, 6, .2, 20e-6
        gx = pp.make_trapezoid("x", flat_area=nx / fov, flat_time=nx * dwell,
                               rise_time=.0001, fall_time=.0001)
        pre_x = pp.make_trapezoid("x", area=-gx.area / 2 - .5 / fov, duration=.001)
        pre_y = pp.make_trapezoid("y", area=-ny / 2 / fov, duration=.001)
        blip = pp.make_trapezoid("y", area=1 / fov, duration=.0002)
        sequence.add_block(pp.make_block_pulse(.2, duration=.0004, use="excitation"))
        sequence.add_block(pre_x, pre_y)
        for row in range(ny):
            readout = pp.make_trapezoid("x", flat_area=(-1) ** row * nx / fov,
                                       flat_time=nx * dwell, rise_time=.0001, fall_time=.0001)
            sequence.add_block(readout, pp.make_adc(nx, dwell=dwell, delay=readout.rise_time))
            if row < ny - 1:
                sequence.add_block(blip)
        # No Matrix/TE/FOV metadata: validate the actual sampled trajectory.
        path = self.folder / "epi.seq"
        sequence.write(str(path))
        acquisition, report = read_sequence(path)
        self.assertEqual(report["sequence_kind"], "epi")
        self.assertEqual(report["image_shape"], [ny, nx])
        np.testing.assert_array_equal(acquisition["polarity"], [1, -1, 1, -1, 1, -1])
        np.testing.assert_allclose(acquisition["FOV_m"], [.2, .2], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
