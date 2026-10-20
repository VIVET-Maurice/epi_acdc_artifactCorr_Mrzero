"""Check physical sub-spin positions, measured interpolation and missing data."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from subspin_sampling import prepare_subspin_inputs, _load_source


def small_inputs(folder):
    """Two-by-two axial object; linear 3D fields and a finer measured B0 map."""
    row, column = np.indices((2, 2))
    X, Y = -.001 + column * .001, -.001 + row * .001
    experiment = {
        "m0_proxy": np.array([[1., 2.], [3., 4.]]),
        "geometry": {"X_m": X, "Y_m": Y,
            "position_mm": np.array([1., -1., 0.]), "spacing_mm": np.ones(2),
            "orientation": np.array([0., 1., 0., -1., 0., 0.]),
            "slice_thickness_mm": 2.},
        "field": {"simulation_mask": np.ones((2, 2), bool),
            "mask": np.ones((2, 2), bool), "b0Hz": np.array([[10., 20.], [30., 40.]])},
        "relaxation": {"T2star_s": np.array([[.01, .02], [.03, np.inf]])},
        "acquisition": {"unchanged": True},
    }
    axis = np.linspace(-.002, .003, 6)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    channel_scale = np.arange(1, 33)
    response = 2 + 100 * xx + 300 * yy + 400 * zz
    source = {"xx": xx, "yy": yy, "zz": zz,
        "fieldmap": response[..., None] * channel_scale,
        "mask": np.ones(xx.shape, bool)}
    path = Path(folder) / "calibration.npz"
    np.savez(path, **source)
    settings = {"units": "Hz/A", "registration_confirmed": True,
        "patient_to_source_m": np.eye(4)}
    native_axis = np.linspace(-.002, .002, 17)
    bx, by = np.meshgrid(native_axis, native_axis)
    native = {"b0Hz": 17 + 1e6 * bx**2 + 1e5 * by**2,
        "X_m": bx, "Y_m": by, "mask": np.ones(bx.shape, bool)}
    return experiment, source, path, settings, native


class SubspinSamplingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.experiment, self.source, self.path, self.settings, self.native = small_inputs(self.folder.name)

    def tearDown(self):
        # Release cached NPZ arrays before removing the fixture directory.
        _load_source.cache_clear()
        self.folder.cleanup()

    def prepare(self, **options):
        return prepare_subspin_inputs(self.experiment, self.path, self.settings, self.native, **options)

    def test_thickness_quadrature_preserves_amplitude_and_real_3d_coil_fields(self):
        original = deepcopy(self.experiment)
        result = self.prepare(counts=(1, 1, 8))
        expanded = result["expanded_experiment"]
        np.testing.assert_allclose(result["weights"].sum(axis=1), 1)
        np.testing.assert_allclose(expanded["m0_proxy"].sum(axis=1), self.experiment["m0_proxy"].ravel())
        np.testing.assert_allclose(result["offsets_m"].mean(axis=0), 0, atol=1e-18)
        np.testing.assert_allclose(np.ptp(result["xyz_m"][..., 2], axis=1), .002 * 7/8)
        np.testing.assert_allclose(result["subspin_b0"], np.repeat(self.experiment["field"]["b0Hz"].ravel()[:, None], 8, axis=1))
        xyz = result["xyz_m"]
        expected = (2 + 100 * xyz[..., 0] + 300 * xyz[..., 1] + 400 * xyz[..., 2])[..., None] * np.arange(1, 33)
        np.testing.assert_allclose(result["fieldbasis"], expected, atol=1e-13)
        self.assertGreater(result["report"]["calibration_thickness_spread_max_Hz_per_A"], 0)
        np.testing.assert_array_equal(self.experiment["m0_proxy"], original["m0_proxy"])
        self.assertIs(expanded["acquisition"], self.experiment["acquisition"])

    def test_native_b0_is_sampled_inside_parent_instead_of_repeating_parent_value(self):
        result = self.prepare(counts=(2, 2, 3), b0_mode="native")
        X, Y = result["xyz_m"][..., 1], -result["xyz_m"][..., 0]
        # Child points coincide with actual finer-grid measurements here.
        expected = 17 + 1e6 * X**2 + 1e5 * Y**2
        np.testing.assert_allclose(result["subspin_b0"], expected, atol=1e-13)
        self.assertGreater(result["report"]["B0_spread_max_Hz"], 0)
        self.assertEqual(result["report"]["native_B0_shape"], [17, 17])
        # The measured 2D B0 is identical at different thickness positions.
        b0 = result["subspin_b0"].reshape(4, 2, 2, 3)
        np.testing.assert_allclose(b0[..., 0], b0[..., 1])
        self.assertFalse(np.allclose(result["subspin_b0"].mean(axis=1), self.experiment["field"]["b0Hz"].ravel()))

    def test_inplane_and_thickness_positions_follow_dicom_axes(self):
        result = self.prepare(counts=(2, 2, 2), b0_mode="native")
        offsets = result["offsets_m"].reshape(2, 2, 2, 3)
        np.testing.assert_allclose(offsets[:, 0, 0, 0], [.00025, -.00025])
        np.testing.assert_allclose(offsets[0, :, 0, 1], [-.00025, .00025])
        np.testing.assert_allclose(offsets[0, 0, :, 2], [-.0005, .0005])
        np.testing.assert_allclose(result["expanded_experiment"]["geometry"]["Z_m"][0], offsets[..., 2].ravel())

    def test_explicit_artificial_z_gradient_has_correct_units(self):
        result = self.prepare(counts=(1, 1, 8), z_gradient_hz_per_m=1000.)
        expected = self.experiment["field"]["b0Hz"].ravel()[:, None] + 1000 * result["offsets_m"][:, 2]
        np.testing.assert_allclose(result["subspin_b0"], expected)
        self.assertAlmostEqual(result["report"]["B0_spread_max_Hz"], 1.75)

    def test_invalid_child_excludes_whole_parent_without_redistributing_amplitude(self):
        # Eliminate the stencil of one x-side of the source volume.
        self.source["mask"][4:, :, :] = False
        np.savez(self.path, **self.source)
        result = self.prepare(counts=(1, 1, 4))
        self.assertEqual(result["report"]["retained_parent_count"], 2)
        self.assertEqual(result["report"]["excluded_parent_count"], 2)
        self.assertEqual(result["expanded_experiment"]["m0_proxy"].shape, (2, 4))
        np.testing.assert_array_equal(result["parent_flat_index"], [2, 3])
        np.testing.assert_allclose(result["expanded_experiment"]["m0_proxy"].sum(axis=1), [3, 4])
        np.testing.assert_allclose(result["weights"], .25)
        self.assertAlmostEqual(result["report"]["retained_amplitude_fraction"], .7)

    def test_native_b0_full_stencil_checks_zero_weight_corner(self):
        # Parent centers coincide with B0 source centers. A masked neighbor is
        # still forbidden by the full-stencil policy, even at zero weight.
        bx = self.native["X_m"][0]
        by = self.native["Y_m"][:, 0]
        column = np.flatnonzero(np.isclose(bx, -.00075))[0]
        row = np.flatnonzero(np.isclose(by, -.00075))[0]
        self.native["mask"][row, column] = False
        result = self.prepare(counts=(1, 1, 2), b0_mode="native")
        self.assertFalse(result["parent_valid_mask"][0, 0])
        self.assertEqual(result["report"]["retained_parent_count"], 3)

    def test_b0_transposed_descending_axes_keep_physical_mapping(self):
        expected = self.prepare(counts=(2, 2, 2), b0_mode="native")
        self.native = {key: value.T[::-1, ::-1] for key, value in self.native.items()}
        actual = self.prepare(counts=(2, 2, 2), b0_mode="native")
        np.testing.assert_allclose(actual["subspin_b0"], expected["subspin_b0"])

    def test_off_grid_native_bilinear_sampling_recovers_known_linear_field(self):
        self.native["b0Hz"] = 30 + 12345 * self.native["X_m"] - 6789 * self.native["Y_m"]
        # Three in-plane midpoints do not coincide with the native grid.
        result = self.prepare(counts=(3, 3, 2), b0_mode="native")
        X, Y = result["xyz_m"][..., 1], -result["xyz_m"][..., 0]
        np.testing.assert_allclose(result["subspin_b0"], 30 + 12345 * X - 6789 * Y, atol=1e-13)

    def test_missing_native_coverage_never_extrapolates_or_keeps_partial_parents(self):
        # Only the parent at the lower-right is fully within this measured ROI.
        self.native = {key: value[6:, 6:] for key, value in self.native.items()}
        result = self.prepare(counts=(2, 2, 2), b0_mode="native")
        np.testing.assert_array_equal(result["parent_flat_index"], [3])
        np.testing.assert_allclose(result["expanded_experiment"]["m0_proxy"].sum(), 4)
        self.assertEqual(result["report"]["parents_missing_B0"], 3)

    def test_registered_affine_applies_at_every_child(self):
        affine = np.eye(4)
        affine[2, 3] = .0002
        self.settings["patient_to_source_m"] = affine
        result = self.prepare(counts=(1, 1, 4))
        xyz = result["xyz_m"]
        expected = 2 + 100 * xyz[..., 0] + 300 * xyz[..., 1] + 400 * (xyz[..., 2] + .0002)
        np.testing.assert_allclose(result["fieldbasis"][..., 0], expected)

    def test_quality_and_t2_are_repeated_without_changing_parent_meaning(self):
        self.experiment["field"]["mask"][0, 1] = False
        result = self.prepare(counts=(1, 1, 4))
        expanded = result["expanded_experiment"]
        np.testing.assert_array_equal(expanded["field"]["mask"].sum(axis=1), [4, 0, 4, 4])
        np.testing.assert_allclose(expanded["relaxation"]["T2star_s"][:, 0], [.01, .02, .03, np.inf])

    def test_preparation_reuses_original_3d_source_cache(self):
        self.prepare()
        self.prepare(counts=(2, 2, 2), b0_mode="native")
        self.assertEqual(_load_source.cache_info().misses, 1)
        self.assertEqual(_load_source.cache_info().hits, 1)


if __name__ == "__main__":
    unittest.main()
