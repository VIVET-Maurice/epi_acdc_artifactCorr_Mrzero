"""Native echo sampling, frame invariance and patient coil coverage checks."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from echo_shim_inputs import prepare_echo_shim_inputs
from subspin_sampling import _load_source


def fixture(folder):
    coordinate = (np.arange(8) - 4) * .001
    X, Y = np.meshgrid(coordinate, coordinate)
    k = (np.arange(4) - 2) / .008
    kx, ky = np.meshgrid(k, k)
    support = np.ones(X.shape, bool)
    support[0, 0] = False
    quaternion = (np.array([1., 0., 0., 1.]) / np.sqrt(2)).tolist()
    metadata = dict(slice_position_mm=[0., 0., 10.], slice_thickness_mm=2.,
                    slice_quaternion_wxyz=quaternion)
    inputs = dict(acquisition=dict(kx_mInv=kx, ky_mInv=ky, FOV_m=np.array([.008, .008]),
                    time_s=np.arange(16).reshape(4, 4) * 1e-4 + .01,
                    TE_s=.02, TR_s=.1),
        b0_raw=dict(X_m=X, Y_m=Y, metadata=deepcopy(metadata)),
        scan_metadata=metadata,
        field=dict(b0Hz=17 + 1000 * X - 700 * Y, mask=np.ones(X.shape, bool)),
        source_sha256={"scan": "declared-test-fixture"})
    first = 2 + 50 * X + 30 * Y
    fit = dict(first_echo_amplitude=first, object_support=support.copy(),
        simulation_support=support.copy(), amplitude_reference_time_s=.002,
        pooled=dict(R2star_sInv=7., target_reliable_mask=True),
        simulation_amplitude_at_reference=first * 1.02,
        simulation_R2star_sInv=10 + 100 * X)
    x, y, z = np.linspace(-.008, .008, 9), np.linspace(-.008, .008, 9), np.linspace(.006, .014, 5)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    response = 3 + 100 * xx + 170 * yy + 200 * zz
    source = dict(xx=xx, yy=yy, zz=zz, mask=np.ones(xx.shape, bool),
                  fieldmap=response[..., None] * np.arange(1, 33))
    path = Path(folder) / "coil_maps.npz"
    np.savez(path, **source)
    settings = dict(units="Hz/A", registration_confirmed=True, patient_to_source_m=np.eye(4))
    return inputs, fit, source, path, settings


class EchoShimInputTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.inputs, self.fit, self.source, self.path, self.settings = fixture(self.folder.name)

    def tearDown(self):
        _load_source.cache_clear()
        self.folder.cleanup()

    def prepare(self, **options):
        return prepare_echo_shim_inputs(self.inputs, self.fit, self.path, self.settings, **options)

    def test_native_amplitude_area_thickness_weights_and_time_anchor(self):
        result = self.prepare(subspins_z=8)
        expanded = result["expanded_experiment"]
        indices = result["parent_flat_index"]
        # Native 1 mm pixels versus EPI 2 mm pixels: area ratio one quarter.
        np.testing.assert_allclose(expanded["m0_proxy"].sum(axis=1),
            self.fit["first_echo_amplitude"].ravel()[indices] / 4.)
        np.testing.assert_allclose(result["weights"].sum(axis=1), 1.)
        np.testing.assert_allclose(expanded["relaxation"]["T2star_s"], 1. / 7.)
        self.assertEqual(expanded["relaxation"]["amplitude_reference_time_s"], .002)
        self.assertEqual(expanded["amplitude_reference_time_s"], .002)
        self.assertEqual(expanded["geometry"]["X_m"].shape, (63, 8))
        self.assertTrue(result["report"]["all_children_in_field_limits"])

    def test_each_thickness_point_queries_original_patient_3d_maps(self):
        result = self.prepare(frame_angle_deg=73., alignment_angle_deg=21., subspins_z=8)
        xyz = result["xyz_m"]
        expected = (3 + 100 * xyz[..., 0] + 170 * xyz[..., 1] + 200 * xyz[..., 2])[..., None] * np.arange(1, 33)
        np.testing.assert_allclose(result["fieldbasis"], expected, atol=1e-12)
        np.testing.assert_allclose(xyz[..., 2].mean(axis=1), .01)
        np.testing.assert_allclose(np.ptp(xyz[..., 2], axis=1), .002 * 7 / 8)
        self.assertGreater(np.ptp(result["fieldbasis"][0, :, 0]), 0.)

    def test_arbitrary_joint_frame_and_mirrors_preserve_encoding_dot_products(self):
        for angle, mx, my in ((37., False, False), (291., True, False), (21., True, True)):
            result = self.prepare(frame_angle_deg=angle, frame_mirror_x=mx,
                alignment_angle_deg=53., alignment_mirror_y=my)
            expanded, indices = result["expanded_experiment"], result["parent_flat_index"]
            original_acq = self.inputs["acquisition"]
            x = self.inputs["b0_raw"]["X_m"].ravel()[indices]
            y = self.inputs["b0_raw"]["Y_m"].ravel()[indices]
            original_dot = original_acq["kx_mInv"][..., None] * x + original_acq["ky_mInv"][..., None] * y
            acq = expanded["acquisition"]
            transformed_dot = (acq["kx_mInv"][..., None] * expanded["geometry"]["X_m"][:, 0]
                + acq["ky_mInv"][..., None] * expanded["geometry"]["Y_m"][:, 0])
            np.testing.assert_allclose(transformed_dot, original_dot, atol=2e-15)
            matrix = result["combined_frame_matrix"]
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(2), atol=1e-15)

    def test_passive_frame_does_not_change_patient_queries_coil_fields_or_support(self):
        baseline = self.prepare(frame_angle_deg=0., alignment_angle_deg=33.)
        changed = self.prepare(frame_angle_deg=273., frame_mirror_x=True, alignment_angle_deg=33.)
        for key in ("xyz_m", "fieldbasis", "parent_valid_mask", "native_B0_Hz"):
            np.testing.assert_array_equal(changed[key], baseline[key])

    def test_imaging_to_coil_alignment_changes_fields_without_reindexing_scalar_data(self):
        baseline = self.prepare(alignment_angle_deg=0.)
        aligned = self.prepare(alignment_angle_deg=90., alignment_mirror_x=True)
        self.assertFalse(np.allclose(aligned["fieldbasis"], baseline["fieldbasis"]))
        for key in ("native_amplitude", "native_rate_sInv", "native_B0_Hz", "parent_valid_mask"):
            np.testing.assert_array_equal(aligned[key], baseline[key])
        np.testing.assert_allclose(aligned["physical_X_m"], -self.inputs["b0_raw"]["Y_m"], atol=1e-18)
        np.testing.assert_allclose(aligned["physical_Y_m"], -self.inputs["b0_raw"]["X_m"], atol=1e-18)

    def test_calibration_holes_drop_complete_parents_without_amplitude_renormalization(self):
        self.source["mask"][5:, :, :] = False
        np.savez(self.path, **self.source)
        result = self.prepare(alignment_angle_deg=0., subspins_z=8)
        self.assertGreater(result["report"]["excluded_missing_ACDC_parent_count"], 0)
        self.assertGreater(result["report"]["excluded_ACDC_amplitude_fraction_of_baseline"], 0.)
        idx = result["parent_flat_index"]
        np.testing.assert_allclose(result["expanded_experiment"]["m0_proxy"].sum(axis=1),
                                   self.fit["first_echo_amplitude"].ravel()[idx] / 4.)
        self.assertTrue(np.isfinite(result["fieldbasis"]).all())

    def test_missing_b0_is_never_filled_or_retained_by_coil_coverage(self):
        self.inputs["field"]["mask"][3, 3] = False
        result = self.prepare()
        self.assertFalse(result["parent_valid_mask"][3, 3])
        self.assertEqual(result["report"]["excluded_missing_B0_parent_count"], 1)
        self.assertGreater(result["report"]["excluded_missing_B0_amplitude_fraction"], 0.)

    def test_field_roi_changes_limits_only_and_is_explicitly_reported(self):
        baseline = self.prepare()
        quality = np.zeros((8, 8), bool)
        quality[3:5, 3:5] = True
        restricted = self.prepare(quality_mask=quality)
        np.testing.assert_array_equal(restricted["parent_valid_mask"], baseline["parent_valid_mask"])
        np.testing.assert_array_equal(restricted["expanded_experiment"]["m0_proxy"], baseline["expanded_experiment"]["m0_proxy"])
        self.assertFalse(restricted["report"]["all_children_in_field_limits"])
        self.assertEqual(restricted["report"]["field_limit_children"], 4 * 8)

    def test_decay_modes_preserve_first_echo_anchor_and_guarded_fallback_values(self):
        no_decay = self.prepare(rate_mode="none")
        self.assertTrue(np.isinf(no_decay["expanded_experiment"]["relaxation"]["T2star_s"]).all())
        guarded = self.prepare(rate_mode="guarded")
        indices = guarded["parent_flat_index"]
        np.testing.assert_allclose(guarded["expanded_experiment"]["m0_proxy"].sum(axis=1),
            self.fit["simulation_amplitude_at_reference"].ravel()[indices] / 4.)
        self.fit["pooled"]["target_reliable_mask"] = False
        rejected = self.prepare(rate_mode="pooled")
        self.assertFalse(rejected["report"]["pooled_rate_accepted"])
        self.assertTrue(np.isinf(rejected["expanded_experiment"]["relaxation"]["T2star_s"]).all())

    def test_inputs_remain_unchanged_and_no_translation_phase_is_implicitly_added(self):
        before_inputs, before_fit = deepcopy(self.inputs), deepcopy(self.fit)
        self.prepare(frame_angle_deg=11.3, alignment_angle_deg=223.4)
        for source in ("field", "b0_raw", "acquisition"):
            for key, value in before_inputs[source].items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(self.inputs[source][key], value)
        for key in ("first_echo_amplitude", "simulation_amplitude_at_reference", "simulation_R2star_sInv"):
            np.testing.assert_array_equal(self.fit[key], before_fit[key])
        with self.assertRaisesRegex(ValueError, "origin-centered"):
            self.prepare(alignment_center_m=(.001, 0.))


if __name__ == "__main__":
    unittest.main()
