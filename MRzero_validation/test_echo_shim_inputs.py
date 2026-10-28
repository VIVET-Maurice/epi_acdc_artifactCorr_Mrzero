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

    def test_finite_voxel_affine_gradients_follow_native_axes_and_physical_alignment(self):
        result = self.prepare(inplane_voxel=True, alignment_angle_deg=31.,
                              alignment_mirror_x=True, frame_angle_deg=47., subspins_z=3)
        voxel = result["expanded_experiment"]["inplane_voxel"]
        native_edges = np.array([[0., .001], [.001, 0.]])
        np.testing.assert_allclose(voxel["edge_vectors_m"],
            native_edges @ result["combined_frame_matrix"].T, atol=1e-18)
        expected_b0 = np.broadcast_to([-.7, 1.], voxel["b0_gradient_Hz_per_pixel"].shape)
        np.testing.assert_allclose(voxel["b0_gradient_Hz_per_pixel"], expected_b0, atol=1e-12)
        alignment = np.asarray(result["report"]["alignment_matrix"])
        aligned_edges = native_edges @ alignment.T
        patient_edges = np.column_stack((-aligned_edges[:, 1], aligned_edges[:, 0], np.zeros(2)))
        expected_coil = ((patient_edges @ [100., 170., 200.])[:, None] * np.arange(1, 33))
        np.testing.assert_allclose(voxel["field_gradient_Hz_per_A_per_pixel"],
            np.broadcast_to(expected_coil, voxel["field_gradient_Hz_per_A_per_pixel"].shape), atol=2e-12)
        corners = np.asarray([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
        corner_xyz = result["xyz_m"][..., None, :] + corners @ patient_edges
        expected_corners = (3 + 100 * corner_xyz[..., 0] + 170 * corner_xyz[..., 1]
                            + 200 * corner_xyz[..., 2])[..., None] * np.arange(1, 33)
        np.testing.assert_allclose(voxel["field_at_corners_Hz_per_A"], expected_corners, atol=2e-12)
        self.assertTrue(voxel["ACDC_corner_coverage_verified"])
        self.assertLess(result["report"]["ACDC_corner_linearization_max_abs_error_Hz_per_A"], 1e-12)

    def test_finite_voxel_passive_frame_preserves_patient_fields_and_native_slopes(self):
        original = self.prepare(inplane_voxel=True, frame_angle_deg=0.)
        changed = self.prepare(inplane_voxel=True, frame_angle_deg=71., frame_mirror_y=True)
        for key in ("xyz_m", "fieldbasis", "parent_valid_mask"):
            np.testing.assert_array_equal(original[key], changed[key])
        for key in ("b0_gradient_Hz_per_pixel", "field_gradient_Hz_per_A_per_pixel"):
            np.testing.assert_array_equal(original["expanded_experiment"]["inplane_voxel"][key],
                                          changed["expanded_experiment"]["inplane_voxel"][key])

    def test_unknown_diagonal_calibration_corner_excludes_complete_voxel(self):
        # This source node is in the diagonal corner stencil of patient (0,0),
        # but not its centre or any of its four axial endpoint stencils.
        self.source["mask"][3, 3, :] = False
        np.savez(self.path, **self.source)
        point = self.prepare(alignment_angle_deg=0., subspins_z=3)
        voxel = self.prepare(alignment_angle_deg=0., subspins_z=3, inplane_voxel=True)
        self.assertTrue(point["parent_valid_mask"][4, 4])
        self.assertFalse(voxel["parent_valid_mask"][4, 4])
        self.assertGreater(voxel["report"]["excluded_missing_ACDC_gradient_or_corner_parent_count"], 0)
        indices = voxel["parent_flat_index"]
        np.testing.assert_allclose(voxel["expanded_experiment"]["m0_proxy"].sum(axis=1),
                                   self.fit["first_echo_amplitude"].ravel()[indices] / 4)

    def test_singular_native_B0_gradient_excludes_parent_instead_of_assuming_zero(self):
        valid = np.zeros((8, 8), bool)
        valid[1:4, 1:4] = True
        valid[7, 7] = True
        self.inputs["field"]["mask"] = valid
        result = self.prepare(inplane_voxel=True)
        self.assertFalse(result["parent_valid_mask"][7, 7])
        self.assertEqual(result["report"]["excluded_missing_B0_gradient_parent_count"], 1)
        self.assertTrue(np.isnan(result["native_B0_gradient_Hz_per_pixel"][7, 7]).all())

    def test_echo_power_uses_only_declared_B0_echoes_and_no_EPI_image(self):
        b0 = self.inputs["field"]["b0Hz"]
        first = self.fit["first_echo_amplitude"]
        te = .002 + np.arange(5) * .002
        coils = first[None, None] * np.exp(-2j * np.pi * te[:, None, None, None] * b0[None, None])
        coils[3:] *= 100  # Later relaxation echoes must not change B0 weights.
        self.inputs["b0_raw"]["coil_images"] = coils
        self.inputs["field_echo_count"] = 3
        self.inputs["scan"] = np.full((4, 4), np.nan)
        result = self.prepare(inplane_voxel=True, b0_weighting="echo_power")
        np.testing.assert_allclose(result["native_B0_weights"], first**2, atol=1e-14)
        report = result["report"]["B0_regularization"]
        self.assertEqual(report["weighting_echo_count"], 3)
        self.assertFalse(report["weighting_uses_EPI"])
        self.assertEqual(result["report"]["retained_parent_count"], 63)

    def test_legacy_defaults_remain_point_model_and_new_options_are_explicit(self):
        point = self.prepare()
        self.assertNotIn("inplane_voxel", point["expanded_experiment"])
        self.assertEqual(point["report"]["B0_weighting"], "uniform")
        with self.assertRaisesRegex(ValueError, "b0_weighting"):
            self.prepare(b0_weighting="fit_to_EPI")
        with self.assertRaisesRegex(ValueError, "inplane_voxel"):
            self.prepare(inplane_voxel=4)
        with self.assertRaisesRegex(ValueError, "coil_images"):
            self.prepare(b0_weighting="echo_power")

    def test_zero_baseline_calibration_preserves_model_arrays_and_inputs(self):
        before=deepcopy(self.inputs)
        old=self.prepare(inplane_voxel=True,subspins_z=3)
        new=self.prepare(inplane_voxel=True,subspins_z=3,baseline_calibration={})
        for key in ('native_X_m','native_Y_m','native_B0_Hz','parent_valid_mask',
                    'fieldbasis','xyz_m','native_amplitude','native_rate_sInv'):
            np.testing.assert_array_equal(old[key],new[key])
        for key in old['expanded_experiment']['inplane_voxel']:
            np.testing.assert_array_equal(old['expanded_experiment']['inplane_voxel'][key],
                                           new['expanded_experiment']['inplane_voxel'][key])
        for key,value in old['expanded_experiment']['acquisition'].items():
            np.testing.assert_array_equal(value,new['expanded_experiment']['acquisition'][key])
        self.prepare(inplane_voxel=True,baseline_calibration=dict(
            frequency_offset_Hz=8.,readout_shift_mm=.7,odd_even_phase_rad=.2))
        for section in ('b0_raw','field','acquisition'):
            for key,value in before[section].items():
                if isinstance(value,np.ndarray):
                    np.testing.assert_array_equal(value,self.inputs[section][key])

    def test_baseline_frequency_offset_preserves_raw_B0_and_gradient_fit(self):
        original=self.prepare(inplane_voxel=True,subspins_z=2)
        changed=self.prepare(inplane_voxel=True,subspins_z=2,
            baseline_calibration={'frequency_offset_Hz':9.5})
        np.testing.assert_allclose(changed['native_B0_Hz'],original['native_B0_Hz']+9.5,atol=1e-14)
        np.testing.assert_array_equal(changed['native_B0_gradient_Hz_per_pixel'],
                                      original['native_B0_gradient_Hz_per_pixel'])
        np.testing.assert_array_equal(changed['native_measured_B0_Hz'],self.inputs['field']['b0Hz'])
        idx=changed['parent_flat_index']
        raw_children=np.repeat(self.inputs['field']['b0Hz'].ravel()[idx,None],2,axis=1)
        np.testing.assert_array_equal(changed['expanded_experiment']['field']['measured_b0Hz'],raw_children)
        self.assertEqual(changed['report']['baseline_calibration']['frequency_offset_Hz'],9.5)
        self.assertEqual(changed['expanded_experiment']['provenance']['baseline_calibration'],
                         changed['report']['baseline_calibration'])

    def test_baseline_readout_shift_moves_patient_coil_centers_endpoints_and_corners(self):
        options=dict(inplane_voxel=True,subspins_z=3,alignment_angle_deg=31.,
                     alignment_mirror_x=True,frame_angle_deg=47.)
        original=self.prepare(**options)
        shifted=self.prepare(**options,baseline_calibration={'readout_shift_mm':.7})
        np.testing.assert_allclose(shifted['native_X_m'],self.inputs['b0_raw']['X_m']+.0007,atol=1e-18)
        np.testing.assert_array_equal(shifted['original_native_X_m'],self.inputs['b0_raw']['X_m'])
        np.testing.assert_array_equal(shifted['original_native_Y_m'],self.inputs['b0_raw']['Y_m'])
        np.testing.assert_array_equal(shifted['native_Y_m'],original['native_Y_m'])
        for key in ('m0_proxy',):
            np.testing.assert_array_equal(shifted['expanded_experiment'][key],original['expanded_experiment'][key])
        np.testing.assert_array_equal(shifted['weights'],original['weights'])
        self.assertEqual(shifted['report']['amplitude_area_ratio'],original['report']['amplitude_area_ratio'])
        self.assertFalse(np.allclose(shifted['fieldbasis'],original['fieldbasis']))
        xyz=shifted['xyz_m']
        expected=(3+100*xyz[...,0]+170*xyz[...,1]+200*xyz[...,2])[...,None]*np.arange(1,33)
        np.testing.assert_allclose(shifted['fieldbasis'],expected,atol=2e-12)
        voxel=shifted['expanded_experiment']['inplane_voxel']
        native_edges=np.array([[0.,.001],[.001,0.]])
        alignment=np.asarray(shifted['report']['alignment_matrix'])
        aligned=native_edges@alignment.T
        patient_edges=np.column_stack((-aligned[:,1],aligned[:,0],np.zeros(2)))
        expected_slopes=(patient_edges@[100.,170.,200.])[:,None]*np.arange(1,33)
        np.testing.assert_allclose(voxel['field_gradient_Hz_per_A_per_pixel'],
            np.broadcast_to(expected_slopes,voxel['field_gradient_Hz_per_A_per_pixel'].shape),atol=2e-12)
        offsets=np.array([[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]])
        corners=xyz[...,None,:]+offsets@patient_edges
        expected_corners=(3+100*corners[...,0]+170*corners[...,1]+200*corners[...,2])[...,None]*np.arange(1,33)
        np.testing.assert_allclose(voxel['field_at_corners_Hz_per_A'],expected_corners,atol=2e-12)

    def test_baseline_odd_even_phase_uses_chronology_and_leaves_nominal_data_untouched(self):
        times=self.inputs['acquisition']['time_s']
        times[1::2]=times[1::2,::-1]
        original_phase=np.linspace(-.2,.2,16).reshape(4,4)
        self.inputs['acquisition']['phase_offset_rad']=original_phase.copy()
        result=self.prepare(frame_angle_deg=37.,baseline_calibration={'odd_even_phase_rad':.6})
        expected=original_phase-.3*np.array([1.,-1.,1.,-1.])[:,None]
        np.testing.assert_array_equal(result['acquisition_native']['phase_offset_rad'],expected)
        np.testing.assert_array_equal(result['expanded_experiment']['acquisition']['phase_offset_rad'],expected)
        np.testing.assert_array_equal(self.inputs['acquisition']['phase_offset_rad'],original_phase)
        self.inputs['acquisition']['time_s'][0,1]=self.inputs['acquisition']['time_s'][0,0]
        with self.assertRaisesRegex(ValueError,'polarity'):
            self.prepare(baseline_calibration={'odd_even_phase_rad':.6})

    def test_invalid_baseline_calibration_is_rejected(self):
        for invalid in ([],{'unknown':1.},{'frequency_offset_Hz':np.nan},
                        {'readout_shift_mm':True},{'odd_even_phase_rad':1j},
                        {'frequency_offset_Hz':[1.]}):
            with self.subTest(invalid=invalid),self.assertRaisesRegex(ValueError,'baseline_calibration'):
                self.prepare(baseline_calibration=invalid)

    def test_native_zero_B0_target_and_current_gradient_include_calibrated_phases(self):
        from experiment_voxel import FiniteVoxelEchoACDCModel
        times=self.inputs['acquisition']['time_s']
        times[1::2]=times[1::2,::-1]
        starts=times.min(axis=1)-.00005
        self.inputs['acquisition'].update(line_start_s=starts,
            line_end_s=np.r_[starts[1:],times[-1].max()+.00005],
            line_index=np.repeat(np.arange(1,5)[:,None],4,axis=1),
            rf_time_s=np.zeros((4,4)),phase_offset_rad=np.zeros((4,4)),sequence_kind='epi')
        params=dict(frequency_offset_Hz=8.,readout_shift_mm=.4,odd_even_phase_rad=.3)
        options=dict(inplane_voxel=True,subspins_z=2,rate_mode='none')
        def make(parameters):
            result=self.prepare(**options,baseline_calibration=parameters)
            return FiniteVoxelEchoACDCModel(result['expanded_experiment'],result['expanded_calibration'],
                profile='cpu',precision='float64',sample_chunk=7)
        nominal=make({})
        frequency=make({'frequency_offset_Hz':8.})
        np.testing.assert_allclose(frequency.ideal_kdata,nominal.ideal_kdata,rtol=1e-12,atol=1e-12)
        self.assertGreater(np.linalg.norm(frequency.forward(np.zeros((32,4)))[1]
                                         -nominal.forward(np.zeros((32,4)))[1]),1e-3)
        phase=make({'odd_even_phase_rad':.3})
        multiplier=np.exp(-.15j*np.array([1.,-1.,1.,-1.]))[:,None]
        np.testing.assert_allclose(phase.ideal_kdata,nominal.ideal_kdata*multiplier,rtol=5e-6,atol=3e-6)
        calibrated=make(params)
        rng=np.random.default_rng(901)
        currents=rng.normal(size=(32,4))*.015
        direction=rng.normal(size=currents.shape);direction/=np.linalg.norm(direction)
        _,gradient=calibrated.objective_and_grad(currents)
        epsilon=1e-5
        difference=(calibrated.objective_and_grad(currents+epsilon*direction)[0]
                    -calibrated.objective_and_grad(currents-epsilon*direction)[0])/(2*epsilon)
        np.testing.assert_allclose(np.sum(gradient*direction),difference,rtol=4e-6,atol=3e-9)


if __name__ == "__main__":
    unittest.main()
