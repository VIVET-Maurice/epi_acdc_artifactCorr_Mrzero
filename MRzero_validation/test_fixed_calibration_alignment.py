"""A fixed scanner-to-coil registration must not move with the observed slice.

Analytic 3D responses provide an independent oracle for centers, thickness
points, finite-voxel edges and corners. No measured acquisition or GPU is needed.
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from echo_shim_inputs import prepare_echo_shim_inputs
from subspin_sampling import _load_source


def _quaternion(rotation):
    """Scipy's xyzw -> the public Siemens wxyz metadata convention."""
    return rotation.as_quat()[[3, 0, 1, 2]].tolist()


def _inplane(angle, mirror_x=False, mirror_y=False):
    # Deliberately independent of production inplane_matrix/slice_basis helpers.
    angle = np.deg2rad(angle)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return rotation @ np.diag([-1. if mirror_x else 1., -1. if mirror_y else 1.])


def _fields(query):
    """Thirty-two different multilinear maps, exactly trilinearly interpolated."""
    x, y, z = np.moveaxis(np.asarray(query), -1, 0)
    channel = np.arange(1., 33.)
    return (channel + x[..., None] * (7. * channel + 11.)
            + y[..., None] * (5. * channel ** 2 - 13.)
            + z[..., None] * (9. * channel + 17.)
            + (x * y)[..., None] * (19. * channel - 2.)
            + (y * z)[..., None] * (23. * channel + 3.)
            + (x * z)[..., None] * (29. - channel)
            + (x * y * z)[..., None] * (31. * channel + 5.))


class FixedCalibrationAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        local = (np.arange(8) - 4) * .001
        self.local_X, self.local_Y = np.meshgrid(local, local)
        kx, ky = np.meshgrid((np.arange(4) - 2) / .008,
                             (np.arange(4) - 2) / .008)
        support = np.ones((8, 8), dtype=bool)
        support[0, 0] = False
        self.inputs = {
            "acquisition": {"kx_mInv": kx, "ky_mInv": ky,
                "FOV_m": np.array([.008, .008]),
                "time_s": np.arange(16).reshape(4, 4) * 1e-4 + .01,
                "TE_s": .02, "TR_s": .1},
            "b0_raw": {"metadata": {}}, "scan_metadata": {},
            "field": {"b0Hz": 17. + 1000. * self.local_X - 700. * self.local_Y,
                      "mask": np.ones((8, 8), dtype=bool)},
            "source_sha256": {"scan": "synthetic-fixed-pose-test"}}
        first = 2. + 50. * self.local_X + 30. * self.local_Y
        self.fit = {"first_echo_amplitude": first,
                    "object_support": support.copy(),
                    "simulation_support": support.copy(),
                    "amplitude_reference_time_s": .002,
                    "pooled": {"R2star_sInv": 7., "target_reliable_mask": True},
                    "simulation_amplitude_at_reference": first * 1.02,
                    "simulation_R2star_sInv": 10. + 100. * self.local_X}
        self._set_pose([0., 0., 0.], Rotation.from_euler("z", 90., degrees=True))
        axis = np.linspace(-.15, .15, 11)
        xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
        self.path = Path(self.directory.name) / "analytic_3d_calibration.npz"
        np.savez(self.path, xx=xx, yy=yy, zz=zz,
                 mask=np.ones(xx.shape, dtype=bool),
                 fieldmap=_fields(np.stack((xx, yy, zz), axis=-1)))
        self.settings = {"units": "Hz/A", "registration_confirmed": True,
                         "patient_to_source_m": np.eye(4)}
        self.reference_rotation = Rotation.from_euler("xyz", [23., -31., 72.], degrees=True)
        self.reference = {
            "slice_position_mm": [11., -13., 17.],
            "slice_quaternion_wxyz": _quaternion(self.reference_rotation)}

    def tearDown(self):
        _load_source.cache_clear()
        self.directory.cleanup()

    def _set_pose(self, center_mm, rotation):
        center = np.asarray(center_mm, dtype=float) * .001
        basis = rotation.as_matrix()
        metadata = {"slice_position_mm": np.asarray(center_mm, float).tolist(),
                    "slice_thickness_mm": 2.,
                    "slice_quaternion_wxyz": _quaternion(rotation)}
        self.inputs["scan_metadata"] = deepcopy(metadata)
        self.inputs["b0_raw"]["metadata"] = deepcopy(metadata)
        origin = center @ basis[:, :2]
        self.inputs["b0_raw"]["X_m"] = self.local_X + origin[0]
        self.inputs["b0_raw"]["Y_m"] = self.local_Y + origin[1]
        return basis

    def _prepare(self, **options):
        return prepare_echo_shim_inputs(self.inputs, self.fit, self.path,
                                         self.settings, **options)

    def _fixed_affine(self, angle, mirror_x=False, mirror_y=False, reference=None):
        reference = self.reference if reference is None else reference
        quat = np.asarray(reference["slice_quaternion_wxyz"])
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        local = np.eye(3)
        local[:2, :2] = _inplane(angle, mirror_x, mirror_y)
        matrix = rotation @ local @ rotation.T
        center = np.asarray(reference["slice_position_mm"]) * .001
        return matrix, center - matrix @ center

    def _physical_points(self, result, count):
        metadata = self.inputs["scan_metadata"]
        quaternion = np.asarray(metadata["slice_quaternion_wxyz"])
        basis = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        center = np.asarray(metadata["slice_position_mm"]) * .001
        indices = result["parent_flat_index"]
        points = np.empty((len(indices), count, 3))
        points[..., 0] = self.local_X.ravel()[indices, None]
        points[..., 1] = self.local_Y.ravel()[indices, None]
        points[..., 2] = ((np.arange(count) + .5) / count - .5) * .002
        return points @ basis.T + center, basis

    def _source_fields(self, aligned_points):
        affine = self.settings["patient_to_source_m"]
        return _fields(aligned_points @ affine[:3, :3].T + affine[:3, 3])

    def test_centers_and_thickness_spins_follow_one_independent_3d_affine(self):
        affine = np.eye(4)
        affine[:3, :3] = (Rotation.from_euler("xyz", [4., -7., 11.], degrees=True).as_matrix()
                          @ np.diag([1.01, .98, 1.02]))
        affine[:3, 3] = [.003, -.004, .005]
        self.settings["patient_to_source_m"] = affine
        for z_mm in (-50., 50.):
            for euler in ([0., 0., 90.], [63., -42., 129.], [90., 0., 0.]):
                for mirror_x, mirror_y in ((False, False), (True, False), (True, True)):
                    with self.subTest(z_mm=z_mm, euler=euler,
                                      mirror_x=mirror_x, mirror_y=mirror_y):
                        self._set_pose([18., -21., z_mm],
                                       Rotation.from_euler("xyz", euler, degrees=True))
                        result = self._prepare(subspins_z=5, alignment_angle_deg=37.,
                            alignment_mirror_x=mirror_x, alignment_mirror_y=mirror_y,
                            alignment_reference_pose=self.reference)
                        points, basis = self._physical_points(result, 5)
                        rotation, translation = self._fixed_affine(37., mirror_x, mirror_y)
                        expected = points @ rotation.T + translation
                        np.testing.assert_allclose(result["xyz_m"], expected,
                                                   rtol=0, atol=7e-17)
                        np.testing.assert_allclose(result["fieldbasis"],
                            self._source_fields(expected), rtol=2e-14, atol=1e-11)
                        normal_step = .002 / 5 * (rotation @ basis[:, 2])
                        np.testing.assert_allclose(np.diff(result["xyz_m"], axis=1),
                            np.broadcast_to(normal_step, (len(points), 4, 3)),
                            rtol=0, atol=5e-17)

    def test_same_patient_point_seen_in_two_different_planes_has_identical_response(self):
        target = np.array([.015, -.022, -.05])
        rotation1 = Rotation.from_euler("xyz", [17., 31., 90.], degrees=True)
        self._set_pose(target * 1000., rotation1)
        options = dict(subspins_z=1, alignment_angle_deg=61., alignment_mirror_y=True,
                       alignment_reference_pose=self.reference)
        first = self._prepare(**options)
        first_index = np.flatnonzero(first["parent_flat_index"] == 4 * 8 + 4).item()

        rotation2 = Rotation.from_euler("xyz", [79., -21., 141.], degrees=True)
        # In the second acquisition the same point is pixel(row=3,column=6).
        local = np.array([.002, -.001, 0.])
        center2 = target - rotation2.as_matrix() @ local
        self._set_pose(center2 * 1000., rotation2)
        second = self._prepare(**options)
        second_index = np.flatnonzero(second["parent_flat_index"] == 3 * 8 + 6).item()
        expected_rotation, expected_translation = self._fixed_affine(61., False, True)
        expected = expected_rotation @ target + expected_translation
        for result, index in ((first, first_index), (second, second_index)):
            np.testing.assert_allclose(result["xyz_m"][index, 0], expected,
                                       rtol=0, atol=4e-17)
            np.testing.assert_allclose(result["fieldbasis"][index, 0],
                                       self._source_fields(expected), rtol=0, atol=1e-11)
        np.testing.assert_allclose(first["fieldbasis"][first_index],
                                   second["fieldbasis"][second_index], rtol=0, atol=1e-11)

    def test_finite_voxel_edges_endpoints_and_corners_use_the_same_fixed_registration(self):
        basis = self._set_pose([18., -21., -50.],
            Rotation.from_euler("xyz", [53., -42., 129.], degrees=True))
        affine = np.eye(4)
        affine[:3, :3] = Rotation.from_euler("xyz", [-3., 12., 9.], degrees=True).as_matrix()
        affine[:3, 3] = [-.003, .002, .004]
        self.settings["patient_to_source_m"] = affine
        result = self._prepare(subspins_z=3, inplane_voxel=True,
            alignment_angle_deg=37., alignment_mirror_x=True,
            alignment_reference_pose=self.reference)
        physical, _ = self._physical_points(result, 3)
        registration, translation = self._fixed_affine(37., True, False)
        expected_centers = physical @ registration.T + translation
        # Pixel row and column are native phase and readout directions.
        native_edges = np.array([[0., .001], [.001, 0.]])
        real_edges = native_edges @ basis[:, :2].T
        aligned_edges = real_edges @ registration.T
        endpoints = expected_centers[..., None, :] + np.stack((
            -.5 * aligned_edges[0], .5 * aligned_edges[0],
            -.5 * aligned_edges[1], .5 * aligned_edges[1]))
        values = self._source_fields(endpoints)
        expected_gradient = np.stack((values[..., 1, :] - values[..., 0, :],
                                      values[..., 3, :] - values[..., 2, :]), axis=-2)
        voxel = result["expanded_experiment"]["inplane_voxel"]
        np.testing.assert_allclose(result["fieldbasis"],
            self._source_fields(expected_centers), rtol=2e-14, atol=1e-11)
        np.testing.assert_allclose(voxel["field_gradient_Hz_per_A_per_pixel"],
                                   expected_gradient, rtol=0, atol=1e-11)
        corners = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
        expected_corners = expected_centers[..., None, :] + corners @ aligned_edges
        np.testing.assert_allclose(voxel["field_at_corners_Hz_per_A"],
            self._source_fields(expected_corners), rtol=2e-14, atol=1e-11)

    def test_reference_equal_to_current_pose_recovers_legacy_queries_and_fourier_grid(self):
        self._set_pose([12., -9., -50.],
                       Rotation.from_euler("xyz", [23., -31., 72.], degrees=True))
        reference = {key: deepcopy(self.inputs["scan_metadata"][key])
                     for key in ("slice_position_mm", "slice_quaternion_wxyz")}
        options = dict(subspins_z=3, inplane_voxel=True, alignment_angle_deg=37.,
                       alignment_mirror_y=True, frame_angle_deg=73., frame_mirror_x=True)
        legacy = self._prepare(**options)
        explicit_none = self._prepare(**options, alignment_reference_pose=None)
        fixed = self._prepare(**options, alignment_reference_pose=reference)
        for key in ("xyz_m", "fieldbasis", "parent_flat_index"):
            np.testing.assert_array_equal(legacy[key], explicit_none[key])
            np.testing.assert_allclose(fixed[key], legacy[key], rtol=2e-14, atol=1e-11)
        for key in ("frame_X_m", "frame_Y_m", "weights", "native_B0_Hz"):
            np.testing.assert_array_equal(fixed[key], legacy[key])
        for key in ("kx_mInv", "ky_mInv", "time_s"):
            np.testing.assert_array_equal(
                fixed["expanded_experiment"]["acquisition"][key],
                legacy["expanded_experiment"]["acquisition"][key])
        for key in ("m0_proxy",):
            np.testing.assert_array_equal(fixed["expanded_experiment"][key],
                                          legacy["expanded_experiment"][key])
        first_voxel = fixed["expanded_experiment"]["inplane_voxel"]
        old_voxel = legacy["expanded_experiment"]["inplane_voxel"]
        for key in ("field_gradient_Hz_per_A_per_pixel", "field_at_corners_Hz_per_A"):
            np.testing.assert_allclose(first_voxel[key], old_voxel[key],
                                       rtol=0, atol=1e-11)

    def test_passive_frame_never_changes_coil_queries_and_encoding_stays_p_times_q(self):
        self._set_pose([18., -21., 50.],
                       Rotation.from_euler("xyz", [53., -42., 129.], degrees=True))
        options = dict(subspins_z=3, inplane_voxel=True,
                       alignment_angle_deg=37., alignment_mirror_y=True,
                       alignment_reference_pose=self.reference)
        first = self._prepare(**options, frame_angle_deg=0.)
        changed = self._prepare(**options, frame_angle_deg=123., frame_mirror_x=True)
        for key in ("xyz_m", "fieldbasis", "parent_flat_index"):
            np.testing.assert_array_equal(changed[key], first[key])
        for key in ("field_gradient_Hz_per_A_per_pixel", "field_at_corners_Hz_per_A"):
            np.testing.assert_array_equal(
                changed["expanded_experiment"]["inplane_voxel"][key],
                first["expanded_experiment"]["inplane_voxel"][key])
        combined = _inplane(123., True, False) @ _inplane(37., False, True)
        native_k = np.stack((self.inputs["acquisition"]["kx_mInv"],
                             self.inputs["acquisition"]["ky_mInv"]), axis=-1)
        expected_k = native_k @ combined.T
        acquisition = changed["expanded_experiment"]["acquisition"]
        np.testing.assert_allclose(acquisition["kx_mInv"], expected_k[..., 0],
                                   rtol=2e-15, atol=1e-12)
        np.testing.assert_allclose(acquisition["ky_mInv"], expected_k[..., 1],
                                   rtol=2e-15, atol=1e-12)
        local = np.stack((self.local_X, self.local_Y), axis=-1) @ combined.T
        np.testing.assert_allclose(changed["frame_X_m"], local[..., 0], rtol=0, atol=2e-17)
        np.testing.assert_allclose(changed["frame_Y_m"], local[..., 1], rtol=0, atol=2e-17)

    def test_opposite_reference_quaternion_sign_defines_the_same_fixed_registration(self):
        self._set_pose([18., -21., -50.],
                       Rotation.from_euler("xyz", [53., -42., 129.], degrees=True))
        reference = deepcopy(self.reference)
        reference["slice_quaternion_wxyz"] = (
            -np.asarray(reference["slice_quaternion_wxyz"])).tolist()
        options = dict(subspins_z=3, inplane_voxel=True, alignment_angle_deg=37.)
        positive = self._prepare(**options, alignment_reference_pose=self.reference)
        negative = self._prepare(**options, alignment_reference_pose=reference)
        for key in ("xyz_m", "fieldbasis", "parent_flat_index"):
            np.testing.assert_array_equal(positive[key], negative[key])

    def test_reference_pose_requires_explicit_finite_center_and_unit_quaternion(self):
        bad_references = [[], "axial", 17, {},
            {"slice_position_mm": [0., 0., 0.]},
            {"slice_quaternion_wxyz": [1., 0., 0., 0.]}]
        for center in ([0., 0.], [0., 0., np.nan], [0., 0., np.inf], [0., 1j, 0.]):
            bad_references.append({**self.reference, "slice_position_mm": center})
        for quaternion in ([1., 0., 0.], [0., 0., 0., 0.], [1., 0., 0., 1.],
                           [np.nan, 0., 0., 1.], [1j, 0., 0., 1.]):
            bad_references.append({**self.reference, "slice_quaternion_wxyz": quaternion})
        for reference in bad_references:
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                self._prepare(alignment_reference_pose=reference)


if __name__ == "__main__":
    unittest.main()
