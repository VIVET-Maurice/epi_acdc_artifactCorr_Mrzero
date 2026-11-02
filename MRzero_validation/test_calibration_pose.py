"""Independent conditional geometry checks for the 3D calibration sampler.

The expected slice axes are declared as readout/phase vectors before encoding
them as a quaternion. Expected fields are evaluated directly from a synthetic
3D affine field, without the production quaternion or interpolation helpers.
These checks validate the geometry for a *given* patient-to-source affine;
they cannot establish the real scanner/coil registration or its z origin.
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from echo_shim_inputs import prepare_echo_shim_inputs
from subspin_sampling import _load_source


def declared_basis(readout, phase):
    """Construct an orthonormal physical basis independently of its quaternion."""
    readout = np.asarray(readout, dtype=float)
    readout = readout / np.linalg.norm(readout)
    phase = np.asarray(phase, dtype=float)
    phase = phase - np.dot(phase, readout) * readout
    phase = phase / np.linalg.norm(phase)
    return np.column_stack((readout, phase, np.cross(readout, phase)))


def planar_transform(angle_deg, signs):
    angle = np.deg2rad(angle_deg)
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]]) @ np.diag(signs)


class CalibrationPoseTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / 'affine_fields.npz'
        channel = np.arange(1, 33, dtype=float)
        self.intercept = 10. + .31 * channel
        # Independent, nonproportional channel slopes distinguish every axis.
        self.slopes = np.column_stack((90. + 3. * channel,
                                       -160. + channel**2 / 4.,
                                       270. - 5. * channel))
        axis = np.linspace(-.12, .12, 17)
        xx, yy, zz = np.meshgrid(axis, axis, axis, indexing='ij')
        source_xyz = np.stack((xx, yy, zz), axis=-1)
        np.savez(self.path, xx=xx, yy=yy, zz=zz,
                 mask=np.ones(xx.shape, dtype=bool),
                 fieldmap=self.intercept + source_xyz @ self.slopes.T)
        self.affine = np.eye(4)
        self.affine[:3, :3] = declared_basis([1., .3, -.2], [-.1, 1., .4]) @ np.diag([.97, 1.02, .94])
        self.affine[:3, 3] = [.004, -.006, .009]
        self.settings = dict(units='Hz/A', registration_confirmed=True,
                             patient_to_source_m=self.affine)
        self.alignment = planar_transform(37., [-1., 1.])
        self.passive = planar_transform(-23., [1., -1.])
        self.options = dict(alignment_angle_deg=37., alignment_mirror_x=True,
                            frame_angle_deg=-23., frame_mirror_y=True,
                            subspins_z=5, inplane_voxel=True, b0_mode='raw')
        x = (np.arange(9) - 4) * .002
        y = (np.arange(8) - 4) * .003
        self.local_x, self.local_y = np.meshgrid(x, y)
        self.bases = {
            'axial': declared_basis([0., 1., 0.], [-1., 0., 0.]),
            'sagittal': declared_basis([0., 1., 0.], [0., 0., 1.]),
            'coronal': declared_basis([1., 0., 0.], [0., 0., 1.]),
            'oblique': declared_basis([1., 2., 3.], [-2., 1., .4]),
            'opposite_normal': declared_basis([1., 0., 0.], [0., -1., 0.]),
        }

    def tearDown(self):
        _load_source.cache_clear()
        self.folder.cleanup()

    def fields_at(self, patient_xyz):
        source_xyz = patient_xyz @ self.affine[:3, :3].T + self.affine[:3, 3]
        return self.intercept + source_xyz @ self.slopes.T

    def prepare(self, basis, center_mm):
        center = np.asarray(center_mm, dtype=float) * 1e-3
        quaternion_xyzw = Rotation.from_matrix(basis).as_quat()
        metadata = dict(slice_position_mm=np.asarray(center_mm).tolist(),
                        slice_thickness_mm=5.,
                        slice_quaternion_wxyz=quaternion_xyzw[[3, 0, 1, 2]].tolist())
        origin = center @ basis[:, :2]
        X, Y = self.local_x + origin[0], self.local_y + origin[1]
        support = np.ones(X.shape, dtype=bool)
        kx, ky = np.meshgrid((np.arange(6)-3)/.024, (np.arange(4)-2)/.024)
        inputs = dict(
            scan_metadata=metadata,
            b0_raw=dict(X_m=X, Y_m=Y, metadata=deepcopy(metadata)),
            field=dict(b0Hz=17. + 900.*self.local_x - 600.*self.local_y,
                       mask=support),
            acquisition=dict(kx_mInv=kx, ky_mInv=ky, FOV_m=np.array([.024, .024]),
                             time_s=.01 + np.arange(kx.size).reshape(kx.shape)*1e-4,
                             TE_s=.02, TR_s=.1))
        fit = dict(first_echo_amplitude=np.ones(X.shape), object_support=support,
                   simulation_support=support, amplitude_reference_time_s=.002,
                   pooled=dict(R2star_sInv=7., target_reliable_mask=True))
        return prepare_echo_shim_inputs(inputs, fit, self.path, self.settings, **self.options)

    def expected_geometry(self, result, basis, center_mm):
        parents = result['parent_flat_index']
        local = np.stack((self.local_x, self.local_y), axis=-1).reshape(-1, 2)[parents]
        # Use the declared physical vectors, rather than the production basis.
        aligned = local @ self.alignment.T
        center_xyz = (np.asarray(center_mm)*1e-3 + aligned[:, 0, None]*basis[:, 0]
                      + aligned[:, 1, None]*basis[:, 1])
        dz = ((np.arange(5)+.5)/5.-.5)*.005
        xyz = center_xyz[:, None, :] + dz[None, :, None]*basis[:, 2]
        native_edges = np.array([[0., .003], [.002, 0.]])
        aligned_edges = native_edges @ self.alignment.T
        patient_edges = (aligned_edges[:, 0, None]*basis[:, 0]
                         + aligned_edges[:, 1, None]*basis[:, 1])
        return xyz, patient_edges, native_edges

    def test_declared_planes_and_signed_50mm_offsets_query_centers_endpoints_corners(self):
        for name, basis in self.bases.items():
            for z_mm in (-50., 0., 50.):
                center_mm = [11., -7., z_mm]
                with self.subTest(plane=name, center_mm=center_mm):
                    result = self.prepare(basis, center_mm)
                    self.assertEqual(len(result['parent_flat_index']), self.local_x.size)
                    xyz, patient_edges, native_edges = self.expected_geometry(result, basis, center_mm)
                    np.testing.assert_allclose(result['xyz_m'], xyz, rtol=0., atol=6e-17)
                    source_xyz = xyz @ self.affine[:3, :3].T + self.affine[:3, 3]
                    np.testing.assert_allclose(result['xyz_source_m'], source_xyz, rtol=0., atol=6e-17)
                    audit = result['report']['calibration_pose']
                    transform = np.asarray(audit['native_local_to_source_m'])
                    parents = result['parent_flat_index']
                    native_xy = np.stack((self.local_x, self.local_y), axis=-1).reshape(-1, 2)[parents]
                    native = np.zeros_like(xyz)
                    native[..., :2] = native_xy[:, None, :]
                    native[..., 2] = ((np.arange(5)+.5)/5.-.5)*.005
                    np.testing.assert_allclose(native @ transform[:3, :3].T + transform[:3, 3],
                                               source_xyz, rtol=0., atol=6e-17)
                    np.testing.assert_allclose(result['fieldbasis'], self.fields_at(xyz),
                                               rtol=0., atol=1e-12)
                    endpoint_offsets = np.stack((-.5*patient_edges[0], .5*patient_edges[0],
                                                 -.5*patient_edges[1], .5*patient_edges[1]))
                    endpoint_fields = self.fields_at(xyz[..., None, :]+endpoint_offsets)
                    gradients = np.stack((endpoint_fields[..., 1, :]-endpoint_fields[..., 0, :],
                                          endpoint_fields[..., 3, :]-endpoint_fields[..., 2, :]), axis=-2)
                    voxel = result['expanded_experiment']['inplane_voxel']
                    np.testing.assert_allclose(voxel['field_gradient_Hz_per_A_per_pixel'],
                                               gradients, rtol=0., atol=1e-12)
                    corners = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
                    corner_fields = self.fields_at(xyz[..., None, :]+corners @ patient_edges)
                    np.testing.assert_allclose(voxel['field_at_corners_Hz_per_A'],
                                               corner_fields, rtol=0., atol=1e-12)
                    np.testing.assert_allclose(voxel['edge_vectors_m'],
                        native_edges @ (self.passive @ self.alignment).T, rtol=0., atol=1e-17)

    def test_translation_changes_generated_fields_without_rotating_the_slice_center(self):
        displacement_mm = np.array([13., -19., 50.])
        displacement = displacement_mm*1e-3
        expected_channel_change = self.slopes @ self.affine[:3, :3] @ displacement
        currents = np.linspace(-.03, .06, 32)
        self.assertGreater(abs(expected_channel_change @ currents), 1.)
        for name, basis in self.bases.items():
            with self.subTest(plane=name):
                original = self.prepare(basis, [0., 0., 0.])
                translated = self.prepare(basis, displacement_mm)
                expected_xyz = original['xyz_m'] + displacement
                np.testing.assert_allclose(translated['xyz_m'], expected_xyz, rtol=0., atol=6e-17)
                actual_change = translated['fieldbasis']-original['fieldbasis']
                np.testing.assert_allclose(actual_change,
                    np.broadcast_to(expected_channel_change, actual_change.shape), rtol=0., atol=1e-12)
                generated_change = translated['fieldbasis'] @ currents-original['fieldbasis'] @ currents
                np.testing.assert_allclose(generated_change, expected_channel_change @ currents,
                                           rtol=0., atol=1e-12)
                for key in ('X_m', 'Y_m', 'Z_m'):
                    np.testing.assert_allclose(translated['expanded_experiment']['geometry'][key],
                        original['expanded_experiment']['geometry'][key], rtol=0., atol=6e-17)


if __name__ == '__main__':
    unittest.main()
