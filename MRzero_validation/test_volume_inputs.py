"""Independent physical-coordinate and full-volume ROI checks."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from volume_inputs import (patient_grid, sample_record, build_volume_bundle,
                           slice_roi, prepare_volume_shim, with_native_pixel_signs)


def record(center=(0., 0., 0.), basis=None):
    axes = tuple((np.arange(6)-3)*.002 for _ in range(3))
    result = dict(axes_m=axes, center_m=np.asarray(center),
                  basis_patient=np.eye(3) if basis is None else np.asarray(basis),
                  metadata=dict(FOV_m=[.012]*3, TE_s=[.005]))
    xyz = patient_grid(result)
    result['magnitude'] = 5. + xyz @ [10., 20., 30.]
    result['b0Hz'] = 17. + xyz @ [80., -60., 120.]
    result['mask'] = np.ones((6, 6, 6), bool)
    return result


class VolumeInputTests(unittest.TestCase):
    def test_declared_pixel_axes_change_affine_without_flipping_or_mutating_data(self):
        original = record(center=[.01, -.02, -.03])
        changed = with_native_pixel_signs(original, [-1, -1, -1], reason='Declared Siemens Cartesian convention.')
        np.testing.assert_allclose(patient_grid(changed), 2*original['center_m']-patient_grid(original))
        np.testing.assert_array_equal(original['basis_patient'], np.eye(3))
        self.assertIs(changed['magnitude'], original['magnitude'])
        self.assertNotIn('native_pixel_signs', original['metadata'])
        for index in ([0, 1, 2], [3, 3, 3]):
            expected = patient_grid(changed)[tuple(index)]
            actual = changed['voxel_to_patient_m'] @ np.r_[index, 1.]
            np.testing.assert_allclose(actual[:3], expected, atol=1e-17)

    def test_identity_resampling_preserves_boundary_centers_and_affine_fields(self):
        source = record(basis=Rotation.from_euler('xyz', [13., 29., -17.], degrees=True).as_matrix())
        points = patient_grid(source)
        values, valid = sample_record(source, source['b0Hz'], points)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(values, source['b0Hz'], rtol=0, atol=1e-12)

    def test_displaced_volume_is_sampled_by_patient_positions_not_same_indices(self):
        source = record(center=[.001, -.001, .001])
        target = record()
        points = patient_grid(target)
        values, valid = sample_record(source, source['b0Hz'], points)
        self.assertTrue(valid.any())
        self.assertFalse(valid.all())
        np.testing.assert_allclose(values[valid], (17. + points @ [80., -60., 120.])[valid])
        self.assertTrue(np.isnan(values[~valid]).all())

    def test_patient_view_has_exact_native_shape_and_roi_keeps_full_object(self):
        measured = record(basis=np.diag([-1., 1., 1.]))
        bundle, native = build_volume_bundle(measured, measured, measured)
        self.assertEqual(bundle['reference_magnitude'].shape, (6, 6, 6))
        selection = dict(axis='x', index=3, coordinate_m=0., generation=0)
        roi = slice_roi(native, selection, thickness_m=.002)
        self.assertEqual(int(roi.sum()), 36)
        self.assertEqual(native['reference_magnitude'].shape, (6, 6, 6))
        self.assertEqual(native['amplitude_reference_time_s'], .005)

    def test_fixed_3d_coil_queries_and_measured_normal_b0_are_preserved(self):
        measured = record()
        _, native = build_volume_bundle(measured, measured, measured)
        axis = np.linspace(-.1, .1, 9)
        xx, yy, zz = np.meshgrid(axis, axis, axis, indexing='ij')
        maps = (10. + 40.*xx + 60.*yy + 90.*zz)[..., None] * np.arange(1, 33)
        settings = dict(units='Hz/A', patient_to_source_m=np.eye(4))
        reference = dict(slice_position_mm=[0., 0., 0.], slice_quaternion_wxyz=[1., 0., 0., 0.])
        selection = dict(axis='z', index=3, coordinate_m=0., generation=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'maps.npz'
            np.savez(path, xx=xx, yy=yy, zz=zz, mask=np.ones(xx.shape, bool), fieldmap=maps)
            volume = prepare_volume_shim(native, selection, path, settings,
                alignment_reference_pose=reference, alignment_angle_deg=90.,
                roi_thickness_m=.002)
        points = native['patient_xyz_m']
        expected = (10. - 40.*points[..., 1] + 60.*points[..., 0] + 90.*points[..., 2])[..., None] * np.arange(1, 33)
        np.testing.assert_allclose(volume['fieldmaps_Hz_per_A'], expected, rtol=1e-6)
        np.testing.assert_allclose(volume['b0_Hz'], measured['b0Hz'], rtol=1e-6)
        self.assertGreater(float(np.ptp(volume['b0_Hz'][3, 3, :])), 0.)
        self.assertEqual(volume['amplitude'].shape, (6, 6, 6))
        self.assertEqual(volume['roi_mask'].sum(), 36)
        self.assertGreater(volume['support'].sum(), volume['roi_mask'].sum())

    def test_missing_b0_is_not_extrapolated(self):
        measured, field = record(), record()
        field['mask'][2:4, 2:4, 2:4] = False
        bundle, native = build_volume_bundle(measured, measured, field)
        self.assertFalse(native['b0_valid'][3, 3, 3])
        self.assertTrue(np.isnan(native['b0_Hz'][3, 3, 3]))
        self.assertFalse(bundle['b0_valid'].all())


if __name__ == '__main__':
    unittest.main()
