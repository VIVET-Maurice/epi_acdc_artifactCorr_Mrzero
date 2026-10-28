"""CPU input-contract tests: frame invariance, echo anchoring and domain masks."""

from copy import deepcopy
from contextlib import ExitStack, contextmanager
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'Experiment'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twin_inputs import (build_candidates, frame_acquisition, model_arguments,
                         prepare_echo_object, read_twin_inputs)


def candidate_fixture():
    x = (np.arange(4) - 2) * .01
    X, Y = np.meshgrid(x, x)
    native_x = (np.arange(8) - 4) * .005
    native_X, native_Y = np.meshgrid(native_x, native_x)
    kx, ky = np.meshgrid((np.arange(4) - 2) / .04, (np.arange(4) - 2) / .04)
    acquisition = dict(kx_mInv=kx, ky_mInv=ky, FOV_m=np.array([.04, .04]),
                       time_s=np.arange(16).reshape(4, 4) * .001 + .02,
                       phase_offset_rad=np.arange(16).reshape(4, 4) * .03,
                       rf_time_s=np.zeros((4, 4)), TE_s=.1)
    gre = dict(magnitude=1. + (X + .02) * 20 + (Y + .02) * 10,
               X_m=X, Y_m=Y, spacing_mm=np.array([10., 10.]),
               position_mm=np.array([20., -20., 0.]),
               orientation=np.array([0., 1., 0., -1., 0., 0.]),
               slice_thickness_mm=5.)
    support = np.ones(native_X.shape, bool)
    support[0, 0] = False
    inputs = dict(gre=gre, acquisition=acquisition,
                  field=dict(b0Hz=11 + 200 * native_X - 90 * native_Y,
                             mask=np.ones(native_X.shape, bool)),
                  b0_raw=dict(X_m=native_X, Y_m=native_Y),
                  optimisation_ROI=np.zeros(native_X.shape, bool),
                  calibration_valid_mask=np.zeros(native_X.shape, bool))
    first = 2 + native_X * 10 + native_Y * 5
    rate = 9 + np.arange(64).reshape(8, 8) * .01
    fitted_anchor = first * 1.03
    fit = dict(simulation_support=support, amplitude_reference_time_s=.002,
               first_echo_amplitude=first, pooled=dict(R2star_sInv=7., target_reliable_mask=True),
               simulation_amplitude_at_reference=fitted_anchor, simulation_R2star_sInv=rate)
    return inputs, fit


class TwinInputTests(unittest.TestCase):
    @contextmanager
    def measured_input_fixture(self, folder, reference_size=64, extra_files=()):
        folder = Path(folder)
        (folder / 'sequences').mkdir()
        for name in ('reference.dat', 'scan.dat', 'b0_map.dat', 'sequences/epi.seq',
                     f'sequences/gre_{reference_size}.seq', 'sequences/b0_map.seq', *extra_files):
            (folder / name).write_bytes(b'test input ' + name.encode())
        coordinate = (np.arange(reference_size) - reference_size // 2) * .2 / reference_size
        X, Y = np.meshgrid(coordinate, coordinate)
        gre = dict(magnitude=np.ones(X.shape), X_m=X, Y_m=Y)
        times = .002 + np.arange(32) * .0018
        raw = {'coil_images': np.ones((32, 2, 4, 4), complex),
               'metadata': {'TE_s': times.tolist()}}
        def sequence(path):
            if Path(path).name in ('b0_map.seq', 'field32.seq'):
                raise ValueError('Multi-echo GRE is not a mono-echo reference.')
            size = reference_size if 'epi' not in Path(path).name else 128
            return ({'TE_s': .078 if size == 128 and 'epi' in Path(path).name else .004,
                     'FOV_m': np.array([.2, .2]), 'kx_mInv': np.zeros((size, size)),
                     'sequence_kind': 'epi' if 'epi' in Path(path).name else 'gre'}, {})
        def reconstruct(diagnostics, acquisition, **kwargs):
            return np.ones(acquisition['kx_mInv'].shape), {}
        with ExitStack() as stack:
            mocks = {}
            for name, kwargs in (
                    ('read_b0_coil_images', dict(return_value=raw)),
                    ('read_sequence', dict(side_effect=sequence)),
                    ('read_gre', dict(return_value=(gre, {'native_reference': True}))),
                    ('read_scan', dict(return_value=(np.ones((128, 128)), {}, {}))),
                    ('reconstruct_receivers', dict(side_effect=reconstruct)),
                    ('reconstruct_b0_echoes', {}),
                    ('_estimate_field', dict(return_value=({}, 6)))):
                mocks[name] = stack.enter_context(patch('twin_inputs.' + name, **kwargs))
            yield mocks, raw, gre

    def test_joint_frame_preserves_encoding_and_leaves_input_unmodified(self):
        inputs, _ = candidate_fixture()
        acquisition = inputs['acquisition']
        before = deepcopy(acquisition)
        reversed_acquisition = frame_acquisition(acquisition, 180)
        rng = np.random.default_rng(3)
        X, Y = rng.normal(size=5), rng.normal(size=5)
        original = (acquisition['kx_mInv'][..., None] * X
                    + acquisition['ky_mInv'][..., None] * Y)
        changed = (reversed_acquisition['kx_mInv'][..., None] * -X
                   + reversed_acquisition['ky_mInv'][..., None] * -Y)
        np.testing.assert_array_equal(original, changed)
        for key, value in before.items():
            np.testing.assert_array_equal(acquisition[key], value)
            if key not in ('kx_mInv', 'ky_mInv'):
                np.testing.assert_array_equal(reversed_acquisition[key], value)
        reversed_acquisition['time_s'][0, 0] = 999
        self.assertNotEqual(acquisition['time_s'][0, 0], 999)

    def test_native_candidates_keep_native_fields_and_support_independent_of_ROI_and_coils(self):
        inputs, fit = candidate_fixture()
        originals = deepcopy(inputs)
        candidates = build_candidates(inputs, fit, 180)
        native = [candidate for candidate in candidates if candidate['name'].startswith('echo128')]
        self.assertEqual(len(native), 3)
        for candidate in native:
            np.testing.assert_array_equal(candidate['B0_Hz'], originals['field']['b0Hz'])
            np.testing.assert_array_equal(candidate['support_mask'], fit['simulation_support'])
            np.testing.assert_array_equal(candidate['X_m'], -originals['b0_raw']['X_m'])
            np.testing.assert_array_equal(candidate['Y_m'], -originals['b0_raw']['Y_m'])
            self.assertGreater(candidate['support_mask'].sum(), inputs['optimisation_ROI'].sum())
            self.assertGreater(candidate['support_mask'].sum(), inputs['calibration_valid_mask'].sum())
        for source in ('gre', 'field', 'b0_raw', 'acquisition'):
            for key, value in originals[source].items():
                np.testing.assert_array_equal(inputs[source][key], value)

    def test_guarded_and_pooled_candidates_use_first_echo_time_anchor(self):
        inputs, fit = candidate_fixture()
        candidates = {candidate['name']: candidate for candidate in build_candidates(inputs, fit)}
        guarded = model_arguments(candidates['echo128_guarded_decay'])
        # Native pixels are 5 mm and acquired EPI pixels are 10 mm: area=1/4.
        np.testing.assert_allclose(guarded['amplitude'], fit['simulation_amplitude_at_reference'] / 4,
                                   rtol=1e-14)
        np.testing.assert_array_equal(guarded['rate_sInv'], fit['simulation_R2star_sInv'])
        anchor = guarded['amplitude_reference_time_s']
        self.assertEqual(anchor, fit['amplitude_reference_time_s'])
        target = inputs['acquisition']['TE_s']
        expected = fit['simulation_amplitude_at_reference'] / 4 * np.exp(
            -fit['simulation_R2star_sInv'] * (target - fit['amplitude_reference_time_s']))
        actual = guarded['amplitude'] * np.exp(-guarded['rate_sInv'] * (target - anchor))
        np.testing.assert_allclose(actual, expected, rtol=1e-14)
        pooled = candidates['echo128_pooled_decay']
        np.testing.assert_array_equal(pooled['rate_sInv'], np.full((8, 8), 7.))
        self.assertEqual(pooled['amplitude_reference_time_s'], anchor)
        fit['pooled']['target_reliable_mask'] = False
        candidates = {candidate['name']: candidate for candidate in build_candidates(inputs, fit)}
        np.testing.assert_array_equal(candidates['echo128_pooled_decay']['rate_sInv'], np.zeros((8, 8)))

    def test_constant_density_zero_k_integral_is_invariant_to_object_refinement(self):
        inputs, _ = candidate_fixture()
        acquisition = inputs['acquisition']
        integrals = []
        for size in (4, 8, 16):
            coordinate = (np.arange(size) - size // 2) * .04 / size
            X, Y = np.meshgrid(coordinate, coordinate)
            for sign in (1, -1):
                original = np.ones((size, size)) * 3.
                candidate = dict(acquisition=acquisition, X_m=sign * X, Y_m=sign * Y,
                                 amplitude=original, B0_Hz=np.zeros((size, size)),
                                 support_mask=np.ones((size, size), bool), rate_sInv=0.,
                                 amplitude_reference_time_s=0.)
                arguments = model_arguments(candidate)
                # At k=0, B0=0 and no decay the Bloch sum is sum(amplitude).
                integrals.append(float(arguments['amplitude'].sum()))
                np.testing.assert_array_equal(candidate['amplitude'], original)
        np.testing.assert_allclose(integrals, 3. * acquisition['kx_mInv'].size, rtol=2e-14)

    def test_echo_support_requires_B0_but_neither_ACDC_calibration_nor_optimisation_ROI(self):
        rng = np.random.default_rng(5)
        x = (np.arange(32) - 16) * .00625
        X, Y = np.meshgrid(x, x)
        object_amplitude = ((X**2 + Y**2) < .045**2).astype(float)
        times = np.array([.002, .004, .006])
        weights = np.array([1., .6j]) / np.sqrt(1 + .6**2)
        coils = (np.exp(-times[:, None, None, None] * 12)
                 * weights[None, :, None, None] * object_amplitude[None, None])
        coils = coils + .001 * (rng.normal(size=coils.shape) + 1j * rng.normal(size=coils.shape))
        measured_valid = np.ones(X.shape, bool)
        measured_valid[16, 16] = False
        inputs = dict(b0_raw=dict(coil_images=coils, X_m=X, Y_m=Y),
                      field=dict(mask=measured_valid, b0Hz=np.zeros(X.shape)),
                      acquisition=dict(TE_s=.1), echo_times_s=times,
                      optimisation_ROI=np.zeros(X.shape, bool),
                      calibration_valid_mask=np.zeros(X.shape, bool))
        fit = prepare_echo_object(inputs)
        self.assertTrue(fit['object_support'][16, 16])
        self.assertFalse(fit['simulation_support'][16, 16])
        self.assertEqual(fit['report']['excluded_missing_B0_voxels'], 1)
        self.assertGreater(fit['simulation_support'].sum(), 100)
        self.assertFalse(fit['report']['ACDC_calibration_used_for_baseline_support'])
        self.assertFalse(fit['report']['optimisation_ROI_used_for_baseline_support'])


    def noise_fixture(self, noise_std=.001):
        x = (np.arange(32) - 16) * .00625
        X, Y = np.meshgrid(x, x)
        amplitude = (X**2 + Y**2 < .045**2).astype(float)
        times = np.array([.002, .004, .006])
        weights = np.array([1., .6j]) / np.sqrt(1 + .6**2)
        coils = np.exp(-times[:, None, None, None] * 12) * weights[None, :, None, None] * amplitude[None, None]
        rng = np.random.default_rng(63)
        coils += noise_std * (rng.normal(size=coils.shape) + 1j * rng.normal(size=coils.shape))
        return dict(b0_raw=dict(coil_images=coils, X_m=X, Y_m=Y),
                    field=dict(mask=np.ones(X.shape, bool), b0Hz=np.zeros(X.shape)),
                    acquisition=dict(TE_s=.1), echo_times_s=times)

    def test_absent_corners_continue_without_fabricating_noise_or_decay(self):
        inputs = self.noise_fixture()
        fit = prepare_echo_object(inputs, background_corner_m=.2)
        noise = fit['report']['noise']
        self.assertFalse(noise['calibrated'])
        self.assertEqual(noise['available_sample_count'], 0)
        self.assertEqual(noise['minimum_sample_count'], 32)
        self.assertIsNotNone(noise['fallback_reason'])
        self.assertFalse(fit['noise_mask'].any())
        np.testing.assert_array_equal(fit['simulation_R2star_sInv'], 0)
        np.testing.assert_array_equal(fit['simulation_amplitude_at_reference'], fit['first_echo_amplitude'])
        self.assertFalse(fit['target_reliable_mask'].any())
        self.assertFalse(fit['pooled']['target_reliable_mask'])
        self.assertTrue(np.isnan(fit['R2star_std_sInv']).all())

    def test_only_finite_receiver_vectors_count_towards_noise_calibration(self):
        inputs = self.noise_fixture()
        raw = inputs['b0_raw']
        corners = (abs(raw['X_m']) > .085) & (abs(raw['Y_m']) > .085)
        raw['coil_images'][1:, :, corners] = np.nan
        fit = prepare_echo_object(inputs)
        self.assertEqual(fit['report']['noise']['available_sample_count'], int(corners.sum()))
        self.assertFalse(fit['report']['noise']['calibrated'])
        self.assertGreater(fit['object_support'].sum(), 100)
        self.assertFalse(fit['pooled']['target_reliable_mask'])

    def test_zero_noise_variance_uses_uncalibrated_fallback(self):
        fit = prepare_echo_object(self.noise_fixture(noise_std=0))
        self.assertGreaterEqual(fit['report']['noise']['available_sample_count'], 32)
        self.assertFalse(fit['report']['noise']['calibrated'])
        self.assertIn('zero', fit['report']['noise']['fallback_reason'])
        np.testing.assert_array_equal(fit['simulation_R2star_sInv'], 0)

    def test_sufficient_corner_noise_retains_original_covariance_and_strict_mode(self):
        from echo_object import fit_echo_object
        from scipy import ndimage
        inputs = self.noise_fixture()
        fit = prepare_echo_object(inputs, allow_uncalibrated_noise=False)
        reference = fit_echo_object(inputs['b0_raw']['coil_images'], inputs['echo_times_s'],
            support_mask=fit['object_support'], noise_mask=fit['noise_mask'],
            target_te_s=.1, pooled_mask=ndimage.binary_erosion(fit['object_support'], iterations=6))
        self.assertTrue(fit['report']['noise']['calibrated'])
        self.assertIsNone(fit['report']['noise']['fallback_reason'])
        for key in ['noise_covariance', 'simulation_R2star_sInv', 'simulation_amplitude_at_reference']:
            np.testing.assert_array_equal(fit[key], reference[key])
        with self.assertRaisesRegex(ValueError, 'finite background samples'):
            prepare_echo_object(inputs, background_corner_m=.2, allow_uncalibrated_noise=False)

    def test_32_receiver_RSS_floor_does_not_consume_noise_corners_or_object_support(self):
        rng = np.random.default_rng(781)
        x = (np.arange(64) - 32) * .2 / 64
        X, Y = np.meshgrid(x, x)
        object_mask = ((X + .025)**2 + Y**2 < .03**2) | ((X - .045)**2 + Y**2 < .012**2)
        object_mask[32, 24] = False  # A measured empty hole must remain empty.
        times = np.array([.002, .004, .006])
        weights = np.exp(1j * np.arange(32)) / np.sqrt(32)
        coils = np.exp(-12 * times[:, None, None, None]) * weights[None, :, None, None] * object_mask
        noise = .04 * (rng.normal(size=coils.shape) + 1j * rng.normal(size=coils.shape))
        # Correlated proper-complex noise must retain its off-diagonal covariance.
        noise[:, 1] += .3j * noise[:, 0]
        coils += noise
        coils[:, 0, 12, 32] += 3.  # Isolated coherent pixel: remove small component.
        inputs = dict(b0_raw=dict(coil_images=coils, X_m=X, Y_m=Y),
            field=dict(mask=np.ones(X.shape, bool), b0Hz=np.zeros(X.shape)),
            acquisition=dict(TE_s=.078), echo_times_s=times)
        fit = prepare_echo_object(inputs, allow_uncalibrated_noise=False)
        report = fit['report']
        self.assertGreater(report['support']['raw_RSS_threshold_voxels'], .99 * X.size)
        self.assertTrue(report['noise']['calibrated'])
        self.assertGreaterEqual(report['noise']['available_sample_count'], 64)
        self.assertTrue(report['support']['SNR_used'])
        self.assertEqual(report['support']['support_noise_overlap_voxels'], 0)
        self.assertEqual(np.count_nonzero(fit['object_support'] & ~object_mask), 0)
        self.assertGreater(np.count_nonzero(fit['object_support'] & object_mask), .95 * object_mask.sum())
        self.assertFalse(fit['object_support'][32, 24])
        self.assertFalse(fit['object_support'][12, 32])
        self.assertEqual(report['support']['retained_component_count'], 2)
        self.assertGreater(report['support']['removed_small_component_voxels'], 0)
        self.assertGreater(abs(fit['noise_covariance'][0, 1].imag), .0005)
        self.assertLess(fit['first_echo_amplitude'][object_mask].mean(),
                        fit['raw_RSS'][0][object_mask].mean())

    def test_small_holes_need_measured_SNR_and_optional_relative_threshold_can_be_disabled(self):
        inputs = self.noise_fixture(noise_std=.005)
        coils = inputs['b0_raw']['coil_images']
        coils[:, :, 16, 16] *= .1  # Still well above noise but below relative threshold.
        fit = prepare_echo_object(inputs, threshold_fraction=.2)
        self.assertTrue(fit['object_support'][16, 16])
        self.assertGreater(fit['report']['support']['added_hole_voxels'], 0)
        disabled = prepare_echo_object(inputs, threshold_fraction=.2, max_hole_voxels=0)
        self.assertFalse(disabled['object_support'][16, 16])
        no_relative_threshold = prepare_echo_object(inputs, threshold_fraction=None)
        self.assertTrue(no_relative_threshold['object_support'][16, 16])
        self.assertIsNone(no_relative_threshold['report']['support']['relative_threshold_fraction'])

    def test_coherent_signal_in_corners_is_flagged_and_an_explicit_clean_mask_is_accepted(self):
        inputs = self.noise_fixture()
        raw = inputs['b0_raw']
        corners = (abs(raw['X_m']) > .085) & (abs(raw['Y_m']) > .085)
        contaminated = corners & (raw['X_m'] > 0)
        raw['coil_images'][:, :, contaminated] += .2
        with self.assertRaisesRegex(ValueError, 'possible signal contamination'):
            prepare_echo_object(inputs, allow_uncalibrated_noise=False)
        fallback = prepare_echo_object(inputs)
        self.assertFalse(fallback['report']['noise']['calibrated'])
        self.assertGreater(fallback['report']['noise']['coherent_mean_max_z'], 8)
        self.assertFalse(fallback['report']['support']['SNR_used'])
        clean = corners & (raw['X_m'] < 0)
        fit = prepare_echo_object(inputs, noise_mask=clean, allow_uncalibrated_noise=False)
        self.assertTrue(fit['report']['noise']['calibrated'])
        np.testing.assert_array_equal(fit['noise_mask'], clean)
        self.assertEqual(fit['report']['noise']['mask_source'], 'explicit signal-free mask')
        self.assertFalse(np.any(fit['object_support'] & clean))

    def test_identically_zero_receiver_does_not_create_false_corner_contamination(self):
        inputs = self.noise_fixture()
        coils = inputs['b0_raw']['coil_images']
        inputs['b0_raw']['coil_images'] = np.concatenate((coils, np.zeros_like(coils[:, :1])), axis=1)
        fit = prepare_echo_object(inputs, allow_uncalibrated_noise=False)
        self.assertTrue(fit['report']['noise']['calibrated'])
        self.assertEqual(fit['report']['noise']['per_receiver_complex_RMS'][-1], 0.)
        self.assertTrue(fit['report']['support']['amplitude_debiased'])

    def test_B0_overrides_are_checked_before_reading_reference_or_EPI(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'sequences').mkdir()
            for name in ('old_b0_map.dat', 'new_echoes.dat', 'one_reference.dat', 'one_scan.dat',
                         'sequences/epi.seq', 'sequences/gre_32.seq', 'sequences/b0_map.seq',
                         'sequences/new32.seq'):
                (folder / name).write_text('fixture')
            with patch('twin_inputs.read_b0_coil_images', side_effect=ValueError(
                    'B0 echo_count: expected 32 echoes from sequence; found 3 in Twix.')) as read_b0, \
                    patch('twin_inputs.read_gre') as read_gre, patch('twin_inputs.read_scan') as read_scan:
                with self.assertRaisesRegex(ValueError, r'new_echoes.dat.*new32.seq.*expected 32.*found 3.*b0_scan_path'):
                    read_twin_inputs(folder, b0_sequence_path='sequences/new32.seq', b0_scan_path='new_echoes.dat')
                read_b0.assert_called_once_with(folder / 'new_echoes.dat', folder / 'sequences/new32.seq')
                read_gre.assert_not_called()
                read_scan.assert_not_called()

    def test_B0_echo_subset_does_not_discard_relaxation_echoes(self):
        import twin_inputs
        coils = np.zeros((32, 2, 4, 4), complex)
        times = .002 + np.arange(32) * .0018
        raw = dict(coil_images=coils, X_m=np.zeros((4, 4)), Y_m=np.zeros((4, 4)))
        with patch('twin_inputs.b0_from_coil_images', return_value={'metadata': {}}) as estimate:
            field, count = twin_inputs._estimate_field(raw, times, 6)
            np.testing.assert_array_equal(estimate.call_args.args[0], coils[:6])
            np.testing.assert_array_equal(estimate.call_args.args[1], times[:6])
            self.assertEqual(count, 6)
            self.assertEqual(field['metadata']['field_echo_indices'], list(range(6)))
            self.assertEqual(field['metadata']['available_relaxation_echo_count'], 32)
            self.assertEqual(raw['coil_images'].shape[0], 32)
        for invalid in (True, 1, 33, 6.5):
            with self.assertRaisesRegex(ValueError, 'b0_echo_count'):
                twin_inputs._estimate_field(raw, times, invalid)

    def test_measured_loading_forwards_reconstruction_device_and_precision(self):
        for kwargs, device, dtype in (({}, 'cpu', torch.float64),
                ({'reconstruction_device': 'cuda:0', 'reconstruction_dtype': torch.float32},
                 'cuda:0', torch.float32)):
            with self.subTest(device=device), TemporaryDirectory() as directory:
                folder = Path(directory)
                times = .002 + np.arange(32) * .0018
                with self.measured_input_fixture(folder) as (mocks, raw, _):
                    result = read_twin_inputs(folder, b0_echo_count=6, **kwargs)
                receivers, echoes = mocks['reconstruct_receivers'], mocks['reconstruct_b0_echoes']
                self.assertEqual(receivers.call_count, 2)
                for call in receivers.call_args_list:
                    self.assertEqual(call.kwargs, {'device': device, 'dtype': dtype})
                self.assertIs(echoes.call_args.args[0], raw)
                self.assertEqual(echoes.call_args.kwargs, {'device': device, 'dtype': dtype})
                self.assertEqual(result['b0_raw']['coil_images'].shape[0], 32)
                np.testing.assert_array_equal(result['echo_times_s'], times)

    def test_explicit_arbitrary_reference_names_keep_own_64_or_128_sequence(self):
        import hashlib
        for size in (64, 128):
            with self.subTest(size=size), TemporaryDirectory() as directory:
                folder = Path(directory)
                arbitrary = 'better_patient_scan.dat'  # Deliberately also matches *scan.dat.
                with self.measured_input_fixture(folder, size, (arbitrary,)) as (mocks, _, gre):
                    original = deepcopy(gre)
                    result = read_twin_inputs(folder, reference_scan_path=arbitrary,
                        reference_sequence_path=f'sequences/gre_{size}.seq')
                self.assertEqual(result['paths']['reference'], folder / arbitrary)
                self.assertEqual(result['paths']['scan'], folder / 'scan.dat')
                reference_acquisition = mocks['read_gre'].call_args.args[1]
                self.assertEqual(reference_acquisition['kx_mInv'].shape, (size, size))
                self.assertEqual(result['acquisition']['kx_mInv'].shape, (128, 128))
                self.assertEqual(result['gre']['magnitude'].shape, (size, size))
                self.assertEqual(result['reference_diagnostics'], {'native_reference': True})
                self.assertEqual(result['source_sha256']['reference'],
                    hashlib.sha256((folder / arbitrary).read_bytes()).hexdigest())
                for name in ('X_m', 'Y_m', 'magnitude'):
                    np.testing.assert_array_equal(gre[name], original[name])

    def test_reference_sequence_autoselection_is_matrix_independent_and_ambiguity_is_explicit(self):
        for size in (32, 64, 128):
            with self.subTest(size=size), TemporaryDirectory() as directory:
                folder = Path(directory)
                with self.measured_input_fixture(folder, size):
                    result = read_twin_inputs(folder)
                    self.assertEqual(result['paths']['reference_sequence'], folder / f'sequences/gre_{size}.seq')
                    (folder / 'sequences/another_reference.seq').write_text('fixture')
                    with self.assertRaisesRegex(ValueError, r'found 2.*reference_sequence_path'):
                        read_twin_inputs(folder)

    def test_explicit_absolute_scan_and_B0_names_are_excluded_from_auto_EPI_selection(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            files = ('renamed_B0_scan.dat', 'preferred_image.dat', 'sequences/field32.seq')
            with self.measured_input_fixture(folder, extra_files=files) as (mocks, _, _):
                result = read_twin_inputs(folder, b0_scan_path=folder / files[0],
                    b0_sequence_path=folder / files[2])
                self.assertEqual(result['paths']['scan'], folder / 'scan.dat')
                mocks['read_b0_coil_images'].assert_called_with(folder / files[0], folder / files[2])
                result = read_twin_inputs(folder, scan_path=folder / files[1],
                    b0_scan_path=folder / files[0], b0_sequence_path=folder / files[2])
                self.assertEqual(result['paths']['scan'], folder / files[1])

    def test_reference_and_B0_slice_disagreement_is_rejected_without_mutating_grids(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.measured_input_fixture(folder) as (_, raw, gre):
                gre['twix_metadata'] = {'slice_position_mm': [0., 0., 5.]}
                raw['metadata']['slice_position_mm'] = [0., 0., 0.]
                original_X = gre['X_m'].copy()
                with self.assertRaisesRegex(ValueError, r'reference GRE and B0.*slice planes'):
                    read_twin_inputs(folder)
                np.testing.assert_array_equal(gre['X_m'], original_X)

    def test_reference_scan_and_sequence_errors_name_the_missing_override(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.measured_input_fixture(folder):
                with self.assertRaisesRegex(FileNotFoundError, 'Reference scan not found'):
                    read_twin_inputs(folder, reference_scan_path='missing.dat')
                (folder / 'another_reference.dat').write_text('fixture')
                with self.assertRaisesRegex(ValueError, r'found 2.*reference_scan_path'):
                    read_twin_inputs(folder)

    def nominal_reference_fixture(self, size):
        rng = np.random.default_rng(size)
        kspace = rng.normal(size=(size, 2, size)) + 1j * rng.normal(size=(size, 2, size))
        metadata = dict(navigator_count=0, polarity=[1] * size, slice_position_mm=[0., 0., 0.],
            slice_quaternion_wxyz=(np.asarray([1, 0, 0, 1]) / np.sqrt(2)).tolist(),
            FOV_m=[.2, .2], slice_thickness_mm=5., TE_s=.00432, TR_s=.015,
            dwell_s=.00001, header_flip_deg=15., image_shape=[size, size])
        return np.ones((size, size)), metadata, dict(kspace=kspace, has_raw_kspace=True)

    def test_incompatible_auto_reference_sequence_uses_own_nominal_header_grid(self):
        for size in (64, 128):
            with self.subTest(size=size), TemporaryDirectory() as directory:
                folder = Path(directory)
                with self.measured_input_fixture(folder, reference_size=32) as (mocks, _, _):
                    nominal = self.nominal_reference_fixture(size)
                    mocks['read_gre'].side_effect = ValueError('Twix image shape disagrees with Pulseq (32, 32).')
                    mocks['read_scan'].side_effect = lambda path, *args, **kwargs: (
                        nominal if path.name == 'reference.dat' else (np.ones((128, 128)), {}, {}))
                    result = read_twin_inputs(folder)
                self.assertEqual(result['gre']['magnitude'].shape, (size, size))
                self.assertEqual(result['reference_acquisition']['kx_mInv'].shape, (size, size))
                self.assertAlmostEqual(result['gre']['TE_s'], .00432)
                self.assertNotIn('reference_sequence', result['paths'])
                self.assertNotIn('reference_sequence', result['source_sha256'])
                report = result['reference_sequence_validation']
                self.assertEqual(report['mode'], 'twix_header_nominal_cartesian')
                self.assertFalse(report['validated'])
                self.assertIn('gre_32.seq', report['reason'])
                self.assertEqual(mocks['read_scan'].call_args_list[0].args[1], None)

    def test_explicit_incompatible_reference_sequence_never_falls_back(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.measured_input_fixture(folder, reference_size=32) as (mocks, _, _):
                mocks['read_gre'].side_effect = ValueError('Twix image shape disagrees with Pulseq (32, 32).')
                with self.assertRaisesRegex(ValueError, 'Twix image shape disagrees'):
                    read_twin_inputs(folder, reference_sequence_path='sequences/gre_32.seq')
                mocks['read_scan'].assert_not_called()

    def test_absent_reference_sequence_reconstructs_validated_nominal_Twix(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.measured_input_fixture(folder, reference_size=64) as (mocks, _, _):
                (folder / 'sequences/gre_64.seq').unlink()
                nominal = self.nominal_reference_fixture(64)
                mocks['read_scan'].side_effect = lambda path, *args, **kwargs: (
                    nominal if path.name == 'reference.dat' else (np.ones((128, 128)), {}, {}))
                result = read_twin_inputs(folder)
                mocks['read_gre'].assert_not_called()
                self.assertNotIn('reference_sequence', result['paths'])
                self.assertTrue(result['reference_sequence_validation']['raw_cartesian_validated'])
                self.assertFalse(result['reference_sequence_validation']['sequence_verified'])
                self.assertEqual(result['reference_acquisition']['TR_s'], .015)

    def test_other_parseable_EPI_sequence_does_not_prevent_nominal_reference_fallback(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.measured_input_fixture(folder, reference_size=64,
                    extra_files=('sequences/another_epi.seq',)) as (mocks, _, _):
                (folder / 'sequences/gre_64.seq').unlink()
                nominal = self.nominal_reference_fixture(64)
                mocks['read_scan'].side_effect = lambda path, *args, **kwargs: (
                    nominal if path.name == 'reference.dat' else (np.ones((128, 128)), {}, {}))
                result = read_twin_inputs(folder, scan_sequence_path='sequences/epi.seq')
                mocks['read_gre'].assert_not_called()
                self.assertNotIn('reference_sequence', result['paths'])
                self.assertEqual(result['reference_sequence_validation']['mode'], 'twix_header_nominal_cartesian')
                self.assertEqual(result['reference_acquisition']['sequence_kind'], 'gre')
                self.assertEqual(result['reference_acquisition']['kx_mInv'].shape, (64, 64))

    def test_nominal_reference_reconstruction_matches_centered_FFT_and_rejects_EPI(self):
        from twin_inputs import _read_nominal_reference
        from measured_continuous import reconstruct_receivers
        for size in (64, 128):
            with self.subTest(size=size):
                nominal = self.nominal_reference_fixture(size)
                with patch('twin_inputs.read_scan', return_value=nominal):
                    gre, diagnostics, acquisition = _read_nominal_reference(Path('any_name.dat'))
                actual, _ = reconstruct_receivers(diagnostics, acquisition)
                axes = (0, 2)
                coils = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(nominal[2]['kspace'], axes=axes), axes=axes), axes=axes)
                expected = np.sqrt(np.sum(abs(coils)**2, axis=1))
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
                self.assertEqual(gre['magnitude'].shape, (size, size))
                nominal[1]['navigator_count'] = 3
                with patch('twin_inputs.read_scan', return_value=nominal):
                    with self.assertRaisesRegex(ValueError, 'mono-echo GRE'):
                        _read_nominal_reference(Path('any_name.dat'))

    def test_synthetic_sequence_defaults_to_manifest_and_accepts_matching_paths(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'acquisition.json').write_text(json.dumps(
                {'format': 'mrzero_synthetic_acquisition'}))
            result = {'paths': {'scan_sequence': folder / 'sequences/manifest64.seq'}}
            # A synthetic acquisition does not select arbitrary files by glob.
            with patch('Mrzero_simulation.synthetic_inputs.read_synthetic_twin_inputs', return_value=result):
                self.assertIs(read_twin_inputs(folder), result)
                for matching in ('sequences/manifest64.seq', result['paths']['scan_sequence']):
                    self.assertIs(read_twin_inputs(folder, scan_sequence_path=matching), result)

    def test_synthetic_reference_B0_and_scan_overrides_respect_manifest(self):
        arguments = {'reference_sequence_path': ('reference_sequence', 'sequences/reference128.seq'),
                     'reference_scan_path': ('reference', 'arbitrary_reference.npz'),
                     'scan_path': ('scan', 'arbitrary_scan.npz'),
                     'b0_sequence_path': ('b0_sequence', 'sequences/field.seq'),
                     'b0_scan_path': ('b0', 'arbitrary_field.npz')}
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'acquisition.json').write_text(json.dumps(
                {'format': 'mrzero_synthetic_acquisition'}))
            result = {'paths': {key: folder / path for key, path in arguments.values()}}
            with patch('Mrzero_simulation.synthetic_inputs.read_synthetic_twin_inputs', return_value=result):
                for argument, (_, path) in arguments.items():
                    self.assertIs(read_twin_inputs(folder, **{argument: path}), result)
                    with self.assertRaisesRegex(ValueError, argument + '=None'):
                        read_twin_inputs(folder, **{argument: 'conflicting_file'})

    def test_synthetic_sequence_conflict_reports_requested_and_manifest_paths(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'acquisition.json').write_text(json.dumps(
                {'format': 'mrzero_synthetic_acquisition'}))
            result = {'paths': {'scan_sequence': folder / 'sequences/manifest64.seq'}}
            with patch('Mrzero_simulation.synthetic_inputs.read_synthetic_twin_inputs', return_value=result):
                with self.assertRaises(ValueError) as caught:
                    read_twin_inputs(folder, scan_sequence_path='sequences/other128.seq')
            message = str(caught.exception)
            self.assertIn('Requested: ' + str((folder / 'sequences/other128.seq').resolve()), message)
            self.assertIn('expected: ' + str((folder / 'sequences/manifest64.seq').resolve()), message)
            self.assertIn('scan_sequence_path=None', message)
            self.assertIn('rerun the input/configuration cells', message)


if __name__ == '__main__':
    unittest.main()
