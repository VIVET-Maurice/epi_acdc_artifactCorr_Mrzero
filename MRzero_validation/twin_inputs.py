"""Inputs and controlled object variants for the measured-EPI validation notebook.

No raw scan is edited. The simulation domain depends on object/B0 coverage,
not on an ACDC optimisation ROI or availability of coil calibration maps.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import sys
import numpy as np
from scipy import ndimage

from prepare_acdc_inputs import read_gre, read_scan
from read_acdc_sequence import read_sequence
from read_acdc_b0 import read_b0_coil_images, b0_from_coil_images
from prepare_acdc_simulation_grid import prepare_simulation_reference
from resample_acdc_maps import resample_to_grid
from measured_continuous import reconstruct_receivers, reconstruct_b0_echoes
from echo_object import fit_echo_object


def _field_echo_count(value, available):
    if value is None:
        return available
    if (isinstance(value, bool) or int(value) != value or not 2 <= value <= available):
        raise ValueError(f'b0_echo_count must be None or an integer from 2 to {available}.')
    return int(value)


def _estimate_field(raw, echo_times, requested_count):
    count = _field_echo_count(requested_count, len(echo_times))
    field = b0_from_coil_images(raw['coil_images'][:count], echo_times[:count],
                                return_intermediates=True)
    field.update(X_m=raw['X_m'], Y_m=raw['Y_m'])
    field['metadata'].update(field_echo_count=count, field_echo_indices=list(range(count)),
        field_echo_times_s=np.asarray(echo_times[:count]).tolist(),
        available_relaxation_echo_count=len(echo_times))
    return field, count


def read_twin_inputs(folder, *, scan_sequence_path=None, b0_sequence_path=None,
                     b0_scan_path=None, b0_echo_count=None):
    """Read native inputs, with explicit EPI and multi-echo B0 selections.

    Relative scan_sequence_path values are resolved under folder. Without an
    override, exactly one sequences/*epi*.seq must exist. Twix/sequence matrix
    validation remains active; changing a filename does not resample a scan.
    ``b0_sequence_path`` defaults to sequences/b0_map.seq and ``b0_scan_path``
    to the unique *b0_map.dat. Overrides are resolved under folder. The B0
    Twix is checked against its sequence before the reference/EPI are read;
    selecting a new 32-echo .seq never converts an old three-echo .dat.
    ``b0_echo_count`` uses only the first N measured echoes to estimate B0;
    all validated echoes remain available for relaxation fitting. None uses all.
    """
    folder = Path(folder)
    manifest = None
    manifest_path = folder / 'acquisition.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        if manifest.get('format') == 'mrzero_synthetic_acquisition':
            project_dir = Path(__file__).resolve().parent.parent
            if str(project_dir) not in sys.path:
                sys.path.insert(0, str(project_dir))
            from Mrzero_simulation.synthetic_inputs import read_synthetic_twin_inputs
            result = read_synthetic_twin_inputs(folder)
            for override, key, argument in ((scan_sequence_path, 'scan_sequence', 'scan_sequence_path'),
                                            (b0_sequence_path, 'b0_sequence', 'b0_sequence_path'),
                                            (b0_scan_path, 'b0', 'b0_scan_path')):
                if override is None:
                    continue
                selected = Path(override)
                selected = selected if selected.is_absolute() else folder / selected
                expected = Path(result['paths'][key]).resolve()
                if selected.resolve() != expected:
                    raise ValueError(
                        f"Synthetic acquisition {key} is fixed by its manifest. "
                        f"Requested: {selected.resolve()}; expected: {expected}. "
                        f"Use {argument}=None to select the manifest file, or provide its matching path. "
                        "In a notebook, rerun the input/configuration cells before reading the data; "
                        "changing a sequence path does not regenerate synthetic samples.")
            if b0_echo_count is not None:
                result['field'], result['field_echo_count'] = _estimate_field(
                    result['b0_raw'], np.asarray(result['echo_times_s']), b0_echo_count)
            return result
    def unique(pattern):
        paths = list(folder.glob(pattern))
        if len(paths) != 1:
            raise ValueError(f'Expected one {pattern}, found {len(paths)}.')
        return paths[0]
    if scan_sequence_path is None:
        candidates = sorted(folder.glob('sequences/*epi*.seq'))
        if len(candidates) != 1:
            names = ', '.join(path.name for path in candidates) or '(none)'
            raise ValueError(f'Expected one EPI sequence, found {len(candidates)}: {names}. '
                             'Set scan_sequence_path explicitly (relative to the acquisition folder, or absolute).')
        scan_sequence_path = candidates[0]
    else:
        scan_sequence_path = Path(scan_sequence_path)
        if not scan_sequence_path.is_absolute():
            scan_sequence_path = folder / scan_sequence_path
    if not scan_sequence_path.is_file():
        raise FileNotFoundError(f'EPI sequence not found: {scan_sequence_path}')
    def selected_path(value, default, label):
        selected = Path(value) if value is not None else default
        if value is not None and not selected.is_absolute():
            selected = folder / selected
        if not selected.is_file():
            raise FileNotFoundError(f'{label} not found: {selected}')
        return selected
    b0_sequence_path = selected_path(b0_sequence_path, folder/'sequences/b0_map.seq', 'B0 sequence')
    b0_scan_path = selected_path(b0_scan_path,
        unique('*b0_map.dat') if b0_scan_path is None else None, 'B0 scan')
    paths = dict(reference=unique('*reference.dat'), b0=b0_scan_path,
        scan=unique('*scan.dat'), scan_sequence=scan_sequence_path,
        reference_sequence=unique('sequences/gre_32*.seq'),
        b0_sequence=b0_sequence_path)
    try:
        raw = read_b0_coil_images(paths['b0'], paths['b0_sequence'])
    except ValueError as error:
        raise ValueError(f"B0 scan {paths['b0']} is incompatible with B0 sequence "
                         f"{paths['b0_sequence']}: {error} Select the matching acquired "
                         ".dat with b0_scan_path; a new .seq does not replace a measurement.") from error
    _field_echo_count(b0_echo_count, len(raw['metadata']['TE_s']))
    acquisition, timing = read_sequence(paths['scan_sequence'])
    reference_acquisition, _ = read_sequence(paths['reference_sequence'])
    gre, gd = read_gre(paths['reference'], reference_acquisition, return_intermediates=True)
    gre['magnitude'], _ = reconstruct_receivers(gd, reference_acquisition)
    scan, scan_metadata, sd = read_scan(paths['scan'], acquisition, return_intermediates=True)
    scan, _ = reconstruct_receivers(sd, acquisition)
    reconstruct_b0_echoes(raw, paths['b0_sequence'])
    echo_times = np.asarray(raw['metadata']['TE_s'])
    field, field_echo_count = _estimate_field(raw, echo_times, b0_echo_count)
    return dict(paths=paths, acquisition=acquisition, timing=timing, gre=gre,
        scan=scan, scan_metadata=scan_metadata, scan_diagnostics=sd, b0_raw=raw,
        field=field, field_echo_count=field_echo_count, echo_times_s=echo_times, manifest=manifest,
        source_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest()
                       for name,path in paths.items()})


def prepare_echo_object(inputs, threshold_fraction=.04, background_corner_m=.085, *,
                        allow_uncalibrated_noise=True, noise_mask=None,
                        support_min_snr=5., min_component_voxels=4,
                        max_hole_voxels=64, noise_mean_z_limit=8.):
    """Estimate noise independently, then form a debiased, SNR-based object.

    ``noise_mask`` explicitly declares signal-free pixels; otherwise use only
    the chosen physical corners, independently of the raw RSS threshold. No
    automatic widening or object-complement substitution is allowed. Complex
    receiver covariance removes the 32-coil RSS floor before segmentation.
    The first-echo amplitude SNR must pass ``support_min_snr``; the optional
    ``threshold_fraction`` (None disables it) is an additional debiased-peak
    threshold. Keep every component of at least ``min_component_voxels``.
    Small enclosed holes are added only where measured SNR also passes.

    A coherent-mean check can reject contaminated background; passing it does
    not prove absence of spatially varying signal. Without valid covariance,
    retain a threshold-only support, first-echo amplitude and zero additional
    decay, all explicitly reported. Strict mode rejects this fallback.
    """
    raw, field = inputs['b0_raw'], inputs['field']
    coils = np.asarray(raw['coil_images'])
    if coils.ndim != 4 or not np.iscomplexobj(coils) or min(coils.shape) < 1:
        raise ValueError('Expected complex coil_images [echo, receiver, row, column].')
    if threshold_fraction is not None and (not np.isfinite(threshold_fraction)
            or not 0 <= threshold_fraction < 1):
        raise ValueError('threshold_fraction must be None or in [0, 1).')
    if not np.isfinite(support_min_snr) or support_min_snr <= 0:
        raise ValueError('support_min_snr must be positive and finite.')
    if (not np.isfinite(background_corner_m) or background_corner_m < 0
            or not np.isfinite(noise_mean_z_limit) or noise_mean_z_limit <= 0):
        raise ValueError('Invalid physical corner threshold or noise_mean_z_limit.')
    for value, label, lower in ((min_component_voxels, 'min_component_voxels', 1),
                                 (max_hole_voxels, 'max_hole_voxels', 0)):
        if isinstance(value, bool) or int(value) != value or value < lower:
            raise ValueError(f'{label} must be an integer >= {lower}.')
    shape = coils.shape[-2:]
    X, Y = np.asarray(raw['X_m']), np.asarray(raw['Y_m'])
    if X.shape != shape or Y.shape != shape or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError('Finite native X_m/Y_m grids must match coil_images.')
    noise_source = 'explicit signal-free mask' if noise_mask is not None else 'explicit physical corners'
    if noise_mask is None:
        noise_mask = (abs(X) > background_corner_m) & (abs(Y) > background_corner_m)
    else:
        noise_mask = np.asarray(noise_mask)
        if noise_mask.shape != shape or not np.isin(noise_mask, [0, 1]).all():
            raise ValueError('noise_mask must be binary and match the native image shape.')
        noise_mask = noise_mask.astype(bool)
    power = np.sum(abs(coils)**2, axis=1)
    rss = np.sqrt(power)
    finite = np.isfinite(coils).all(axis=(0, 1))
    receivers = coils.shape[1]
    samples = coils[:, :, noise_mask].transpose(1, 0, 2).reshape(receivers, -1)
    samples = samples[:, np.isfinite(samples).all(axis=0)]
    sample_count = samples.shape[1]
    minimum_samples = max(32, 2 * receivers)
    noise_fallback_reason = None
    covariance = None
    mean_z = None
    if sample_count < minimum_samples:
        noise_fallback_reason = f"Only {sample_count} finite background samples; need {minimum_samples}."
    elif np.var(samples, axis=1, ddof=1).real.sum() <= 0:
        noise_fallback_reason = "Background samples have zero receiver-noise variance."
    else:
        centered = samples - samples.mean(axis=1, keepdims=True)
        covariance = centered @ centered.conj().T / (sample_count - 1)
        # Separate echoes and connected patches avoid cancellation between
        # opposite corners or receiver phases. Spatial independence is an
        # explicit diagnostic assumption, never an estimated effective N.
        labels, _ = ndimage.label(noise_mask)
        patch_means = []
        within_patch_power = np.zeros(receivers)
        within_patch_dof = 0
        for echo in coils:
            for label in range(1, int(labels.max()) + 1):
                patch = echo[:, labels == label]
                patch = patch[:, np.isfinite(patch).all(axis=0)]
                if patch.shape[1] < 4:
                    continue
                patch_mean = patch.mean(axis=1)
                patch_means.append((patch.shape[1], patch_mean))
                within_patch_power += np.sum(abs(patch - patch_mean[:, None])**2, axis=1)
                within_patch_dof += patch.shape[1] - 1
        # Estimate the screening variance after removing patch means: a bright
        # coherent corner must not inflate its own denominator and pass.
        receiver_power = (within_patch_power / within_patch_dof if within_patch_dof
                          else np.diag(covariance).real)
        mean_scores = []
        for count, patch_mean in patch_means:
            score = np.divide(np.sqrt(count) * abs(patch_mean), np.sqrt(receiver_power),
                out=np.zeros(receivers), where=receiver_power > 0)
            score[(receiver_power == 0) & (abs(patch_mean) > 0)] = np.inf
            mean_scores.append(float(score.max()))
        mean_z = max(mean_scores, default=0.)
        if mean_z > noise_mean_z_limit:
            noise_fallback_reason = (f'Background coherent mean z={mean_z:.3g} exceeds '
                f'{noise_mean_z_limit:g}; possible signal contamination. Inspect the declared '
                'corners or supply an explicit signal-free noise_mask.')
            covariance = None
    if noise_fallback_reason is not None and not allow_uncalibrated_noise:
        raise ValueError(noise_fallback_reason)
    if covariance is not None:
        # Match the fitter's proper-complex RSS-power debiasing and variance.
        amplitude = np.sqrt(np.maximum(power[0] - np.trace(covariance).real, 0.))
        trace_squared = float(np.sum(abs(covariance)**2))
        first = np.where(np.isfinite(coils[0]), coils[0], 0).astype(complex)
        quadratic = np.einsum('cyx,cd,dyx->yx', first.conj(), covariance, first,
                              optimize=True).real
        power_variance = np.maximum(2 * quadratic - trace_squared, trace_squared)
        amplitude_variance = np.divide(power_variance, 4 * amplitude**2,
            out=np.full(shape, np.inf), where=amplitude > 0)
        snr = np.divide(amplitude, np.sqrt(amplitude_variance),
            out=np.zeros(shape), where=amplitude_variance > 0)
        signal_evidence = finite & (snr >= support_min_snr)
        support_method = 'debiased first-echo RSS amplitude SNR and optional relative threshold'
    else:
        amplitude, snr = rss[0], np.full(shape, np.nan)
        signal_evidence = finite & (amplitude > 0)
        support_method = 'uncalibrated first-echo RSS threshold; SNR unavailable'
    finite_amplitude = amplitude[np.isfinite(amplitude)]
    peak = float(finite_amplitude.max()) if finite_amplitude.size else 0.
    relative_threshold = 0. if threshold_fraction is None else float(threshold_fraction) * peak
    seeds = signal_evidence & (amplitude > relative_threshold) & ~noise_mask
    labels, component_count = ndimage.label(seeds)
    sizes = np.bincount(labels.ravel())
    support = seeds & (sizes[labels] >= min_component_voxels)
    removed = int(seeds.sum() - support.sum())
    filled = np.zeros(shape, bool)
    if covariance is not None and max_hole_voxels:
        holes = ndimage.binary_fill_holes(support) & ~support
        hole_labels, _ = ndimage.label(holes)
        hole_sizes = np.bincount(hole_labels.ravel())
        filled = (holes & (hole_sizes[hole_labels] <= max_hole_voxels)
                  & signal_evidence & ~noise_mask)
        support |= filled
    if not support.any():
        raise ValueError('No object component remains after the declared noise/SNR/amplitude thresholds; '
                         'inspect the noise mask and support settings.')
    interior = ndimage.binary_erosion(support, iterations=6)
    fit = fit_echo_object(coils, inputs['echo_times_s'],
        support_mask=support, noise_mask=noise_mask if noise_fallback_reason is None else None,
        target_te_s=inputs['acquisition']['TE_s'], pooled_mask=interior)
    fit['report']['noise'].update(
        available_sample_count=int(sample_count), minimum_sample_count=minimum_samples,
        requested_background_voxels=int(noise_mask.sum()),
        background_corner_m=float(background_corner_m), fallback_reason=noise_fallback_reason,
        mask_source=noise_source, mask_independent_of_object_threshold=True,
        coherent_mean_max_z=mean_z, coherent_mean_z_limit=float(noise_mean_z_limit),
        contamination_screen='Per-echo connected-patch complex means; spatial independence assumed; '
            'cannot certify absence of spatially varying signal.',
        per_receiver_complex_RMS=(np.sqrt(np.diag(covariance).real).tolist()
                                  if covariance is not None else None),
        RSS_noise_RMS=(float(np.sqrt(np.trace(covariance).real))
                       if covariance is not None else None))
    valid = support & field['mask'] & np.isfinite(field['b0Hz'])
    fit['object_support'] = support
    fit['simulation_support'] = valid
    fit['noise_mask'] = noise_mask
    fit['support_snr'] = snr
    fit['support_debiased_amplitude'] = amplitude
    fit['report']['support'] = dict(method=support_method,
        SNR_used=covariance is not None, amplitude_debiased=covariance is not None,
        min_amplitude_SNR=float(support_min_snr),
        relative_threshold_fraction=threshold_fraction, relative_amplitude_threshold=relative_threshold,
        raw_RSS_threshold_voxels=int(np.count_nonzero(finite & (rss[0] >
            (0. if threshold_fraction is None else threshold_fraction * np.nanmax(rss[0]))))),
        seed_voxels=int(seeds.sum()), seed_component_count=int(component_count),
        retained_component_count=int(ndimage.label(support)[1]),
        removed_small_component_voxels=removed, min_component_voxels=int(min_component_voxels),
        added_hole_voxels=int(filled.sum()), max_hole_voxels=int(max_hole_voxels),
        hole_policy='Only small enclosed holes whose measured amplitude SNR also passes; disabled without noise calibration.',
        excluded_noise_mask_voxels=int(noise_mask.sum()),
        support_noise_overlap_voxels=int(np.count_nonzero(support & noise_mask)),
        support_voxels=int(support.sum()), support_independent_of_B0=True)
    fit['report'].update(object_threshold_fraction=threshold_fraction,
        object_voxels=int(support.sum()), simulated_voxels=int(valid.sum()),
        excluded_missing_B0_voxels=int((support & ~valid).sum()),
        excluded_missing_B0_amplitude_fraction=float(
            fit['first_echo_amplitude'][support & ~valid].sum()/fit['first_echo_amplitude'][support].sum()),
        ACDC_calibration_used_for_baseline_support=False,
        optimisation_ROI_used_for_baseline_support=False)
    return fit


def frame_acquisition(acquisition, rotation_deg=180):
    """Joint XY frame change: transform both k and spin coordinates, not data."""
    if rotation_deg not in (0,180):
        raise ValueError('Frame rotation must be 0 or 180 degrees.')
    result = deepcopy(acquisition)
    sign = 1 if rotation_deg == 0 else -1
    for key in ('kx_mInv','ky_mInv'):
        result[key] = sign*np.asarray(acquisition[key])
    return result


def build_candidates(inputs, fit, frame_rotation_deg=180):
    """Ablations isolate legacy geometry, native amplitude and echo relaxation.

    The legacy case is an explicit old-convention diagnostic (GRE interpolated
    to 64, fixed-grid array rotation). It does not claim to reproduce every ROI
    and subspin setting of a previously executed notebook.
    """
    sign = 1 if frame_rotation_deg == 0 else -1
    gre, acq, field, raw = (inputs[k] for k in ('gre','acquisition','field','b0_raw'))
    modern_acq = frame_acquisition(acq, frame_rotation_deg)
    candidates = []
    for legacy in (True,False):
        source = deepcopy(gre)
        if legacy:
            source['magnitude'] = np.rot90(source['magnitude'],2).copy()
        refined, _ = prepare_simulation_reference(source, acq)
        b0_values = np.rot90(field['b0Hz'],2) if legacy else field['b0Hz']
        b0_mask = np.rot90(field['mask'],2) if legacy else field['mask']
        b0, valid = resample_to_grid(b0_values,raw['X_m'],raw['Y_m'],b0_mask,
                                   refined['X_m'],refined['Y_m'])
        amp = refined['magnitude']
        requested = ndimage.binary_fill_holes((amp > .04*amp.max()) & refined['valid_mask'])
        support = requested & valid & np.isfinite(b0)
        candidates.append(dict(name='legacy_GRE64' if legacy else 'geometry_GRE64',
            acquisition=acq if legacy else modern_acq,
            X_m=refined['X_m']*(1 if legacy else sign),
            Y_m=refined['Y_m']*(1 if legacy else sign),
            amplitude=amp, B0_Hz=np.nan_to_num(b0), support_mask=support,
            rate_sInv=np.zeros(amp.shape), amplitude_reference_time_s=0.,
            legacy_output_reindex=legacy,
            note='Old fixed-grid rotation diagnostic' if legacy else 'Consistent joint frame; GRE proxy, no decay'))
    native = dict(acquisition=modern_acq, X_m=sign*raw['X_m'],Y_m=sign*raw['Y_m'],
        B0_Hz=np.nan_to_num(field['b0Hz']),support_mask=fit['simulation_support'],
        amplitude_reference_time_s=fit['amplitude_reference_time_s'],legacy_output_reindex=False)
    shape=fit['first_echo_amplitude'].shape
    candidates.append(dict(native,name='echo128_no_decay',amplitude=fit['first_echo_amplitude'],
                           rate_sInv=np.zeros(shape),note='Native first echo; no additional decay'))
    pooled=fit['pooled']
    pooled_rate=float(pooled['R2star_sInv']) if pooled['target_reliable_mask'] else 0.
    candidates.append(dict(native,name='echo128_pooled_decay',amplitude=fit['first_echo_amplitude'],
        rate_sInv=np.full(shape,pooled_rate),note='One conservative ROI-mean effective decay rate'))
    candidates.append(dict(native,name='echo128_guarded_decay',
        amplitude=fit['simulation_amplitude_at_reference'],rate_sInv=fit['simulation_R2star_sInv'],
        note='Native voxel fits only where long-TE uncertainty passes; explicit zero-decay fallback'))
    if 'regularized_B0_Hz' in fit:
        regularized = dict(native, B0_Hz=fit['regularized_B0_Hz'])
        candidates.append(dict(regularized,name='echo128_regularized_B0',
            amplitude=fit['first_echo_amplitude'],rate_sInv=np.zeros(shape),
            note='Native amplitude; measured B0 local-plane regularisation at fixed spatial scale'))
        candidates.append(dict(regularized,name='echo128_regularized_B0_pooled',
            amplitude=fit['first_echo_amplitude'],rate_sInv=np.full(shape,pooled_rate),
            note='Regularised measured B0 and conservative pooled effective decay'))
    return candidates


def model_arguments(candidate):
    keys=('acquisition','X_m','Y_m','amplitude','B0_Hz','support_mask','rate_sInv',
          'amplitude_reference_time_s')
    result = {key:candidate[key] for key in keys}
    # Quadrature weights: a refined object has smaller physical pixels.
    X,Y = np.asarray(candidate['X_m']),np.asarray(candidate['Y_m'])
    column = [X[0,1]-X[0,0],Y[0,1]-Y[0,0]]
    row = [X[1,0]-X[0,0],Y[1,0]-Y[0,0]]
    object_area = abs(float(np.linalg.det(np.stack((column,row)))))
    acq = candidate['acquisition']
    image_area = float(np.prod(acq['FOV_m'])/np.asarray(acq['kx_mInv']).size)
    result['amplitude'] = np.asarray(candidate['amplitude'])*(object_area/image_area)
    return result


__all__=['read_twin_inputs','prepare_echo_object','frame_acquisition','build_candidates','model_arguments']
