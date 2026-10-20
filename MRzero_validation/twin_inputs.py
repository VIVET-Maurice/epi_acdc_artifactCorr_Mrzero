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


def read_twin_inputs(folder, *, scan_sequence_path=None):
    """Read native inputs; optionally select the EPI .seq explicitly.

    Relative scan_sequence_path values are resolved under folder. Without an
    override, exactly one sequences/*epi*.seq must exist. Twix/sequence matrix
    validation remains active; changing a filename does not resample a scan.
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
            if scan_sequence_path is not None:
                selected = Path(scan_sequence_path)
                selected = selected if selected.is_absolute() else folder / selected
                if selected.resolve() != Path(result['paths']['scan_sequence']).resolve():
                    raise ValueError("Synthetic acquisition sequence is fixed by its manifest; the requested .seq differs.")
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
    paths = dict(reference=unique('*reference.dat'), b0=unique('*b0_map.dat'),
        scan=unique('*scan.dat'), scan_sequence=scan_sequence_path,
        reference_sequence=unique('sequences/gre_32*.seq'),
        b0_sequence=folder/'sequences/b0_map.seq')
    acquisition, timing = read_sequence(paths['scan_sequence'])
    reference_acquisition, _ = read_sequence(paths['reference_sequence'])
    gre, gd = read_gre(paths['reference'], reference_acquisition, return_intermediates=True)
    gre['magnitude'], _ = reconstruct_receivers(gd, reference_acquisition)
    scan, scan_metadata, sd = read_scan(paths['scan'], acquisition, return_intermediates=True)
    scan, _ = reconstruct_receivers(sd, acquisition)
    raw = read_b0_coil_images(paths['b0'], paths['b0_sequence'])
    reconstruct_b0_echoes(raw, paths['b0_sequence'])
    echo_times = np.asarray(raw['metadata']['TE_s'])
    field = b0_from_coil_images(raw['coil_images'], echo_times, return_intermediates=True)
    field.update(X_m=raw['X_m'], Y_m=raw['Y_m'])
    return dict(paths=paths, acquisition=acquisition, timing=timing, gre=gre,
        scan=scan, scan_metadata=scan_metadata, scan_diagnostics=sd, b0_raw=raw,
        field=field, echo_times_s=echo_times, manifest=manifest,
        source_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest()
                       for name,path in paths.items()})


def prepare_echo_object(inputs, threshold_fraction=.04, background_corner_m=.085, *,
                        allow_uncalibrated_noise=True):
    """Fit native echoes, with an explicit fallback if corner noise is unavailable.

    Keep the declared corner region: never replace it with all non-object pixels.
    Without enough finite noise samples (or with zero measured noise variance),
    retain first-echo amplitude and mark additional decay as uncalibrated/zero.
    Set allow_uncalibrated_noise=False to require a valid noise calibration.
    """
    raw, field = inputs['b0_raw'], inputs['field']
    rss = np.sqrt(np.sum(abs(raw['coil_images'])**2, axis=1))
    support = ndimage.binary_fill_holes(rss[0] > threshold_fraction*rss[0].max())
    # Explicit corner patches, inspected in the notebook; never treat ROI
    # complement as noise because it may contain omitted object signal.
    noise_mask = ((abs(raw['X_m']) > background_corner_m)
                  & (abs(raw['Y_m']) > background_corner_m) & ~support)
    # Count exactly the finite complex receiver vectors used by the fitter.
    coils = np.asarray(raw['coil_images'])
    receivers = coils.shape[1]
    samples = coils[:, :, noise_mask].transpose(1, 0, 2).reshape(receivers, -1)
    samples = samples[:, np.isfinite(samples).all(axis=0)]
    sample_count = samples.shape[1]
    minimum_samples = max(32, 2 * receivers)
    noise_fallback_reason = None
    if sample_count < minimum_samples:
        noise_fallback_reason = f"Only {sample_count} finite background samples; need {minimum_samples}."
    elif np.var(samples, axis=1, ddof=1).real.sum() <= 0:
        noise_fallback_reason = "Background samples have zero receiver-noise variance."
    if noise_fallback_reason is not None and not allow_uncalibrated_noise:
        raise ValueError(noise_fallback_reason)
    interior = ndimage.binary_erosion(support, iterations=6)
    fit = fit_echo_object(coils, inputs['echo_times_s'],
        support_mask=support, noise_mask=noise_mask if noise_fallback_reason is None else None,
        target_te_s=inputs['acquisition']['TE_s'], pooled_mask=interior)
    fit['report']['noise'].update(
        available_sample_count=int(sample_count), minimum_sample_count=minimum_samples,
        requested_background_voxels=int(noise_mask.sum()),
        background_corner_m=float(background_corner_m), fallback_reason=noise_fallback_reason)
    valid = support & field['mask'] & np.isfinite(field['b0Hz'])
    fit['object_support'] = support
    fit['simulation_support'] = valid
    fit['noise_mask'] = noise_mask
    fit['report'].update(object_threshold_fraction=threshold_fraction,
        object_voxels=int(support.sum()), simulated_voxels=int(valid.sum()),
        excluded_missing_B0_voxels=int((support & ~valid).sum()),
        excluded_missing_B0_amplitude_fraction=float(rss[0][support & ~valid].sum()/rss[0][support].sum()),
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
