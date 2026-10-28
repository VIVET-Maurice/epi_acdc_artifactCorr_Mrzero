"""Describe BOTH signs of EPI magnitude mismatch with a supplied, frozen gain.

A positive residual means the simulation is brighter than the measurement;
a negative residual means the simulation is darker. Neither sign identifies
the physical cause. Full-FOV scores include measured zero-signal pixels.
Optional reference envelopes are diagnostic regions, not ground-truth masks.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _magnitude(value, name, *, measured=False):
    array = np.asarray(value)
    if (array.ndim != 2 or min(array.shape) < 1 or not np.issubdtype(array.dtype, np.number)
            or not np.isfinite(array).all() or (measured and np.iscomplexobj(array))):
        raise ValueError(f'{name} must be a finite numeric 2D image; measured RSS must be real.')
    if not np.iscomplexobj(array) and np.any(array < 0):
        raise ValueError(f'{name} magnitude must be nonnegative.')
    return np.abs(array).astype(float)


def _mask(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or not np.isin(array, [0, 1]).all():
        raise ValueError(f'{name} must be a binary mask matching the images.')
    return array.astype(bool)


def fixed_comparison_envelope(reference_image, threshold_fraction=.1):
    """Fill enclosed holes in a thresholded fixed reference magnitude.

    This region is independent of the measured/simulated EPI brightness only
    when the caller's reference is independent. A zero-B0 simulation reference
    still inherits object coverage and modeling assumptions; it is not truth.
    """
    reference = _magnitude(reference_image, 'reference_image')
    if (isinstance(threshold_fraction, (bool, np.bool_))
            or not np.isscalar(threshold_fraction) or not np.isfinite(threshold_fraction)
            or not 0 <= threshold_fraction < 1):
        raise ValueError('threshold_fraction must be in [0, 1).')
    return ndimage.binary_fill_holes(reference > float(threshold_fraction) * reference.max())


def signal_disagreement(measured, simulated, *, gain, analysis_mask=None,
                        subsets=None, reference_scale=None, bright_fraction=.2,
                        ratios=(.25, .5)):
    """Return ``{'report': ..., 'arrays': ...}`` without fitting any quantity.

    ``gain`` scales the simulated magnitude and is supplied by the caller.
    With a measured brightness threshold T, opposite ratio flags are
    S<q*M and M>=T, versus M<q*S and S>=T. Thus dark measured pixels remain
    eligible whenever the simulation is bright. Counts are descriptive flags,
    not labels of anatomical holes or proven acquisition artifacts.

    Full-FOV residual scores are always reported. Array masks and optional
    subset scores use ``analysis_mask`` (default all pixels). Reference scale
    defaults to max(measured); supplied scales must be finite nonnegative.
    Zero measured energy gives null normalized errors, retaining absolute SSE.
    """
    measured = _magnitude(measured, 'measured', measured=True)
    simulated = _magnitude(simulated, 'simulated')
    if simulated.shape != measured.shape:
        raise ValueError('Measured and simulated images must have matching shapes.')
    if (isinstance(gain, (bool, np.bool_)) or not np.isscalar(gain)
            or not np.isrealobj(gain) or not np.isfinite(gain) or gain <= 0):
        raise ValueError('gain must be a supplied positive finite scalar.')
    if (isinstance(bright_fraction, (bool, np.bool_)) or not np.isscalar(bright_fraction)
            or not np.isfinite(bright_fraction) or not 0 < bright_fraction < 1):
        raise ValueError('bright_fraction must be between zero and one.')
    scale = float(measured.max()) if reference_scale is None else reference_scale
    if (isinstance(scale, (bool, np.bool_)) or not np.isscalar(scale)
            or not np.isrealobj(scale) or not np.isfinite(scale) or scale < 0):
        raise ValueError('reference_scale must be a finite nonnegative scalar.')
    values = np.asarray(ratios)
    if (values.ndim != 1 or values.size < 1 or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or np.any(values <= 0) or np.any(values >= 1)
            or np.unique(values).size != values.size):
        raise ValueError('ratios must contain distinct numeric values between zero and one.')
    full = np.ones(measured.shape, bool)
    analysis = full.copy() if analysis_mask is None else _mask(analysis_mask, measured.shape, 'analysis_mask')
    if subsets is None:
        subsets = {}
    if not isinstance(subsets, dict):
        raise ValueError('subsets must be a dictionary of named binary masks.')
    selected_subsets = {str(name): _mask(mask, measured.shape, f'subsets[{name}]') & analysis
                        for name, mask in subsets.items()}
    scaled = float(gain) * simulated
    if not np.isfinite(scaled).all():
        raise ValueError('Scaled simulated magnitude overflows.')
    difference = scaled - measured
    threshold = float(bright_fraction) * float(scale)
    measured_bright, simulated_bright = measured >= threshold, scaled >= threshold
    both_low = (measured <= threshold) & (scaled <= threshold)
    direction_masks = {}
    arrays = {'measured_magnitude': measured, 'simulated_scaled_magnitude': scaled,
              'difference': difference, 'analysis_mask': analysis,
              'both_low_mask': both_low & analysis}
    for ratio in values.astype(float):
        label = format(100 * ratio, '.12g').replace('.', 'p')
        under = (scaled < ratio * measured) & measured_bright
        over = (measured < ratio * scaled) & simulated_bright
        direction_masks[label] = (ratio, under, over)
        arrays[f'simulated_low_measured_high_{label}_mask'] = under & analysis
        arrays[f'measured_low_simulated_high_{label}_mask'] = over & analysis

    def score(mask):
        error = difference[mask]
        energy = float(np.sum(measured[mask]**2))
        over_sse = float(np.sum(np.maximum(error, 0)**2))
        under_sse = float(np.sum(np.minimum(error, 0)**2))
        sse = over_sse + under_sse
        low_sse = float(np.sum(difference[mask & both_low]**2))
        count = int(mask.sum())
        result = {'pixels': count, 'measured_energy': energy, 'SSE': sse,
            'normalized_SSE': sse / energy if energy > 0 else None,
            'NRMSE_percent': 100 * float(np.sqrt(sse / energy)) if energy > 0 else None,
            'overprediction_SSE': over_sse, 'underprediction_SSE': under_sse,
            'overprediction_normalized_SSE': over_sse / energy if energy > 0 else None,
            'underprediction_normalized_SSE': under_sse / energy if energy > 0 else None,
            'overprediction_fraction_of_SSE': over_sse / sse if sse > 0 else 0.,
            'underprediction_fraction_of_SSE': under_sse / sse if sse > 0 else 0.,
            'measured_zero_pixels': int(np.count_nonzero(mask & (measured == 0))),
            'both_low_pixels': int(np.count_nonzero(mask & both_low)),
            'both_low_SSE': low_sse, 'both_low_fraction_of_SSE': low_sse / sse if sse > 0 else 0.,
            'ratios': {}}
        for label, (ratio, under, over) in direction_masks.items():
            result['ratios'][label] = {'ratio': ratio,
                'simulated_low_measured_high_pixels': int(np.count_nonzero(mask & under)),
                'measured_low_simulated_high_pixels': int(np.count_nonzero(mask & over)),
                'simulated_low_measured_high_fraction': float(np.count_nonzero(mask & under) / count) if count else None,
                'measured_low_simulated_high_fraction': float(np.count_nonzero(mask & over) / count) if count else None}
        return result

    report = {'gain': float(gain), 'gain_fitted': False, 'reference_scale': float(scale),
        'reference_scale_source': 'measured maximum' if reference_scale is None else 'supplied fixed scale',
        'bright_fraction': float(bright_fraction), 'brightness_threshold': threshold,
        'full_FOV': score(full), 'analysis_mask': score(analysis),
        'subsets': {name: score(mask) for name, mask in selected_subsets.items()},
        'policy': {'signed_residual': 'scaled simulated magnitude minus measured magnitude',
            'overprediction': 'simulation brighter than measurement; includes measured dark/zero pixels',
            'underprediction': 'simulation darker than measurement; the previous one-sided hole count',
            'full_FOV_is_primary': True, 'measured_zero_pixels_excluded': False,
            'subset_masks_intersect_analysis_mask': True,
            'array_masks_use_analysis_mask': True,
            'both_low': 'both magnitudes at or below the fixed brightness threshold',
            'both_low_is_not_agreement': True,
            'causal_interpretation': 'Descriptive mismatch only; no physical cause or ground truth inferred.',
            'reference_envelope': 'Optional diagnostic region; not an independent ground-truth mask.'}}
    return {'report': report, 'arrays': arrays}


__all__ = ['signal_disagreement', 'fixed_comparison_envelope']
