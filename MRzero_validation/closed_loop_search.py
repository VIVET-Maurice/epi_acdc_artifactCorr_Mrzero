"""Small-data alpha search driven by experimental costs, without scanner I/O.

The empirical discrepancy is affine between the initial 0/1 anchors, then
piecewise linear through every measured anchor. This introduces no oscillatory
polynomial and never extrapolates outside [0, 1]. It does not identify a physical
MRzero correction, estimate experimental uncertainty, or guarantee improvement.

Waveform fingerprints must describe commands AFTER the hardware quantization.
Hardware feasibility is supplied by the caller; this module does not certify it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np


def _nonnegative_scalar(value, name):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isrealobj(value) or not np.isfinite(value) or value < 0):
        raise ValueError(f'{name} must be a finite nonnegative scalar.')
    return float(value)


def quantized_waveform_fingerprint(commands, *, timing_s=None, channel_order=None):
    """Hash exact numeric commands already quantized by the controller pipeline.

    Identical numeric values have the same hash across integer/float dtypes.
    This function does not quantize or round currents. Include timing/channel
    metadata whenever it can differ between candidates. The caller must use the
    same fingerprint convention for measured and proposed waveforms.
    """
    values = np.asarray(commands)
    if (values.ndim != 2 or min(values.shape) == 0 or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all()):
        raise ValueError('commands must be a finite real 2D quantized command array.')
    canonical = values.astype('<f8', copy=True)
    canonical[canonical == 0] = 0.  # normalize signed zero only, without rounding
    digest = hashlib.sha256()
    digest.update(b'acdc-quantized-command-v1\0')
    digest.update(json.dumps(list(canonical.shape)).encode())
    digest.update(canonical.tobytes(order='C'))
    if timing_s is not None:
        times = np.asarray(timing_s)
        if (times.ndim != 1 or times.size == 0 or times.dtype.kind not in 'fiu'
                or not np.isfinite(times).all()):
            raise ValueError('timing_s must be a finite real 1D array.')
        times = times.astype('<f8', copy=True)
        times[times == 0] = 0.
        digest.update(b'\0timing_s\0')
        digest.update(json.dumps(list(times.shape)).encode())
        digest.update(times.tobytes(order='C'))
    if channel_order is not None:
        order = list(channel_order)
        if len(order) != len(set(map(str, order))):
            raise ValueError('channel_order must contain unique channel identifiers.')
        digest.update(b'\0channel_order\0')
        digest.update(json.dumps(list(map(str, order)), separators=(',', ':')).encode())
    return digest.hexdigest()


def _measured_records(measured_points):
    if not isinstance(measured_points, Sequence) or isinstance(measured_points, (str, bytes)):
        raise ValueError('measured_points must be a sequence of acquisition dictionaries.')
    if not 2 <= len(measured_points) <= 4:
        raise ValueError('Require the two initial anchors and zero to two additional measured scans.')
    records = []
    for point in measured_points:
        if not isinstance(point, Mapping):
            raise ValueError('Each measured point must be a dictionary.')
        if any(name not in point for name in ('id', 'alpha', 'experimental_J', 'fingerprint')):
            raise ValueError('Each measured point requires id, alpha, experimental_J, fingerprint.')
        alpha = _nonnegative_scalar(point['alpha'], 'measured alpha')
        if alpha > 1:
            raise ValueError('Measured alphas must remain in [0, 1].')
        cost = _nonnegative_scalar(point['experimental_J'], 'experimental_J')
        if (not isinstance(point['id'], str) or not point['id'].strip()
                or not isinstance(point['fingerprint'], str) or not point['fingerprint'].strip()):
            raise ValueError('Measured ids and waveform fingerprints must be nonempty strings.')
        record = dict(point)
        record.update(alpha=alpha, experimental_J=cost,
                      experimental_NRMSE_percent=100*float(np.sqrt(cost)))
        records.append(record)
    alphas = [point['alpha'] for point in records]
    if alphas.count(0.) != 1 or alphas.count(1.) != 1:
        raise ValueError('Exactly one experimental anchor at alpha=0 and alpha=1 is required.')
    if len(set(alphas)) != len(alphas):
        raise ValueError('Measured alpha values must be distinct; repeats need explicit aggregation upstream.')
    if len({point['id'] for point in records}) != len(records):
        raise ValueError('Measured acquisition ids must be unique.')
    return records


def select_measured_best(measured_points, *, incumbent_id=None, uncertainty_floor_pp=.5):
    """Separate the nominal experimental minimum from a conservative incumbent.

    The floor is a user-selected practical tolerance in NRMSE percentage points,
    not an estimated confidence interval. Default incumbent is the better of the
    historical alpha=0/1 scans. Only experimental J can change either selection.
    """
    records = _measured_records(measured_points)
    floor = _nonnegative_scalar(uncertainty_floor_pp, 'uncertainty_floor_pp')
    nominal = min(records, key=lambda item: item['experimental_J'])
    if incumbent_id is None:
        incumbent = min((item for item in records if item['alpha'] in (0., 1.)),
                        key=lambda item: item['experimental_J'])
    else:
        matches = [item for item in records if item['id'] == incumbent_id]
        if not matches:
            raise ValueError('incumbent_id must identify one of the measured acquisitions.')
        incumbent = matches[0]
    gain = incumbent['experimental_NRMSE_percent'] - nominal['experimental_NRMSE_percent']
    robust = nominal['id'] != incumbent['id'] and gain > 0 and gain >= floor
    retained = nominal if robust else incumbent
    return {'nominal_best': nominal, 'retained_best': retained, 'previous_incumbent': incumbent,
            'nominal_improvement_over_incumbent_pp': float(gain),
            'robust_improvement': bool(robust), 'uncertainty_floor_pp': floor,
            'uncertainty_policy': 'Configured practical tolerance; no repeat-scan uncertainty was estimated.',
            'selection_uses_experimental_cost_only': True}


def propose_closed_loop_candidate(alpha_grid, simulated_J, measured_points, *,
                                 waveform_fingerprints, admissible=None,
                                 max_additional_scans=2, trust_radius=.25,
                                 min_predicted_gain_pp=.5, uncertainty_floor_pp=.5,
                                 allow_exploration=False, incumbent_id=None):
    """Calibrate a quantized alpha bank and optionally propose one next scan.

    Arrays share a strictly increasing alpha grid containing 0, 1 and every
    measured alpha. simulated_J contains relative MSE; an inadmissible candidate
    may have NaN simulated cost. No negative simulated or experimental J is
    accepted. Negative *calibrated* J remains flagged, never square-rooted, and
    can only be proposed as explicitly requested exploration without a predicted
    NRMSE. Exploratory selection maximizes distance from existing alpha anchors
    locally, rather than rewarding an impossible negative predicted cost.

    Trust is centered on the nominal best measured alpha; the separately
    returned conservative incumbent may remain unchanged for a tiny gain.
    Fingerprint deduplication compares actual quantized commands, not alpha.
    """
    records = _measured_records(measured_points)
    selection = select_measured_best(records, incumbent_id=incumbent_id,
                                     uncertainty_floor_pp=uncertainty_floor_pp)
    if (isinstance(max_additional_scans, (bool, np.bool_))
            or not isinstance(max_additional_scans, (int, np.integer))
            or not 0 <= max_additional_scans <= 2):
        raise ValueError('max_additional_scans must be an integer from zero through two.')
    if not isinstance(allow_exploration, (bool, np.bool_)):
        raise ValueError('allow_exploration must be an explicit boolean.')
    radius = _nonnegative_scalar(trust_radius, 'trust_radius')
    if radius > 1:
        raise ValueError('trust_radius must not exceed one alpha unit.')
    minimum_gain = _nonnegative_scalar(min_predicted_gain_pp, 'min_predicted_gain_pp')
    alphas = np.asarray(alpha_grid)
    costs = np.asarray(simulated_J)
    if (alphas.ndim != 1 or alphas.size < 2 or alphas.dtype.kind not in 'fiu'
            or not np.isfinite(alphas).all() or np.any(np.diff(alphas) <= 0)
            or alphas[0] != 0 or alphas[-1] != 1):
        raise ValueError('alpha_grid must increase strictly from exactly zero to exactly one.')
    alphas = alphas.astype(float)
    if costs.shape != alphas.shape or costs.dtype.kind not in 'fiu':
        raise ValueError('simulated_J must be a real array matching alpha_grid.')
    costs = costs.astype(float)
    if np.isinf(costs).any() or np.any(costs[np.isfinite(costs)] < 0):
        raise ValueError('simulated_J cannot contain infinity or negative costs.')
    if admissible is None:
        feasible = np.ones(alphas.size, bool)
    else:
        feasible = np.asarray(admissible)
        if feasible.shape != alphas.shape or not np.isin(feasible, [0, 1]).all():
            raise ValueError('admissible must be a binary mask matching alpha_grid.')
        feasible = feasible.astype(bool)
    if not np.isfinite(costs[feasible]).all():
        raise ValueError('Every admissible candidate requires a finite simulated_J.')
    fingerprints = list(waveform_fingerprints)
    if (len(fingerprints) != alphas.size or any(not isinstance(item, str) or not item.strip()
                                               for item in fingerprints)):
        raise ValueError('waveform_fingerprints must supply one nonempty string per alpha.')
    indices = []
    for point in records:
        matching = np.flatnonzero(np.isclose(alphas, point['alpha'], rtol=0, atol=1e-12))
        if matching.size != 1 or not np.isfinite(costs[matching[0]]):
            raise ValueError('Every measured alpha requires one explicitly simulated grid point.')
        index = int(matching[0])
        if fingerprints[index] != point['fingerprint']:
            raise ValueError(f"Measured/grid waveform fingerprints disagree at alpha={point['alpha']}.")
        indices.append(index)
    order = np.argsort([point['alpha'] for point in records])
    knots = np.array([records[i]['alpha'] for i in order])
    experimental = np.array([records[i]['experimental_J'] for i in order])
    anchor_simulated = np.array([costs[indices[i]] for i in order])
    discrepancies = experimental-anchor_simulated
    calibrated = costs + np.interp(alphas, knots, discrepancies)
    # Preserve observed values exactly, including an experimental cost of zero.
    calibrated[np.asarray(indices)[order]] = experimental
    valid = np.isfinite(calibrated) & (calibrated >= 0)
    predicted_nrmse = np.full(alphas.shape, np.nan)
    predicted_nrmse[valid] = 100*np.sqrt(calibrated[valid])
    simulated_nrmse = np.full(alphas.shape, np.nan)
    finite_simulated = np.isfinite(costs)
    simulated_nrmse[finite_simulated] = 100*np.sqrt(costs[finite_simulated])
    center = selection['nominal_best']['alpha']
    measured_fingerprints = {point['fingerprint'] for point in records}
    already_measured = np.array([item in measured_fingerprints for item in fingerprints])
    inside_trust = abs(alphas-center) <= radius+1e-12
    available = feasible & finite_simulated & np.isfinite(calibrated) & inside_trust & ~already_measured
    duplicates = np.zeros(alphas.size, bool)
    kept = {}
    for index in sorted(np.flatnonzero(available), key=lambda i: (abs(alphas[i]-center), alphas[i])):
        fingerprint = fingerprints[index]
        if fingerprint in kept:
            duplicates[index] = True
        else:
            kept[fingerprint] = int(index)
    available &= ~duplicates
    ordinary = available & valid
    best_nrmse = selection['nominal_best']['experimental_NRMSE_percent']
    predicted_gain = best_nrmse-predicted_nrmse
    promising = ordinary & (predicted_gain > 0) & (predicted_gain >= minimum_gain)
    count_new = len(records)-2
    remaining = max(0, int(max_additional_scans)-count_new)
    candidate = None
    if remaining == 0:
        stop_reason = 'additional_scan_budget_exhausted'
    elif np.any(promising):
        index = min(np.flatnonzero(promising), key=lambda i: (calibrated[i], abs(alphas[i]-center), alphas[i]))
        stop_reason = 'promising_local_candidate'
        candidate = {'mode': 'predicted_improvement', 'exploratory': False, 'index': int(index)}
    elif allow_exploration and np.any(available):
        distances = np.min(abs(alphas[:, None]-knots[None, :]), axis=1)
        index = min(np.flatnonzero(available), key=lambda i: (-distances[i], abs(alphas[i]-center), alphas[i]))
        stop_reason = 'explicit_exploration_without_promised_gain'
        candidate = {'mode': 'explicit_exploration', 'exploratory': True, 'index': int(index)}
    elif not np.any(available):
        stop_reason = 'no_unmeasured_local_waveform'
    elif not np.any(ordinary):
        stop_reason = 'no_nonnegative_local_prediction'
    else:
        stop_reason = 'no_promising_local_candidate'
    if candidate is not None:
        i = candidate['index']
        candidate.update(alpha=float(alphas[i]), fingerprint=fingerprints[i],
            simulated_J=float(costs[i]), simulated_NRMSE_percent=float(simulated_nrmse[i]),
            calibrated_J=float(calibrated[i]), calibrated_cost_valid=bool(valid[i]),
            predicted_NRMSE_percent=float(predicted_nrmse[i]) if valid[i] else None,
            predicted_gain_pp=float(predicted_gain[i]) if valid[i] else None,
            best_experimental_NRMSE_percent=float(best_nrmse),
            hardware_admissible_as_supplied=bool(feasible[i]), improvement_guaranteed=False)
    return {'alpha_grid': alphas, 'simulated_J': costs, 'calibrated_J': calibrated,
        'simulated_NRMSE_percent': simulated_nrmse, 'predicted_NRMSE_percent': predicted_nrmse,
        'valid_calibrated_mask': valid, 'negative_calibrated_mask': np.isfinite(calibrated) & (calibrated < 0),
        'ordinary_eligible_mask': ordinary, 'promising_mask': promising,
        'exploration_eligible_mask': available, 'already_measured_mask': already_measured,
        'duplicate_quantized_waveform_mask': duplicates, 'admissible_mask': feasible,
        'candidate': candidate, 'stop_reason': stop_reason, 'measured_selection': selection,
        'budget': {'maximum_additional_scans': int(max_additional_scans),
                   'completed_additional_scans': count_new, 'remaining_additional_scans': remaining},
        'trust_region': {'center_alpha': float(center), 'radius': radius,
                         'lower': max(0., center-radius), 'upper': min(1., center+radius),
                         'center_policy': 'nominal best experimental alpha'},
        'calibration': {'method': 'affine anchored discrepancy' if len(records) == 2 else 'piecewise linear anchored discrepancy',
                        'alpha_knots': knots, 'experimental_J_knots': experimental,
                        'simulated_J_knots': anchor_simulated, 'discrepancy_knots': discrepancies,
                        'extrapolation_allowed': False, 'physical_correction_identified': False},
        'policy': {'min_predicted_gain_pp': minimum_gain, 'explicit_exploration_enabled': bool(allow_exploration),
                   'candidate_is_prediction_only': True, 'hardware_checks_performed_by_caller': True,
                   'negative_calibrated_cost_is_not_zero_clipped': True,
                   'deduplication_uses_quantized_waveform_fingerprints': True}}


__all__ = ['propose_closed_loop_candidate', 'select_measured_best', 'quantized_waveform_fingerprint']
