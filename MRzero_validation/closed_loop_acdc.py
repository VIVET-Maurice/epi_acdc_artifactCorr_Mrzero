"""Persistent manual ACDC feedback loop on a fixed, quantized alpha bank.

Experimental costs use one GRE, one ROI and one baseline-derived intensity
normalization for the entire session. MRzero magnitude is calibrated in scale
against the unshimmed EPI once. The empirical correction acts on the scalar
cost, never on measured k-space, object density or the physical forward model.
No scanner control is performed. CSVs use the original hardware export path.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from closed_loop_search import propose_closed_loop_candidate
from measured_scan_comparison import (compare_measured_scan_set, _check_scan,
    _check_patient_geometry, _scan_grid, _registered_image)
from measured_continuous import reconstruct_receivers
from prepare_acdc_inputs import read_scan

SCHEMA = 'acdc-manual-closed-loop-v1'
SEARCH_KEYS = ('min_predicted_gain_pp', 'uncertainty_floor_pp', 'trust_radius', 'allow_exploration')
DEFAULTS = dict(alpha_min=0., alpha_max=1., alpha_points=21, max_additional_scans=2,
    min_predicted_gain_pp=.5, uncertainty_floor_pp=.5, trust_radius=.25,
    allow_exploration=False, navigator_phase_branch=0, reference_threshold_fraction=.1,
    previous_scan=None, previous_csv=None, previous_npz=None, previous_metadata=None,
    association_confirmed=False, data_kind='measured', waveform_output_dir=None)


def _json(value):
    if isinstance(value, dict): return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json(v) for v in value]
    if isinstance(value, np.ndarray): return _json(value.tolist())
    if isinstance(value, np.generic): return _json(value.item())
    if isinstance(value, Path): return str(value.resolve())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def _canonical(value):
    return json.dumps(_json(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def _file(path, *, suffix=None):
    if path is None: raise ValueError('Un chemin de fichier explicite est requis.')
    path = Path(path).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    if suffix and path.suffix.lower() != suffix: raise ValueError(f'Expected {suffix}: {path}')
    return {'path': str(path), 'sha256': sha256(path)}


def _check_file(record):
    if sha256(record['path']) != record['sha256']:
        raise ValueError(f"Fichier modifié depuis son association à la session : {record['path']}")


def twix_data_kind(path):
    """Read protocol headers, including synthetic exports without a manifest."""
    import twixtools
    with redirect_stdout(io.StringIO()):
        scans = twixtools.read_twix(str(path), parse_data=False, verbose=False)
    if not scans or not isinstance(scans[-1].get('hdr'), dict):
        raise ValueError(f'Missing Twix protocol headers: {path}')
    header = scans[-1]['hdr']
    protocol = str(header.get('MeasYaps', {}).get('tProtocolName', ''))
    description = str(header.get('Config', {}).get('SequenceDescription', ''))
    return 'synthetic' if 'MRZERO_SYNTHETIC' in (protocol+' '+description).upper() else 'measured'


def _require_data_kind(path, kind):
    actual = twix_data_kind(path)
    if actual != kind:
        raise ValueError(f'Data origin mismatch: {path} is {actual}, session is {kind}. '
                         'A copied real scan cannot calibrate a synthetic baseline (or conversely).')


def relative_image_cost(image, reference, mask):
    image, reference, mask = np.asarray(image), np.asarray(reference), np.asarray(mask)
    if (image.shape != reference.shape or mask.shape != reference.shape or image.ndim != 2
            or not np.isin(mask, [0, 1]).all() or not mask.any()
            or np.iscomplexobj(image) or np.iscomplexobj(reference)):
        raise ValueError('Real magnitude images and a nonempty binary common ROI are required.')
    selected = mask.astype(bool)
    if (not np.isfinite(image[selected]).all() or not np.isfinite(reference[selected]).all()
            or (image[selected] < 0).any() or (reference[selected] < 0).any()):
        raise ValueError('Nonnegative finite magnitudes are required on the frozen ROI.')
    energy = float(np.sum(reference[selected]**2))
    if energy <= 0: raise ValueError('The GRE reference has no energy on the frozen ROI.')
    cost = float(np.sum((image[selected]-reference[selected])**2)/energy)
    return {'J': cost, 'NRMSE_percent': 100*float(np.sqrt(cost)), 'ROI_pixels': int(selected.sum())}


def _config(configuration):
    config = dict(DEFAULTS, **(configuration or {}))
    # Initial experimental anchors define this deliberately limited search.
    if config['alpha_min'] != 0 or config['alpha_max'] != 1:
        raise ValueError('Cette première boucle reste strictement dans 0 <= alpha <= 1.')
    for name, low, high in (('alpha_points', 3, 1001), ('max_additional_scans', 0, 2)):
        value = config[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in [{low}, {high}].')
    for name in ('association_confirmed', 'allow_exploration'):
        if not isinstance(config[name], (bool, np.bool_)): raise ValueError(f'{name} must be an explicit boolean.')
    for name in ('min_predicted_gain_pp', 'uncertainty_floor_pp', 'trust_radius', 'reference_threshold_fraction'):
        value = config[name]
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative.')
    if config['trust_radius'] > 1 or config['reference_threshold_fraction'] >= 1:
        raise ValueError('Trust radius must be <= 1; GRE threshold must be < 1.')
    if config['navigator_phase_branch'] not in (-1, 0, 1): raise ValueError('Invalid navigator phase branch.')
    if config['data_kind'] not in ('measured', 'synthetic'):
        raise ValueError('data_kind must explicitly be measured or synthetic.')
    for name in ('previous_scan', 'previous_csv', 'previous_npz', 'previous_metadata'):
        if config[name] is not None: config[name] = str(Path(config[name]).expanduser().resolve())
    if config['waveform_output_dir'] is not None:
        output = config['waveform_output_dir']
        if not isinstance(output, (str, os.PathLike)) or not str(output).strip():
            raise ValueError('waveform_output_dir must be a nonempty directory path or None.')
        output = Path(output).expanduser().resolve()
        if output.exists() and not output.is_dir():
            raise ValueError('waveform_output_dir must name a directory, not a file.')
        config['waveform_output_dir'] = str(output)
    return _json(config)


@contextmanager
def _session_lock(folder):
    folder = Path(folder).expanduser().resolve(); folder.mkdir(parents=True, exist_ok=True)
    path = folder/'.writing.lock'
    try: descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f'Session occupée : {path}. Après un arrêt brutal, vérifier qu’aucun calcul ne tourne avant de supprimer ce verrou.') from error
    try:
        os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
        yield folder
    finally: path.unlink(missing_ok=True)


def _write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, suffix='.tmp', delete=False, encoding='utf-8') as stream:
        temporary = Path(stream.name); stream.write(json.dumps(_json(value), indent=2, ensure_ascii=False, allow_nan=False))
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save(state):
    folder = Path(state['session_dir'])
    revision = int(state['manifest'].get('revision', -1))+1
    arrays_path = folder/f'state_{revision:04d}.npz'
    manifest_path = folder/f'state_{revision:04d}.json'
    if arrays_path.exists() or manifest_path.exists():
        raise FileExistsError('Incomplete state revision exists; preserve it and recover the preceding session before proceeding.')
    with tempfile.NamedTemporaryFile('wb', dir=folder, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name); np.savez_compressed(stream, **state['arrays'])
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, arrays_path)
    state['manifest'].update(revision=revision, updated_UTC=datetime.now(timezone.utc).isoformat(),
        arrays_file=arrays_path.name, arrays_sha256=sha256(arrays_path))
    _write_json(manifest_path, state['manifest'])
    _write_json(folder/'session.json', {'schema': SCHEMA, 'state_file': manifest_path.name,
                                      'state_sha256': sha256(manifest_path)})


def load_closed_loop(session_dir, *, verify_sources=True):
    """Resume without GRE/B0 preparation, MRzero construction or new simulation."""
    folder = Path(session_dir).expanduser().resolve()
    pointer = json.loads((folder/'session.json').read_text())
    if pointer['schema'] != SCHEMA: raise ValueError('Unsupported closed-loop session schema.')
    path = folder/pointer['state_file']
    if sha256(path) != pointer['state_sha256']: raise ValueError('Session metadata checksum mismatch.')
    manifest = json.loads(path.read_text())
    arrays_path = folder/manifest['arrays_file']
    if sha256(arrays_path) != manifest['arrays_sha256']: raise ValueError('Session numerical archive checksum mismatch.')
    with np.load(arrays_path, allow_pickle=False) as archive: arrays = {key: archive[key] for key in archive.files}
    if verify_sources:
        for record in manifest['sources'].values(): _check_file(record)
        _check_file(manifest['command_context'])
        for proposal in manifest['proposals']:
            for record in proposal['export_files'].values(): _check_file(record)
        for point in manifest['measured_points']: _check_file(point['scan'])
    return {'session_dir': str(folder), 'manifest': manifest, 'arrays': arrays}


def _resume_config(state, configuration):
    if configuration is None: return
    current = _config(configuration)
    saved = state['manifest']['configuration']
    # Omitted historical paths are restored from the persistent session.
    for key, value in current.items():
        if key in SEARCH_KEYS or key == 'association_confirmed': continue
        if key.startswith('previous_') and value is None: continue
        if _canonical(value) != _canonical(saved.get(key)):
            raise ValueError(f'Configuration de session modifiée : {key}. Conserver cette session ou choisir un nouveau dossier explicitement.')
    for key in SEARCH_KEYS: saved[key] = current[key]


def _check_epi_metadata(reference_metadata, metadata):
    _check_patient_geometry(reference_metadata, metadata, 'Feedback EPI')
    for name, tolerance in (('slice_thickness_mm', 1e-3), ('header_flip_deg', 1e-3)):
        if name not in reference_metadata or name not in metadata or not np.isclose(reference_metadata[name], metadata[name], rtol=0, atol=tolerance):
            raise ValueError(f'EPI geometry/acquisition mismatch: {name}.')
    if metadata.get('receiver_count') != reference_metadata.get('receiver_count'):
        raise ValueError('Receiver count changed: the frozen RSS gain is not transferable.')


def _read_feedback_image(state, path):
    manifest, arrays = state['manifest'], state['arrays']
    acquisition, timing = manifest['acquisition'], manifest['timing']
    _, metadata, diagnostics = read_scan(Path(path), acquisition, return_intermediates=True,
        navigator_phase_branch=manifest['configuration']['navigator_phase_branch'])
    _check_scan(metadata, acquisition, timing)
    _check_epi_metadata(manifest['baseline_metadata'], metadata)
    image, reconstruction = reconstruct_receivers(diagnostics, acquisition)
    X, Y = _scan_grid(image, metadata, acquisition)
    registered, valid = _registered_image({'image': image, 'X_m': X, 'Y_m': Y},
        arrays['native_X_m'], arrays['native_Y_m'])
    if not np.all(valid[arrays['mask']]):
        raise ValueError('The new scan does not cover the entire frozen ROI; it cannot change the comparison mask.')
    scaled = manifest['normalization']['experimental_gain']*registered/manifest['normalization']['reference_scale']
    return scaled, {'metadata': metadata, 'reconstruction': reconstruction,
        'navigator_fit': diagnostics.get('navigator_fit'), 'shared_gain_refitted': False,
        'ROI_refitted': False, 'raw_complex_comparison_performed': False}


def _search(state):
    m, a = state['manifest'], state['arrays']; config = m['configuration']
    result = propose_closed_loop_candidate(a['alpha_grid'], a['simulated_J'], m['measured_points'],
        waveform_fingerprints=m['fingerprints'], admissible=a['admissible'],
        max_additional_scans=config['max_additional_scans'], trust_radius=config['trust_radius'],
        min_predicted_gain_pp=config['min_predicted_gain_pp'], uncertainty_floor_pp=config['uncertainty_floor_pp'],
        allow_exploration=config['allow_exploration'], incumbent_id=m.get('incumbent_id'))
    m['search'] = _json(result)
    m['incumbent_id'] = result['measured_selection']['retained_best']['id']
    # The nominal best is always selected from actually acquired scans. The
    # separately retained incumbent flags improvements below the practical floor.
    m['best_measured_id'] = result['measured_selection']['nominal_best']['id']
    for name, selection in (('best_measured', 'nominal_best'), ('retained', 'retained_best')):
        selected = result['measured_selection'][selection]
        index = int(np.flatnonzero(np.isclose(a['alpha_grid'], selected['alpha'], rtol=0, atol=1e-12))[0])
        a[name+'_currents_A'] = a['candidate_currents_A'][index].copy()
    return result


def _proposal_directory(state, round_index):
    """Keep CSV and its verified command metadata in one immutable directory.

    Legacy sessions retain their original session-local export directory.
    A configured acquisition waveform folder groups new sessions below
    ``closed_loop/<readable session name>_<absolute-path hash>/round_N``.
    The hash distinguishes different sessions that share a directory name.
    """
    session = Path(state['session_dir']).resolve()
    root = state['manifest']['configuration'].get('waveform_output_dir')
    if root is None:
        return session / f'round_{round_index}'
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', session.name).strip('._-') or 'session'
    identifier = hashlib.sha256(str(session).encode('utf-8')).hexdigest()[:12]
    return Path(root) / 'closed_loop' / f'{name}_{identifier}' / f'round_{round_index}'


def _propose(state):
    from closed_loop_commands import load_command_context, evaluate_candidate, export_candidate
    m, a = state['manifest'], state['arrays']
    result = _search(state)
    pending = [item for item in m['proposals'] if not item.get('completed')]
    if pending:
        m['pending_proposal_id'] = pending[0]['id']
        m['search']['candidate'] = pending[0]['candidate']
        m['search']['stop_reason'] = 'existing_export_waiting_for_its_scan'
        return
    candidate = result['candidate']
    m['pending_proposal_id'] = None
    if candidate is None: return
    context = load_command_context(m['command_context']['path'])
    evaluated = evaluate_candidate(context, a['historical_quantized_currents_A'], candidate['alpha'])
    if not evaluated['valid'] or evaluated['fingerprint'] != candidate['fingerprint']:
        raise ValueError('Candidate hardware/fingerprint changed since the frozen simulation bank; no CSV exported.')
    index = candidate['index']
    if not np.array_equal(evaluated['quantized_currents_A'], a['candidate_currents_A'][index]):
        raise ValueError('Exported DAC commands would differ from the simulated candidate.')
    round_index = len(m['measured_points'])-1
    output = _proposal_directory(state, round_index)
    exported = export_candidate(context, evaluated, output)
    # Preserve every generated command/metadata artifact, not just its filename.
    files = {path.name: _file(path) for path in output.rglob('*') if path.is_file()}
    csv_path = Path(exported['csv_file']).resolve()
    if not csv_path.is_file() or str(csv_path) not in [item['path'] for item in files.values()]:
        raise ValueError('The verified export did not return a CSV inside its round folder.')
    proposal = {'id': f'round_{round_index}', 'round_index': round_index, 'candidate': _json(candidate),
        'export': _json(exported), 'export_files': files, 'csv': _file(csv_path), 'completed': False,
        'hardware_before_DAC': _json(evaluated['hardware_before_DAC']),
        'hardware_after_DAC': _json(evaluated['hardware_after_DAC']),
        'acquisition_instruction': 'Apply exactly this CSV with the frozen sequence/trigger; then confirm the scan/CSV association.'}
    m['proposals'].append(proposal); m['pending_proposal_id'] = proposal['id']


def run_round0(session_dir, *, configuration, inputs=None, model=None, prepared=None,
               frame_matrix=None, simulation_metadata=None):
    """Initialize once, or resume the exact existing session without a model."""
    with _session_lock(session_dir) as folder:
        if (folder/'session.json').is_file():
            state = load_closed_loop(folder); _resume_config(state, configuration)
            _propose(state); _save(state); return state
        config = _config(configuration)
        if not config['association_confirmed']:
            raise ValueError('Confirmer explicitement l’association scan historique / CSV ou NPZ / même séquence avant le Round 0.')
        if inputs is None or model is None or prepared is None:
            raise ValueError('First Round 0 needs the notebook’s own prepared inputs and MRzero model; later rounds resume from session.json.')
        is_synthetic = (inputs.get('manifest') or {}).get('format') in ('mrzero_synthetic_acquisition', 'mrzero_synthetic_twix')
        if is_synthetic and config['data_kind'] != 'synthetic':
            raise ValueError('Synthetic inputs must be explicitly labeled data_kind="synthetic"; they are not experimental evidence.')
        previous = _file(config['previous_scan'], suffix='.dat')
        baseline = _file(inputs['paths']['scan'], suffix='.dat')
        if previous['sha256'] == baseline['sha256']:
            raise ValueError('The previous ACDC scan duplicates the unshimmed baseline.')
        for path in (previous['path'], baseline['path'], inputs['paths']['reference'], inputs['paths']['b0']):
            _require_data_kind(path, config['data_kind'])
        from closed_loop_commands import (build_command_context, save_command_context,
            load_historical_commands, evaluate_candidate)
        settings = config['waveform_parameters']
        context = build_command_context(model, inputs, prepared, waveform_parameters=settings,
            csv_time_origin=config['csv_time_origin'], trigger_channel=config['trigger_channel'],
            total_current_limit_A=settings['total_current_limit_A'], field_limit_Hz=config['field_limit_Hz'])
        historical = load_historical_commands(context, csv_path=config['previous_csv'], npz_path=config['previous_npz'],
            metadata_path=config['previous_metadata'])
        old = historical['quantized_currents_A']
        if not np.any(old): raise ValueError('Historical waveform is zero; an amplitude search would contain no new experiment.')
        comparison = compare_measured_scan_set(inputs, [previous['path']], frame_matrix=frame_matrix,
            roi_mask=config.get('roi_mask'), normalize=True,
            navigator_phase_branch=config['navigator_phase_branch'], reference_threshold_fraction=config['reference_threshold_fraction'])
        _check_epi_metadata(inputs['scan_metadata'], comparison['metadata'][2])
        if not np.isclose(inputs['gre']['twix_metadata']['slice_thickness_mm'], inputs['scan_metadata']['slice_thickness_mm'], rtol=0, atol=1e-3):
            raise ValueError('GRE and EPI slice thickness differ; a shared 2D reference is not verified.')
        images = comparison['images']; mask = comparison['mask']
        # Native MRzero and measured array order must have a verified common grid.
        for name, actual in (('X_m', model.reconstruction.X_m), ('Y_m', model.reconstruction.Y_m)):
            if not np.allclose(actual, comparison[name], rtol=0, atol=1e-9):
                raise ValueError('MRzero reconstruction grid differs from the registered measured EPI grid.')
        grid = np.linspace(0., 1., config['alpha_points'])
        candidates = [evaluate_candidate(context, old, float(alpha)) for alpha in grid]
        if not candidates[0]['valid'] or not candidates[-1]['valid']:
            raise ValueError('Baseline/historical anchors fail current hardware checks; do not scale or export them silently.')
        if not np.array_equal(candidates[-1]['quantized_currents_A'], old):
            raise ValueError('Alpha=1 must reproduce the historical applied DAC commands exactly.')
        simulation_images, cached = [], {}
        for index, item in enumerate(candidates):
            if not item['valid']:
                simulation_images.append(np.full(images[0].shape, np.nan)); continue
            fingerprint = item['fingerprint']
            if fingerprint not in cached:
                simulated = np.abs(model.forward(item['quantized_currents_A'])[0])
                if simulated.shape != mask.shape or not np.isfinite(simulated).all():
                    raise ValueError('MRzero returned an invalid image.')
                cached[fingerprint] = simulated
            simulation_images.append(cached[fingerprint])
            print(f'MRzero alpha={grid[index]:.3f} ({index+1}/{len(grid)}), commandes DAC vérifiées', flush=True)
        sim0 = simulation_images[0][mask]; epi0 = comparison['native_images'][1][mask]
        denominator = float(np.sum(sim0**2))
        if denominator <= 0: raise ValueError('No simulated baseline signal on the frozen GRE ROI.')
        simulation_to_epi_gain = float(np.sum(sim0*epi0)/denominator)
        simulated_gain = simulation_to_epi_gain*comparison['gain']/comparison['reference_scale']
        if not np.isfinite(simulated_gain) or simulated_gain <= 0: raise ValueError('Invalid frozen MRzero baseline gain.')
        costs = np.array([relative_image_cost(simulated_gain*image, images[0], mask)['J'] if candidate['valid'] else np.nan
                         for image, candidate in zip(simulation_images, candidates)])
        arrays = {'alpha_grid': grid, 'simulated_J': costs, 'admissible': np.array([item['valid'] for item in candidates]),
            'candidate_currents_A': np.stack([item['quantized_currents_A'] for item in candidates]),
            'candidate_images': simulated_gain*np.stack(simulation_images), 'historical_quantized_currents_A': old,
            'reference_image': images[0], 'image_baseline': images[1], 'image_historical': images[2], 'mask': mask,
            'native_X_m': comparison['native_X_m'], 'native_Y_m': comparison['native_Y_m'],
            'X_m': comparison['X_m'], 'Y_m': comparison['Y_m']}
        sources = {name: _file(path) for name, path in inputs['paths'].items()}
        sources['historical_scan'] = previous
        for name in ('previous_csv', 'previous_npz', 'previous_metadata'):
            if config[name] is not None: sources[name] = _file(config[name])
        for key in ('previous_npz', 'previous_csv'):
            if config[key] and Path(config[key]).with_suffix('.json').is_file():
                sources[key+'_sidecar'] = _file(Path(config[key]).with_suffix('.json'))
        context_path = folder/'command_context.npz'
        if context_path.exists(): raise FileExistsError('An incomplete initialization already has a command context; choose a clean session folder.')
        save_command_context(context, context_path)
        points = []
        for name, alpha, image, scan, fingerprint in (
                ('baseline', 0., images[1], baseline, candidates[0]['fingerprint']),
                ('historical', 1., images[2], previous, candidates[-1]['fingerprint'])):
            score = relative_image_cost(image, images[0], mask)
            points.append({'id': name, 'alpha': alpha, 'experimental_J': score['J'],
                'experimental_NRMSE_percent': score['NRMSE_percent'], 'fingerprint': fingerprint, 'scan': scan,
                'image_key': 'image_'+name, 'association_confirmed': True,
                'association_basis': 'Explicit operator confirmation; headers cannot prove executed controller commands.',
                'data_kind': config['data_kind']})
        manifest = {'schema': SCHEMA, 'revision': -1, 'created_UTC': datetime.now(timezone.utc).isoformat(),
            'configuration': config, 'sources': sources, 'command_context': _file(context_path),
            'acquisition': _json(inputs['acquisition']), 'timing': _json(inputs['timing']),
            'baseline_metadata': _json(inputs['scan_metadata']), 'comparison_report': _json(comparison['report']),
            'normalization': {'experimental_gain': comparison['gain'], 'reference_scale': comparison['reference_scale'],
                'simulation_to_baseline_EPI_gain': simulation_to_epi_gain, 'simulated_gain_to_normalized_GRE': simulated_gain,
                'gain_fit_source': 'Only unshimmed EPI and its MRzero simulation; frozen for every alpha and later acquisition.'},
            'historical_commands': _json(historical['report']), 'simulation_metadata': _json(simulation_metadata),
            'fingerprints': [item['fingerprint'] for item in candidates],
            'candidate_hardware': [_json({k: v for k, v in item.items() if k not in
                ('command_currents_A', 'quantized_currents_A', 'target_currents_A', 'dac_codes')}) for item in candidates],
            'measured_points': points, 'proposals': [],
            'scope': 'One-dimensional scaling of an identified historical waveform; not general 32-channel optimization.',
            'code_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                ('closed_loop_acdc.py', 'closed_loop_commands.py', 'closed_loop_search.py', 'measured_scan_comparison.py')}}
        state = {'session_dir': str(folder), 'manifest': manifest, 'arrays': arrays}
        # Persist a resumable baseline even if a subsequent export is interrupted.
        _save(state); _propose(state); _save(state)
        return state


def run_feedback_round(session_dir, *, round_index, scan_path=None, association_confirmed=False, configuration=None):
    """Import one explicitly associated scan, then optionally export the next CSV."""
    if (isinstance(round_index, (bool, np.bool_)) or not isinstance(round_index, (int, np.integer))
            or round_index not in (1, 2)):
        raise ValueError('round_index must be the integer 1 or 2.')
    with _session_lock(session_dir) as folder:
        state = load_closed_loop(folder); _resume_config(state, configuration)
        if scan_path is None: return state
        if association_confirmed is not True:
            raise ValueError('Confirmer que ce scan utilise exactement le CSV de ce round et la séquence figée.')
        record = _file(scan_path, suffix='.dat'); m = state['manifest']
        _require_data_kind(record['path'], m['configuration']['data_kind'])
        existing = [point for point in m['measured_points'] if point['id'] == f'round_{round_index}']
        if existing:
            if existing[0]['scan'] != record: raise ValueError('This round already contains a different scan; its association is immutable.')
            _propose(state); _save(state); return state
        if any(point['scan']['sha256'] == record['sha256'] for point in m['measured_points']):
            raise ValueError('This Twix is already recorded; it cannot stand for another waveform.')
        if len(m['measured_points'])-2 != round_index-1:
            raise ValueError('Import experimental rounds in order; no missing acquisition is synthesized.')
        proposals = [p for p in m['proposals'] if p['round_index'] == round_index and not p['completed']]
        if len(proposals) != 1: raise ValueError('No verified CSV awaits an acquisition for this round.')
        proposal = proposals[0]; candidate = proposal['candidate']
        image, diagnostics = _read_feedback_image(state, record['path'])
        score = relative_image_cost(image, state['arrays']['reference_image'], state['arrays']['mask'])
        before = min(point['experimental_NRMSE_percent'] for point in m['measured_points'])
        predicted = candidate['predicted_NRMSE_percent']
        point = {'id': f'round_{round_index}', 'alpha': candidate['alpha'], 'experimental_J': score['J'],
            'experimental_NRMSE_percent': score['NRMSE_percent'], 'fingerprint': candidate['fingerprint'],
            'scan': record, 'csv': proposal['csv'], 'association_confirmed': True,
            'association_basis': 'Operator confirms this exact scan/CSV hash pair and the frozen sequence.',
            'predicted_NRMSE_percent_before_scan': predicted,
            'prediction_error_pp': None if predicted is None else score['NRMSE_percent']-predicted,
            'improvement_over_previous_best_pp': before-score['NRMSE_percent'],
            'image_key': f'image_round_{round_index}', 'diagnostics': _json(diagnostics),
            'data_kind': m['configuration']['data_kind']}
        state['arrays'][point['image_key']] = image
        m['measured_points'].append(point); proposal['completed'] = True
        _save(state); _propose(state); _save(state)
        return state


def print_closed_loop_summary(state):
    m = state['manifest']; measured_label = 'acquisitions synthétiques' if m['configuration']['data_kind'] == 'synthetic' else 'acquisitions mesurées'
    print(f'Session persistante : {state["session_dir"]} — {measured_label}')
    for point in m['measured_points']:
        print(f"{point['id']}: alpha={point['alpha']:.3f}, NRMSE acquisition={point['experimental_NRMSE_percent']:.3f} %")
        if 'prediction_error_pp' in point:
            print(f"  gain sur meilleur précédent={point['improvement_over_previous_best_pp']:.3f} points ; erreur de prédiction={point['prediction_error_pp']}")
    search = m.get('search')
    if not search: return
    selection = search['measured_selection']
    print('Meilleure acquisition observée :', selection['nominal_best']['id'])
    print('Configuration retenue avec seuil pratique :', selection['retained_best']['id'],
          f"(seuil {selection['uncertainty_floor_pp']} point ; variabilité non estimée sans répétitions)")
    proposal = next((p for p in m['proposals'] if not p['completed']), None)
    if proposal:
        candidate = proposal['candidate']
        print(f"Proposition {proposal['id']}: alpha={candidate['alpha']:.3f} ; mode={candidate['mode']}")
        print('NRMSE simulée :', candidate['simulated_NRMSE_percent'], '% ; prédite :', candidate['predicted_NRMSE_percent'], '%')
        print('Gain prédit, non garanti :', candidate['predicted_gain_pp'], 'points')
        print('CSV à appliquer manuellement :', proposal['csv']['path'])
        print('Contrôles matériels après DAC :', proposal['hardware_after_DAC'])
    else: print('Pas de nouvel export :', search['stop_reason'])


def plot_closed_loop(state):
    """Four clearly labeled figures; measured images/residuals share their scales."""
    import matplotlib.pyplot as plt
    m, a = state['manifest'], state['arrays']; search = m.get('search', {})
    synthetic = m['configuration']['data_kind'] == 'synthetic'
    label = 'acquisitions synthétiques' if synthetic else 'mesures réelles'
    figures = []
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(a['alpha_grid'], 100*np.sqrt(a['simulated_J']), '--', label='Simulation MRzero')
    if search:
        values = np.array([np.nan if x is None else x for x in search['predicted_NRMSE_percent']])
        ax.plot(a['alpha_grid'], values, label='Prédiction calibrée (empirique)')
    points = m['measured_points']
    ax.scatter([p['alpha'] for p in points], [p['experimental_NRMSE_percent'] for p in points], c='black', zorder=3, label=label)
    pending = next((p for p in m['proposals'] if not p['completed']), None)
    if pending:
        c = pending['candidate']; ax.axvline(c['alpha'], color='tab:orange', label='Prochain essai proposé')
        if c['predicted_NRMSE_percent'] is not None: ax.scatter(c['alpha'], c['predicted_NRMSE_percent'], marker='*', s=140, color='tab:orange')
    ax.set(xlabel='Amplitude alpha des commandes historiques', ylabel='NRMSE contre GRE [%]', title='Simulation, prédiction et acquisitions distinctes')
    ax.legend(); figures.append(fig)
    X, Y = a['X_m']*1000, a['Y_m']*1000
    image_items = [('GRE réelle' if not synthetic else 'GRE synthétique', a['reference_image'])]+[
        (f"{p['id']} : {p['experimental_NRMSE_percent']:.2f} %", a[p['image_key']]) for p in points]
    vmax = max(float(np.nanmax(image)) for _, image in image_items)
    fig, axes = plt.subplots(1, len(image_items), figsize=(4*len(image_items), 4), layout='constrained', squeeze=False)
    for ax, (title, image) in zip(axes.ravel(), image_items):
        artist = ax.pcolormesh(X, Y, image, shading='auto', cmap='gray', vmin=0, vmax=vmax)
        ax.contour(X, Y, a['mask'], levels=[.5], colors='gold', linewidths=.5)
        ax.set(title=title, xlabel='X [mm]', ylabel='Y [mm]', aspect='equal')
    fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=.7, label='Gain baseline commun, unités GRE normalisées')
    figures.append(fig)
    residuals = [a[p['image_key']]-a['reference_image'] for p in points]
    scale = max(max(float(np.nanmax(abs(image))) for image in residuals), np.finfo(float).tiny)
    fig, axes = plt.subplots(1, len(points), figsize=(4*len(points), 4), layout='constrained', squeeze=False)
    for ax, point, image in zip(axes.ravel(), points, residuals):
        artist = ax.pcolormesh(X, Y, image, shading='auto', cmap='RdBu_r', vmin=-scale, vmax=scale)
        ax.set(title=point['id']+' − GRE', xlabel='X [mm]', ylabel='Y [mm]', aspect='equal')
    fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=.7, label='Résidu signé, même échelle')
    figures.append(fig)
    from closed_loop_commands import load_command_context, sample_command_waveform
    context = load_command_context(m['command_context']['path'])
    old = a['historical_quantized_currents_A']
    if pending:
        selected = a['candidate_currents_A'][pending['candidate']['index']]
        title = f"Proposition alpha={pending['candidate']['alpha']:.3f}"
    else:
        selected = a['best_measured_currents_A']
        title = 'Meilleure configuration acquise'
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout='constrained', sharex=True)
    for column, (commands, title) in enumerate(((old, 'Historique'), (selected, title))):
        data = sample_command_waveform(context, commands)
        times = data['CSV_time_s']*1000
        polarity = np.asarray(data['channel_polarity'])[:, None]
        actual = data['actual_A']*polarity
        commanded = data['commanded_A']*polarity
        for channel in range(commands.shape[0]):
            axes[0, column].step(times, commanded[channel], where='post', color='gray', alpha=.2,
                                 label='Commandes DAC' if channel == 0 else None)
            axes[0, column].plot(times, actual[channel], alpha=.5,
                                 label='Courants physiques prédits' if channel == 0 else None)
        limit = float(np.max(data['max_current_A']))
        axes[0, column].axhline(limit, color='red', ls='--'); axes[0, column].axhline(-limit, color='red', ls='--')
        axes[0, column].set(title=title, ylabel='Courant de sortie [A]', ylim=(-1.05*limit, 1.05*limit))
        axes[0, column].legend(fontsize=8)
        axes[1, column].plot(times, np.abs(actual).sum(0), label='Somme absolue physique prédite')
        axes[1, column].step(times, np.abs(commanded).sum(0), where='post', color='gray', ls='--', label='Somme absolue DAC')
        axes[1, column].axhline(data['total_current_limit_A'], color='red', ls='--', label='Limite totale')
        axes[1, column].set(xlabel="Temps selon l'origine archivée du modèle [ms] (avant trigger inclus)", ylabel='Somme |I| [A]')
        axes[1, column].legend(fontsize=8)
    fig.suptitle('Commandes et réponse temporelle du contrôleur — prédiction, sans télémétrie')
    figures.append(fig)
    for name, fig in zip(('cost', 'images', 'residuals', 'currents'), figures):
        fig.savefig(Path(state['session_dir'])/f'round_{len(points)-2}_{name}.png', dpi=120)
    return figures


__all__ = ['run_round0', 'run_feedback_round', 'load_closed_loop', 'plot_closed_loop',
           'print_closed_loop_summary', 'relative_image_cost']
