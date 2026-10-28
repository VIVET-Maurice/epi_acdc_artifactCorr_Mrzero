"""Report both EPI signal-mismatch directions from frozen simulation archives.

No simulation, calibration, gain fitting, registration or optimization runs.
The measured image and archived gains are reused. Signed errors without a
brightness cutoff are primary; symmetric ratio flags at 10% AND 20% brightness
are secondary diagnostics. Optional approximate user regions are illustrations,
not ground-truth masks or parameter-fitting regions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from signal_disagreement import signal_disagreement, fixed_comparison_envelope


ROOT = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = ROOT / 'results_baseline_calibration/baseline_acquisition019_20261022_final/calibration.npz'
DEFAULT_LEGACY = ROOT / 'results_epi_fidelity/fidelity_acquisition019_20261021_final/comparison.npz'


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _orientation_check(calibration, legacy, report):
    for name in ('measured_image', 'X_image_m', 'Y_image_m'):
        if not np.allclose(calibration[name], legacy[name], rtol=1e-11, atol=1e-15):
            raise ValueError(f'Archives do not share the same {name}; do not compare by array index.')
    X, Y = calibration['X_image_m'], calibration['Y_image_m']
    frame = np.asarray(report['before_preparation']['combined_frame_matrix'])
    fov = np.asarray(report['before_model']['runtime']['reconstruction']['FOV_m'])
    ny, nx = X.shape
    x, y = np.meshgrid((np.arange(nx)-nx//2)*fov[0]/nx, (np.arange(ny)-ny//2)*fov[1]/ny)
    transformed = np.stack((x, y), axis=-1) @ frame.T
    error = float(np.max(abs(transformed - np.stack((X, Y), axis=-1))))
    if error > 1e-10:
        raise ValueError('Saved display coordinates do not match the declared joint frame.')
    return {'same_measured_image': True, 'same_output_coordinates': True,
            'joint_frame_output_grid_max_error_m': error,
            'joint_frame_matrix': frame.tolist(), 'native_array_order_retained': True,
            'array_flips_or_registration_applied': False,
            'reference': 'Uncalibrated zero-B0 target shares native image-array ordering; frame changes coordinates jointly.'}


def _figures(folder, arrays, regions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    def colorbar(figure, artist, axis, values, low, high):
        below, above = np.min(values) < low, np.max(values) > high
        extend = 'both' if below and above else 'min' if below else 'max' if above else 'neither'
        figure.colorbar(artist, ax=axis, shrink=.75, extend=extend)

    X, Y = arrays['X_image_m']*1000, arrays['Y_image_m']*1000
    measured = arrays['measured_image']
    before, after = arrays['finite_before_scaled_image'], arrays['finite_applied_scaled_image']
    scale = float(measured.max())
    difference_before, difference_after = before-measured, after-measured
    flags = np.zeros(measured.shape)
    flags[arrays['finite_applied_b10_simulated_low_measured_high_50_mask']] = -1
    flags[arrays['finite_applied_b10_measured_low_simulated_high_50_mask']] = 1
    cmap = ListedColormap(['#2166ac', '#f7f7f7', '#b2182b'])
    panels = [
        (measured, 'EPI mesurée', 'gray', 0, scale),
        (before, 'Voxels finis avant calibration', 'gray', 0, scale),
        (after, 'Voxels finis après calibration', 'gray', 0, scale),
        (difference_before, 'Avant : simulation − mesure\nRouge : simulation trop claire', 'RdBu_r', -.4*scale, .4*scale),
        (difference_after, 'Après : simulation − mesure\nBleu : simulation trop sombre', 'RdBu_r', -.4*scale, .4*scale),
        (flags, 'Deux sens, ratio 1/2 et luminosité 10 %\nRouge : mesure plus sombre ; bleu : simulation plus sombre', cmap, -1, 1),
    ]
    fig, axes = plt.subplots(2+len(regions), 3, figsize=(16, 5*(2+len(regions))),
                             layout='constrained', squeeze=False)
    for axis, (values, title, colors, low, high) in zip(axes[:2].ravel(), panels):
        artist = axis.pcolormesh(X, Y, values, shading='auto', cmap=colors, vmin=low, vmax=high)
        axis.set(title=title, xlabel='X [mm]', ylabel='Y [mm]', aspect='equal')
        colorbar(fig, artist, axis, values, low, high)
    for row, region in enumerate(regions, 2):
        cx, cy = region
        for axis, values, title, colors, low, high in zip(axes[row],
                (measured, after, difference_after),
                ('Mesure : zone illustrative', 'Simulation : même zone', 'Simulation − mesure, sans seuil'),
                ('gray', 'gray', 'RdBu_r'), (0, 0, -.4*scale), (scale, scale, .4*scale)):
            artist = axis.pcolormesh(X, Y, values, shading='auto', cmap=colors, vmin=low, vmax=high)
            axis.add_patch(plt.Circle((cx, cy), 5, fill=False, color='orange', linewidth=1.2))
            axis.set(xlim=(cx-14, cx+14), ylim=(cy-14, cy+14), title=title,
                     xlabel='X [mm]', ylabel='Y [mm]', aspect='equal')
            colorbar(fig, artist, axis, values, low, high)
    fig.suptitle('Désaccord de signal dans les deux sens — zones approximatives, sans étiquette de vérité terrain')
    fig.savefig(folder/'signed_disagreement.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), layout='constrained', sharex=True, sharey=True)
    for axis, key, title in zip(axes,
            ('measured_image', 'legacy_point_scaled_image', 'finite_before_scaled_image', 'finite_applied_scaled_image'),
            ('EPI mesurée', 'Ancien modèle ponctuel', 'Voxels finis avant calibration', 'Voxels finis après calibration')):
        artist = axis.pcolormesh(X, Y, arrays[key], shading='auto', cmap='gray', vmin=0, vmax=scale)
        axis.set(title=title, xlabel='X [mm]', ylabel='Y [mm]', aspect='equal')
        colorbar(fig, artist, axis, arrays[key], 0, scale)
    fig.savefig(folder/'stages_shared_scale.png', dpi=130)
    plt.close(fig)


def _markdown(summary):
    labels = {'legacy_point': 'Ancien modèle ponctuel',
              'finite_before': 'Voxels finis avant calibration',
              'finite_applied': 'Voxels finis après calibration'}
    lines = ['# Comparaison du signal dans les deux sens', '',
        'Le résidu est simulation × gain archivé − mesure. Les erreurs positives '
        'incluent les zones mesurées sombres que la simulation laisse trop claires. '
        'Aucun gain ni paramètre n’a été ajusté pour ce rapport.', '',
        'Les coordonnées et la mesure sont identiques dans les deux archives ; '
        'aucun retournement ni recalage des images n’a été appliqué. Les gains '
        'historiques ont des politiques différentes (ancien : champ complet ; '
        'voxels finis : apprentissage). La comparaison des trois étapes est descriptive.', '']
    for domain, title in (('full_FOV', 'Champ complet — score principal sans seuil de luminosité'),
                          ('analysis_mask', 'Enveloppe de référence fixe — diagnostic secondaire')):
        lines += ['## '+title, '',
            '| Modèle | NRMSE (%) | SSE simulation trop claire / énergie mesurée (%) | '
            'SSE simulation trop sombre / énergie mesurée (%) |',
            '| --- | ---: | ---: | ---: |']
        for name, model in summary['models'].items():
            score = model['b10'][domain]
            lines.append(f"| {labels[name]} | {score['NRMSE_percent']:.3f} | "
                         f"{100*score['overprediction_normalized_SSE']:.3f} | "
                         f"{100*score['underprediction_normalized_SSE']:.3f} |")
        lines += ['', 'Comptes au ratio 1/2 : simulation > 2 × mesure et '
            'simulation < mesure / 2. Le seuil de luminosité s’applique au plus '
            'clair des deux signaux ; les pixels mesurés sombres restent admissibles.', '',
            '| Modèle | Seuil fixe (% pic mesuré) | Simulation trop claire | Simulation trop sombre |',
            '| --- | ---: | ---: | ---: |']
        for name, model in summary['models'].items():
            for brightness in ('b10', 'b20'):
                score = model[brightness][domain]['ratios']['50']
                lines.append(f"| {labels[name]} | {int(brightness[1:])} | "
                    f"{score['measured_low_simulated_high_pixels']} | "
                    f"{score['simulated_low_measured_high_pixels']} |")
        lines.append('')
    initial = summary['models']['legacy_point']['b10']['full_FOV']
    final = summary['models']['finite_applied']['b10']['full_FOV']
    over_delta = final['overprediction_normalized_SSE'] - initial['overprediction_normalized_SSE']
    under_delta = final['underprediction_normalized_SSE'] - initial['underprediction_normalized_SSE']
    lines += [f'Du modèle ponctuel à la dernière étape, la SSE normalisée « simulation '
        f'trop claire » varie de {100*over_delta:+.3f} points d’énergie mesurée, et '
        f'celle du côté « simulation trop sombre » de {100*under_delta:+.3f} points. '
        'Le score global ne prouve donc pas à lui seul que les pertes de signal '
        'réelles sont mieux reproduites.', '',
        'L’enveloppe provient de la cible sans B0 avant calibration, avec seuil '
        'de 10 % et remplissage des trous. Elle hérite de la couverture GRE/B0 '
        'du modèle et n’est pas une vérité terrain indépendante. Les scores hors '
        'enveloppe sont conservés dans le JSON.', '',
        'Les pixels où les deux signaux sont faibles peuvent encore présenter '
        'un désaccord relatif important : leur SSE reste comptée sans seuil. '
        'Les régions cerclées sont approximatives et illustratives ; elles '
        'n’ont servi à aucun ajustement.', '',
        f"Le modèle archivé utilise ici des taux résiduels de relaxation entre "
        f"{summary['physical_assumptions']['residual_rate_min_sInv']:.6g} et "
        f"{summary['physical_assumptions']['residual_rate_max_sInv']:.6g} s⁻¹, et "
        'un B0 2D uniforme dans l’épaisseur. Les trois échos GRE courts ne '
        'déterminent pas à eux seuls la relaxation ni le déphasage dans '
        'l’épaisseur au TE EPI long. Aucune cause physique n’est inférée par '
        'ce rapport.', '',
        'Détails et provenance : [JSON](signal_disagreement.json), '
        '[archives numériques](signal_disagreement.npz), '
        '[comparaison des trois étapes](stages_shared_scale.png), '
        '[résidus signés et zooms](signed_disagreement.png).', '']
    return '\n'.join(lines)


def report_archives(arguments):
    calibration_path, legacy_path = arguments.calibration_archive.resolve(), arguments.legacy_archive.resolve()
    calibration_report_path, legacy_report_path = calibration_path.with_suffix('.json'), legacy_path.with_suffix('.json')
    calibration_report = json.loads(calibration_report_path.read_text())
    legacy_report = json.loads(legacy_report_path.read_text())
    with np.load(calibration_path, allow_pickle=False) as archive:
        calibration = {key: archive[key] for key in archive.files}
    with np.load(legacy_path, allow_pickle=False) as archive:
        legacy = {key: archive[key] for key in archive.files}
    geometry = _orientation_check(calibration, legacy, calibration_report)
    measured = calibration['measured_image']
    reference = abs(calibration['before_ideal_image'])
    envelope = fixed_comparison_envelope(reference, .1)
    subsets = {name: calibration[f'{name}_mask'] for name in ('fit', 'validation', 'test')}
    images = {'legacy_point': legacy['legacy_image'], 'finite_before': calibration['before_image'],
              'finite_applied': calibration['applied_image']}
    gains = {'legacy_point': legacy_report['metrics']['legacy']['full_FOV']['gain'],
             'finite_before': calibration_report['metrics']['before']['gain_from_fit_pixels_only'],
             'finite_applied': calibration_report['metrics']['applied']['gain_from_fit_pixels_only']}
    arrays = {'measured_image': measured, 'X_image_m': calibration['X_image_m'],
              'Y_image_m': calibration['Y_image_m'], 'fixed_reference_image': reference,
              'fixed_reference_envelope': envelope}
    models = {}
    for name, image in images.items():
        models[name] = {}
        arrays[f'{name}_scaled_image'] = abs(image)*gains[name]
        for brightness in (.1, .2):
            result = signal_disagreement(measured, image, gain=gains[name], analysis_mask=envelope,
                subsets=subsets, reference_scale=float(measured.max()), bright_fraction=brightness)
            outside = signal_disagreement(measured, image, gain=gains[name], analysis_mask=~envelope,
                reference_scale=float(measured.max()), bright_fraction=brightness)
            result['report']['outside_reference_envelope'] = outside['report']['analysis_mask']
            label = f'b{round(brightness*100)}'
            models[name][label] = result['report']
            for key, value in result['arrays'].items():
                arrays[f'{name}_{label}_{key}'] = value
    regions = [] if arguments.no_illustrative_regions else [(-20., 0.), (25., -15.)]
    region_reports = []
    for cx, cy in regions:
        mask = ((arrays['X_image_m']*1000-cx)**2+(arrays['Y_image_m']*1000-cy)**2) <= 25
        region_reports.append({'center_mm': [cx, cy], 'radius_mm': 5., 'pixels': int(mask.sum()),
            'selection': 'Approximate user-mentioned coordinates; never fitted or treated as ground truth.',
            'models': {name: signal_disagreement(measured, image, gain=gains[name], analysis_mask=mask,
                reference_scale=float(measured.max()), bright_fraction=.1)['report']['analysis_mask']
                for name, image in images.items()}})
    output = arguments.output.resolve() if arguments.output else (
        ROOT/'results_signal_disagreement'/datetime.now(timezone.utc).strftime('disagreement_%Y%m%d_%H%M%S_%f_UTC'))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a new output directory; preserve existing artifacts in {output}.')
    output.mkdir(parents=True, exist_ok=True)
    evidence = []
    if arguments.roi_evidence_dir is not None:
        for name in ('roi_report.json', 'native_echo_ROIs.png'):
            source = arguments.roi_evidence_dir/name
            if source.is_file():
                target = output/('illustrative_'+name)
                shutil.copy2(source, target)
                evidence.append({'source': str(source.resolve()), 'artifact': target.name, 'sha256': _hash(source),
                    'interpretation': 'Illustrative native-GRE/echo evidence; approximate ROI coordinates, not fit labels.'})
    summary = {'created_UTC': datetime.now(timezone.utc).isoformat(), 'models': models,
        'geometry_checks': geometry, 'gains': gains,
        'gain_policy': {'legacy_point': 'Frozen historic full-FOV evaluation gain from fidelity archive.',
            'finite_before': 'Frozen training-pixel gain from baseline calibration archive.',
            'finite_applied': 'Frozen training-pixel gain from native rebuilt calibration archive.',
            'refitted_for_this_report': False,
            'comparability': 'Historic gain policies differ; stage comparisons are descriptive, not a new held-out model-selection test.'},
        'reference_envelope': {'source': 'Uncalibrated finite-voxel zero-B0 target, threshold 10% and filled enclosed holes.',
            'pixels': int(envelope.sum()), 'is_independent_ground_truth': False,
            'limitations': 'Inherits GRE amplitude and B0/calibration coverage; full-FOV signed scores remain primary.'},
        'illustrative_regions': region_reports, 'native_echo_evidence': evidence,
        'physical_assumptions': {'residual_rate_min_sInv': calibration_report['before_preparation']['rate_min_sInv'],
            'residual_rate_max_sInv': calibration_report['before_preparation']['rate_max_sInv'],
            'through_plane_B0': 'Measured 2D B0 is uniform through the slice; no measured z susceptibility gradient.'},
        'input_artifacts': {str(path): _hash(path) for path in
            (calibration_path, calibration_report_path, legacy_path, legacy_report_path)},
        'code_sha256': {Path(__file__).name: _hash(__file__), 'signal_disagreement.py': _hash(ROOT/'signal_disagreement.py')},
        'policy': ['Signed SSE includes both signs without a brightness cutoff, including measured zero pixels.',
            'Ratio flags use two fixed brightness thresholds, 10% and 20%; neither threshold proves a causal signal void.',
            'Both-low pixels can still contain large relative discrepancies; their SSE is reported.',
            'The former one-sided hole metric counted simulated-low/measured-high pixels only.',
            'No simulation, parameter fitting, gain fitting, registration or controller operation was performed.']}
    np.savez_compressed(output/'signal_disagreement.npz', **arrays)
    _figures(output, arrays, regions)
    (output/'signal_disagreement.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    (output/'signal_disagreement.md').write_text(_markdown(summary))
    print(json.dumps({'output': str(output), 'models': {
        name: {'full_FOV': value['b10']['full_FOV'], 'reference_envelope': value['b10']['analysis_mask']}
        for name, value in models.items()}}, indent=2, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-archive', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--legacy-archive', type=Path, default=DEFAULT_LEGACY)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--roi-evidence-dir', type=Path, default=Path('/tmp/acdc_epi_real_voids'))
    parser.add_argument('--no-illustrative-regions', action='store_true')
    report_archives(parser.parse_args())
