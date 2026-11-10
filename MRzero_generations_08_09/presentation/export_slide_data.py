"""Export presentation facts from generated JSON; never type numerical results.

This standard-library-only bridge is used by the figure notebook and deck builder.
It deliberately requires the scientific summary and figure manifest, so stale
README numbers or incomplete placeholders cannot silently enter the final deck.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def load(path):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing generated scientific artifact: {path}. Run 00_generate_weekly_figures.ipynb first.")
    return json.loads(path.read_text())


def export_slide_data(folder=HERE):
    folder=Path(folder).resolve()
    summary=load(folder/'presentation_summary.json')
    manifest=load(folder/'figures_manifest.json')
    reconstruction=summary['reconstruction']
    off=summary['off']
    geometry=summary['geometry']
    prepared=summary['prepared_report']
    receiver=summary['receiver_audit']
    equivalence=summary['equivalence']
    runtime=summary['runtime']
    aperture=summary['ADC_window']
    uniform=summary['uniform_phantom']
    scan=summary['scan_metadata']
    echoes=summary['echo_times_s']
    rows={row['operator']:row for row in reconstruction['rows']}
    direct=rows['direct_actual_k']
    error=direct['complex_relative_image_error']
    off_nrmse=off['NRMSE_percent']
    if abs(off['gain']-reconstruction['shared_OFF_gain'])>1e-10*max(1.,abs(off['gain'])):
        raise AssertionError('OFF and matched-reconstruction reporting gains differ.')
    count=equivalence['common_settings']['ADC_sample_count']
    voxels=equivalence['common_settings']['voxel_count']
    ny,nx=reconstruction['operators']['auto']['image_shape']
    fov=geometry['FOV_m']
    thickness=geometry['slice_thickness_mm']
    physical_ids=receiver['measured_channels']['EPI_OFF']['physical_IDs']
    receiver_count=receiver['measured_channels']['EPI_OFF']['count']
    receiver_summary=(f"{receiver_count} measured MRI streams; EPI forward: "
        f"{receiver['current_EPI_forward_receiver_count']} uniform receiver, no sensitivity maps.")
    zero=next(row for row in equivalence['rows'] if row['case']=='zero')
    max_error=max(row['signal_complex_relative_L2'] for row in equivalence['rows'])
    timing=runtime['common_settings']; provenance=runtime['provenance']
    repeats=sorted(set(r['repetitions'] for r in runtime['records']))
    warmups=sorted(set(r['warmup_count'] for r in runtime['records'] if r['operation'].startswith('warm_')))
    aperture_row=max(aperture['quadrature_rows'],key=lambda row:row['quadrature_order'])
    aliases={
        'legacy':'slide02_legacy_pattern','off_comparison':'slide03_off_validation',
        'off_main':'slide10_gre_off_comparison','reconstruction':'slide08_reconstruction_control',
        'equivalence':'slide11_signal_equivalence','runtime':'slide11_runtime_benchmarks',
        'adc_window':'slide12_adc_window','uniform':'uniform_phantom_b0_control',
        'sampling_controls':'slide12_sampling_controls',
        'coordinates':'slide13_trajectory_timing',
    }
    figures={}
    for alias,name in aliases.items():
        entry=manifest['figures'][name]
        path=Path(entry['files']['png'])
        if not path.is_file():
            path=folder/'figures'/path.name
        if not path.is_file():raise FileNotFoundError(path)
        figures[alias]=dict(path=str(path.relative_to(folder)),manifest_key=name,
            source_manifest='figures_manifest.json',sha256=sha256(path.read_bytes()).hexdigest())
    figure_results={
        'figures':figures,
        'off':dict(off,
            title=f"Fresh Generation-08 OFF prediction has {off_nrmse:.2f}% magnitude NRMSE",
            nrmse_percent=off_nrmse,
            mask_summary=f"M: {summary['OFF_mask_pixels']:,} finite full-FOV pixels; one positive least-squares gain; no registration.",
            context_summary=(f"Acquisition {Path(summary['source_paths']['scan']).parent.name}; {voxels:,} retained voxels; "
                f"{nx}×{ny}, FOV {fov[0]*1000:g}×{fov[1]*1000:g} mm, slice {thickness:g} mm. "
                f"TE model {geometry['sequence_TE_ms']:.4f} ms / scan {scan['TE_s']*1000:.3f} ms; before EPI calibration.")),
        'geometry':dict(geometry,
            summary=f"{nx}×{ny} image grid · {fov[0]*1000:g}×{fov[1]*1000:g} mm FOV · {thickness:g} mm slice\nRF time, nominal Pulseq k and registered shim maps",
            matrix_label=f"{nx} × {ny} pixels",sample_label=f"{count:,} samples"),
        'receiver':dict(receiver,slide_summary=receiver_summary,physical_receive_IDs=physical_ids),
        'reconstruction':dict(reconstruction,
            title=("Pulseq-coordinate reconstruction leaves the intensity pattern unchanged" if error<1e-10 else
                   "The matched actual-coordinate reconstruction test reveals a difference"),
            complex_relative_error=error,
            magnitude_nrmse_percent=direct['magnitude_difference_NRMSE_percent'],
            synthetic_relative_error=reconstruction['sanity']['FFT_vs_direct_relative_L2']),
        'equivalence':dict(equivalence,
            title=f"08 and 09 agree within {max_error*1e6:.2g} ppm in the tested cases",
            max_signal_relative_error=max_error,
            zero_current_relative_error=zero['signal_complex_relative_L2'],
            case_summary=(f"Zero current plus a deterministic nonoptimized waveform, peak |I| "
                f"{equivalence['current_configuration']['projected_peak_A']:.3f} A; not validation over the full "
                f"±{summary['configuration']['MAX_CURRENT_A']:g} A range.")),
        'runtime':dict(runtime,context_summary=(
            f"{provenance['cpu_model']}; CPU; PyTorch {provenance['torch_threads']} threads (BLAS thread env unset); model {timing['dtype']}, native MRzero {timing['native_MRzero_precision']}. "
            f"{count:,} ADCs × {voxels:,} voxels; repetitions {', '.join(map(str,repeats))}; warm-up "
            f"{', '.join(map(str,warmups))} for repeated operations / 0 for setup. Points: medians; bars: IQR. "
            "Warm 08/08 V2 use native CPU blocks; 09 has no signal cache. File I/O and projection/guidance excluded; GPU memory unavailable.")),
        'adc_window':dict(aperture,summary=(
            f"{aperture['dwell_s']*1e6:g} µs rectangular window: magnitude change "
            f"{aperture_row['magnitude_image_NRMSE_percent']:.3f}% (order {aperture_row['quadrature_order']}). "
            "Zero ACDC currents; scanner receive filtering is unknown.")),
        'uniform':dict(uniform,summary=(f"Uniform interior CV: {uniform['B0_zero']['CV_percent']:.3f}% → "
            f"{uniform['measured_B0']['CV_percent']:.2f}% with measured B0; {uniform['interior_mask']['interior_pixels']:,} fixed pixels.")),
        'assumptions':dict(rf=f"Scan flip {scan['header_flip_deg']:g}°; ideal model flip 90°; no GRE steady state",
            extrapolation=(f"MEGRE TE {min(echoes)*1000:.3f}–{max(echoes)*1000:.3f} ms; "
                f"EPI ADC up to {geometry['ADC_elapsed_max_ms']:.3f} ms: extrapolation.")),
        'legacy':summary['legacy'],
        'optimization_replay':summary.get('optimization_replay',{}),
        'provenance':dict(summary['provenance'],source_paths=summary['source_paths'],
            input_sha256=manifest['input_sha256'],configuration=summary['configuration'],
            figure_manifest=str((folder/'figures_manifest.json').resolve()),
            summary_path=str((folder/'presentation_summary.json').resolve()),
            summary_sha256=sha256((folder/'presentation_summary.json').read_bytes()).hexdigest(),
            limitations_summary=("Receive-sensitive EPI forward remains planned. Dwell aperture is hypothetical; scanner filtering is unknown. "
                "Parity is numerical validation, and archived-current replay is not a new optimization or measured ACDC-ON validation.")),
    }
    path=folder/'presentation_slide_data.json'
    path.write_text(json.dumps(figure_results,indent=2,allow_nan=False))
    return figure_results


if __name__=='__main__':
    export_slide_data()
    print(HERE/'presentation_slide_data.json')
