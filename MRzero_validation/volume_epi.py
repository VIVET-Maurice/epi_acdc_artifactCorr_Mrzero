"""Receiver-at-a-time 3D EPI reconstruction with three navigators per shot.

The raw encoding axes are readout, phase and partition. This module performs
no anatomical rotation, physical registration, reformatting or optimization.
Geometry is the responsibility of the volume reader using these images.
"""

from __future__ import annotations

from collections import defaultdict
import ctypes
import hashlib
import mmap
from pathlib import Path
import sys

import numpy as np

EXPERIMENT = Path(__file__).resolve().parent.parent / 'Experiment'
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))
from prepare_acdc_inputs import centered_fft, estimate_epi_navigator_phase


def _layout(scan):
    """Validate a full Cartesian volume and retain each shot's own navigators."""
    blocks = scan.get('mdb', [])
    images = [(index, block) for index, block in enumerate(blocks) if block.is_image_scan()]
    navigators = [(index, block) for index, block in enumerate(blocks) if block.is_flag_set('PHASCOR')]
    if not images:
        raise ValueError('3D EPI needs image MDHs and three navigators per partition.')
    header = scan['hdr']['MeasYaps']
    grid = header.get('sKSpace', {})
    first = images[0][1]
    nx, receivers = int(first.mdh.SamplesInScan), int(first.mdh.UsedChannels)
    ny, nz = int(grid.get('lPhaseEncodingLines', 0)), int(grid.get('lPartitions', 0))
    if (int(grid.get('ucDimension', 0)) != 4 or nx < 4 or receivers < 1 or ny < 2 or nz < 2
            or int(grid.get('lBaseResolution', 0)) != nx):
        raise ValueError('3D EPI requires full Cartesian readout/phase/partition encoding without implicit oversampling removal.')
    if len(images) != ny*nz or len(navigators) != 3*nz:
        raise ValueError('3D EPI requires every phase/partition pair and exactly three navigators per partition.')
    relevant = images + navigators
    if any(int(block.mdh.SamplesInScan) != nx or int(block.mdh.UsedChannels) != receivers
           for _, block in relevant):
        raise ValueError('3D EPI receiver and sample counts must remain constant.')
    for counter in ('cAve', 'cSlc', 'cEco', 'cPhs', 'cRep', 'cSet', 'cSeg'):
        if len({getattr(block, counter) for _, block in relevant}) != 1:
            raise ValueError('3D EPI must contain one slab, echo, average, repetition and segment.')
    by_partition, nav_by_partition = defaultdict(list), defaultdict(list)
    for index, block in images:
        if (not 0 <= int(block.cPar) < nz or not 0 <= int(block.cLin) < ny
                or int(block.mdh.CenterCol) != nx//2 or int(block.mdh.CenterLin) != ny//2
                or int(block.mdh.CenterPar) != nz//2):
            raise ValueError('3D EPI MDHs must describe a full centered phase/partition grid.')
        by_partition[int(block.cPar)].append((index, block))
    for index, block in navigators:
        if not 0 <= int(block.cPar) < nz:
            raise ValueError('3D EPI navigator has an invalid partition counter.')
        nav_by_partition[int(block.cPar)].append((index, block))
    shots = []
    for partition in range(nz):
        indexed = by_partition[partition]
        lines = sorted((block for _, block in indexed), key=lambda block: int(block.cLin))
        if [int(block.cLin) for block in lines] != list(range(ny)):
            raise ValueError('3D EPI phase rows must be unique and complete within every partition.')
        indexed_nav = nav_by_partition[partition]
        nav = [block for _, block in indexed_nav]
        if len(nav) != 3 or [block.is_flag_set('REFLECT') for block in nav] != [False, True, False]:
            raise ValueError('Each 3D EPI partition needs +,-,+ navigators in acquisition order.')
        if max(index for index, _ in indexed_nav) >= min(index for index, _ in indexed):
            raise ValueError('3D EPI navigators must precede the imaging train of their partition.')
        negative = np.asarray([block.is_flag_set('REFLECT') for block in lines], dtype=bool)
        if not negative.any() or not (~negative).any():
            raise ValueError('3D EPI imaging rows must include both readout polarities.')
        shots.append((lines, nav, negative))
    return shots, (nx, ny, nz), receivers


def _receiver_selection(values, count):
    if values is None:
        return list(range(count))
    if isinstance(values, (int, np.integer)) and not isinstance(values, (bool, np.bool_)):
        values = [int(values)]
    try:
        values = list(values)
    except TypeError:
        raise ValueError('receiver_indices must contain integer receiver indices.') from None
    if (not values or any(isinstance(value, (bool, np.bool_))
                          or not isinstance(value, (int, np.integer))
                          or not 0 <= value < count for value in values)
            or len(set(values)) != len(values)):
        raise ValueError('receiver_indices must be unique integer indices present in the Twix.')
    return [int(value) for value in values]


def _ordered_receiver(block, receiver, mapping):
    """Access one receiver without materializing the other 31 raw channels."""
    samples = int(block.mdh.SamplesInScan)
    if mapping is not None and hasattr(block, 'mem_pos') and getattr(block, 'version_is_ve', False):
        from twixtools.mdh_def import Channel_header
        offset = (int(block.mem_pos) + ctypes.sizeof(block.mdh)
                  + receiver*int(block.block_len) + ctypes.sizeof(Channel_header))
        if offset < 0 or offset + samples*np.dtype(np.complex64).itemsize > mapping.size:
            raise ValueError('3D EPI receiver data extend beyond the Twix file.')
        data = np.ndarray((samples,), dtype=np.complex64, buffer=mapping, offset=offset)
    else:
        # Supports in-memory test/local MDBs and older Twix layouts.
        data = np.asarray(block.data)[receiver]
    if data.shape != (samples,) or not np.iscomplexobj(data) or not np.isfinite(data).all():
        raise ValueError('3D EPI raw receiver data must be finite complex samples.')
    return data[::-1] if block.is_flag_set('REFLECT') else data


def _reconstruct_receiver(shots, shape, receiver, mapping, phase_branch):
    nx, ny, nz = shape
    # Readout is transformed per partition; the other two Fourier transforms
    # are applied only after all measured partitions are present.
    hybrid_volume = np.empty((nz, ny, nx), dtype=np.complex128)
    corrections, residuals, fitted = [], [], []
    for partition, (lines, navigators, negative) in enumerate(shots):
        kspace = np.stack([_ordered_receiver(block, receiver, mapping) for block in lines])
        nav_kspace = np.stack([_ordered_receiver(block, receiver, mapping) for block in navigators])[:, None, :]
        nav = centered_fft(nav_kspace, (2,), inverse=True)
        correction, fit = estimate_epi_navigator_phase(nav, phase_branch=phase_branch)
        hybrid = centered_fft(kspace, (1,), inverse=True)
        hybrid[negative] *= np.exp(-1j*correction[0])[None, :]
        hybrid_volume[partition] = hybrid
        corrections.append(correction[0])
        residuals.append(float(fit['fit_residual_rad'][0]))
        fitted.append(bool(fit['fitted_channels'][0]))
    volume = centered_fft(hybrid_volume, (0, 1), inverse=True).transpose(2, 1, 0)
    finite_residuals = np.asarray(residuals)[np.isfinite(residuals)]
    report = dict(array_axes=['readout', 'phase', 'partition'], image_shape=list(shape),
        receiver_index=receiver, navigator_count_per_partition=3, partition_count=nz,
        navigator_phase_branch=phase_branch, navigator_fit_partition_count=int(np.sum(fitted)),
        navigator_phase_correction_rad=np.asarray(corrections).tolist(),
        navigator_fit_residual_RMS_rad=[value if np.isfinite(value) else None for value in residuals],
        navigator_fit_residual_median_rad=(float(np.median(finite_residuals)) if finite_residuals.size else None),
        uncalibrated_navigator_partition_indices=np.flatnonzero(~np.asarray(fitted)).tolist(),
        reconstruction='REFLECT readout ordering; independent +,-,+ navigator fit per partition; centered 3D inverse FFT',
        navigator_policy='Fit each partition independently; retain zero correction and report partitions without sufficient navigator signal.',
        dtype=str(volume.dtype))
    return volume, report


def iter_epi3d_receivers(path, receiver_indices=None, *, scan=None, navigator_phase_branch=0):
    """Yield ``(receiver_index, complex_xyz, report)`` one receiver at a time.

    ``xyz`` denotes encoding readout/phase/partition axes, not patient axes.
    Supply an already parsed final Twix measurement as ``scan`` to avoid a
    second header parse. With Siemens VD/VE files, raw samples are memory
    mapped by receiver; no array of all receiver volumes is constructed.

    Three equally spaced +,-,+ navigators before each partition's single EPI
    train are required. The caller must validate this timing using the actual
    sequence. The phase estimate retains the existing pi-branch convention.
    """
    if isinstance(navigator_phase_branch, (bool, np.bool_)) or navigator_phase_branch not in (0, 1):
        raise ValueError('navigator_phase_branch must be 0 or 1.')
    path = Path(path)
    if scan is None:
        import twixtools
        scan = twixtools.read_twix(str(path), include_scans=[-1], parse_geometry=False,
                                  parse_pmu=False, verbose=False)[-1]
    shots, shape, receiver_count = _layout(scan)
    selected = _receiver_selection(receiver_indices, receiver_count)
    any_file_backed = any(hasattr(block, 'mem_pos') and getattr(block, 'version_is_ve', False)
                          for lines, _, _ in shots for block in lines)
    mapping = np.memmap(path, dtype=np.uint8, mode='r') if any_file_backed else None
    try:
        for receiver in selected:
            volume, report = _reconstruct_receiver(shots, shape, receiver, mapping, navigator_phase_branch)
            report['available_receiver_count'] = receiver_count
            yield receiver, volume, report
            if mapping is not None and hasattr(mapping._mmap, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
                # The interleaved raw file can otherwise become resident in
                # its entirety while visiting receivers one at a time.
                mapping._mmap.madvise(mmap.MADV_DONTNEED)
    finally:
        if mapping is not None:
            mapping._mmap.close()


def read_epi3d_sequence(path, *, expected_signature_md5=None, trigger_channel='ext1'):
    """Read the exact repeated-shot 3D Cartesian EPI encoding and command clock.

    Return a per-shot acquisition and full sequence timing. The reduced
    acquisition is valid only because every partition has the same XY
    trajectory, RF-relative ADC times, RF/ADC phase difference and command
    windows. Partition encoding remains explicit in ``kz_partition_mInv``;
    it is not a collection of independently excited physical slices.

    ADC arrays are phase-sorted with ascending readout coordinates, matching
    ``read_acdc_sequence``. Current windows remain chronological. Full CSV
    event timing includes an explicit zero command after every shot, so RF,
    navigators and inter-shot gaps are not silently assigned the last line's
    currents. No waveform or hardware action is performed here.
    """
    import pypulseq as pp
    from read_acdc_sequence import _events, _cartesian

    path = Path(path)
    sequence = pp.Sequence()
    sequence.read(str(path))
    signature = getattr(sequence, 'signature_value', None)
    if expected_signature_md5 is not None and signature != expected_signature_md5:
        raise ValueError('3D EPI Pulseq signature differs from the signature declared by the Twix.')
    matrix = np.asarray(sequence.definitions.get('Matrix', []), dtype=float).ravel()
    fov = np.asarray(sequence.definitions.get('FOV', []), dtype=float).ravel()
    if (matrix.shape != (3,) or not np.isfinite(matrix).all()
            or np.any(matrix < 2) or not np.equal(matrix, np.floor(matrix)).all()
            or np.any(matrix.astype(int) % 2)
            or fov.shape != (3,) or not np.isfinite(fov).all() or np.any(fov <= 0)
            or float(sequence.definitions.get('AcquisitionDimension', 0)) != 3):
        raise ValueError('3D EPI needs an explicit positive FOV and even three-dimensional Matrix.')
    nx, ny, nz = matrix.astype(int)
    declared_nav = sequence.definitions.get('NavigatorEchoes', 0)
    if declared_nav != 3:
        raise ValueError('This 3D EPI reader requires three leading navigators in every partition.')
    rf, adc, duration = _events(sequence, pp)
    centers = np.asarray([event['center'] for event in rf])
    k_adc, _, excitation_centers, refocusing_centers, adc_times = sequence.calculate_kspace()
    if len(refocusing_centers) or not np.allclose(excitation_centers, centers, rtol=0, atol=1e-10):
        raise ValueError('3D EPI supports excitation RF only, with unambiguous RF centers.')
    dwell = adc[0]['dwell']
    if (dwell <= 0 or any(event['samples'] != nx for event in adc)
            or not np.allclose([event['dwell'] for event in adc], dwell, rtol=0, atol=1e-12)
            or k_adc.shape != (3, len(adc)*nx) or not np.isfinite(k_adc).all()):
        raise ValueError('3D EPI ADC sample counts, dwells and trajectory dimensions disagree.')
    sample_times = np.asarray(adc_times).reshape(-1, nx)
    adc_begin = np.asarray([event['start']+event['delay'] for event in adc])
    if not np.allclose(sample_times[:, 0], adc_begin+dwell/2, rtol=0, atol=1e-10):
        raise ValueError('3D EPI ADC event windows disagree with the calculated sample centers.')
    owners = np.searchsorted(centers, adc_begin, side='right')-1
    if np.any(owners < 0):
        raise ValueError('3D EPI ADC occurs before its excitation.')
    if (np.any(adc_begin < np.asarray([event['end'] for event in rf])[owners]-1e-12)
            or not np.array_equal(np.searchsorted(centers, sample_times[:, -1], side='right')-1, owners)):
        raise ValueError('3D EPI ADC overlaps an RF event or spans excitations.')
    counts = np.bincount(owners, minlength=len(rf))
    active_rf = np.flatnonzero(counts)
    if (len(active_rf) != nz or np.any(counts[active_rf] != ny+3)
            or not np.array_equal(active_rf, np.arange(active_rf[0], len(rf)))):
        raise ValueError('3D EPI needs one equally sized imaging train per partition, with leading dummies only.')
    tr = float(sequence.definitions.get('TR', 0))
    if tr <= 0 or not np.allclose(np.diff(centers), tr, rtol=0, atol=1e-8):
        raise ValueError('3D EPI RF centers must be regularly spaced by the declared TR.')
    trajectories = k_adc.reshape(3, len(adc), nx)
    kxy_shots, time_shots, phase_shots = [], [], []
    starts_shots, ends_shots, global_starts, global_ends = [], [], [], []
    nav_centers_shots, kz_shots, orders, polarities = [], [], [], []
    imaging_adc_indices = []
    for rf_index in active_rf:
        indices = np.flatnonzero(owners == rf_index)
        imaging = indices[3:]
        origin = centers[rf_index]
        starts = np.asarray([adc[index]['start'] for index in imaging])-origin
        ends = np.asarray([adc[index]['end'] for index in imaging])-origin
        ends[:-1] = starts[1:]
        times = sample_times[imaging]-origin
        if (np.any(starts <= 0) or np.any(ends <= starts) or np.any(np.diff(starts) <= 0)
                or np.any(times.min(axis=1) < starts-1e-12)
                or np.any(times.max(axis=1) >= ends)):
            raise ValueError('3D EPI command windows must contain their ADC samples chronologically.')
        kz = trajectories[2, imaging]
        if not np.allclose(kz, kz[0, 0], rtol=0, atol=1e-3):
            raise ValueError('3D EPI kz must remain constant during one partition train.')
        kx, ky, sorted_times, polarity, order, measured_fov, offset = _cartesian(
            trajectories[0, imaging].copy(), trajectories[1, imaging].copy(), times.copy(), 'epi')
        if not np.allclose(measured_fov, fov[:2], rtol=1e-4, atol=1e-6):
            raise ValueError('3D EPI measured XY FOV differs from the sequence definition.')
        phases = np.asarray([rf[rf_index]['phase']-adc[index]['phase'] for index in imaging])[order]
        phases = np.angle(np.exp(1j*phases))
        nav_centers = sample_times[indices[:3]].mean(axis=1)-origin
        if not np.allclose(np.diff(nav_centers), np.diff(nav_centers)[0], rtol=0, atol=1e-9):
            raise ValueError('3D EPI symmetric navigator correction needs equally spaced echoes.')
        nav_polarities = np.sign(np.diff(trajectories[0, indices[:3]], axis=1).mean(axis=1))
        if not np.array_equal(nav_polarities, [1, -1, 1]):
            raise ValueError('3D EPI navigators must have +,-,+ readout polarities.')
        kxy_shots.append(np.stack((kx, ky)))
        time_shots.append(sorted_times)
        phase_shots.append(phases)
        starts_shots.append(starts)
        ends_shots.append(ends)
        global_starts.append(starts+origin)
        global_ends.append(ends+origin)
        nav_centers_shots.append(nav_centers)
        kz_shots.append(float(np.mean(kz)))
        orders.append(order)
        polarities.append(polarity)
        imaging_adc_indices.extend(imaging.tolist())
    comparisons = ((kxy_shots, 1e-3, 'XY trajectories'), (time_shots, 1e-9, 'ADC timing'),
                   (phase_shots, 1e-8, 'RF/ADC phase differences'),
                   (starts_shots, 1e-9, 'command starts'), (ends_shots, 1e-9, 'command ends'),
                   (nav_centers_shots, 1e-9, 'navigator timing'))
    errors = {}
    for values, tolerance, name in comparisons:
        errors[name] = float(max(np.max(np.abs(value-values[0])) for value in values))
        if errors[name] > tolerance:
            raise ValueError(f'3D EPI partitions have different {name}; a shared repeated waveform model is invalid.')
    if any(not np.array_equal(value, orders[0]) for value in orders):
        raise ValueError('3D EPI phase-line acquisition order changes between partitions.')
    kz = np.asarray(kz_shots)
    expected_kz = (np.arange(nz)-nz//2)/fov[2]
    if not np.allclose(kz, expected_kz, rtol=0, atol=1e-3):
        raise ValueError('3D EPI partitions must form the complete ascending centered kz grid.')
    # Validate actual output events, not just the declared output count.
    elapsed, triggers = 0., []
    for block_number in sequence.block_events:
        block = sequence.get_block(block_number)
        for event in (getattr(block, 'trigger', None) or {}).values():
            if event.type == 'output' and event.channel == trigger_channel:
                triggers.append(elapsed+float(event.delay))
        elapsed += float(sequence.block_durations[block_number])
    starts_global = np.asarray(global_starts)
    ends_global = np.asarray(global_ends)
    if len(triggers) != nz*ny or not np.allclose(triggers, starts_global.ravel(), rtol=0, atol=1e-9):
        raise ValueError('3D EPI needs exactly one matching output trigger at every imaging-command start.')
    order = orders[0]
    times = time_shots[0]
    kx, ky = kxy_shots[0]
    te = float(sequence.definitions.get('TE', np.nan))
    measured_te = float(np.interp(0., kx[ny//2], times[ny//2]))
    if not np.isfinite(te) or abs(te-measured_te) > 1e-7:
        raise ValueError('3D EPI declared TE disagrees with the sampled k-space center.')
    acquisition = dict(sequence_kind='epi3d', nx=int(nx), ny=int(ny), nz=int(nz),
        kx_mInv=kx, ky_mInv=ky, kz_partition_mInv=kz,
        time_s=times, rf_time_s=np.zeros_like(times),
        phase_offset_rad=np.repeat(phase_shots[0][:, None], nx, axis=1),
        line_index=np.repeat((order+1)[:, None], nx, axis=1),
        line_start_s=starts_shots[0][:, None], line_end_s=ends_shots[0][:, None],
        polarity=polarities[0], dwell_s=dwell, TE_s=te, TR_s=tr,
        FOV_m=fov, matrix_size=matrix.astype(int), image_shape=(int(ny), int(nx)),
        navigator_echo_center_s=nav_centers_shots[0],
        echo_spacing_s=float(np.median(np.diff(starts_shots[0]))))
    event_times, event_states = [], []
    for shot_starts, shot_ends in zip(starts_global, ends_global):
        event_times.extend(shot_starts.tolist()+[float(shot_ends[-1])])
        event_states.extend(list(range(ny))+[-1])
    first_trigger = float(triggers[0])
    timing = dict(sequence_kind='epi3d', matrix_size=matrix.astype(int).tolist(),
        state_start_s=starts_shots[0].tolist(), state_end_s=ends_shots[0].tolist(),
        state_edges_s=np.r_[starts_shots[0], ends_shots[0][-1]].tolist(),
        navigator_count_per_partition=3, dummy_rf_count=int(active_rf[0]), partition_count=int(nz),
        shared_partition_checks=errors, repeated_waveform_per_partition=True,
        RF_relative_imaging_origin_s=float(starts_shots[0][0]),
        active_rf_center_sequence_s=centers[active_rf].tolist(),
        active_rf_start_sequence_s=[rf[index]['start'] for index in active_rf],
        global_state_start_sequence_s=starts_global.ravel().tolist(),
        global_state_end_sequence_s=ends_global.ravel().tolist(),
        trigger_channel=trigger_channel, trigger_count=len(triggers), verified_per_line_triggers=True,
        trigger_sequence_s=triggers, first_imaging_trigger_sequence_s=first_trigger,
        waveform_event_time_sequence_s=event_times,
        waveform_event_time_first_trigger_s=(np.asarray(event_times)-first_trigger).tolist(),
        waveform_event_state_index=event_states,
        waveform_zero_state_index=-1,
        waveform_contract='Repeat the same chronological line commands in every partition; insert explicit zero after each train and throughout RF/navigators/gaps.',
        sequence_duration_s=duration, TR_s=tr, TE_s=te,
        sequence_file=str(path.resolve()), sequence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sequence_signature_md5=signature,
        full_3D_excitation='One whole-slab excitation per kz partition; a partition is not an RF-selected anatomical slice.')
    return acquisition, timing


__all__ = ['iter_epi3d_receivers', 'read_epi3d_sequence']
