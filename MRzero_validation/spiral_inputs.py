"""Measured spiral inputs for notebook 07, with independent GRE/B0 validation.

Spiral Twix segments are transport chunks of one continuous ADC. They are
concatenated in acquisition order and matched to the actual Pulseq samples.
The three Cartesian navigators remain available for inspection; an EPI
odd/even-line phase correction is never applied to a spiral.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
from pathlib import Path
import sys

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "Experiment"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from continuous_reconstruction import ContinuousReconstruction
from measured_continuous import reconstruct_b0_echoes, reconstruct_receivers
from prepare_acdc_inputs import check_physical_geometry, read_gre
from read_acdc_b0 import read_b0_coil_images
from read_acdc_sequence import read_sequence
from read_spiral_sequence import read_spiral_sequence
from receiver_channels import receiver_channel_ids
from slice_geometry import encoding_origin, normalized_quaternion, slice_basis
from twin_inputs import _check_slice_geometry, _estimate_field, _field_echo_count


def _spiral_twix_data(scan, acquisition):
    """Validate one single-shot raw measurement without Cartesian assumptions."""
    acquired = [block for block in scan["mdb"] if block.is_image_scan()]
    navigators = [block for block in scan["mdb"] if block.is_flag_set("PHASCOR")]
    expected_samples = np.asarray(acquisition["kx_mInv"]).size
    expected_navigators = int(acquisition["navigator_count"])
    if not acquired or len(navigators) != expected_navigators:
        raise ValueError("Spiral Twix imaging/navigator ADC counts disagree with the selected Pulseq sequence.")
    first_data = np.asarray(acquired[0].data)
    if first_data.ndim != 2 or first_data.shape[0] < 1:
        raise ValueError("Spiral Twix requires at least one receiver.")
    channels = first_data.shape[0]
    lengths = []
    for block in acquired:
        data = np.asarray(block.data)
        if (data.ndim != 2 or data.shape[0] != channels or data.shape[1] < 1
                or not np.isfinite(data).all() or block.is_flag_set("REFLECT")
                or block.mdh.CutOff.Pre or block.mdh.CutOff.Post):
            raise ValueError("Spiral imaging segments require finite chronological samples, constant receivers and no REFLECT/cutoff.")
        lengths.append(data.shape[1])
    if sum(lengths) != expected_samples:
        raise ValueError(f"Spiral Twix has {sum(lengths)} imaging samples; Pulseq requires {expected_samples}. "
                         "Select matching raw data and sequence; samples are not cropped or padded.")
    segment_limit = int(acquisition.get("max_adc_segment_length", expected_samples))
    if max(lengths) > segment_limit:
        raise ValueError("Spiral Twix ADC segments exceed Pulseq MaxAdcSegmentLength.")
    for counter in ("cAve", "cSlc", "cPar", "cEco", "cPhs", "cRep", "cLin", "cSeg"):
        if len({getattr(block, counter) for block in acquired}) != 1:
            raise ValueError(f"Single-shot spiral Twix changes {counter}; multiple shots/slices/echoes are unsupported.")
    segment_counters = [int(block.cSet) for block in acquired]
    if segment_counters != list(range(len(acquired))):
        raise ValueError("Spiral Twix cSet must enumerate the contiguous ADC transport segments in file order.")
    for block, expected in zip(navigators, acquisition["navigator_sample_count"]):
        if (np.asarray(block.data).shape != (channels, expected)
                or not np.isfinite(block.data).all() or block.mdh.CutOff.Pre or block.mdh.CutOff.Post):
            raise ValueError("Spiral navigator receiver/sample counts disagree with Pulseq.")
    if navigators and [block.is_flag_set("REFLECT") for block in navigators] != [False, True, False]:
        raise ValueError("The three spiral navigators must have +,-,+ acquisition polarities.")
    metadata_blocks = acquired + navigators
    channel_ids = receiver_channel_ids(metadata_blocks, channels, label='Spiral Twix')
    entry = metadata_blocks[0].mdh.SliceData
    position = np.asarray([entry.SlicePos.Sag, entry.SlicePos.Cor, entry.SlicePos.Tra], float)
    quaternion = normalized_quaternion(list(entry.Quaternion))
    if not np.isfinite(position).all():
        raise ValueError("Spiral Twix slice position must be finite.")
    basis = slice_basis({"slice_quaternion_wxyz": quaternion})
    for block in metadata_blocks[1:]:
        entry = block.mdh.SliceData
        other_position = [entry.SlicePos.Sag, entry.SlicePos.Cor, entry.SlicePos.Tra]
        if (not np.allclose(other_position, position, rtol=0, atol=1e-3)
                or not np.allclose(slice_basis({"slice_quaternion_wxyz": list(entry.Quaternion)}),
                                   basis, rtol=0, atol=1e-6)):
            raise ValueError("Spiral Twix geometry changes between ADC segments or navigators.")
    header = scan["hdr"]["MeasYaps"]
    slices = header["sSliceArray"]["asSlice"]
    if len(slices) != 1:
        raise ValueError("Spiral Twix must describe exactly one slice.")
    slice_header = slices[0]
    fov = np.asarray([slice_header["dReadoutFOV"], slice_header["dPhaseFOV"]], float) * 1e-3
    shape = list(acquisition["image_shape"])
    if not np.allclose(fov, acquisition["FOV_m"], rtol=0, atol=1e-6):
        raise ValueError("Spiral Twix FOV disagrees with the selected Pulseq sequence.")
    grid_header = header.get("sKSpace", {})
    for key, expected in (("lBaseResolution", shape[1]), ("lPhaseEncodingLines", shape[0])):
        if key in grid_header and int(grid_header[key]) != expected:
            raise ValueError("Spiral Twix nominal image matrix disagrees with the selected Pulseq sequence.")
    header_te = float(header["alTE"][0]) * 1e-6
    header_tr = float(header["alTR"][0]) * 1e-6
    if (not np.isclose(header_te, acquisition["TE_s"], rtol=0, atol=1e-6)
            or not np.isclose(header_tr, acquisition["TR_s"], rtol=0, atol=1e-6)):
        raise ValueError("Spiral Twix TE/TR disagree with the selected Pulseq sequence.")
    header_dwell = float(header["sRXSPEC"]["alDwellTime"][0]) * 1e-9
    # On the supplied scanner this header stores the first navigator dwell,
    # not the later imaging dwell. Never stretch the spiral using that value.
    first_dwell = (acquisition["navigator_dwell_s"][0] if navigators else acquisition["dwell_s"])
    if not np.isclose(header_dwell, first_dwell, rtol=0, atol=1e-12):
        raise ValueError("Spiral Twix header dwell disagrees with the first Pulseq ADC event.")
    metadata = dict(TE_s=header_te, TR_s=header_tr, dwell_s=float(acquisition["dwell_s"]),
        header_first_adc_dwell_s=header_dwell, image_shape=shape, receiver_count=channels,
        navigator_count=len(navigators), header_flip_deg=float(header["adFlipAngleDegree"][0]),
        FOV_m=fov.tolist(), slice_position_mm=position.tolist(),
        slice_quaternion_wxyz=quaternion.tolist(), slice_thickness_mm=float(slice_header["dThickness"]),
        imaging_sample_count=expected_samples, imaging_adc_segment_count=len(acquired),
        imaging_adc_segment_samples=lengths, imaging_adc_segment_cSet=segment_counters,
        trajectory_source="Pulseq imaging ADC; Twix segments concatenated in file order",
        navigator_phase_correction_applied=False,
        phase_correction="None; raw spiral complex samples are retained.")
    if "flip_deg" in acquisition:
        metadata.update(pulseq_flip_deg=float(acquisition["flip_deg"]),
                        flip_angle_source="Actual Pulseq excitation; scanner protocol header is advisory.")
    if not np.isfinite(metadata["slice_thickness_mm"]) or metadata["slice_thickness_mm"] <= 0:
        raise ValueError("Spiral Twix slice thickness must be finite and positive.")
    metadata["encoding_origin_m"] = encoding_origin(metadata).tolist()
    if channel_ids is not None:
        metadata['receiver_channel_ids'] = channel_ids
        metadata['receiver_channel_id_source'] = 'Twix ChannelId headers in complex-data receiver-row order'
    # A leading shot axis keeps sample layouts uniform across the simulator,
    # reconstruction and measured-data diagnostics. No line sorting occurs.
    samples = np.concatenate([np.asarray(block.data, complex) for block in acquired], axis=1)[None]
    navigator_data = (np.stack([np.asarray(block.data, complex) for block in navigators])
                      if navigators else np.empty((0, channels, 0), complex))
    diagnostics = dict(source_format="Siemens Twix", has_raw_kspace=True,
        array_axes=["shot", "receiver", "sample"], kspace=samples, corrected_kspace=samples,
        navigator_kspace=navigator_data, navigator_phase_correction_applied=False,
        navigator_handling="Retained as acquired; EPI odd/even-line correction is inapplicable to the spiral.",
        imaging_segment_samples=lengths, imaging_segment_cSet=segment_counters,
        sample_order="Chronological acquisition order, excluding navigators and SYNCDATA.")
    return metadata, diagnostics


def read_spiral_scan(path, acquisition, *, reconstruction_device="cpu",
                     reconstruction_dtype=torch.float64, return_intermediates=False):
    """Read and reconstruct one spiral Twix using its validated ADC trajectory."""
    import twixtools

    path = Path(path)
    if path.suffix.lower() != ".dat" or acquisition.get("sequence_kind") != "spiral":
        raise ValueError("Spiral scan loading requires a Twix .dat and a parsed spiral acquisition.")
    with contextlib.redirect_stdout(io.StringIO()):
        scans = twixtools.read_twix(str(path), verbose=False)
    metadata, diagnostics = _spiral_twix_data(scans[-1], acquisition)
    actual_acquisition = dict(acquisition, encoding_origin_m=np.asarray(metadata["encoding_origin_m"]))
    operator = ContinuousReconstruction(actual_acquisition, reconstruction_device, reconstruction_dtype)
    # Operator input has receiver-leading axes: (receiver, shot, sample).
    coils = operator.numpy_image(np.moveaxis(diagnostics["kspace"], 1, 0))
    image = np.sqrt(np.sum(np.abs(coils) ** 2, axis=0))
    diagnostics.update(coil_images=np.moveaxis(coils, 0, 1),
        uncorrected_coil_images=np.moveaxis(coils, 0, 1), uncorrected_image=image,
        continuous_reconstruction=dict(operator.report, receiver_count=coils.shape[0],
            receiver_phase_reapplied=False, navigator_phase_correction_applied=False))
    return (image, metadata, diagnostics) if return_intermediates else (image, metadata)


def read_spiral_inputs(folder, *, scan_sequence_path=None, scan_path=None,
                       reference_sequence_path=None, reference_scan_path=None,
                       b0_sequence_path=None, b0_scan_path=None, b0_echo_count=None,
                       reconstruction_device="cpu", reconstruction_dtype=torch.float64,
                       reconstruction_method="nufft", density_compensation="pipe_menon"):
    """Read notebook 07 inputs while retaining the notebook 06 object contract.

    GRE and B0 retain their own Cartesian trajectories and existing validation.
    Explicit sequence paths are recommended when the folder contains multiple
    candidates. All relative paths are interpreted relative to ``folder``.
    """
    folder = Path(folder)

    def selected(value, pattern, label):
        if value is None:
            candidates = sorted(folder.glob(pattern))
            if len(candidates) != 1:
                names = ", ".join(path.name for path in candidates) or "(none)"
                raise ValueError(f"Expected one {label}, found {len(candidates)}: {names}. Set its path explicitly.")
            path = candidates[0]
        else:
            path = Path(value)
            if not path.is_absolute():
                path = folder / path
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path

    paths = dict(scan_sequence=selected(scan_sequence_path, "sequences/*spiral*.seq", "spiral sequence"),
        scan=selected(scan_path, "*spiral*scan.dat", "spiral scan"),
        reference=selected(reference_scan_path, "*reference*.dat", "reference GRE scan"),
        b0=selected(b0_scan_path, "*b0_map.dat", "B0 scan"),
        b0_sequence=selected(b0_sequence_path, "sequences/*32echo.seq", "32-echo B0 sequence"))
    if len({paths[key].resolve() for key in ("scan", "reference", "b0")}) != 3:
        raise ValueError("Spiral, reference GRE and B0 must use distinct measurement files.")
    acquisition, timing = read_spiral_sequence(paths["scan_sequence"])
    if reconstruction_method not in ("direct", "nufft"):
        raise ValueError("reconstruction_method must be 'direct' or 'nufft'.")
    acquisition["reconstruction_method"] = reconstruction_method
    if density_compensation == "pipe_menon":
        from continuous_reconstruction import compute_spiral_density_weights
        weights, report = compute_spiral_density_weights(acquisition, device=reconstruction_device)
        acquisition["density_compensation_weights"] = weights
        acquisition["density_compensation_report"] = report
    elif density_compensation == "radial":
        if not acquisition["uniform_radial_density"]:
            raise ValueError("Radial density weights require a declared uniform radial-density spiral.")
        radius = np.hypot(acquisition["kx_mInv"], acquisition["ky_mInv"])
        # A finite central cell avoids zero-weighting the sampled k-space
        # centre; absolute weight scale cancels in the normalized adjoint.
        weights = np.maximum(radius, .5 / np.max(acquisition["FOV_m"]))
        acquisition["density_compensation_weights"] = weights / weights.mean()
        acquisition["density_compensation_report"] = dict(method="radial", central_cell_floor_mInv=float(.5 / np.max(acquisition["FOV_m"])))
    elif density_compensation == "none":
        acquisition["density_compensation_report"] = dict(method="none")
    else:
        raise ValueError("density_compensation must be 'pipe_menon', 'radial' or 'none'.")
    if reference_sequence_path is None:
        candidates = []
        for path in sorted((folder / "sequences").glob("*.seq")):
            if path.resolve() in {paths["scan_sequence"].resolve(), paths["b0_sequence"].resolve()}:
                continue
            try:
                candidate, _ = read_sequence(path)
                if candidate.get("sequence_kind") == "gre":
                    candidates.append(path)
            except ValueError:
                continue
        if len(candidates) != 1:
            raise ValueError("Set reference_sequence_path explicitly: the folder does not contain a unique mono-echo GRE sequence.")
        reference_sequence_path = candidates[0]
    paths["reference_sequence"] = selected(reference_sequence_path, "", "reference GRE sequence")
    reference_acquisition, reference_timing = read_sequence(paths["reference_sequence"])
    if reference_acquisition.get("sequence_kind") != "gre":
        raise ValueError("The reference sequence must be mono-echo Cartesian GRE.")
    raw = read_b0_coil_images(paths["b0"], paths["b0_sequence"])
    _field_echo_count(b0_echo_count, len(raw["metadata"]["TE_s"]))
    gre, gd = read_gre(paths["reference"], reference_acquisition, return_intermediates=True)
    reference_acquisition["encoding_origin_m"] = encoding_origin(gre["twix_metadata"])
    gre["magnitude"], _ = reconstruct_receivers(gd, reference_acquisition,
        device=reconstruction_device, dtype=reconstruction_dtype)
    scan, scan_metadata, sd = read_spiral_scan(paths["scan"], acquisition,
        reconstruction_device=reconstruction_device, reconstruction_dtype=reconstruction_dtype,
        return_intermediates=True)
    acquisition["encoding_origin_m"] = encoding_origin(scan_metadata)
    check_physical_geometry(gre, scan_metadata, acquisition,
        np.isfinite(gre["magnitude"]), gre["X_m"], gre["Y_m"], dicom_input=False)
    try:
        geometry_checks = _check_slice_geometry(gre["twix_metadata"], scan_metadata, raw["metadata"])
    except ValueError as error:
        raise ValueError(str(error).replace("EPI", "spiral")) from error
    geometry_checks = [check.replace("EPI", "spiral") for check in geometry_checks]
    reconstruct_b0_echoes(raw, paths["b0_sequence"], device=reconstruction_device, dtype=reconstruction_dtype)
    echo_times = np.asarray(raw["metadata"]["TE_s"])
    field, field_echo_count = _estimate_field(raw, echo_times, b0_echo_count)
    reference_report = dict(mode="pulseq", validated=True, selection="explicit")
    gre["reference_sequence_validation"] = reference_report
    return dict(paths=paths, acquisition=acquisition, timing=timing, gre=gre,
        scan=scan, scan_metadata=scan_metadata, scan_diagnostics=sd, b0_raw=raw,
        reference_acquisition=reference_acquisition, reference_timing=reference_timing,
        reference_diagnostics=gd, reference_sequence_validation=reference_report,
        geometry_checks=geometry_checks, field=field, field_echo_count=field_echo_count,
        echo_times_s=echo_times, manifest=None,
        source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()})


__all__ = ["read_spiral_inputs", "read_spiral_scan"]
