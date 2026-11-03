"""Cartesian 3D Twix images on their own measured physical grids.

No anatomical registration or orientation search is performed. The default
geometry follows the Siemens-to-ISMRMRD converter: the MDH quaternion has PRS
columns; our arrays have RPS axes. In particular, it is not the historical
2D notebook's direct interpretation of quaternion column zero as readout.
An explicitly declared native-to-patient transform can override this convention.

Primary reference:
https://github.com/ismrmrd/siemens_to_ismrmrd/blob/master/main.cpp#L1054-L1065
Encoded/reconstructed FOV:
https://github.com/ismrmrd/siemens_to_ismrmrd/blob/master/parameter_maps/IsmrmrdParameterMap_Siemens.xsl#L401-L454
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import inspect
import io
import json
import mmap
import re
from pathlib import Path
import tempfile

import numpy as np
from scipy import fft, ndimage
from scipy.spatial.transform import Rotation


RECONSTRUCTION_VERSION = "cartesian3d_twix_rps_mrd_v3"
_KINDS = {"gre": "gre", "reference": "gre", "epi": "epi", "scan": "epi", "b0": "b0"}


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(*values, default=None):
    for value in values:
        try:
            if isinstance(value, (bool, list, tuple, dict)) or np.iscomplexobj(value):
                continue
            result = float(value)
            if np.isfinite(result):
                return result
        except (ValueError, TypeError):
            pass
    return default


def _integer(value, name):
    number = _number(value)
    if number is None or number < 1 or number != int(number):
        raise ValueError(f"{name} must be a positive integer.")
    return int(number)


def _receiver_selection(indices, channels):
    if indices is None:
        return list(range(channels))
    try:
        values = list(indices)
    except TypeError:
        raise ValueError("receiver_indices must contain unique zero-based integer receiver indices.") from None
    if (not values or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                          for v in values)
            or len(set(values)) != len(values) or min(values) < 0 or max(values) >= channels):
        raise ValueError("receiver_indices must contain unique zero-based integer receiver indices in range.")
    return [int(v) for v in values]


def _geometry(scan, blocks, patient_from_native_m=None):
    hdr = scan["hdr"]
    yaps = hdr["MeasYaps"]
    kspace = yaps["sKSpace"]
    slices = yaps["sSliceArray"]["asSlice"]
    if len(slices) != 1 or int(kspace.get("ucDimension", 4)) != 4:
        raise ValueError("read_twix_volume needs one Cartesian 3D slab.")
    slab = slices[0]
    first = blocks[0].mdh
    position = np.asarray([first.SliceData.SlicePos.Sag, first.SliceData.SlicePos.Cor,
                           first.SliceData.SlicePos.Tra], dtype=float) * .001
    quaternion = np.asarray(first.SliceData.Quaternion, dtype=float)
    if (position.shape != (3,) or not np.isfinite(position).all()
            or quaternion.shape != (4,) or not np.isfinite(quaternion).all()
            or not np.isclose(np.linalg.norm(quaternion), 1., rtol=0, atol=1e-6)):
        raise ValueError("Twix volume needs a finite MDH center and unit WXYZ quaternion.")
    quaternion /= np.linalg.norm(quaternion)
    prs = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    basis = prs[:, [1, 0, 2]]
    raw_shape = np.array([int(first.SamplesInScan), max(b.cLin for b in blocks) + 1,
                          max(b.cPar for b in blocks) + 1])
    if np.any(raw_shape < 2) or np.any(raw_shape % 2):
        raise ValueError("A full even Cartesian 3D matrix is required.")
    meas, config, dicom = (hdr.get(k, {}) for k in ("Meas", "Config", "Dicom"))
    recon_shape = np.array([
        _integer(_number(config.get("N0FImageColumns"), config.get("ImageColumns"),
                         config.get("NImageCols"), kspace.get("lBaseResolution")), "reconstructed readout"),
        _integer(_number(config.get("N0FImageLines"), config.get("ImageLines"),
                         config.get("NImageLins"), kspace.get("lBaseResolution")), "reconstructed phase"),
        _integer(kspace.get("lImagesPerSlab", raw_shape[2]), "reconstructed partitions")])
    fov = np.array([slab["dReadoutFOV"], slab["dPhaseFOV"], slab["dThickness"]], dtype=float) * .001
    if not np.isfinite(fov).all() or np.any(fov <= 0):
        raise ValueError("The slab must declare three finite positive FOV dimensions.")
    if np.any(recon_shape > raw_shape) or np.any((raw_shape - recon_shape) % 2):
        raise ValueError("Only an integral centered image crop is supported; no interpolation or partial Fourier.")
    spacing = fov / recon_shape
    encoded_fov = raw_shape * spacing
    # Actual acquisition fields take precedence over the stale Config OS field
    # carried by the Pulseq scans 198/199 (Config says2, actual Meas says1).
    readout_os = _number(meas.get("flReadoutOSFactor"), dicom.get("flReadoutOSFactor"),
                         config.get("ReadoutOversamplingFactor"), default=raw_shape[0] / recon_shape[0])
    phase_os = _number(config.get("phaseOversampling"), dicom.get("PhaseOversampling"),
                       meas.get("PhaseOversampling"), default=raw_shape[1] / recon_shape[1] - 1)
    partition_os = _number(kspace.get("dSliceOversamplingForDialog"),
                           default=raw_shape[2] / recon_shape[2] - 1)
    if not np.allclose(raw_shape / recon_shape, [readout_os, 1 + phase_os, 1 + partition_os],
                       rtol=0, atol=1e-6):
        raise ValueError("Actual Cartesian matrix and reconstructed FOV disagree with declared oversampling.")
    for block in blocks:
        mdh = block.mdh
        center = np.asarray([mdh.SliceData.SlicePos.Sag, mdh.SliceData.SlicePos.Cor,
                             mdh.SliceData.SlicePos.Tra]) * .001
        q = np.asarray(mdh.SliceData.Quaternion, dtype=float)
        if (not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1., rtol=0, atol=1e-6)
                or not np.allclose(center, position, rtol=0, atol=1e-6)
                or not (np.allclose(q / np.linalg.norm(q), quaternion, rtol=0, atol=1e-6)
                        or np.allclose(q / np.linalg.norm(q), -quaternion, rtol=0, atol=1e-6))):
            raise ValueError("MDH slab geometry changes within the volume.")
        if (int(mdh.SamplesInScan) != raw_shape[0]
                or [int(mdh.CenterCol), int(mdh.CenterLin), int(mdh.CenterPar)] != (raw_shape // 2).tolist()):
            raise ValueError("Volume must use a full centered Cartesian k-space matrix.")
    native_to_patient = np.eye(4)
    native_to_patient[:3, :3], native_to_patient[:3, 3] = basis, position
    overridden = patient_from_native_m is not None
    if overridden:
        if np.iscomplexobj(patient_from_native_m):
            raise ValueError("patient_from_native_m must be a finite orthogonal 4x4 transform in metres.")
        try:
            supplied = np.asarray(patient_from_native_m, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("patient_from_native_m must be a finite orthogonal 4x4 transform in metres.") from None
        if (supplied.shape != (4, 4) or not np.isfinite(supplied).all()
                or not np.allclose(supplied[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
                or not np.allclose(supplied[:3, :3].T @ supplied[:3, :3], np.eye(3), rtol=0, atol=1e-8)):
            raise ValueError("patient_from_native_m must be a finite orthogonal 4x4 transform in metres.")
        native_to_patient = supplied.copy()
        basis, position = supplied[:3, :3].copy(), supplied[:3, 3].copy()
    crop_start = (raw_shape - recon_shape) // 2
    crop = tuple(slice(int(a), int(a + n)) for a, n in zip(crop_start, recon_shape))
    return {"raw_shape": raw_shape, "shape": recon_shape, "spacing_m": spacing,
            "FOV_m": fov, "encoded_FOV_m": encoded_fov, "crop": crop,
            "basis_patient": basis, "center_m": position,
            "native_to_patient_m": native_to_patient,
            "MDH_quaternion_wxyz": quaternion.tolist(),
            "geometry_override": overridden,
            "geometry_convention": ("Explicit native-to-patient transform in metres." if overridden else
                "Siemens-to-ISMRMRD: MDH quaternion columns are (phase,readout,slice); data arrays are (readout,phase,slice). No anatomical orientation fit or additional pixel flip."),
            "pulseq_geometry_limitation": "The Twix slab pose is retained as declared; a Pulseq sequence's gradient-axis mapping and trajectory are not inferred from image anatomy."}


def _read_receiver(block, receiver, source_bytes):
    """Read one channel without fetching all32 channels for every receiver pass."""
    if source_bytes is None or getattr(block, "mem_pos", None) is None:
        return np.asarray(block.data[receiver], dtype=np.complex64)
    from twixtools import mdh_def
    if block.version_is_ve:
        initial = ctypes.sizeof(mdh_def.Scan_header)
        channel_header = ctypes.sizeof(mdh_def.Channel_header)
    else:
        initial = 0
        channel_header = ctypes.sizeof(mdh_def.VB17_header)
    offset = int(block.mem_pos) + initial + receiver * int(block.block_len) + channel_header
    return np.frombuffer(source_bytes, dtype=np.complex64, count=int(block.mdh.SamplesInScan), offset=offset)


def _coil_volume(blocks, receiver, geometry, source_bytes):
    nro, npe, npar = geometry["raw_shape"]
    kspace = np.empty((int(nro), int(npe), int(npar)), dtype=np.complex64)
    for block in blocks:
        data = _read_receiver(block, receiver, source_bytes)
        kspace[:, block.cLin, block.cPar] = data[::-1] if block.is_flag_set("REFLECT") else data
    image = fft.fftshift(fft.ifftn(fft.ifftshift(kspace), workers=1, overwrite_x=True))
    return np.ascontiguousarray(image[geometry["crop"]])


def _unwrap_b0(pair, pair_weight, first_power, second_power, delta_te):
    from skimage.restoration import unwrap_phase
    magnitude = np.sqrt(first_power)
    second_magnitude = np.sqrt(second_power)
    coherence = np.divide(abs(pair), pair_weight, out=np.zeros_like(pair_weight), where=pair_weight > 0)
    mask = ((magnitude > .05 * magnitude.max())
            & (second_magnitude > .05 * second_magnitude.max()) & (coherence >= .2)
            & np.isfinite(pair) & (pair_weight > 0))
    labels, number = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    allowed = sizes >= 8
    allowed[0] = False
    mask &= allowed[labels]
    wrapped = np.angle(pair)
    unwrapped = np.full(pair.shape, np.nan, dtype=np.float64)
    options = {"rng": 0} if "rng" in inspect.signature(unwrap_phase).parameters else {"seed": 0}
    components = 0
    for label, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None or not allowed[label]:
            continue
        selected = labels[box] == label
        phase = np.ma.array(wrapped[box], mask=~selected)
        squeezed = phase.squeeze()
        if squeezed.ndim == 1:
            values = np.unwrap(np.asarray(squeezed)).reshape(phase.shape)
        else:
            values = np.ma.getdata(unwrap_phase(squeezed, **options)).reshape(phase.shape)
        weights = np.where(selected, pair_weight[box], -1)
        anchor = np.unravel_index(np.argmax(weights), weights.shape)
        # The component has an unknown integer frequency branch. Preserve the
        # principal phase at one strong voxel; never infer an absolute branch.
        values -= 2 * np.pi * np.rint((values[anchor] - wrapped[box][anchor]) / (2 * np.pi))
        unwrapped[box][selected] = values[selected]
        components += 1
    field = -unwrapped / (2 * np.pi * delta_te)
    principal = np.where(mask, -wrapped / (2 * np.pi * delta_te), np.nan)
    return field.astype(np.float32), principal.astype(np.float32), mask, coherence.astype(np.float32), components


def read_twix_volume(path, kind, receiver_indices=None, cache_dir=None, *, patient_from_native_m=None):
    """Return RSS magnitude and optionally B0 on native (readout,phase,slice) axes.

    ``axes_m`` contains three local vectors, ``basis_patient`` their direction
    columns and ``center_m`` the measured center: patient= center+basis@local.
    ``native_to_patient_m`` is their homogeneous affine. A geometry override is
    explicit, applies only to this record, and is recorded in metadata/cache.

    Reconstruct one receiver and one echo at a time. The returned record does
    not retain all coil images in RAM. No extrapolation, common-grid resampling,
    field subtraction or coil-current assumption is performed.
    """
    path = Path(path).resolve()
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ValueError("kind must be gre/reference, epi/scan or b0.")
    kind = _KINDS[kind]
    if receiver_indices is not None:
        try:
            receiver_indices = list(receiver_indices)
        except TypeError:
            raise ValueError("receiver_indices must contain unique zero-based integer receiver indices.") from None
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               for v in receiver_indices):
            raise ValueError("receiver_indices must contain unique zero-based integer receiver indices.")
        receiver_indices = [int(v) for v in receiver_indices]
    if patient_from_native_m is not None:
        if np.iscomplexobj(patient_from_native_m):
            raise ValueError("patient_from_native_m must be a finite orthogonal 4x4 transform in metres.")
        try:
            patient_from_native_m = np.asarray(patient_from_native_m, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("patient_from_native_m must be a finite orthogonal 4x4 transform in metres.") from None
        if (patient_from_native_m.shape != (4, 4) or not np.isfinite(patient_from_native_m).all()
                or not np.allclose(patient_from_native_m[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
                or not np.allclose(patient_from_native_m[:3, :3].T @ patient_from_native_m[:3, :3],
                                   np.eye(3), rtol=0, atol=1e-8)):
            raise ValueError("patient_from_native_m must be a finite orthogonal 4x4 transform in metres.")
    digest = _hash(path)
    override_json = None if patient_from_native_m is None else np.asarray(patient_from_native_m).tolist()
    signature = json.dumps({"version": RECONSTRUCTION_VERSION, "source": digest, "kind": kind,
        "receivers": None if receiver_indices is None else list(receiver_indices),
        "patient_from_native_m": override_json}, sort_keys=True)
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / (hashlib.sha256(signature.encode()).hexdigest() + ".npz")
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as stored:
                result = {key: stored[key] for key in stored.files if key not in {"metadata_json", "axis0_m", "axis1_m", "axis2_m"}}
                result["axes_m"] = tuple(stored[f"axis{i}_m"] for i in range(3))
                result["metadata"] = json.loads(stored["metadata_json"].item())
            result["metadata"]["cache_hit"] = True
            return result
    import twixtools
    with contextlib.redirect_stdout(io.StringIO()):
        scan = twixtools.read_twix(str(path), parse_geometry=False, verbose=False)[-1]
    blocks = [block for block in scan["mdb"] if block.is_image_scan()]
    if not blocks:
        raise ValueError("No Cartesian volume image data found.")
    for name in ("cAve", "cSlc", "cPhs", "cRep", "cSet", "cSeg"):
        if len({getattr(block, name) for block in blocks}) != 1:
            raise ValueError("Select one slab, average, repetition, phase and set before reading a volume.")
    echoes = sorted({block.cEco for block in blocks})
    if echoes != list(range(len(echoes))) or (kind != "b0" and len(echoes) != 1) or (kind == "b0" and len(echoes) < 2):
        raise ValueError("GRE/EPI need one echo; B0 needs two or more contiguous echoes.")
    channels = int(blocks[0].mdh.UsedChannels)
    if any(int(block.mdh.UsedChannels) != channels for block in blocks):
        raise ValueError("Receiver count changes within the Cartesian volume.")
    receivers = _receiver_selection(receiver_indices, channels)
    geometry = _geometry(scan, blocks, patient_from_native_m)
    shape = tuple(int(v) for v in geometry["shape"])
    nro, npe, npar = (int(v) for v in geometry["raw_shape"])
    grouped = []
    for echo in echoes:
        group = [block for block in blocks if block.cEco == echo]
        keys = {(block.cLin, block.cPar) for block in group}
        if len(group) != npe * npar or len(keys) != len(group) or keys != {(y, z) for y in range(npe) for z in range(npar)}:
            raise ValueError("Missing or duplicate Cartesian phase/partition rows; no partial Fourier or zero filling.")
        grouped.append(group)
    yaps = scan["hdr"]["MeasYaps"]
    te = np.asarray(yaps["alTE"][:len(echoes)], dtype=float) * 1e-6
    if te.shape != (len(echoes),) or not np.isfinite(te).all() or np.any(te <= 0):
        raise ValueError("The actual echo count needs finite positive echo times.")
    if kind == "b0" and (np.any(np.diff(te) <= 0) or not np.allclose(np.diff(te), np.diff(te)[0], rtol=0, atol=1e-8)):
        raise ValueError("B0 reconstruction requires equally spaced increasing echo times.")
    power = np.zeros(shape, dtype=np.float64)
    pair = np.zeros(shape, dtype=np.complex128) if kind == "b0" else None
    pair_weight = np.zeros(shape, dtype=np.float64) if kind == "b0" else None
    second_power = np.zeros(shape, dtype=np.float64) if kind == "b0" else None
    epi_report = None
    if kind == "epi":
        from volume_epi import iter_epi3d_receivers
        receiver_reports = []
        for receiver, coil, report in iter_epi3d_receivers(path, receivers, scan=scan):
            if coil.shape != tuple(geometry["raw_shape"]):
                raise ValueError("EPI reconstructor returned a matrix different from its actual raw Cartesian grid.")
            coil = coil[geometry["crop"]]
            power += abs(coil) ** 2
            receiver_reports.append({key: value for key, value in report.items()
                                     if key != "navigator_phase_correction_rad"})
        epi_report = {"policy": "Independent three-navigator correction for every receiver and partition.",
                      "receiver_reports": receiver_reports,
                      "phase_correction_arrays_retained": False}
    else:
        source_bytes = np.memmap(path, dtype=np.uint8, mode="r") if all(getattr(b, "mem_pos", None) is not None for b in blocks) else None
        for receiver in receivers:
            previous = None
            for echo, group in enumerate(grouped):
                coil = _coil_volume(group, receiver, geometry, source_bytes)
                if echo == 0:
                    power += abs(coil) ** 2
                if previous is not None:
                    pair += coil * previous.conj()
                    pair_weight += abs(coil) * abs(previous)
                    second_power += abs(coil) ** 2
                previous = coil
            if source_bytes is not None and hasattr(source_bytes._mmap, "madvise"):
                # Mapped pages otherwise accumulate in RSS over32 receiver
                # passes, despite allocating just one Cartesian cube.
                source_bytes._mmap.madvise(mmap.MADV_DONTNEED)
        del source_bytes
    axes = tuple((np.arange(n) - n // 2) * step for n, step in zip(shape, geometry["spacing_m"]))
    native_to_patient = geometry["native_to_patient_m"]
    voxel_to_patient = native_to_patient.copy()
    voxel_to_patient[:3, :3] = geometry["basis_patient"] @ np.diag(geometry["spacing_m"])
    voxel_to_patient[:3, 3] = geometry["center_m"] - voxel_to_patient[:3, :3] @ (np.asarray(shape) // 2)
    nuclei = yaps.get("sTXSPEC", {}).get("asNucleusInfo", [{}])
    frequency = _number(nuclei[0].get("lFrequency") if nuclei else None,
                        scan["hdr"].get("Dicom", {}).get("lFrequency"))
    wip = yaps.get("sWipMemBlock", yaps.get("sWiPMemBlock", {}))
    integers = wip.get("alFree", [])
    signature_word = _number(integers[63]) if len(integers) > 63 else None
    sequence_file = str(yaps.get("tSequenceFileName", scan["hdr"].get("Config", {}).get("SequenceFileName", "")))
    variant = scan["hdr"].get("Meas", {}).get("tSequenceVariant", scan["hdr"].get("Dicom", {}).get("tSequenceVariant", ""))
    signature_md5 = (variant.lower() if "pulseq" in sequence_file.lower()
                     and isinstance(variant, str) and re.fullmatch(r"[0-9a-fA-F]{32}", variant) else None)
    metadata = {"format": RECONSTRUCTION_VERSION, "source_file": str(path), "source_sha256": digest,
        "kind": kind, "array_axes": ["readout", "phase", "slice"], "coordinate_units": "m",
        "raw_shape": geometry["raw_shape"].tolist(), "image_shape": list(shape),
        "FOV_m": geometry["FOV_m"].tolist(), "encoded_FOV_m": geometry["encoded_FOV_m"].tolist(),
        "spacing_m": geometry["spacing_m"].tolist(),
        "image_crop_start": [s.start for s in geometry["crop"]],
        "image_crop_stop": [s.stop for s in geometry["crop"]],
        "receiver_count": channels, "receiver_indices": receivers, "echo_times_s": te.tolist(),
        "TE_s": float(te[0]), "frequency_Hz": None if frequency is None else int(frequency),
        "pulseq_sequence_file": wip.get("tFree"),
        "sequence_signature_md5": signature_md5,
        "pulseq_signature_word32": None if signature_word is None else int(signature_word),
        "sequence_identity_verified": False,
        "sequence_identity_note": ("Pulseq MD5 extracted from Twix Meas.tSequenceVariant; compare to the loaded .seq signature before simulation."
            if signature_md5 else "No complete Pulseq MD5 declared in Meas.tSequenceVariant; a WIP32-bit signature word alone is insufficient."),
        "center_patient_m": geometry["center_m"].tolist(), "basis_patient": geometry["basis_patient"].tolist(),
        "MDH_quaternion_wxyz": geometry["MDH_quaternion_wxyz"],
        "geometry_convention": geometry["geometry_convention"],
        "geometry_override": geometry["geometry_override"],
        "pulseq_geometry_limitation": geometry["pulseq_geometry_limitation"],
        "spatial_registration_fitted": False, "navigator_correction": epi_report,
        "applied_ACDC_current_known": False,
        "current_note": "Protocol/file names such as CAL_p1_ch0 do not establish the actually applied ACDC currents.",
        "fft_convention": "fftshift(ifftn(ifftshift(kspace))); REFLECT readouts reversed before transform; no extra RF/ADC phase.",
        "rawdata_correction": "No receiver gain correction applied; receive-constant phase cancels in inter-echo products.",
        "memory_policy": "One receiver/echo Cartesian cube at a time; RSS and echo-pair accumulators only.",
        "cache_hit": False}
    result = {"magnitude": np.sqrt(power).astype(np.float32), "axes_m": axes,
              "basis_patient": geometry["basis_patient"], "center_m": geometry["center_m"],
              "native_to_patient_m": native_to_patient, "voxel_to_patient_m": voxel_to_patient,
              "metadata": metadata}
    if kind == "b0":
        dt = float(te[1] - te[0])
        b0, principal, mask, coherence, components = _unwrap_b0(pair, pair_weight, power, second_power, dt)
        result.update(b0Hz=b0, wrapped_b0Hz=principal, mask=mask, echo_pair_coherence=coherence)
        metadata.update(b0_signal_convention="signal(TE)=amplitude*exp(-2*pi*i*b0Hz*TE)",
            delta_TE_s=dt, frequency_alias_period_Hz=1 / dt,
            principal_frequency_interval_Hz=[-.5 / dt, .5 / dt],
            unwrapped_components=components,
            phase_unwrap_policy="Masked 3D unwrap separately in connected signal components of at least8 voxels; no background fill. Highest-pair-weight voxel anchors each component to its principal frequency branch.",
            absolute_frequency_branch_known=False,
            b0_mask_policy="Both echoes above5% of their respective peak RSS; receive-coil echo coherence>=0.2; components>=8 voxels.")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        values = {key: value for key, value in result.items() if key not in {"axes_m", "metadata"}}
        values.update({f"axis{i}_m": axis for i, axis in enumerate(axes)})
        values["metadata_json"] = json.dumps(metadata)
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **values)
        temporary.replace(cache_path)
    return result


__all__ = ["read_twix_volume", "RECONSTRUCTION_VERSION"]
