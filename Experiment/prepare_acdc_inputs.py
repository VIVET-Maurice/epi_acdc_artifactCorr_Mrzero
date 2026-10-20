"""Prepare a GRE/B0/scan folder for the existing MATLAB experiment optimizer.

The image reconstruction and Pulseq timing follow 01_reconstruction_et_B0.ipynb.
Unlike that historical notebook, the independent GRE supplies the object
amplitude. No B0 value is extrapolated, and no registration is inferred.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from prepare_acdc_calibration import sha256
from read_acdc_sequence import read_sequence
from resample_acdc_maps import resample_to_grid


DEFAULT_FILES = {"gre": "gre_reference.dat", "b0": "b0_map.dat",
                 "scan": "scan.dat", "sequence": "sequences/scan.seq"}


def read_arrays(path: Path) -> dict:
    """Read simple named arrays; pickle-based objects are deliberately unsupported."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if path.suffix.lower() == ".mat":
        return {key: value for key, value in loadmat(path, simplify_cells=True).items()
                if not key.startswith("__")}
    raise ValueError(f"Expected an NPZ or MATLAB v5/v7 file: {path.name}")


def real_array(value, shape: tuple, name: str, *, finite=True) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array) or array.shape != shape:
        raise ValueError(f"{name} must be a real array with shape {shape}.")
    array = array.astype(float)
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def binary_mask(value, shape: tuple, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != shape or not np.isin(mask, [0, 1]).all():
        raise ValueError(f"{name} must be a binary array with shape {shape}.")
    return mask.astype(bool)


def dicom_value(dataset, keyword: str):
    # Enhanced DICOM stores geometry in functional groups, as in the notebook.
    for element in dataset.iterall():
        if element.keyword == keyword:
            return element.value
    raise ValueError(f"DICOM metadata missing: {keyword}")


def read_gre(path: Path, reference_acquisition: dict | None = None,
             *, return_intermediates=False):
    """Read one reference magnitude image, reconstructing mono-echo GRE Twix.

    A raw ``.dat`` reference requires its own parsed GRE Pulseq acquisition.
    Native Twix image axes are ``(phase, readout)``. The supported axial frame
    matches the existing simulator: readout is patient +y, phase is patient -x.
    The default return remains one dictionary. For notebook figures, request
    ``(reference, diagnostics)`` with ``return_intermediates=True``.
    """
    path = Path(path)
    if path.suffix.lower() == ".dat":
        if reference_acquisition is None:
            raise ValueError("A raw reference GRE Twix requires its own reference_sequence acquisition.")
        if reference_acquisition.get("sequence_kind") != "gre":
            raise ValueError("The reference Twix requires a mono-echo 2D GRE/FLASH reference_sequence.")
        reconstructed = read_scan(path, reference_acquisition,
                                  return_intermediates=return_intermediates)
        magnitude, metadata = reconstructed[:2]
        if metadata["navigator_count"] != 0:
            raise ValueError("The reference Twix must be mono-echo GRE/FLASH without EPI navigators.")
        expected_quaternion = np.asarray([1, 0, 0, 1], float) / np.sqrt(2)
        if not np.isclose(abs(np.dot(metadata["slice_quaternion_wxyz"], expected_quaternion)),
                          1, atol=1e-8):
            raise ValueError("The reference Twix adapter supports the existing axial orientation only; "
                             "export registered NPZ/MAT coordinates for another orientation.")
        center = np.asarray(metadata["slice_position_mm"], float)
        if np.any(np.abs(center[:2]) > 1e-3):
            raise ValueError("The reference Twix adapter requires a centered in-plane origin; "
                             "export registered NPZ/MAT coordinates for an offset acquisition.")
        fov = np.asarray(metadata["FOV_m"], float)
        thickness = float(metadata["slice_thickness_mm"])
        if (fov.shape != (2,) or not np.isfinite(fov).all() or np.any(fov <= 0)
                or not np.isfinite(thickness) or thickness <= 0):
            raise ValueError("Reference Twix FOV and slice thickness must be finite and positive.")
        ny, nx = magnitude.shape
        dx_mm, dy_mm = fov * 1000 / np.asarray([nx, ny])
        # FFT zero is at n//2, not (n-1)/2. SlicePos is the slice center,
        # whereas position_mm below is the physical center of pixel [0, 0].
        x = (np.arange(nx) - nx // 2) * dx_mm * 1e-3 + center[1] * 1e-3
        y = (np.arange(ny) - ny // 2) * dy_mm * 1e-3 - center[0] * 1e-3
        X, Y = np.meshgrid(x, y)
        result = {"magnitude": magnitude, "X_m": X, "Y_m": Y,
                  "position_mm": center + [ny // 2 * dy_mm, -nx // 2 * dx_mm, 0],
                  "spacing_mm": np.asarray([dy_mm, dx_mm]),
                  "orientation": np.asarray([0, 1, 0, -1, 0, 0], float),
                  "slice_thickness_mm": thickness,
                  "TE_s": metadata["TE_s"], "TR_s": metadata["TR_s"],
                  "flip_deg": metadata["header_flip_deg"], "twix_metadata": metadata}
        check_reference_sequence(result, reference_acquisition)
        return (result, reconstructed[2]) if return_intermediates else result
    if path.suffix.lower() != ".dcm":
        result = read_arrays(path)
        for key in ("TE_s", "TR_s", "flip_deg"):
            if key not in result:
                continue
            value = np.asarray(result[key])
            try:
                if value.ndim != 0 or np.iscomplexobj(value):
                    raise ValueError
                scalar = float(value)
                if not np.isfinite(scalar):
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"GRE {key} must be a finite real scalar.") from None
            result[key] = scalar
        diagnostics = {"source_format": "image archive", "has_raw_kspace": False}
        return (result, diagnostics) if return_intermediates else result
    import pydicom

    dataset = pydicom.dcmread(path)
    if "M" not in dataset.ImageType or "P" in dataset.ImageType:
        raise ValueError("GRE DICOM must be a single magnitude image, not phase.")
    pixels = np.asarray(dataset.pixel_array).squeeze()
    if pixels.ndim != 2 or min(pixels.shape) < 2:
        raise ValueError("Select one 2D GRE magnitude slice and one echo.")
    orientation = np.asarray(dicom_value(dataset, "ImageOrientationPatient"), float)
    if not np.allclose(orientation, [1, 0, 0, 0, 1, 0], rtol=0, atol=1e-8):
        raise ValueError("This DICOM adapter supports the existing axial geometry only; "
                         "export registered NPZ coordinates for another orientation.")
    position = np.asarray(dicom_value(dataset, "ImagePositionPatient"), float)
    spacing = np.asarray(dicom_value(dataset, "PixelSpacing"), float)
    if (position.shape != (3,) or not np.isfinite(position).all() or spacing.shape != (2,)
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
        raise ValueError("DICOM position and positive pixel spacing must define a finite physical grid.")
    rows, columns = np.indices(pixels.shape)
    patient_x = position[0] + columns * spacing[1]
    patient_y = position[1] + rows * spacing[0]
    try:
        slope = float(dicom_value(dataset, "RescaleSlope"))
        intercept = float(dicom_value(dataset, "RescaleIntercept"))
    except ValueError:
        slope, intercept = 1.0, 0.0
    magnitude = pixels.astype(float) * slope + intercept
    result = {"magnitude": magnitude, "X_m": patient_y * 1e-3, "Y_m": -patient_x * 1e-3,
              "position_mm": position, "spacing_mm": spacing, "orientation": orientation,
              "slice_thickness_mm": float(dicom_value(dataset, "SliceThickness"))}
    for name, keywords, scale in (("TE_s", ("EffectiveEchoTime", "EchoTime"), .001),
                                  ("TR_s", ("RepetitionTime",), .001),
                                  ("flip_deg", ("FlipAngle",), 1)):
        for keyword in keywords:
            try:
                result[name] = float(dicom_value(dataset, keyword)) * scale
                break
            except ValueError:
                pass
    diagnostics = {"source_format": "DICOM magnitude", "has_raw_kspace": False}
    return (result, diagnostics) if return_intermediates else result


def centered_fft(array, axes, inverse=False):
    transform = np.fft.ifftn if inverse else np.fft.fftn
    return np.fft.fftshift(transform(np.fft.ifftshift(array, axes=axes), axes=axes), axes=axes)


def estimate_epi_navigator_phase(nav, signal_fraction=0.10, phase_branch=0):
    """Fit an affine negative-minus-positive phase from three +,-,+ echoes.

    Equally spaced echoes and approximately constant B0 are assumed. Average
    the two pairwise phases, not the complex positive echoes: the latter can
    cancel when B0 rotates their phases. Unwrap only a reliable contiguous run.

    The symmetric estimate is ambiguous modulo pi. Branch 0 selects the small
    mismatch at the strongest signal sample; branch 1 adds pi. This convention
    requires experimental verification and never uses image registration.
    """
    nav = np.asarray(nav, dtype=complex)
    if (nav.ndim != 3 or nav.shape[0] != 3 or nav.shape[1] < 1
            or nav.shape[2] < 4 or not np.isfinite(nav).all()):
        raise ValueError("Navigators must be finite arrays shaped (3, receivers, readout), with at least four samples.")
    if not np.isscalar(signal_fraction) or not 0 < signal_fraction < 1:
        raise ValueError("Navigator signal_fraction must be between 0 and 1.")
    if isinstance(phase_branch, (bool, np.bool_)) or phase_branch not in (0, 1):
        raise ValueError("Navigator phase_branch must be 0 (small mismatch) or 1 (add pi).")

    channels, nx = nav.shape[1:]
    x = np.linspace(-1, 1, nx)
    correction = np.zeros((channels, nx))
    fit_mask = np.zeros((channels, nx), dtype=bool)
    pair_phase = np.full((2, channels, nx), np.nan)
    phase_estimate = np.full((channels, nx), np.nan)
    residual = np.full(channels, np.nan)
    branch_adjustment = np.zeros(channels)
    amplitude = np.abs(nav)
    outer_sum = amplitude[0] + amplitude[2]
    retention = np.divide(np.abs(nav[0] + nav[2]), outer_sum,
                          out=np.zeros_like(outer_sum), where=outer_sum > 0)

    for channel in range(channels):
        # All three echoes must have signal; a cancelled complex average must
        # not supply either the mask or the weights.
        reliable = np.all(amplitude[:, channel] > signal_fraction
                          * amplitude[:, channel].max(axis=1, keepdims=True), axis=0)
        indices = np.flatnonzero(reliable)
        runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        runs = [run for run in runs if run.size >= 4]
        if not runs:
            continue  # Retain zero correction and report this receiver as unfitted.
        weight = amplitude[1, channel] * np.sqrt(amplitude[0, channel] * amplitude[2, channel])
        selected = max(runs, key=lambda run: weight[run].sum())
        left = np.unwrap(np.angle(nav[1, channel, selected] * nav[0, channel, selected].conj()))
        right = np.unwrap(np.angle(nav[1, channel, selected] * nav[2, channel, selected].conj()))
        averaged = (left + right) / 2
        anchor = int(np.argmax(weight[selected]))
        adjustment = -np.pi * np.floor((averaged[anchor] + np.pi / 2) / np.pi) + phase_branch * np.pi
        averaged += adjustment
        slope, intercept = np.polyfit(x[selected], averaged, 1, w=np.sqrt(weight[selected]))
        correction[channel] = intercept + slope * x
        fit_mask[channel, selected] = True
        pair_phase[:, channel, selected] = np.stack((left, right))
        phase_estimate[channel, selected] = averaged
        residual[channel] = np.sqrt(np.average((averaged - correction[channel, selected])**2,
                                              weights=weight[selected]))
        branch_adjustment[channel] = adjustment

    diagnostics = {"fit_mask": fit_mask, "fitted_channels": fit_mask.any(axis=1),
                   "pair_phase_rad": pair_phase, "phase_estimate_rad": phase_estimate,
                   "fit_residual_rad": residual, "outer_average_retention": retention,
                   "branch_adjustment_rad": branch_adjustment, "phase_branch": int(phase_branch)}
    return correction, diagnostics


def read_scan(path: Path, acquisition: dict | None = None,
              *, return_intermediates=False, navigator_phase_branch=0) -> tuple:
    """Reconstruct one full Cartesian 2D Twix scan (GRE or three-navigator EPI).

    Twix samples already include the receiver's phase reference; do not apply
    Pulseq RF/ADC phases a second time when reconstructing measured data.
    With ``return_intermediates=True``, return ``(image, metadata, diagnostics)``
    instead of the default ``(image, metadata)``. K-space and coil images are
    shaped ``(phase, receiver, readout)`` after REFLECT sample reordering.
    """
    expected_shape = (np.asarray(acquisition["kx_mInv"]).shape if acquisition else None)
    if path.suffix.lower() != ".dat":
        image = np.asarray(read_arrays(path)["image"])
        if image.ndim != 2 or min(image.shape) < 2:
            raise ValueError("scan.image must be a full 2D magnitude image.")
        image = real_array(image, expected_shape or image.shape, "scan.image")
        if return_intermediates:
            return image, {}, {"source_format": "image archive", "has_raw_kspace": False}
        return image, {}
    import twixtools

    with contextlib.redirect_stdout(io.StringIO()):
        scans = twixtools.read_twix(str(path), verbose=False)
    scan = scans[-1]
    acquired = [block for block in scan["mdb"] if block.is_image_scan()]
    lines = sorted(acquired, key=lambda b: b.cLin)
    navigators = [block for block in scan["mdb"] if block.is_flag_set("PHASCOR")]
    if len(lines) < 2 or np.asarray(lines[0].data).ndim != 2:
        raise ValueError("Twix must contain one full Cartesian 2D image.")
    ny = len(lines)
    channels, nx = lines[0].data.shape
    if (nx < 2 or channels < 1 or len(navigators) not in (0, 3)
            or [block.cLin for block in lines] != list(range(ny))
            or any(block.data.shape != (channels, nx) for block in lines + navigators)):
        raise ValueError("Twix needs unique complete phase rows, constant receiver/sample counts, "
                         "and either zero GRE navigators or three EPI navigators.")
    if expected_shape is not None and (ny, nx) != expected_shape:
        raise ValueError(f"Twix image shape {(ny, nx)} disagrees with Pulseq {expected_shape}; "
                         "partial Fourier and readout oversampling are not resampled implicitly.")
    for counter in ("cAve", "cSlc", "cPar", "cEco", "cPhs", "cRep", "cSet"):
        if len({getattr(block, counter) for block in lines}) != 1:
            raise ValueError("Twix must contain one slice, echo, average and repetition.")
    header = scan["hdr"]["MeasYaps"]
    grid_header = header.get("sKSpace", {})
    if (int(grid_header.get("lBaseResolution", nx)) != nx
            or int(grid_header.get("lPhaseEncodingLines", ny)) != ny
            or any(block.mdh.CenterCol != nx // 2 or block.mdh.CenterLin != ny // 2
                   for block in lines)):
        raise ValueError("Twix must be a full centered Cartesian grid without implicit oversampling removal.")
    if navigators and [block.is_flag_set("REFLECT") for block in navigators] != [False, True, False]:
        raise ValueError("Expected +,-,+ navigator polarities from the existing acquisition.")

    def ordered(block):
        data = np.asarray(block.data, dtype=complex)
        return data[:, ::-1] if block.is_flag_set("REFLECT") else data

    kspace = np.stack([ordered(block) for block in lines])
    negative = np.asarray([block.is_flag_set("REFLECT") for block in lines])
    corrected = kspace
    correction = np.zeros((channels, nx))
    nav_kspace = nav = navigator_fit = None
    if navigators:
        if acquisition is not None and "navigator_echo_center_s" in acquisition:
            echo_centers = np.asarray(acquisition["navigator_echo_center_s"]).ravel()
            spacing = np.diff(echo_centers)
            if (echo_centers.shape != (3,) or not np.isfinite(echo_centers).all()
                    or np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=0, atol=1e-9)):
                raise ValueError("Symmetric navigator phase correction requires three equally spaced echoes.")
        nav_kspace = np.stack([ordered(block) for block in navigators])
        nav = centered_fft(nav_kspace, (2,), inverse=True)
        correction, navigator_fit = estimate_epi_navigator_phase(nav, phase_branch=navigator_phase_branch)
        hybrid = centered_fft(kspace, (2,), inverse=True)
        hybrid[negative] *= np.exp(-1j * correction)[None, :, :]
        corrected = centered_fft(hybrid, (2,))
    coil_images = centered_fft(corrected, (0, 2), inverse=True)
    image = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=1))
    slices = header["sSliceArray"]["asSlice"]
    if len(slices) != 1:
        raise ValueError("The Twix adapter supports exactly one slice.")
    slice_header = slices[0]
    slice_data = [block.mdh.SliceData for block in lines + navigators]
    def slice_position(entry):
        return np.asarray([entry.SlicePos.Sag, entry.SlicePos.Cor, entry.SlicePos.Tra], float)

    position = slice_position(slice_data[0])
    quaternion = np.asarray(list(slice_data[0].Quaternion), float)
    if (not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-6)):
        raise ValueError("Twix slice position/quaternion is invalid.")
    quaternion /= np.linalg.norm(quaternion)
    for entry in slice_data[1:]:
        if (not np.allclose(slice_position(entry), position, rtol=0, atol=1e-3)
                or not np.isclose(abs(np.dot(list(entry.Quaternion), quaternion)), 1, atol=1e-6)):
            raise ValueError("Twix slice geometry changes within this acquisition.")
    metadata = {"TE_s": float(header["alTE"][0]) * 1e-6,
                "TR_s": float(header["alTR"][0]) * 1e-6,
                "dwell_s": float(header["sRXSPEC"]["alDwellTime"][0]) * 1e-9,
                "polarity": np.where(negative, -1, 1).tolist(), "navigator_count": len(navigators),
                "image_shape": [ny, nx], "receiver_count": channels,
                "acquired_phase_rows": [int(block.cLin) for block in acquired],
                "header_flip_deg": float(header["adFlipAngleDegree"][0]),
                "FOV_m": [float(slice_header["dReadoutFOV"]) * 1e-3,
                          float(slice_header["dPhaseFOV"]) * 1e-3],
                "slice_position_mm": position.tolist(), "slice_quaternion_wxyz": quaternion.tolist(),
                "slice_thickness_mm": float(slice_header["dThickness"])}
    if return_intermediates:
        uncorrected_coils = centered_fft(kspace, (0, 2), inverse=True) if navigators else coil_images
        diagnostics = {
            "source_format": "Siemens Twix", "has_raw_kspace": True,
            "array_axes": ["phase", "receiver", "readout"],
            "kspace": kspace, "corrected_kspace": corrected,
            "coil_images": coil_images, "uncorrected_coil_images": uncorrected_coils,
            "uncorrected_image": np.sqrt(np.sum(np.abs(uncorrected_coils) ** 2, axis=1)),
            "negative_readout_rows": negative,
            "navigator_kspace": nav_kspace, "navigator_hybrid": nav,
            "navigator_phase_correction_rad": correction,
            "navigator_fit": navigator_fit,
        }
        return image, metadata, diagnostics
    return image, metadata


def check_physical_geometry(gre: dict, scan_metadata: dict, acquisition: dict,
                            support: np.ndarray, X: np.ndarray, Y: np.ndarray,
                            *, dicom_input: bool) -> None:
    """Reject observable geometry disagreements even when registration was confirmed."""
    fov = np.asarray(acquisition["FOV_m"], float)
    if (np.any(np.abs(X[support]) > fov[0] / 2 + 1e-6)
            or np.any(np.abs(Y[support]) > fov[1] / 2 + 1e-6)):
        raise ValueError("GRE ROI extends outside the scan field of view; check coordinates/registration "
                         "or explicitly select an ROI inside the acquired FOV.")
    if not scan_metadata:
        return
    if not np.allclose(scan_metadata["FOV_m"], fov, rtol=1e-4, atol=1e-6):
        raise ValueError("Twix header field of view disagrees with the Pulseq ADC trajectory.")
    reference_metadata = gre.get("twix_metadata")
    if reference_metadata:
        reference_position = np.asarray(reference_metadata["slice_position_mm"], float)
        scan_position = np.asarray(scan_metadata["slice_position_mm"], float)
        if not np.allclose(reference_position, scan_position, rtol=0, atol=1e-3):
            raise ValueError("Reference GRE Twix and scan Twix describe different slice planes/origins.")
        if not np.isclose(abs(np.dot(reference_metadata["slice_quaternion_wxyz"],
                                     scan_metadata["slice_quaternion_wxyz"])), 1, atol=1e-8):
            raise ValueError("Reference GRE Twix and scan Twix describe different readout/phase orientations.")
        return
    if not dicom_input:
        return  # The explicit NPZ coordinates already refer to the sequence axes.
    # This DICOM adapter deliberately supports the exact axial frame of the
    # existing notebook. A different Twix rotation/offset needs a registered
    # coordinate export; a boolean cannot override a contradictory header.
    expected_quaternion = np.asarray([1, 0, 0, 1], float) / np.sqrt(2)
    actual = np.asarray(scan_metadata["slice_quaternion_wxyz"], float)
    if not np.isclose(abs(np.dot(actual, expected_quaternion)), 1, atol=1e-8):
        raise ValueError("Twix readout/phase orientation differs from the supported DICOM adapter; "
                         "export registered GRE coordinates as NPZ/MAT.")
    position = np.asarray(scan_metadata["slice_position_mm"], float)
    if np.any(np.abs(position[:2]) > 1e-3):
        raise ValueError("The DICOM adapter requires a centered Twix in-plane origin; "
                         "export registered GRE coordinates as NPZ/MAT for an offset acquisition.")
    if abs(position[2] - float(gre["position_mm"][2])) > 1e-3:
        raise ValueError("GRE DICOM and Twix describe different slice planes.")


def check_reference_sequence(gre: dict, acquisition: dict) -> None:
    """Validate the reference against its own sequence, irrespective of scan size."""
    X, Y = np.asarray(gre["X_m"]), np.asarray(gre["Y_m"])
    if (np.allclose(X, X[:, :1], atol=1e-9, rtol=0)
            and np.allclose(Y, Y[:1, :], atol=1e-9, rtol=0)):
        x, y = X[:, 0], Y[0, :]
    elif (np.allclose(X, X[:1, :], atol=1e-9, rtol=0)
          and np.allclose(Y, Y[:, :1], atol=1e-9, rtol=0)):
        x, y = X[0, :], Y[:, 0]
    else:
        raise ValueError("Reference sequence validation needs a rectilinear GRE X_m/Y_m grid.")
    shape = np.asarray(acquisition["kx_mInv"]).shape
    if (len(y), len(x)) != shape:
        raise ValueError("GRE reference matrix disagrees with its reference_sequence ADC trajectory.")
    steps = []
    for axis in (x, y):
        delta = np.diff(axis)
        if not np.allclose(delta, delta[0], atol=1e-9, rtol=0) or delta[0] == 0:
            raise ValueError("GRE reference coordinates must have constant nonzero spacing.")
        steps.append(abs(delta[0]) * len(axis))
    if not np.allclose(steps, acquisition["FOV_m"], rtol=1e-4, atol=1e-6):
        raise ValueError("GRE reference FOV disagrees with its reference_sequence ADC trajectory.")
    for key in ("TE_s", "TR_s"):
        if key in gre and not np.isclose(gre[key], acquisition[key], rtol=0, atol=1e-6):
            raise ValueError(f"GRE reference and reference_sequence disagree on {key}.")
    metadata = gre.get("twix_metadata")
    if not metadata:
        return
    if acquisition.get("sequence_kind") != "gre" or metadata["navigator_count"] != 0:
        raise ValueError("The reference Twix must match a mono-echo GRE/FLASH sequence without EPI navigators.")
    if not np.isclose(metadata["dwell_s"], acquisition["dwell_s"], rtol=0, atol=1e-9):
        raise ValueError("Reference Twix and reference_sequence disagree on dwell_s.")
    if not np.array_equal(metadata["polarity"], acquisition["polarity"]):
        raise ValueError("Reference Twix readout polarity disagrees with reference_sequence.")
    expected_rows = np.argsort(np.asarray(acquisition["line_index"])[:, 0]).tolist()
    if metadata["acquired_phase_rows"] != expected_rows:
        raise ValueError("Reference Twix chronological phase-row order disagrees with reference_sequence.")


def prepare_inputs(input_dir: Path, output_dir: Path, config: dict) -> dict:
    """Validate inputs, then write the existing experiment struct and its provenance."""
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    settings = config.get("preparation", {})
    for name in ("geometry_confirmed", "b0_sign_confirmed"):
        if settings.get(name) is not True:
            raise ValueError(f"preparation.{name} must be true after its experimental check.")
    if settings.get("b0_units") != "Hz":
        raise ValueError("Declare preparation.b0_units='Hz'; convert rad/s or ppm before importing.")
    files = {**DEFAULT_FILES, **config.get("files", {})}
    paths = {key: (input_dir / files[key]).resolve() for key in DEFAULT_FILES}
    for key in ("reference_sequence", "b0_sequence"):
        if files.get(key):
            paths[key] = (input_dir / files[key]).resolve()
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Missing experimental input: {path}")
    acquisition, timing = read_sequence(paths["sequence"])
    if timing["sequence_kind"] == "gre" and "reference_sequence" not in paths:
        raise ValueError("Conventional GRE requires files.reference_sequence for the reference GRE/FLASH.")
    reference_acquisition = None
    if "reference_sequence" in paths:
        reference_acquisition, reference_timing = read_sequence(paths["reference_sequence"])
        if reference_timing["sequence_kind"] != "gre":
            raise ValueError("reference_sequence must describe a mono-echo 2D GRE/FLASH acquisition.")
    gre = read_gre(paths["gre"], reference_acquisition)
    shape = np.asarray(gre["magnitude"]).shape
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("GRE magnitude must be a single 2D image.")
    magnitude = real_array(gre["magnitude"], shape, "GRE magnitude")
    if np.any(magnitude < 0) or magnitude.max() <= 0:
        raise ValueError("GRE magnitude must be nonnegative with nonzero signal.")
    X = real_array(gre["X_m"], magnitude.shape, "GRE X_m")
    Y = real_array(gre["Y_m"], magnitude.shape, "GRE Y_m")
    # Require a two-dimensional physical grid, not two degenerate coordinate images.
    if np.linalg.matrix_rank(np.stack([X.ravel() - X.mean(), Y.ravel() - Y.mean()])) != 2:
        raise ValueError("GRE X_m/Y_m must describe a nondegenerate two-dimensional physical grid.")
    threshold = float(settings.get("reference_threshold_fraction", .03))
    if not np.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("reference_threshold_fraction must lie in [0,1).")
    support = (binary_mask(gre["mask"], magnitude.shape, "GRE mask") if "mask" in gre
               else magnitude > threshold * magnitude.max())
    roi_path = None
    if settings.get("roi_mask"):
        roi_path = (input_dir / settings["roi_mask"]).resolve()
        roi = (np.load(roi_path, allow_pickle=False) if roi_path.suffix.lower() == ".npy"
               else read_arrays(roi_path)["mask"])
        support = binary_mask(roi, magnitude.shape, "Explicit GRE ROI mask")
    support &= magnitude > 0
    if not support.any():
        raise ValueError("The GRE ROI contains no positive signal.")
    b0_metadata = None
    if paths["b0"].suffix.lower() == ".dat":
        from read_acdc_b0 import read_b0_twix
        if "b0_sequence" not in paths:
            raise ValueError("A raw B0 Twix requires files.b0_sequence (the multi-echo GRE Pulseq file).")
        reconstruction = settings.get("b0_reconstruction", {})
        if not isinstance(reconstruction, dict) or "phase_sign" in reconstruction:
            raise ValueError("Use preparation.b0_sign for the B0 sign; b0_reconstruction must be an object.")
        b0_arrays = read_b0_twix(paths["b0"], paths["b0_sequence"], **reconstruction)
        b0_metadata = b0_arrays["metadata"]
    elif paths["b0"].suffix.lower() == ".npy":
        b0_arrays = {"b0Hz": np.load(paths["b0"], allow_pickle=False)}
    else:
        b0_arrays = read_arrays(paths["b0"])
    b0_shape = np.asarray(b0_arrays["b0Hz"]).shape
    if len(b0_shape) != 2 or min(b0_shape) < 2:
        raise ValueError("B0 b0Hz must be a single 2D measured map.")
    b0 = real_array(b0_arrays["b0Hz"], b0_shape, "B0 b0Hz", finite=False)
    if "units" in b0_arrays and str(np.asarray(b0_arrays["units"]).item()) != "Hz":
        raise ValueError("B0 archive units must be Hz.")
    quality = (binary_mask(b0_arrays["mask"], b0_shape, "B0 mask")
               if "mask" in b0_arrays else np.isfinite(b0))
    quality &= np.isfinite(b0)
    if "X_m" in b0_arrays and "Y_m" in b0_arrays:
        b0_X = real_array(b0_arrays["X_m"], b0_shape, "B0 X_m")
        b0_Y = real_array(b0_arrays["Y_m"], b0_shape, "B0 Y_m")
        b0, quality = resample_to_grid(b0, b0_X, b0_Y, quality, X, Y)
    elif "X_m" in b0_arrays or "Y_m" in b0_arrays:
        raise ValueError("B0 coordinates require both X_m and Y_m.")
    elif settings.get("b0_on_gre_grid") is not True or b0_shape != magnitude.shape:
        raise ValueError("B0 needs X_m/Y_m, or preparation.b0_on_gre_grid=true and the GRE shape.")
    if np.any(support & ~quality):
        missing = int(np.count_nonzero(support & ~quality))
        raise ValueError(f"{missing} GRE ROI points lack reliable measured B0. Improve B0 coverage "
                         "or supply an explicit GRE mask for the intended ROI; no extrapolation is applied.")
    sign = settings.get("b0_sign", 1)
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError("preparation.b0_sign must be +1 or -1 for the model exp(-2*pi*i*B0*t).")
    b0 *= sign
    reference_report = {"validated": False, "reason": "Legacy configuration has no reference_sequence."}
    if reference_acquisition is not None:
        check_reference_sequence(gre, reference_acquisition)
        reference_report = {"validated": True, "image_shape": list(np.asarray(gre["magnitude"]).shape),
                            "sequence_kind": reference_timing["sequence_kind"],
                            "FOV_m": np.asarray(reference_acquisition["FOV_m"]).tolist(),
                            "TE_s": reference_acquisition["TE_s"], "TR_s": reference_acquisition["TR_s"]}
    scan, scan_metadata = read_scan(paths["scan"], acquisition)
    if not np.isfinite(scan).all() or np.any(scan < 0) or scan.max() <= 0:
        raise ValueError("Scan image must be a finite, nonnegative, nonzero magnitude image.")
    check_physical_geometry(gre, scan_metadata, acquisition, support, X, Y,
                            dicom_input=paths["gre"].suffix.lower() == ".dcm")
    if b0_metadata is not None:
        # The B0 reader exports physical axial X/Y coordinates. A 2D
        # resampling must never silently match maps from a different plane.
        b0_position = np.asarray(b0_metadata["slice_position_mm"], float)
        if "position_mm" in gre and abs(b0_position[2] - gre["position_mm"][2]) > 1e-3:
            raise ValueError("B0 Twix and reference GRE describe different slice planes.")
        if scan_metadata:
            b0_normal = np.asarray(b0_metadata["slice_quaternion_wxyz"], float)
            scan_normal = np.asarray(scan_metadata["slice_quaternion_wxyz"], float)
            if (abs(b0_position[2] - scan_metadata["slice_position_mm"][2]) > 1e-3
                    or not np.isclose(abs(np.dot(b0_normal, scan_normal)), 1, atol=1e-8)):
                raise ValueError("B0 Twix and scan Twix describe different slice planes/orientations.")
    for key in ("TE_s", "TR_s", "dwell_s"):
        tolerance = 1e-9 if key == "dwell_s" else 1e-6
        if key in scan_metadata and abs(scan_metadata[key] - acquisition[key]) > tolerance:
            raise ValueError(f"Twix and Pulseq disagree on {key}.")
    if scan_metadata and (scan_metadata["navigator_count"] != timing["navigator_count"]
                          or not np.array_equal(scan_metadata["polarity"], acquisition["polarity"])):
        raise ValueError("Twix navigator count / line polarity disagrees with Pulseq.")
    if scan_metadata and "acquired_phase_rows" in scan_metadata:
        expected_rows = np.argsort(np.asarray(acquisition["line_index"])[:, 0]).tolist()
        if scan_metadata["acquired_phase_rows"] != expected_rows:
            raise ValueError("Twix chronological phase-row order disagrees with Pulseq.")
    t2star = float(settings.get("T2star_s", np.inf))
    if np.isnan(t2star) or t2star <= 0:
        raise ValueError("T2star_s must be positive (seconds); omit it for no relaxation model.")
    # Its absolute scale cancels in the normalized objective; retain it for audit.
    scale = float(magnitude[support].max())
    amplitude = np.where(support, magnitude / scale, 0)
    geometry = {"X_m": X, "Y_m": Y, "axes": "X=sequence readout; Y=sequence phase; metres"}
    for key in ("position_mm", "spacing_mm", "orientation", "slice_thickness_mm"):
        if key in gre:
            geometry[key] = gre[key]
    report = {**timing, "input_files": {key: str(path) for key, path in paths.items()},
              "input_sha256": {key: sha256(path) for key, path in paths.items()},
              "support_voxels": int(support.sum()), "reliable_B0_voxels": int(quality.sum()),
              "excluded_GRE_signal_fraction": float(magnitude[~support].sum() / magnitude.sum()),
              "roi_source": (str(roi_path) if roi_path else
                             "explicit GRE mask" if "mask" in gre else "GRE magnitude threshold"),
              "GRE_amplitude_normalization": scale, "B0_sign": int(sign),
              "reference_sequence_validation": reference_report,
              "reference_image_shape": list(magnitude.shape), "B0_source_shape": list(b0_shape),
              "b0_reconstruction": b0_metadata,
              "reference_metadata": {key: gre[key] for key in ("TE_s", "TR_s", "flip_deg") if key in gre},
              "reference_twix_metadata": gre.get("twix_metadata"),
              "scan_metadata": scan_metadata, "geometry_confirmed": True, "b0_sign_confirmed": True,
              "experiment_input": str(output_dir / "experiment_input.mat")}
    if roi_path:
        report["input_files"]["roi_mask"] = str(roi_path)
        report["input_sha256"]["roi_mask"] = sha256(roi_path)
    assumptions = ["Independent GRE magnitude is an effective transverse-amplitude proxy, not proton density.",
                   "Only the explicitly selected GRE support participates; all its B0 values are measured.",
                   "The normal scan is a measured comparison, not the optimizer's object amplitude.",
                   "A single uniform receiver is modeled; reference/scan contrast differences are not calibrated.",
                   "Currents are zero during RF/navigators and constant within each imaging state.",
                   "Current transitions are instantaneous in the existing model."]
    experiment = {"images": {"gre_reference": magnitude, "scan": scan, "perturbed": scan},
                  "field": {"b0Hz": np.where(support, b0, 0),
                            "measured_b0Hz": np.where(quality, b0, np.nan),
                            "mask": quality, "simulation_mask": support},
                  "geometry": geometry, "acquisition": acquisition, "m0_proxy": amplitude,
                  "relaxation": {"T2star_s": t2star},
                  "assumptions": np.asarray(assumptions, dtype=object),
                  "provenance": {"sequence_sha256": timing["sequence_sha256"],
                                 "input_sha256": report["input_sha256"], "B0_sign": int(sign),
                                 "GRE_amplitude_normalization": scale},
                  "optimization_ready": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("experiment_input.mat", "preparation_report.json")):
        raise ValueError("Prepared output already exists; choose a new output directory.")
    if b0_metadata is not None:
        reconstructed = {key: value for key, value in b0_arrays.items() if key != "metadata"}
        reconstructed.update(b0Hz=np.asarray(b0_arrays["b0Hz"]) * sign,
                             units="Hz", applied_b0_sign=sign)
        np.savez_compressed(output_dir / "b0_reconstructed.npz", **reconstructed)
    savemat(output_dir / "experiment_input.mat", {"experiment": experiment},
            do_compression=True, long_field_names=True)
    (output_dir / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, help="Default: INPUT_DIR/config.json")
    args = parser.parse_args()
    config_path = args.config or args.input_dir / "config.json"
    config = json.loads(config_path.read_text())
    report = prepare_inputs(args.input_dir, args.output_dir, config)
    print(f"Prepared {report['support_voxels']} GRE/B0 points: {report['experiment_input']}")


if __name__ == "__main__":
    main()
