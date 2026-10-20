"""Compare measured GRE, unshimmed EPI and explicitly selected corrected scans.

Images stay in native array order. Registered native coordinates are used for
interpolation onto the EPI grid; a common orthogonal XY matrix affects display
coordinates only. One baseline-to-GRE gain and one common mask serve every
corrected scan. No acquisition is selected by this module.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "Experiment"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from prepare_acdc_inputs import read_scan
from read_acdc_sequence import read_sequence
from resample_acdc_maps import resample_to_grid
from measured_continuous import reconstruct_receivers


def _frame_matrix(value):
    matrix = np.eye(2) if value is None else np.asarray(value, dtype=float)
    if (matrix.shape != (2, 2) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(2), rtol=0, atol=1e-10)):
        raise ValueError("frame_matrix must be a finite orthogonal XY rotation/mirror matrix.")
    return matrix


def _registered_image(record, targetX, targetY):
    image = np.asarray(record["image"])
    if image.ndim != 2 or np.iscomplexobj(image) or np.any(image[np.isfinite(image)] < 0):
        raise ValueError("Measured images must be nonnegative real 2D magnitudes.")
    return resample_to_grid(image, record["X_m"], record["Y_m"], np.isfinite(image), targetX, targetY)


def compare_registered_images(reference, baseline, corrected, *, frame_matrix=None,
                              roi_mask=None, normalize=True, reference_threshold_fraction=0.1):
    """Compare image records on the baseline EPI grid without registration.

    Each record has ``image``, native ``X_m``, ``Y_m`` and optional ``label``.
    ``corrected`` is an explicit list of records. ``roi_mask`` is either a
    mask on the EPI grid or a dictionary with ``mask``, ``X_m`` and ``Y_m``;
    a sampled ROI requires every contributing mask value to be true.
    With no explicit ROI, threshold GRE at ``reference_threshold_fraction``
    of its interpolated peak; zero selects the full common image coverage.
    Return common-scaled ``images``, ``labels``, ``rows`` (RMSE/NRMSE), common
    ``mask``, display ``X_m``/``Y_m`` and a serializable ``report``.
    """
    matrix = _frame_matrix(frame_matrix)
    targetX, targetY = np.asarray(baseline["X_m"], float), np.asarray(baseline["Y_m"], float)
    records = [reference, baseline] + list(corrected)
    labels = [record.get("label", default) for record, default in zip(
        records, ["GRE reference", "Without ACDC"] + [f"Corrected {i + 1}" for i in range(len(records) - 2)])]
    native_images, valid_masks = [], []
    for record in records:
        image, valid = _registered_image(record, targetX, targetY)
        native_images.append(image)
        valid_masks.append(valid)
    coverage = np.logical_and.reduce(valid_masks)
    if not np.isfinite(reference_threshold_fraction) or not 0 <= reference_threshold_fraction < 1:
        raise ValueError("reference_threshold_fraction must lie in [0, 1).")
    if roi_mask is None:
        requested = (native_images[0] > reference_threshold_fraction * np.nanmax(native_images[0])
                     if reference_threshold_fraction > 0 else np.ones(targetX.shape, bool))
        roi_source = "GRE magnitude threshold" if reference_threshold_fraction > 0 else "full image coverage"
    elif isinstance(roi_mask, dict):
        values = np.asarray(roi_mask["mask"])
        if not np.isin(values, [0, 1]).all():
            raise ValueError("ROI mask must be binary.")
        sampled, valid = resample_to_grid(values.astype(float), roi_mask["X_m"], roi_mask["Y_m"],
                                         np.ones(values.shape, bool), targetX, targetY)
        requested = valid & (sampled >= 1 - 1e-12)
        roi_source = "explicit mask sampled in native physical coordinates"
    else:
        values = np.asarray(roi_mask)
        if values.shape != targetX.shape or not np.isin(values, [0, 1]).all():
            raise ValueError("ROI mask must be binary and match the EPI grid, or supply its physical coordinates.")
        requested = values.astype(bool)
        roi_source = "explicit EPI-grid mask"
    mask = requested & coverage
    if not mask.any():
        raise ValueError("No common valid comparison pixels remain.")
    reference_roi, baseline_roi = native_images[0][mask], native_images[1][mask]
    reference_energy = float(np.sum(reference_roi**2))
    baseline_energy = float(np.sum(baseline_roi**2))
    if reference_energy <= 0 or baseline_energy <= 0:
        raise ValueError("GRE and baseline need nonzero signal on the comparison mask.")
    gain = float(np.sum(reference_roi * baseline_roi) / baseline_energy) if normalize else 1.
    scale = float(np.max(reference_roi)) if normalize else 1.
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("The shared baseline-to-GRE intensity gain must be positive.")
    images = [native_images[0] / scale] + [gain * image / scale for image in native_images[1:]]
    reference_rms = float(np.sqrt(np.mean(images[0][mask]**2)))
    rows = []
    for label, image in zip(labels[1:], images[1:]):
        rmse = float(np.sqrt(np.mean((image[mask] - images[0][mask])**2)))
        rows.append({"label": label, "RMSE": rmse, "NRMSE_percent": 100 * rmse / reference_rms})
    displayX = matrix[0, 0] * targetX + matrix[0, 1] * targetY
    displayY = matrix[1, 0] * targetX + matrix[1, 1] * targetY
    report = {
        "comparison_grid": "native unshimmed EPI grid", "image_shape": list(targetX.shape),
        "requested_pixels": int(requested.sum()), "common_pixels": int(mask.sum()),
        "ROI_source": roi_source, "reference_threshold_fraction": float(reference_threshold_fraction),
        "excluded_missing_coverage_pixels": int(np.sum(requested & ~coverage)),
        "shared_baseline_to_GRE_gain": gain, "GRE_reference_scale": scale,
        "normalize": bool(normalize), "RMSE_units": "GRE-normalized units" if normalize else "Twix RSS units",
        "gain_fitted_from": "unshimmed baseline only" if normalize else "no gain fitting",
        "same_gain_for_all_corrected_scans": True, "same_mask_for_all_scans": True,
        "frame_matrix": matrix.tolist(), "native_array_order_preserved": True,
        "interpolation": "linear in native physical coordinates; no extrapolation",
        "registration_applied": False, "nonrigid_registration_applied": False,
        "waveform_match_verified": False,
        "waveform_provenance": "not checked against current optimization",
    }
    return {"images": images, "labels": labels, "rows": rows, "mask": mask,
            "requested_mask": requested, "common_coverage": coverage,
            "X_m": displayX, "Y_m": displayY, "native_X_m": targetX, "native_Y_m": targetY,
            "gain": gain, "reference_scale": scale, "native_images": native_images, "report": report}


def _scan_grid(image, metadata, acquisition):
    ny, nx = np.asarray(image).shape
    fov = np.asarray(acquisition["FOV_m"], float)
    sag, cor, _ = np.asarray(metadata["slice_position_mm"], float) * .001
    return np.meshgrid((np.arange(nx) - nx // 2) * fov[0] / nx + cor,
                       (np.arange(ny) - ny // 2) * fov[1] / ny - sag)


def _check_scan(metadata, acquisition, timing):
    for key, tolerance in (("TE_s", 1e-6), ("TR_s", 1e-6), ("dwell_s", 1e-9)):
        if abs(float(metadata[key]) - float(acquisition[key])) > tolerance:
            raise ValueError(f"Measured scan and parsed sequence disagree on {key}.")
    if (not np.allclose(metadata["FOV_m"], acquisition["FOV_m"], rtol=1e-4, atol=1e-6)
            or tuple(metadata["image_shape"]) != np.asarray(acquisition["kx_mInv"]).shape):
        raise ValueError("Measured scan and sequence disagree on FOV or image shape.")
    if (int(metadata["navigator_count"]) != int(timing["navigator_count"])
            or not np.array_equal(metadata["polarity"], acquisition["polarity"])):
        raise ValueError("Measured scan and sequence disagree on navigators or readout polarity.")
    expected_rows = np.argsort(np.asarray(acquisition["line_index"])[:, 0]).tolist()
    if list(metadata["acquired_phase_rows"]) != expected_rows:
        raise ValueError("Measured scan and sequence disagree on chronological phase rows.")


def _check_patient_geometry(reference, other, label):
    if not np.allclose(reference["slice_position_mm"], other["slice_position_mm"], rtol=0, atol=1e-3):
        raise ValueError(f"{label} and GRE slice centres differ; comparison does not perform registration.")
    q0, q1 = (np.asarray(item["slice_quaternion_wxyz"], float) for item in (reference, other))
    if (q0.shape != (4,) or q1.shape != (4,) or not np.isfinite(q0).all() or not np.isfinite(q1).all()
            or not np.isclose(abs(np.dot(q0, q1)), 1, rtol=0, atol=1e-8)):
        raise ValueError(f"{label} and GRE slice orientations differ; comparison does not perform registration.")


def compare_measured_scan_set(inputs, corrected_scan_paths, *, corrected_sequence_path=None,
                              frame_matrix=None, roi_mask=None, normalize=True,
                              navigator_phase_branch=0, reference_threshold_fraction=0.1,
                              skip_incompatible=False):
    """Read every explicitly supplied corrected Twix and compare to GRE/EPI.

    ``inputs`` is the native ``read_twin_inputs`` result. Its GRE and baseline
    are reused. Supply all desired result paths explicitly; there is no glob,
    inferred best result, or claim that historical scans used current currents.
    Unless a corrected sequence is supplied, the baseline sequence is assumed
    and checked against corrected Twix headers/order. All results share one
    coverage mask and one baseline-derived gain.
    ``skip_incompatible=True`` retains validation and permission failures in
    ``report.skipped_scans``; no continuous reconstruction uses a failed scan.
    If all fail, return empty ``images``/``rows`` and ``comparison_available=False``.
    """
    paths = [Path(path) for path in corrected_scan_paths]
    if not paths:
        raise ValueError("Supply at least one explicit corrected scan path.")
    if len(set(path.resolve() for path in paths)) != len(paths):
        raise ValueError("Corrected scan paths must be distinct.")
    gre, baseline_metadata = inputs["gre"], inputs["scan_metadata"]
    reference_metadata = gre["twix_metadata"]
    acquisition, timing = inputs["acquisition"], inputs["timing"]
    _check_scan(baseline_metadata, acquisition, timing)
    _check_patient_geometry(reference_metadata, baseline_metadata, "Baseline")
    baseline = np.asarray(inputs["scan"], float)
    reconstruction_reports = {}
    parity = None
    diagnostics = inputs.get("scan_diagnostics", {})
    baseline_branch = (diagnostics.get("navigator_fit") or {}).get("phase_branch")
    if baseline_branch is not None and int(baseline_branch) != int(navigator_phase_branch):
        raise ValueError("Baseline and corrected navigator branches differ; reconstruct the baseline consistently first.")
    if "kspace" in diagnostics:
        reconstructed, reconstruction_reports["baseline"] = reconstruct_receivers(dict(diagnostics), acquisition)
        denominator = float(np.linalg.norm(baseline))
        parity = float(np.linalg.norm(reconstructed - baseline) / denominator) if denominator else None
        baseline = reconstructed
    X, Y = _scan_grid(baseline, baseline_metadata, acquisition)
    reference = {"image": gre["magnitude"], "X_m": gre["X_m"], "Y_m": gre["Y_m"], "label": "GRE reference"}
    base_record = {"image": baseline, "X_m": X, "Y_m": Y, "label": "Without ACDC"}
    if corrected_sequence_path is None:
        corrected_acquisition, corrected_timing = acquisition, timing
    else:
        corrected_acquisition, corrected_timing = read_sequence(Path(corrected_sequence_path))
    corrected, metadata_records, valid_paths, skipped = [], [reference_metadata, baseline_metadata], [], []
    for path in paths:
        metadata = None
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix.lower() != ".dat":
                raise ValueError("Corrected measured scans must be Siemens Twix .dat files.")
            _, metadata, diagnostics = read_scan(path, corrected_acquisition, return_intermediates=True,
                                                navigator_phase_branch=navigator_phase_branch)
            _check_scan(metadata, corrected_acquisition, corrected_timing)
            _check_patient_geometry(reference_metadata, metadata, path.name)
        except (ValueError, PermissionError) as error:
            if not skip_incompatible:
                if isinstance(error, PermissionError):
                    raise
                raise ValueError(f"{path.name}: {error}") from error
            record = {"path": str(path.absolute()), "filename": path.name,
                      "reason": str(error), "error_type": type(error).__name__}
            if metadata is not None:
                record.update(TE_s=float(metadata["TE_s"]), TR_s=float(metadata["TR_s"]),
                              dwell_s=float(metadata["dwell_s"]))
            skipped.append(record)
            continue
        image, reconstruction_reports[path.name] = reconstruct_receivers(diagnostics, corrected_acquisition)
        scanX, scanY = _scan_grid(image, metadata, corrected_acquisition)
        corrected.append({"image": image, "X_m": scanX, "Y_m": scanY, "label": path.name})
        metadata_records.append(metadata)
        valid_paths.append(path)
    if not corrected:
        return {"images": [], "labels": [], "rows": [], "mask": None, "requested_mask": None,
                "common_coverage": None, "X_m": None, "Y_m": None, "gain": None,
                "reference_scale": None, "metadata": metadata_records,
                "report": {"comparison_available": False, "compatible_scan_count": 0,
                           "reason": "No corrected scan could be read and pass sequence and physical geometry validation.",
                           "requested_corrected_scan_paths": [str(path.resolve()) for path in paths],
                           "corrected_scan_paths": [], "skipped_scans": skipped,
                           "baseline_reader_RSS_relative_error": parity,
                           "waveform_match_verified": False,
                           "waveform_provenance": "not checked against current optimization"}}
    result = compare_registered_images(reference, base_record, corrected, frame_matrix=frame_matrix,
                                       roi_mask=roi_mask, normalize=normalize,
                                       reference_threshold_fraction=reference_threshold_fraction)
    result["metadata"] = metadata_records
    result["report"].update(
        comparison_available=True, compatible_scan_count=len(corrected),
        requested_corrected_scan_paths=[str(path.resolve()) for path in paths],
        corrected_scan_paths=[str(path.resolve()) for path in valid_paths], skipped_scans=skipped,
        corrected_sequence_assumed_same_as_baseline=corrected_sequence_path is None,
        baseline_reader_RSS_relative_error=parity,
        baseline_navigator_phase_branch=None if baseline_branch is None else int(baseline_branch),
        navigator_phase_branch=int(navigator_phase_branch),
        geometry_checks="Twix versus sequence: TE/TR/dwell/FOV/matrix/navigators/polarity/phase order; all slice centres and orientations versus GRE",
        slice_thickness_mm=[float(metadata["slice_thickness_mm"]) for metadata in metadata_records],
        TE_s=[float(metadata["TE_s"]) for metadata in metadata_records],
        TR_s=[float(metadata["TR_s"]) for metadata in metadata_records],
        reconstruction_reports=reconstruction_reports)
    return result


def compare_measured_scans(inputs, corrected_scan_path, **options):
    """Convenience wrapper for one explicitly selected corrected acquisition."""
    return compare_measured_scan_set(inputs, [corrected_scan_path], **options)


__all__ = ["compare_registered_images", "compare_measured_scan_set", "compare_measured_scans"]
