"""Compare explicitly selected spiral measurements with one shared GRE scale."""

from pathlib import Path

import numpy as np

from measured_scan_comparison import (
    compare_registered_images, _check_patient_geometry, _scan_grid,
)
from read_spiral_sequence import read_spiral_sequence
from slice_geometry import encoding_origin
from spiral_inputs import read_spiral_scan


def _matching_trajectory(baseline, candidate):
    """A corrected scan must repeat the nominal acquisition being optimized."""
    if tuple(candidate["image_shape"]) != tuple(baseline["image_shape"]):
        raise ValueError("Corrected spiral and baseline image matrices differ.")
    for key in ("kx_mInv", "ky_mInv", "time_s", "rf_time_s", "line_index",
                "line_start_s", "line_end_s", "FOV_m", "phase_offset_rad"):
        a, b = np.asarray(baseline[key]), np.asarray(candidate[key])
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-9):
            raise ValueError(f"Corrected spiral and baseline disagree on {key}.")
    for key in ("TE_s", "TR_s", "dwell_s", "navigator_count"):
        if not np.isclose(candidate[key], baseline[key], rtol=0, atol=1e-9):
            raise ValueError(f"Corrected spiral and baseline disagree on {key}.")


def compare_measured_scan_set(inputs, corrected_scan_paths, *,
        corrected_sequence_path=None, frame_matrix=None, roi_mask=None,
        normalize=True, navigator_phase_branch=0,
        reference_threshold_fraction=0.1, skip_incompatible=False):
    """Keep the same physical grid, DCF, coverage mask and gain for every scan.

    A raw-data filename does not prove that the current waveform was used.
    That provenance remains explicitly unverified in the returned report.
    Spiral imaging samples never receive an EPI navigator phase correction.
    """
    if navigator_phase_branch != 0:
        raise ValueError("An EPI navigator phase branch is inapplicable to spiral imaging.")
    paths = [Path(path) for path in corrected_scan_paths]
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Supply one or more distinct explicit corrected spiral scan paths.")
    acquisition, gre = inputs["acquisition"], inputs["gre"]
    if acquisition.get("sequence_kind") != "spiral":
        raise ValueError("Spiral measured comparison requires spiral inputs.")
    if corrected_sequence_path is not None:
        candidate, _ = read_spiral_sequence(corrected_sequence_path)
        _matching_trajectory(acquisition, candidate)
    baseline_metadata = inputs["scan_metadata"]
    reference_metadata = gre["twix_metadata"]
    _check_patient_geometry(reference_metadata, baseline_metadata, "Baseline")
    image = np.asarray(inputs["scan"], float)
    X, Y = _scan_grid(image, baseline_metadata, acquisition)
    reference = dict(image=gre["magnitude"], X_m=gre["X_m"], Y_m=gre["Y_m"], label="GRE reference")
    baseline = dict(image=image, X_m=X, Y_m=Y, label="Without ACDC",
                    encoding_origin_m=encoding_origin(baseline_metadata))
    corrected, skipped, metadata, valid_paths = [], [], [reference_metadata, baseline_metadata], []
    reconstruction_reports = {}
    for path in paths:
        try:
            scan, info, diagnostics = read_spiral_scan(path, acquisition, return_intermediates=True)
            _check_patient_geometry(reference_metadata, info, path.name)
            if not np.isclose(info["slice_thickness_mm"], baseline_metadata["slice_thickness_mm"], rtol=0, atol=1e-3):
                raise ValueError("Corrected spiral and baseline slice thicknesses differ.")
        except (ValueError, PermissionError, FileNotFoundError) as error:
            if not skip_incompatible:
                raise
            skipped.append(dict(path=str(path.absolute()), filename=path.name,
                                reason=str(error), error_type=type(error).__name__))
            continue
        scanX, scanY = _scan_grid(scan, info, acquisition)
        corrected.append(dict(image=scan, X_m=scanX, Y_m=scanY, label=path.name))
        metadata.append(info)
        valid_paths.append(str(path.resolve()))
        reconstruction_reports[path.name] = diagnostics["continuous_reconstruction"]
    report = dict(comparison_available=bool(corrected), compatible_scan_count=len(corrected),
        requested_corrected_scan_paths=[str(path.absolute()) for path in paths],
        corrected_scan_paths=valid_paths, skipped_scans=skipped,
        corrected_sequence_assumed_same_as_baseline=corrected_sequence_path is None,
        waveform_match_verified=False, waveform_provenance="not checked against current optimization",
        navigator_phase_correction_applied=False,
        geometry_checks="Matched Pulseq trajectory and turn windows; Twix samples, FOV, matrix, TE/TR, navigator count, slice centre, orientation and thickness",
        reconstruction_reports=reconstruction_reports)
    if not corrected:
        return dict(images=[], labels=[], rows=[], mask=None, requested_mask=None,
                    common_coverage=None, X_m=None, Y_m=None, gain=None,
                    reference_scale=None, metadata=metadata, report=report)
    result = compare_registered_images(reference, baseline, corrected,
        frame_matrix=frame_matrix, roi_mask=roi_mask, normalize=normalize,
        reference_threshold_fraction=reference_threshold_fraction)
    result["metadata"] = metadata
    result["report"].update(report, comparison_grid="native unshimmed spiral grid")
    return result


__all__ = ["compare_measured_scan_set"]
