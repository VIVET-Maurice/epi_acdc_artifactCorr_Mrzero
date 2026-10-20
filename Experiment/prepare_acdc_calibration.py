"""Audit and interpolate an ACDC Hz/A volume on the prepared experiment grid.

The phantom scans were acquired without ACDC coils. The intended experiment
is a VIRTUAL correction using the supplied 3T basis and a documented virtual
placement. Identity placement means NPZ (xx, yy, zz) are set equal to patient
(x, y, z) in metres; it is not a measured physical registration. An explicit
patient-to-source affine can instead be supplied with --transform.

virtual_optimization_ready depends on coverage, not physical registration.
The separate legacy optimization_ready flag remains false unless a physical
registration has been independently confirmed with --confirm-registration.

No response is invented outside the source mask.  A missing calibration stencil
is represented by NaN.  If --restrict-support is explicitly selected, a NEW
experimental input is saved with these voxels excluded from its simulation;
the existing input, measurements and raw calibration are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_grid(source: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the actual xyz axis order instead of assuming an image order."""
    fieldmap, mask = source["fieldmap"], source["mask"]
    if fieldmap.ndim != 4 or fieldmap.shape[-1] != 32:
        raise ValueError("Expected fieldmap[nx, ny, nz, 32] in Hz/A.")
    if mask.shape != fieldmap.shape[:3] or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected a binary source mask with the fieldmap grid shape.")
    axes = []
    for dimension, name in enumerate(("xx", "yy", "zz")):
        coordinates = source[name]
        if coordinates.shape != mask.shape or not np.isfinite(coordinates).all():
            raise ValueError(f"{name} must be a finite coordinate volume.")
        selector = [0, 0, 0]
        selector[dimension] = slice(None)
        axis = coordinates[tuple(selector)].copy()
        shape = [1, 1, 1]
        shape[dimension] = len(axis)
        if not np.allclose(coordinates, axis.reshape(shape), rtol=0, atol=1e-10):
            raise ValueError(f"{name} is not aligned with array axis {dimension}; no implicit flip is permitted.")
        if not np.all(np.diff(axis) > 0):
            raise ValueError(f"{name} axis must be strictly increasing.")
        axes.append(axis)
    return tuple(axes)


def target_patient_grid(experiment: dict) -> np.ndarray:
    """DICOM: column advances along first IOP vector, row along the second."""
    geometry = experiment["geometry"]
    X_m = np.asarray(geometry["X_m"], dtype=float)
    Y_m = np.asarray(geometry["Y_m"], dtype=float)
    if X_m.shape != Y_m.shape or X_m.ndim != 2:
        raise ValueError("Invalid prepared X_m/Y_m geometry.")
    orientation = np.asarray(geometry["orientation"], dtype=float).reshape(6)
    column_direction, row_direction = orientation[:3], orientation[3:]
    if not np.allclose([np.linalg.norm(column_direction), np.linalg.norm(row_direction),
                        np.dot(column_direction, row_direction)], [1, 1, 0], atol=1e-8):
        raise ValueError("DICOM orientation vectors must be orthonormal.")
    position_m = np.asarray(geometry["position_mm"], dtype=float).reshape(3) * 1e-3
    spacing_m = np.asarray(geometry["spacing_mm"], dtype=float).reshape(2) * 1e-3
    row, column = np.indices(X_m.shape)
    patient = (position_m + column[..., None] * spacing_m[1] * column_direction
               + row[..., None] * spacing_m[0] * row_direction)
    # The prepared simulator uses readout +patient-y and phase -patient-x.
    if not (np.allclose(X_m, patient[..., 1], rtol=0, atol=1e-10)
            and np.allclose(Y_m, -patient[..., 0], rtol=0, atol=1e-10)):
        raise ValueError("Prepared simulator axes differ from X=patient+y,Y=patient-x; review geometry explicitly.")
    return patient


def load_transform(path: Path | None) -> np.ndarray:
    if path is None:
        return np.eye(4)
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value["patient_to_source_m"]
    affine = np.asarray(value, dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1])
            or abs(np.linalg.det(affine[:3, :3])) < 1e-10):
        raise ValueError("Transform must be an invertible 4x4 affine in metre coordinates.")
    return affine


def interpolate_maps(source: dict, axes: tuple, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Require all eight trilinear source-mask corners, with no extrapolation."""
    weights = RegularGridInterpolator(axes, source["mask"].astype(float),
                                      method="linear", bounds_error=False, fill_value=0)(query)
    valid = weights >= 1 - 1e-10
    maps = np.full(query.shape[:2] + (source["fieldmap"].shape[-1],), np.nan)
    for channel in range(maps.shape[-1]):
        sampled = RegularGridInterpolator(axes, source["fieldmap"][..., channel],
                                          method="linear", bounds_error=False, fill_value=np.nan)(query)
        maps[..., channel] = np.where(valid, sampled, np.nan)
    valid &= np.isfinite(maps).all(axis=-1)
    maps[~valid] = np.nan
    return maps, valid


def save_geometry_figure(path: Path, experiment: dict, maps: np.ndarray, valid: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    support = experiment["field"]["simulation_mask"].astype(bool)
    quality = experiment["field"]["mask"].astype(bool)
    category = np.zeros(valid.shape, dtype=int)
    category[valid] = 1
    category[support & valid] = 2
    category[support & ~valid] = 3
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].imshow(experiment["m0_proxy"], cmap="gray")
    axes[0, 0].set_title("Prepared object amplitude (GRE grid)")
    axes[0, 1].imshow(category, cmap=ListedColormap(["white", "#cccccc", "#62aa69", "#d3483b"]),
                      vmin=0, vmax=3, interpolation="nearest")
    axes[0, 1].set_title("Calibration: valid support green; missing red")
    axes[0, 2].imshow(quality & valid, cmap="gray", interpolation="nearest")
    axes[0, 2].set_title("Measured B0 quality within valid calibration")
    for axis, channel in zip(axes[1], [0, 15, 31]):
        scale = np.nanmax(np.abs(maps[..., channel]))
        artist = axis.imshow(maps[..., channel], cmap="RdBu_r", vmin=-scale, vmax=scale)
        axis.set_title(f"ACDC channel {channel + 1}: candidate [Hz/A]")
        fig.colorbar(artist, ax=axis, shrink=.8)
    for axis in axes.flat:
        axis.set_xlabel("GRE column: +patient x")
        axis.set_ylabel("GRE row: +patient y")
    fig.suptitle("Virtual ACDC placement on the phantom grid; no ACDC coils during acquisition")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def prepare(args: argparse.Namespace) -> dict:
    source_path, input_path = args.source.resolve(), args.input.resolve()
    with np.load(source_path, allow_pickle=False) as archive:
        required = ("fieldmap", "mask", "xx", "yy", "zz")
        source = {name: archive[name] for name in required}
    axes = source_grid(source)
    experiment = loadmat(input_path, simplify_cells=True)["experiment"]
    patient = target_patient_grid(experiment)
    affine = load_transform(args.transform)
    query = patient @ affine[:3, :3].T + affine[:3, 3]
    maps, valid = interpolate_maps(source, axes, query)
    support = np.asarray(experiment["field"]["simulation_mask"], dtype=bool)
    quality = np.asarray(experiment["field"]["mask"], dtype=bool)
    m0 = np.asarray(experiment["m0_proxy"], dtype=float)
    missing = support & ~valid
    active_support = support & valid if args.restrict_support else support
    registration_confirmed = bool(args.confirm_registration)
    virtual_ready = not np.any(active_support & ~valid) and active_support.any()
    ready = registration_confirmed and virtual_ready
    reasons = []
    if not registration_confirmed:
        reasons.append("NPZ coordinate frame/origin and alignment to the experimental object are not confirmed.")
    if np.any(active_support & ~valid):
        reasons.append("Simulation support contains voxels with no valid source calibration stencil.")
    if not active_support.any():
        reasons.append("There is no simulation support with valid calibration.")
    report = {
        "source_file": str(source_path), "source_sha256": sha256(source_path),
        "source_reference": args.source_reference,
        "source_installation_note": "User identifies supplied 3T maps as Bay5 and 7T maps as Bay2; registration is separately checked.",
        "input_file": str(input_path), "input_sha256": sha256(input_path),
        "units": "Hz/A", "units_provenance": "Explicit confirmation supplied by the user; NPZ has no units key.",
        "study_kind": "Virtual ACDC correction of phantom data acquired without ACDC coils.",
        "ACDC_coils_present_during_scan": False,
        "virtual_placement": "Patient-to-source affine defines a virtual coil placement; not measured physical registration.",
        "source_fieldmap_shape": list(source["fieldmap"].shape),
        "source_array_axis_order": ["patient/source x", "patient/source y", "patient/source z", "channel"],
        "source_axes_m": [[float(axis[0]), float(axis[-1])] for axis in axes],
        "source_coordinate_separability_tolerance_m": 1e-10,
        "target_grid_shape": list(support.shape), "channel_count": int(maps.shape[-1]),
        "target_patient_bounds_m": [[float(patient[..., i].min()), float(patient[..., i].max())] for i in range(3)],
        "target_simulator_axes": "X_m=patient+y (readout),Y_m=patient-x (phase)",
        "patient_to_source_m": affine.tolist(), "registration_confirmed": registration_confirmed,
        "registration_note": args.registration_note or (
            "Coordinate frame/origin and registration explicitly confirmed by the user."
            if registration_confirmed else "Virtual placement is explicitly defined by the affine; physical registration is not applicable to the phantom acquisition without ACDC."),
        "transform_file": str(args.transform.resolve()) if args.transform else None,
        "transform_sha256": sha256(args.transform) if args.transform else None,
        "interpolation": "Trilinear, no extrapolation; source mask must cover the full interpolation stencil.",
        "invalid_response_policy": "NaN, never a zero-valued substitute response.",
        "valid_grid_voxels": int(valid.sum()), "grid_voxels": int(valid.size),
        "original_support_voxels": int(support.sum()), "covered_support_voxels": int((support & valid).sum()),
        "uncovered_support_voxels": int(missing.sum()),
        "coverage_support_fraction": float((support & valid).sum() / support.sum()),
        "quality_B0_voxels": int(quality.sum()), "covered_quality_B0_voxels": int((quality & valid).sum()),
        "uncovered_m0_fraction": float(m0[missing].sum() / m0[support].sum()),
        "uncovered_m0_energy_fraction": float((m0[missing] ** 2).sum() / (m0[support] ** 2).sum()),
        "restrict_support_explicitly_requested": bool(args.restrict_support),
        "active_support_voxels": int(active_support.sum()), "optimization_ready": bool(ready),
        "virtual_optimization_ready": bool(virtual_ready),
        "virtual_blocking_reasons": [reason for reason in reasons if not reason.startswith("NPZ coordinate")],
        "blocking_reasons": reasons,
        "channel_order": "NPZ fieldmap fourth-axis order, channel indices 1..32; no channel permutation applied.",
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration_file = output / "acdc_calibration_candidate.mat"
    metadata = {
        "registration_confirmed": registration_confirmed, "optimization_ready": bool(ready),
        "virtual_optimization_ready": bool(virtual_ready),
        "study_kind": report["study_kind"], "virtual_placement": report["virtual_placement"],
        "ACDC_coils_present_during_scan": False,
        "patient_to_source_m": affine, "source_file": str(source_path),
        "source_sha256": report["source_sha256"], "registration_note": report["registration_note"],
        "source_reference": args.source_reference,
        "source_channel_order": np.arange(1, maps.shape[-1] + 1),
        "support_restricted": bool(args.restrict_support),
        "uncovered_original_support_voxels": int(missing.sum()),
    }
    savemat(calibration_file, {
        "acdc_maps": maps, "X_m": experiment["geometry"]["X_m"],
        "Y_m": experiment["geometry"]["Y_m"], "units": "Hz/A", "spatial_valid_mask": valid,
        "registration_confirmed": registration_confirmed, "optimization_ready": bool(ready),
        "virtual_optimization_ready": bool(virtual_ready),
        "metadata": metadata,
    }, do_compression=True, long_field_names=True)
    report["calibration_file"] = str(calibration_file)
    # Plot original support before any explicitly requested support restriction.
    figure = output / "acdc_calibration_geometry.png"
    save_geometry_figure(figure, experiment, maps, valid)
    report["geometry_figure"] = str(figure)
    if args.restrict_support:
        # The new ideal target/reference must subsequently be computed using
        # this same cropped object.  Measured B0 and image arrays are preserved.
        experiment["field"]["simulation_mask"] = active_support
        experiment["m0_proxy"] = np.where(active_support, m0, 0)
        experiment["optimization_ready"] = bool(ready)
        experiment["virtual_optimization_ready"] = bool(virtual_ready)
        previous = [value for value in np.asarray(experiment["assumptions"], dtype=object).ravel().tolist()
                    if "Experimental ACDC maps in Hz/A are absent" not in str(value)]
        previous.append("ACDC maps supplied in Hz/A; coordinate registration status is recorded in calibration provenance.")
        previous.append("No ACDC coils were present during the phantom scans; supplied 3T response maps define a virtual coil placement for simulation only.")
        previous.append("ACDC source-mask intersection explicitly selected; excluded object mass is recorded in calibration provenance.")
        experiment["assumptions"] = np.asarray(previous, dtype=object)
        experiment["provenance"]["acdc_calibration"] = metadata
        cropped_input = output / "experiment_input_acdc_support.mat"
        savemat(cropped_input, {"experiment": experiment}, do_compression=True, long_field_names=True)
        report["restricted_input_file"] = str(cropped_input)
    else:
        report["restricted_input_file"] = None
    (output / "acdc_calibration_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "calibration_file", "covered_support_voxels", "original_support_voxels", "uncovered_support_voxels",
        "covered_quality_B0_voxels", "quality_B0_voxels", "uncovered_m0_fraction",
        "registration_confirmed", "optimization_ready", "virtual_optimization_ready", "blocking_reasons",
        "virtual_blocking_reasons")}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "ACDC_maps" / "ACDC_3T.npz")
    parser.add_argument("--source-reference", default="https://github.com/bughht/ACDC_Optim",
                        help="User-provided provenance reference; does not certify geometric registration.")
    parser.add_argument("--input", type=Path, default=ROOT / "prepared" / "experiment_input.mat")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prepared" / "acdc_calibration")
    parser.add_argument("--transform", type=Path, help="JSON 4x4 patient-to-source affine in metres.")
    parser.add_argument("--confirm-registration", action="store_true",
                        help="Explicitly confirm source coordinates, origin and object alignment.")
    parser.add_argument("--registration-note", default=None, help="Record how source-to-experiment registration was established.")
    parser.add_argument("--restrict-support", action="store_true",
                        help="Explicitly export a separate input restricted to source-calibration coverage.")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
