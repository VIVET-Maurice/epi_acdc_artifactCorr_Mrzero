"""Read registered ACDC maps from the existing MAT or the original 3D NPZ.

The NPZ stores responses in channel order, with source coordinates in metres.
Units and the confirmed patient-to-source placement are supplied explicitly in
config.calibration because the original NPZ does not contain that metadata.
Interpolation reuses the established calibration preparation code.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from prepare_acdc_calibration import (interpolate_maps, sha256, source_grid,
                                     target_patient_grid)


def read_calibration(path, experiment, settings=None):
    """Return the MAT-compatible calibration dictionary on the reference grid.

    MAT inputs retain their own metadata and are resampled by the existing caller.
    A 2D NPZ may store the same arrays and scalar metadata as the MAT. Raw 3D
    NPZ inputs contain fieldmap[nx,ny,nz,32], mask, xx, yy and zz. Their settings
    require units='Hz/A', registration_confirmed=true and patient_to_source_m
    (a 4x4 affine in metre coordinates). No source file is modified.
    """
    path = Path(path)
    if path.suffix.lower() == ".mat":
        return loadmat(path, simplify_cells=True)
    if path.suffix.lower() != ".npz":
        raise ValueError("ACDC calibration must be a .mat or .npz file.")

    with np.load(path, allow_pickle=False) as archive:
        if "acdc_maps" in archive:
            required = ("acdc_maps", "X_m", "Y_m", "units", "registration_confirmed")
            if not all(key in archive for key in required):
                raise ValueError("2D ACDC NPZ requires acdc_maps, X_m, Y_m, units and registration_confirmed.")
            calibration = {key: archive[key] for key in required}
            for key in ("units", "registration_confirmed"):
                if calibration[key].size != 1:
                    raise ValueError(f"2D ACDC NPZ {key} must be a scalar.")
                calibration[key] = calibration[key].item()
            if "spatial_valid_mask" in archive:
                calibration["spatial_valid_mask"] = archive["spatial_valid_mask"]
            calibration["metadata"] = {"source_file": str(path.resolve()),
                                        "source_sha256": sha256(path)}
            return calibration

    settings = {} if settings is None else settings
    if not isinstance(settings, dict):
        raise ValueError("config.calibration must be an object.")
    if settings.get("units") != "Hz/A":
        raise ValueError("NPZ calibration requires config.calibration.units='Hz/A'.")
    if settings.get("registration_confirmed") is not True:
        raise ValueError("NPZ calibration requires config.calibration.registration_confirmed=true.")
    affine = np.asarray(settings.get("patient_to_source_m"), dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or abs(np.linalg.det(affine[:3, :3])) < 1e-10):
        raise ValueError("config.calibration.patient_to_source_m must be an invertible 4x4 affine in metres.")

    source_hash = sha256(path)
    expected_hash = settings.get("source_sha256")
    if expected_hash is not None and expected_hash != source_hash:
        raise ValueError("NPZ calibration source_sha256 differs from the confirmed source in config.calibration.")

    geometry = experiment["geometry"]
    if not all(key in geometry for key in ("orientation", "position_mm", "spacing_mm")):
        raise ValueError("3D NPZ calibration needs the reference's patient geometry: "
                         "orientation, position_mm and spacing_mm (available in DICOM).")
    patient = target_patient_grid(experiment)
    with np.load(path, allow_pickle=False) as archive:
        required = ("fieldmap", "mask", "xx", "yy", "zz")
        if not all(key in archive for key in required):
            raise ValueError("ACDC NPZ requires fieldmap, mask, xx, yy and zz arrays.")
        source = {key: archive[key] for key in required}
    if np.iscomplexobj(source["fieldmap"]):
        raise ValueError("ACDC responses must be real values in Hz/A.")
    axes = source_grid(source)
    query = patient @ affine[:3, :3].T + affine[:3, 3]
    maps, valid = interpolate_maps(source, axes, query)
    metadata = {
        "source_file": str(path.resolve()), "source_sha256": source_hash,
        "source_fieldmap_shape": np.asarray(source["fieldmap"].shape),
        "source_channel_order": np.arange(1, maps.shape[-1] + 1),
        "units": "Hz/A", "registration_confirmed": True,
        "patient_to_source_m": affine,
        "registration_note": settings.get("registration_note", "Confirmed in config.calibration."),
        "interpolation": "Trilinear on registered coordinates; full valid source stencil; no extrapolation.",
    }
    return {"acdc_maps": maps, "X_m": geometry["X_m"], "Y_m": geometry["Y_m"],
            "units": "Hz/A", "spatial_valid_mask": valid,
            "registration_confirmed": True, "metadata": metadata}
