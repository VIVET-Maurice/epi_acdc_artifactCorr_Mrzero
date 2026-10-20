"""Read synthetic raw acquisitions with the V4 reconstruction conventions.

The manifest identifies portable NPZ raw-data files and matching Pulseq
sequences. Ground truth is deliberately never loaded by this adapter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
VALIDATION_DIR = PROJECT_DIR / "MRzero_validation"
for directory in (PROJECT_DIR / "Experiment", VALIDATION_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from measured_continuous import reconstruct_b0_echoes, reconstruct_receivers
from read_acdc_b0 import b0_from_coil_images, read_b0_sequence
from read_acdc_sequence import read_sequence

FORMAT_NAME = "mrzero_synthetic_acquisition"
REQUIRED_FILES = (
    "scan", "reference", "b0", "scan_sequence", "reference_sequence", "b0_sequence",
)


def _sha256(path):
    """Hash a file without holding its complete contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_paths(folder, manifest):
    """Resolve existing files, rejecting paths outside the acquisition folder."""
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError(f"Expected acquisition manifest format {FORMAT_NAME!r}.")
    if manifest.get("schema_version") != 1:
        raise ValueError("Only synthetic acquisition schema_version=1 is supported.")
    files = manifest.get("files")
    if not isinstance(files, dict) or any(name not in files for name in REQUIRED_FILES):
        raise ValueError(f"Manifest files must include {', '.join(REQUIRED_FILES)}.")
    paths = {}
    for name, filename in files.items():
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Manifest files.{name} must be a nonempty relative path.")
        relative = Path(filename)
        if relative.is_absolute():
            raise ValueError(f"Manifest files.{name} must be relative to the acquisition folder.")
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder):
            raise ValueError(f"Manifest files.{name} escapes the acquisition folder.")
        if not path.is_file():
            raise FileNotFoundError(f"Missing synthetic acquisition file: {path}")
        paths[name] = path
    return paths


def _validate_hashes(paths, manifest):
    """Check optional files_sha256 entries; return every observed file hash."""
    observed = {name: _sha256(path) for name, path in paths.items()}
    expected = manifest.get("files_sha256", {})
    if not isinstance(expected, dict):
        raise ValueError("Manifest files_sha256 must be an object keyed like files.")
    for name, value in expected.items():
        if (name not in paths or not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value.lower())):
            raise ValueError(f"Invalid manifest SHA-256 entry for {name!r}.")
        if observed[name] != value.lower():
            raise ValueError(f"SHA-256 mismatch for synthetic input {name}: {paths[name].name}")
    return observed


def _complex_array(value, expected_shape, name):
    """Validate finite complex samples with documented acquisition axes."""
    value = np.asarray(value)
    if (not np.iscomplexobj(value) or value.shape != expected_shape
            or not np.isfinite(value).all()):
        raise ValueError(f"{name} must be finite complex data with shape {expected_shape}.")
    return value


def _read_receivers(path, acquisition, navigator_count=0):
    """Reconstruct NPZ k-space ordered as [phase, receiver, readout].

    Ideal synthetic samples have no odd/even receiver mismatch. Optional
    navigators are preserved, without applying a second phase correction.
    """
    ny, nx = np.asarray(acquisition["kx_mInv"]).shape
    with np.load(path, allow_pickle=False) as archive:
        if "kspace" not in archive:
            raise ValueError(f"{path.name} is missing kspace.")
        kspace = np.asarray(archive["kspace"])
        receivers = kspace.shape[1] if kspace.ndim == 3 else 0
        if receivers < 1:
            raise ValueError(f"{path.name}.kspace needs [phase, receiver, readout] axes.")
        kspace = _complex_array(kspace, (ny, receivers, nx), f"{path.name}.kspace")
        navigators = None
        if "navigator_kspace" in archive:
            navigators = _complex_array(archive["navigator_kspace"],
                (navigator_count, receivers, nx), f"{path.name}.navigator_kspace")
    diagnostics = {
        "source_format": "MRzero synthetic NPZ", "has_raw_kspace": True,
        "array_axes": ["phase", "receiver", "readout"],
        "kspace": kspace, "corrected_kspace": kspace,
        "negative_readout_rows": np.asarray(acquisition["polarity"]) < 0,
        "navigator_kspace": navigators, "navigator_hybrid": None,
        "navigator_phase_correction_rad": np.zeros((receivers, nx)),
        "navigator_fit": None, "navigator_correction_applied": False,
        "receiver_phase_reapplied": False,
    }
    image, _ = reconstruct_receivers(diagnostics, acquisition)
    return image, diagnostics


def _geometry(shape, fov):
    """Build centered axial coordinates with the Fourier origin at n//2."""
    ny, nx = shape
    dx, dy = np.asarray(fov) / np.asarray([nx, ny])
    X, Y = np.meshgrid((np.arange(nx) - nx // 2) * dx,
                       (np.arange(ny) - ny // 2) * dy)
    return {
        "X_m": X, "Y_m": Y,
        "position_mm": np.asarray([ny // 2 * dy, -nx // 2 * dx, 0.]) * 1000,
        "spacing_mm": np.asarray([dy, dx]) * 1000,
        "orientation": np.asarray([0., 1., 0., -1., 0., 0.]),
        "slice_thickness_mm": 5.,
    }


def _scan_metadata(acquisition, timing, receivers):
    """Describe the known centered synthetic slice without scanner headers."""
    return {
        "source_format": "MRzero synthetic NPZ", "TE_s": acquisition["TE_s"],
        "TR_s": acquisition["TR_s"], "dwell_s": acquisition["dwell_s"],
        "FOV_m": np.asarray(acquisition["FOV_m"]).tolist(),
        "image_shape": list(np.asarray(acquisition["kx_mInv"]).shape),
        "receiver_count": receivers, "navigator_count": timing["navigator_count"],
        "polarity": np.asarray(acquisition["polarity"]).tolist(),
        "slice_position_mm": [0., 0., 0.],
        "slice_quaternion_wxyz": (np.asarray([1., 0., 0., 1.]) / np.sqrt(2)).tolist(),
        "slice_thickness_mm": 5., "receiver_phase_reapplied": False,
    }


def read_synthetic_twin_inputs(folder):
    """Read raw NPZ data into the full V4 input dictionary.

    Images and estimated B0 use the same reconstruction as measured V4 inputs.
    Optional ground-truth and calibration files are validated and hashed, but
    their arrays never participate in reconstruction or object fitting.
    """
    folder = Path(folder).resolve()
    manifest_path = folder / "acquisition.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    paths = _manifest_paths(folder, manifest)
    hashes = _validate_hashes(paths, manifest)
    hashes["manifest"] = _sha256(manifest_path)

    acquisition, timing = read_sequence(paths["scan_sequence"])
    reference_acquisition, reference_timing = read_sequence(paths["reference_sequence"])
    if acquisition["sequence_kind"] != "epi" or reference_acquisition["sequence_kind"] != "gre":
        raise ValueError("Synthetic V4 inputs need EPI scan and mono-echo GRE reference sequences.")
    scan, diagnostics = _read_receivers(paths["scan"], acquisition, timing["navigator_count"])
    reference, reference_diagnostics = _read_receivers(paths["reference"], reference_acquisition)
    metadata = _scan_metadata(acquisition, timing, diagnostics["kspace"].shape[1])
    reference_metadata = _scan_metadata(reference_acquisition, reference_timing,
                                        reference_diagnostics["kspace"].shape[1])
    gre = dict(_geometry(reference.shape, reference_acquisition["FOV_m"]),
        magnitude=reference, TE_s=reference_acquisition["TE_s"],
        TR_s=reference_acquisition["TR_s"], twix_metadata=reference_metadata)

    b0_info = read_b0_sequence(paths["b0_sequence"])
    ny, nx = b0_info["image_shape"]
    echo_count = b0_info["echo_count"]
    with np.load(paths["b0"], allow_pickle=False) as archive:
        if "kspace" not in archive:
            raise ValueError(f"{paths['b0'].name} is missing kspace.")
        kspace = np.asarray(archive["kspace"])
        receivers = kspace.shape[1] if kspace.ndim == 4 else 0
        if receivers < 1:
            raise ValueError("B0 kspace needs [echo, receiver, phase, readout] axes.")
        kspace = _complex_array(kspace, (echo_count, receivers, ny, nx), "B0 kspace")
    geometry = _geometry((ny, nx), b0_info["FOV_m"])
    raw = {
        "kspace": kspace, "X_m": geometry["X_m"], "Y_m": geometry["Y_m"],
        "array_axes": ["echo", "receiver", "phase", "readout"],
        "metadata": dict(b0_info, source_format="MRzero synthetic NPZ",
                         receiver_count=receivers, sequence_info=b0_info,
                         source_sha256=hashes["b0"], slice_thickness_mm=5.),
    }
    reconstruct_b0_echoes(raw, paths["b0_sequence"])
    echo_times = np.asarray(b0_info["TE_s"])
    field = b0_from_coil_images(raw["coil_images"], echo_times, return_intermediates=True)
    field.update(X_m=raw["X_m"], Y_m=raw["Y_m"])
    return {
        "paths": paths, "acquisition": acquisition, "timing": timing,
        "gre": gre, "scan": scan, "scan_metadata": metadata,
        "scan_diagnostics": diagnostics, "reference_diagnostics": reference_diagnostics,
        "b0_raw": raw, "field": field, "echo_times_s": echo_times,
        "source_sha256": hashes, "manifest": manifest,
    }


__all__ = ["read_synthetic_twin_inputs"]
