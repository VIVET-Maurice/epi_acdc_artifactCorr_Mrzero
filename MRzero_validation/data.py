"""Load portable ACDC inputs and export the original MATLAB slice selection.

The packaged NPZ files contain the exact cropped arrays, so the notebook runs
without the large shared MAT archives. MATLAB v7.3 source files are read with
h5py only when exporting another subject or slice.
"""

import json
from pathlib import Path

import numpy as np


def _input_path(subject, slice_index, directory):
    """Use three-digit subject identifiers and MATLAB's 1-based slice number."""
    subject = f"{int(subject):03d}"
    slice_index = int(slice_index)
    if slice_index < 1:
        raise ValueError("slice_index uses MATLAB's 1-based convention and must be positive.")
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
    return directory / f"subject{subject}_slice{slice_index}_inputs.npz"


def load_subject_inputs(subject="006", slice_index=120, directory=None):
    """Return independent NumPy arrays from a packaged subject/slice archive.

    slice_index is the MATLAB 1-based index. Spatial arrays follow MATLAB's
    [x, y] order; shim maps have shape [x, y, channel] and units Hz/A. X_m and
    Y_m are encoding coordinates in meters, including the original -10 mm Y
    shift. No image normalization, resampling, or additional masking is done.
    """
    path = _input_path(subject, slice_index, directory)
    with np.load(path, allow_pickle=False) as archive:
        inputs = {name: archive[name].copy() for name in archive.files}
    return inputs


def default_params(inputs):
    """Return the original subject-006 simulation and optimization protocol."""
    return dict(
        X=inputs["X_m"], Y=inputs["Y_m"], FOV=0.22, matrix_size=32,
        dt=8e-6, Gmax=0.08, Smax=200.0, T1=1.2, T2=0.080,
        Imax=5.0, gammaHz=42.577e6, shim_opt=3, niters=500,
    )


def export_subject_inputs(source_directory, subject="006", slice_index=120, directory=None):
    """Export the exact run_code.m selection from original MATLAB v7.3 files.

    source_directory contains b0maps/b0db_XXX.mat and acdc32_basis.mat. Select
    the bounding box of field_map_data.mask on the chosen slice, then retain
    every third voxel in both dimensions. The constraint mask keeps every
    third selected head voxel, matching mask2(1:3:end, 1:3:end). Save a small
    NPZ archive plus an English provenance JSON and return the NPZ path.
    """
    import h5py

    path = _input_path(subject, slice_index, directory)
    subject = f"{int(subject):03d}"
    slice_index = int(slice_index)
    source_directory = Path(source_directory)
    field_path = source_directory / "b0maps" / f"b0db_{subject}.mat"
    basis_path = source_directory / "acdc32_basis.mat"
    z = slice_index - 1

    with h5py.File(field_path, "r") as field_file:
        field = field_file["field_map_data"]
        if z >= field["mask"].shape[0]:
            raise ValueError(f"slice_index={slice_index} exceeds the source volume depth.")
        # HDF5 stores MATLAB dimensions in reverse order: [z, y, x].
        full_mask = field["mask"][z].T.astype(bool)
        x_indices, y_indices = np.nonzero(full_mask)
        if x_indices.size == 0:
            raise ValueError("The selected source slice has an empty head mask.")
        x_start, x_stop = int(x_indices.min()), int(x_indices.max())
        y_start, y_stop = int(y_indices.min()), int(y_indices.max())
        xs = slice(x_start, x_stop + 1, 3)
        ys = slice(y_start, y_stop + 1, 3)

        def selected(name):
            return np.asarray(field[name][z, ys, xs].T, dtype=np.float64)

        head_mask = selected("mask").astype(bool)
        constraint_mask = np.zeros_like(head_mask)
        constraint_mask[::3, ::3] = head_mask[::3, ::3]
        inputs = dict(
            img=selected("mprage"),
            b0_Hz=selected("field_map_ppm") * 297 * head_mask,
            X_m=selected("x_pos_mm") * 0.001,
            Y_m=(selected("y_pos_mm") - 10) * 0.001,
            Z_m=selected("z_pos_mm") * 0.001,
            head_mask=head_mask,
            brain_mask=selected("brain_mask").astype(bool),
            constraint_mask=constraint_mask,
        )
        spatial_shape = list(reversed(field["mask"].shape))

    with h5py.File(basis_path, "r") as basis_file:
        # Basis dimensions on disk are [channel, z, y, x].
        basis = basis_file["acdc_basis"]
        inputs["shim_Hz_per_A"] = basis[:, z, ys, xs].transpose(2, 1, 0) * 42.6e6
        basis_shape = list(reversed(basis.shape))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **inputs)
    provenance = dict(
        description="Exact original MATLAB inputs, without resampling or recalibration.",
        subject=subject, slice_MATLAB_1based=slice_index, slice_python_0based=z,
        source_files=dict(field_map=str(field_path), acdc_basis=str(basis_path)),
        source_dataset_shapes_MATLAB=dict(field_map_data=spatial_shape, acdc_basis=basis_shape),
        crop_MATLAB_inclusive_1based=dict(
            x_start=x_start + 1, x_stop=x_stop + 1,
            y_start=y_start + 1, y_stop=y_stop + 1, step=3,
        ),
        arrays={name: dict(shape=list(value.shape), dtype=str(value.dtype))
                for name, value in inputs.items()},
        axis_order=dict(spatial_arrays="[x, y] matching the original MATLAB arrays",
                        shim_Hz_per_A="[x, y, channel]; Python channels 0..31 are MATLAB channels 1..32"),
        units_and_operations=dict(
            img="Unnormalized MPRAGE magnitude; no head masking applied.",
            b0_Hz="field_map_ppm * 297 * head_mask",
            shim_Hz_per_A="acdc_basis * 42.6e6; original Bz basis conversion to Hz/A",
            X_m="x_pos_mm * 0.001", Y_m="(y_pos_mm - 10) * 0.001", Z_m="z_pos_mm * 0.001",
            head_mask="mask selected from its bounding box with spatial step 3",
            brain_mask="brain_mask selected with the same bounding box and spatial step",
            constraint_mask="head_mask retained only at selected array indices [0::3, 0::3]",
        ),
        geometry_limitations=(
            "Scanner-coordinate arrays and the supplied coil basis are preserved. "
            "These exports contain no DICOM orientation matrix or declared RAS/LPS frame. "
            "The method assumes the coil basis and field maps share their coordinate frame."
        ),
        mask_voxels={name: int(inputs[name].sum())
                     for name in ("head_mask", "brain_mask", "constraint_mask")},
    )
    provenance_path = path.with_name(path.name.replace("_inputs.npz", "_provenance.json"))
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return path
