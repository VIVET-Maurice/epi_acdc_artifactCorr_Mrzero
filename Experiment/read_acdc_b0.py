"""Reconstruct a measured B0 map from one Cartesian multi-echo GRE Twix.

The echo-pair products cancel each receive coil's static phase before coil
combination. No missing field values are extrapolated. The field convention is
``signal(TE) = amplitude * exp(-2j*pi*b0Hz*TE)``; scanner sign confirmation is
still handled by the folder pipeline's existing ``b0_sign`` option.
"""

from __future__ import annotations

import contextlib
import argparse
import hashlib
import inspect
import io
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from read_acdc_sequence import _events


def read_b0_sequence(path: Path) -> dict:
    """Read actual ADC trajectories/times for a full, monopolar multi-echo GRE.

    Two or more equally spaced echoes are supported. Leading dummy RF pulses
    are allowed; every acquired RF must have the same echo train. A tolerance
    of 0.002 k-space samples allows decimal rounding in exported .seq files.
    """
    import pypulseq as pp

    path = Path(path)
    sequence = pp.Sequence()
    sequence.read(str(path))
    rf, adc, duration = _events(sequence, pp)
    kspace, _, centers, refocusing, times = sequence.calculate_kspace()
    centers = np.asarray(centers)
    nx = adc[0]["samples"]
    dwell = adc[0]["dwell"]
    if (len(refocusing) or nx < 2 or nx % 2 or not np.isfinite(dwell) or dwell <= 0
            or any(event["samples"] != nx for event in adc)
            or not np.allclose([event["dwell"] for event in adc], dwell, rtol=0, atol=1e-12)
            or not np.allclose(centers, [event["center"] for event in rf], rtol=0, atol=1e-10)):
        raise ValueError("B0 sequence needs equal even ADC lengths/dwell and excitation RF only.")
    if kspace.shape != (3, len(adc) * nx) or not np.isfinite(kspace).all():
        raise ValueError("B0 sequence has invalid ADC trajectory dimensions.")
    times = np.asarray(times).reshape(-1, nx)
    rf_index = np.searchsorted(centers, times[:, 0], side="right") - 1
    if (np.any(rf_index < 0)
            or not np.array_equal(rf_index, np.searchsorted(centers, times[:, -1], side="right") - 1)
            or any(times[i, 0] < rf[index]["end"] for i, index in enumerate(rf_index))):
        raise ValueError("B0 ADC windows must follow one excitation and finish before the next RF.")
    counts = np.bincount(rf_index, minlength=len(rf))
    active = np.flatnonzero(counts)
    ne = int(counts[active[0]])
    ny = len(active)
    if (ne < 2 or ny < 2 or ny % 2 or np.any(counts[active[0]:] != ne)
            or not np.array_equal(rf_index, np.repeat(active, ne))):
        raise ValueError("B0 requires a complete even phase matrix, equal multi-echo trains and leading dummies only.")
    trajectory = kspace.reshape(3, ny, ne, nx)
    times = times.reshape(ny, ne, nx)
    if np.max(np.abs(trajectory[2])) > 1e-3:
        raise ValueError("B0 supports a single 2D slice with zero slice k-space at the ADC.")
    polarity = np.sign(np.mean(np.diff(trajectory[0], axis=-1), axis=-1)).astype(int)
    if np.any(polarity != polarity[0, 0]) or polarity[0, 0] == 0:
        raise ValueError("B0 requires monopolar echoes; bipolar echo phase corrections are not modeled.")
    if polarity[0, 0] < 0:
        trajectory = trajectory[..., ::-1]
        times = times[..., ::-1]
    order = np.argsort(trajectory[1, :, 0, 0])
    trajectory, times = trajectory[:, order], times[order]
    # Fit the full span: adjacent differences amplify decimal quantization of
    # phase-gradient amplitudes in text .seq files.
    dx = float(np.polyfit(np.arange(nx), trajectory[0].mean(axis=(0, 1)), 1)[0])
    dy = float(np.polyfit(np.arange(ny), trajectory[1, :, 0, 0], 1)[0])
    offset = float(np.median(trajectory[0, :, :, nx // 2]))
    grid_offset = min((0.0, dx / 2), key=lambda value: abs(offset - value))
    expected_x = (np.arange(nx) - nx // 2) * dx + grid_offset
    expected_y = (np.arange(ny) - ny // 2) * dy
    if (dx <= 0 or dy <= 0
            or not np.allclose(trajectory[0], expected_x[None, None, :], rtol=0, atol=.002 * dx)
            or not np.allclose(trajectory[1], expected_y[:, None, None], rtol=0, atol=.002 * dy)):
        raise ValueError("B0 requires full uniform Cartesian sampling, without partial Fourier or ramp sampling.")
    fov = np.asarray([1 / dx, 1 / dy])
    echo_times = np.asarray([[np.interp(0, trajectory[0, row, echo], times[row, echo])
                              - centers[active[order[row]]] for echo in range(ne)] for row in range(ny)])
    te = np.median(echo_times, axis=0)
    if not np.allclose(echo_times, te[None, :], rtol=0, atol=1e-7):
        raise ValueError("B0 echo times must be constant across all phase lines.")
    _echo_spacing(te)
    tr = float(np.median(np.diff(centers)))
    if not np.allclose(np.diff(centers), tr, rtol=0, atol=1e-7):
        raise ValueError("B0 sequence requires constant TR, including dummy excitations.")
    phases = np.asarray([rf[index]["phase"] - event["phase"] for index, event in zip(rf_index, adc)])
    if np.max(np.abs(np.angle(np.exp(1j * (phases - phases[0]))))) > 1e-6:
        raise ValueError("B0 needs matching RF/ADC phase offsets across echoes and lines.")
    for name, measured, tolerance in (("TE", te, 1e-7), ("TR", [tr], 1e-7),
                                       ("Matrix", [nx, ny], 0), ("FOV", fov, 1e-6)):
        if name in sequence.definitions:
            declared = np.asarray(sequence.definitions[name], float).reshape(-1)
            if name == "FOV":
                declared = declared[:2]
            if declared.size != len(measured) or not np.allclose(declared, measured, rtol=0, atol=tolerance):
                raise ValueError(f"B0 Pulseq {name} disagrees with its actual ADC trajectory/timing.")
    return {"image_shape": [ny, nx], "matrix_size": [nx, ny], "echo_count": ne,
            "TE_s": te.tolist(), "TR_s": tr, "dwell_s": dwell, "FOV_m": fov.tolist(),
            "polarity": int(polarity[0, 0]), "acquired_phase_rows": np.argsort(order).tolist(),
            "dummy_rf_count": int(active[0]), "sequence_duration_s": duration,
            "readout_grid_offset_mInv": grid_offset, "trajectory_tolerance_samples": .002,
            "sequence_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _echo_spacing(te):
    te = np.asarray(te, float)
    if (te.ndim != 1 or te.size < 2 or not np.isfinite(te).all() or np.any(te <= 0)
            or np.any(np.diff(te) <= 0)
            or not np.allclose(np.diff(te), np.diff(te)[0], rtol=0, atol=1e-7)):
        raise ValueError("B0 requires at least two positive, increasing, equally spaced echo times.")
    return float(np.mean(np.diff(te)))


def b0_from_coil_images(coils, echo_times_s, *, phase_sign=-1, magnitude_fraction=.05,
                        max_phase_residual_rad=.5, min_component_pixels=10,
                        frequency_offset_hz=0, return_intermediates=False) -> dict:
    """Estimate B0 from complex images shaped (echo, receiver, phase, readout).

    Each spatially connected component is unwrapped independently and anchored
    to the principal phase branch at its strongest voxel. This is an explicit
    frequency-branch assumption, not a measurement of the absolute branch.
    ``frequency_offset_hz`` is an independently known correction, never fitted.
    Set ``return_intermediates=True`` to retain echo products and masks for
    notebook inspection; the estimated field and default return are unchanged.
    """
    from skimage.restoration import unwrap_phase

    coils = np.asarray(coils)
    dt = _echo_spacing(echo_times_s)
    if (coils.ndim != 4 or coils.shape[0] != len(echo_times_s)
            or min(coils.shape[1:]) < 1 or not np.isfinite(coils).all()):
        raise ValueError("B0 coil images must be finite (echo, receiver, phase, readout) arrays.")
    if (phase_sign not in (-1, 1) or not 0 < magnitude_fraction < 1
            or not 0 < max_phase_residual_rad <= np.pi
            or isinstance(min_component_pixels, bool) or int(min_component_pixels) != min_component_pixels
            or min_component_pixels < 1 or not np.isfinite(frequency_offset_hz)):
        raise ValueError("Invalid B0 reconstruction thresholds, phase sign or frequency offset.")
    magnitude = np.sqrt(np.sum(np.abs(coils) ** 2, axis=1))
    cross = np.sum(coils[1:] * coils[:-1].conj(), axis=1)
    increment = np.angle(np.sum(cross, axis=0))
    residual = np.max(np.abs(np.angle(cross * cross[0].conj()[None])), axis=0)
    strength = np.min(magnitude, axis=0)
    magnitude_mask = strength > magnitude_fraction * strength.max()
    residual_mask = residual <= max_phase_residual_rad
    quality = magnitude_mask & residual_mask & (np.min(np.abs(cross), axis=0) > 0)
    initial_quality = quality.copy() if return_intermediates else None
    labels, _ = ndimage.label(quality)
    sizes = np.bincount(labels.ravel())
    quality &= sizes[labels] >= min_component_pixels
    labels, count = ndimage.label(quality)
    if not count:
        raise ValueError("No reliable B0 component remains; inspect echo magnitudes and phase residuals.")
    field = np.full(strength.shape, np.nan)
    anchor_values = []
    unwrap_options = {"rng": 0} if "rng" in inspect.signature(unwrap_phase).parameters else {"seed": 0}
    for label, box in enumerate(ndimage.find_objects(labels), start=1):
        component = labels[box] == label
        phase = np.ma.array(increment[box], mask=~component)
        unwrapped = np.ma.getdata(unwrap_phase(phase, **unwrap_options))
        anchor = np.unravel_index(np.argmax(np.where(component, strength[box], -np.inf)), component.shape)
        # Remove only the arbitrary 2π constant introduced by spatial unwrapping.
        unwrapped -= 2 * np.pi * np.round((unwrapped[anchor] - increment[box][anchor]) / (2 * np.pi))
        values = phase_sign * unwrapped / (2 * np.pi * dt) + frequency_offset_hz
        field[box][component] = values[component]
        anchor_values.append(float(values[anchor]))
    result = {"b0Hz": field, "mask": quality, "magnitude": magnitude[0], "units": "Hz",
            "phase_residual_rad": residual,
            "metadata": {"phase_sign": phase_sign, "echo_spacing_s": dt,
                         "frequency_offset_hz": float(frequency_offset_hz),
                         "principal_frequency_band_hz": [-.5 / dt, .5 / dt],
                         "frequency_branch": "Each component's strongest voxel is in the principal band before explicit offset; absolute branch is not measured.",
                         "component_anchor_hz": anchor_values, "component_count": count,
                         "reliable_voxels": int(quality.sum()), "magnitude_fraction": magnitude_fraction,
                         "max_phase_residual_rad": max_phase_residual_rad,
                         "min_component_pixels": min_component_pixels,
                         "method": "RSS mask; sum of receive-coil adjacent-echo conjugate products; masked spatial phase unwrap; no extrapolation",
                         "signal_convention": "exp(-2*pi*i*b0Hz*TE) when phase_sign=-1"}}
    if return_intermediates:
        result["intermediates"] = {
            "echo_axes": ["echo", "phase", "readout"],
            "echo_pair_axes": ["adjacent echo pair", "phase", "readout"],
            "echo_magnitude": magnitude,
            "echo_cross_products": cross,
            "wrapped_phase_increment_rad": increment,
            "magnitude_mask": magnitude_mask,
            "phase_residual_mask": residual_mask,
            "quality_before_component_filter": initial_quality,
            "component_labels": labels,
        }
    return result


def read_b0_coil_images(path: Path, sequence_path: Path | None = None) -> dict:
    """Read and reconstruct each B0 echo before estimating its field.

    ``kspace`` and ``coil_images`` have dimensions
    ``(echo, receiver, phase, readout)``. This first stage allows the notebook
    to display the measured echoes even when no reliable B0 mask remains.
    """
    import twixtools

    path = Path(path)
    with contextlib.redirect_stdout(io.StringIO()):
        scans = twixtools.read_twix(str(path), verbose=False)
    scan = scans[-1]
    lines = [block for block in scan["mdb"] if block.is_image_scan()]
    if not lines or np.asarray(lines[0].data).ndim != 2:
        raise ValueError("B0 Twix must contain a full Cartesian multi-echo image measurement.")
    header = scan["hdr"]["MeasYaps"]
    channels, nx = lines[0].data.shape
    ny, ne = len({block.cLin for block in lines}), len({block.cEco for block in lines})
    if (nx < 2 or ny < 2 or nx % 2 or ny % 2 or ne < 2 or len(lines) != ny * ne
            or {(block.cLin, block.cEco) for block in lines} != {(y, e) for y in range(ny) for e in range(ne)}
            or any(block.data.shape != (channels, nx) for block in lines)):
        raise ValueError("B0 Twix needs unique complete even Cartesian rows at each echo and constant receiver/sample counts.")
    for counter in ("cAve", "cSlc", "cPar", "cPhs", "cRep", "cSet"):
        if len({getattr(block, counter) for block in lines}) != 1:
            raise ValueError("B0 Twix must contain one slice, average and repetition.")
    grid = header.get("sKSpace", {})
    if (int(grid.get("lBaseResolution", nx)) != nx or int(grid.get("lPhaseEncodingLines", ny)) != ny
            or any(block.mdh.CenterCol != nx // 2 or block.mdh.CenterLin != ny // 2 for block in lines)):
        raise ValueError("B0 Twix requires a full centered grid without implicit oversampling removal.")
    reflect = {block.is_flag_set("REFLECT") for block in lines}
    if len(reflect) != 1:
        raise ValueError("B0 Twix requires monopolar echoes; bipolar phase correction is unsupported.")
    polarity = -1 if reflect == {True} else 1
    slices = header["sSliceArray"]["asSlice"]
    if len(slices) != 1:
        raise ValueError("B0 Twix supports exactly one slice.")
    def slice_position(geometry):
        return np.asarray([geometry.SlicePos.Sag, geometry.SlicePos.Cor, geometry.SlicePos.Tra], float)

    first_geometry = lines[0].mdh.SliceData
    position = slice_position(first_geometry)
    quaternion = np.asarray(list(first_geometry.Quaternion), float)
    if (not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-6)):
        raise ValueError("B0 Twix has invalid slice geometry.")
    quaternion /= np.linalg.norm(quaternion)
    expected_quaternion = np.asarray([1, 0, 0, 1], float) / np.sqrt(2)
    if not np.isclose(abs(np.dot(quaternion, expected_quaternion)), 1, atol=1e-8):
        raise ValueError("B0 Twix supports the existing axial readout +patient y / phase -patient x frame only; "
                         "export a registered B0 NPZ for another orientation.")
    for block in lines:
        geometry = block.mdh.SliceData
        if (not np.allclose(slice_position(geometry), position, rtol=0, atol=1e-3)
                or not np.isclose(abs(np.dot(list(geometry.Quaternion), quaternion)), 1, atol=1e-6)):
            raise ValueError("B0 slice position/orientation changes within the measurement.")
    te = np.asarray(header["alTE"][:ne], float) * 1e-6
    _echo_spacing(te)
    metadata = {"source_format": "Siemens Twix", "image_shape": [ny, nx], "receiver_count": channels,
                "echo_count": ne, "TE_s": te.tolist(), "TR_s": float(header["alTR"][0]) * 1e-6,
                "dwell_s": float(header["sRXSPEC"]["alDwellTime"][0]) * 1e-9,
                "FOV_m": [float(slices[0]["dReadoutFOV"]) * 1e-3, float(slices[0]["dPhaseFOV"]) * 1e-3],
                "slice_position_mm": position.tolist(), "slice_quaternion_wxyz": quaternion.tolist(),
                "slice_thickness_mm": float(slices[0]["dThickness"]), "polarity": polarity,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if sequence_path is not None:
        sequence = read_b0_sequence(sequence_path)
        for key, tolerance in (("image_shape", 0), ("echo_count", 0), ("polarity", 0),
                               ("TE_s", 1e-6), ("TR_s", 1e-6), ("dwell_s", 1e-10), ("FOV_m", 1e-6)):
            if not np.allclose(metadata[key], sequence[key], rtol=0, atol=tolerance):
                raise ValueError(f"B0 Twix {key} disagrees with the supplied B0 sequence.")
        expected_order = [(row, echo) for row in sequence["acquired_phase_rows"] for echo in range(ne)]
        if [(block.cLin, block.cEco) for block in lines] != expected_order:
            raise ValueError("B0 Twix echo/phase-line acquisition order disagrees with its sequence.")
        te = np.asarray(sequence["TE_s"])
        metadata["sequence_info"] = sequence
    kspace = np.empty((ne, channels, ny, nx), complex)
    for block in lines:
        data = np.asarray(block.data)
        kspace[block.cEco, :, block.cLin] = data[:, ::-1] if polarity < 0 else data
    axes = (2, 3)
    coils = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    fov = metadata["FOV_m"]
    x = (np.arange(nx) - nx // 2) * fov[0] / nx + position[1] * 1e-3
    y = (np.arange(ny) - ny // 2) * fov[1] / ny - position[0] * 1e-3
    X, Y = np.meshgrid(x, y)
    return {"kspace": kspace, "coil_images": coils, "X_m": X, "Y_m": Y,
            "array_axes": ["echo", "receiver", "phase", "readout"],
            "metadata": metadata}


def read_b0_twix(path: Path, sequence_path: Path | None = None,
                 *, return_intermediates=False, **options) -> dict:
    """Reconstruct B0 while retaining the existing folder-pipeline interface."""
    echoes = read_b0_coil_images(path, sequence_path)
    result = b0_from_coil_images(echoes["coil_images"], echoes["metadata"]["TE_s"],
                                 return_intermediates=return_intermediates, **options)
    result["X_m"], result["Y_m"] = echoes["X_m"], echoes["Y_m"]
    result["metadata"].update(echoes["metadata"])
    if return_intermediates:
        result["intermediates"].update(kspace=echoes["kspace"],
                                        coil_images=echoes["coil_images"],
                                        coil_axes=echoes["array_axes"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("twix", type=Path)
    parser.add_argument("--sequence", type=Path, required=True, help="Matching multi-echo B0 .seq file")
    parser.add_argument("--output", type=Path, required=True, help="Derived .npz map; originals are preserved")
    parser.add_argument("--magnitude-fraction", type=float, default=.05)
    parser.add_argument("--max-phase-residual-rad", type=float, default=.5)
    parser.add_argument("--min-component-pixels", type=int, default=10)
    parser.add_argument("--frequency-offset-hz", type=float, default=0)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("The derived NPZ/JSON already exists; choose another output name.")
    result = read_b0_twix(args.twix, args.sequence, magnitude_fraction=args.magnitude_fraction,
                          max_phase_residual_rad=args.max_phase_residual_rad,
                          min_component_pixels=args.min_component_pixels,
                          frequency_offset_hz=args.frequency_offset_hz)
    metadata = result.pop("metadata")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"B0 map: {args.output} ({int(result['mask'].sum())} reliable voxels)")
    print("Inspect magnitude, phase_residual_rad and mask; confirm scanner sign and absolute frequency branch.")


if __name__ == "__main__":
    main()
