"""Readable, serializable snapshot of the first uncalibrated MRzero baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from twin_geometry import compare_magnitude


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach") or isinstance(value, np.ndarray):
        return _json_safe(_array(value).tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return "+inf" if value > 0 else "-inf" if value < 0 else "NaN"
    return value


def _statistics(value):
    """Retain infinite relaxation times explicitly instead of losing them to null."""
    values = _array(value).reshape(-1)
    if np.iscomplexobj(values):
        values = np.abs(values)
    finite = values[np.isfinite(values)]
    result = {"count": int(values.size), "finite_count": int(finite.size),
              "positive_infinity_count": int(np.isposinf(values).sum()),
              "negative_infinity_count": int(np.isneginf(values).sum()),
              "NaN_count": int(np.isnan(values).sum())}
    for name, operation in (("min", np.min), ("median", np.median),
                            ("max", np.max), ("mean", np.mean), ("std", np.std)):
        result[name] = float(operation(finite)) if finite.size else None
    return result


def _range_text(stats):
    if stats["finite_count"]:
        text = f"{stats['min']:.6g} / {stats['median']:.6g} / {stats['max']:.6g}"
        if stats["positive_infinity_count"]:
            text += f"; +inf: {stats['positive_infinity_count']}"
        return text
    if stats["positive_infinity_count"] == stats["count"]:
        return "+inf"
    return "unavailable"


def _text(value):
    if value is None:
        return "unavailable"
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(_text(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def _phantom_snapshot(model):
    # Generation 09 retains the phantom actually passed to its first native
    # simulation. Its inherited point-phantom builder must not be used instead.
    phantom = getattr(model, "first_native_phantom", None)
    if phantom is not None:
        return phantom
    if hasattr(model, "first_native_phantom"):
        raise ValueError("Run the zero-current B0-enabled baseline before creating its MRzero report.")
    # Generation 08 releases its small phantoms after caching MRzero signals.
    # Rebuild the same inputs without running MRzero or changing the cache.
    cache = dict(getattr(model, "_native_phantoms", {}))
    try:
        return model._phantom(True)
    finally:
        if hasattr(model, "_native_phantoms"):
            model._native_phantoms.clear()
            model._native_phantoms.update(cache)


def first_simulation_report(model, prepared, inputs, measured_image,
                            baseline_image, baseline_kdata, generation):
    """Return ``(report, rows)`` before any EPI nuisance calibration.

    ``rows`` has three columns: parameter, simulated value, and scan value.
    ``report`` is directly usable with ``json.dumps(..., allow_nan=False)``.
    The scan column comes from acquisition metadata; unavailable quantitative
    tissue maps are stated explicitly. Image gain is a reporting metric only.
    """
    runtime = dict(model.runtime_info)
    native_run = dict(getattr(model, "first_native_simulation", {}) or {})
    phantom = _phantom_snapshot(model)
    acquisition = inputs["acquisition"]
    scan = inputs.get("scan_metadata", {})
    preparation = prepared.get("report", {})
    object_metadata = inputs.get("b0_raw", {}).get("metadata", {})

    parameters = {name: _statistics(getattr(phantom, name))
                  for name in ("PD", "B0", "T1", "T2", "T2dash", "D", "B1")
                  if hasattr(phantom, name)}
    if hasattr(model, "_rate"):
        parameters["R2_sInv"] = _statistics(model._rate)
    if hasattr(model, "_amplitude"):
        parameters["amplitude_at_reference"] = _statistics(model._amplitude)

    time = _array(model.time_s)
    rf_time = _array(model.rf_time_s)
    elapsed = time - rf_time
    voxel_size = native_run.get("voxel_size_m", runtime.get("native_voxel_size_m"))
    if voxel_size is None:
        voxel_size = runtime.get("subspins", {}).get("native_voxel_size_m")
    # SimData stores the dephasing kernel rather than the original voxel size.
    # Generation 08's explicit point phantom always uses zero native size.
    if voxel_size is None and not native_run:
        voxel_size = 0.0
    graph_options = runtime.get("native_graph_options", {
        "max_state_count": 16, "min_state_mag": 0.0,
        "min_emitted_signal": 0.0, "min_latent_signal": 0.0})

    weights = _array(prepared.get("weights", []))
    measured_B0 = prepared.get("native_measured_B0_Hz")
    used_B0 = prepared.get("native_B0_Hz")
    native_mask = prepared.get("parent_valid_mask")

    def map_statistics(value):
        if value is None:
            return None
        array = _array(value)
        if native_mask is not None and _array(native_mask).shape == array.shape:
            array = array[_array(native_mask).astype(bool)]
        return _statistics(array)

    image_comparison = compare_magnitude(measured_image, baseline_image)
    kdata = _array(baseline_kdata)
    report = {
        "generation": str(generation),
        "stage": "first zero-current B0-enabled baseline, before EPI nuisance calibration",
        "native_parameters": parameters,
        "native_parameter_statistics_order": "finite minimum / median / maximum over active simulated voxels; infinite counts reported separately",
        "amplitude_interpretation": runtime.get("object_amplitude"),
        "PD_interpretation": "effective amplitude extrapolated to RF; not a measured proton-density map",
        "amplitude_reference_time_s": getattr(model, "amplitude_reference_time_s", None),
        "native_RF": {
            "flip_angle_deg": runtime.get("native_flip_angle_deg", 90.0),
            "phase_rad": runtime.get("native_RF_phase_rad", float(np.pi / 2)),
            "selective": False,
            "model": runtime.get("native_RF_model", "independent ideal 90-degree excitation per RF history"),
            "RF_history_count": int(np.unique(rf_time).size),
            "steady_state_simulated": False,
            "slice_profile_simulated": False,
        },
        "timing": {
            "ADC_time_since_RF_min_s": float(elapsed.min()),
            "ADC_time_since_RF_max_s": float(elapsed.max()),
            "model_TE_s": float(model.TE_s),
            "sequence_TE_s": acquisition.get("TE_s"),
            "sequence_TR_s": acquisition.get("TR_s"),
            "native_TR_policy": "independent RF histories; sequence TR is hardware timing, not a simulated GRE steady state",
            "sequence_dwell_s": acquisition.get("dwell_s"),
            "sequence_echo_spacing_s": acquisition.get("echo_spacing_s"),
        },
        "geometry": {
            "native_object_matrix_yx": list(prepared["native_X_m"].shape),
            "retained_parent_voxels": int(np.count_nonzero(prepared["parent_valid_mask"])),
            "active_MRzero_voxels": int(np.count_nonzero(_array(model.support))),
            "z_samples_per_parent": int(weights.shape[1]) if weights.ndim == 2 else None,
            "native_voxel_size_m": voxel_size,
            "native_voxel_shape": native_run.get("voxel_shape", "box"),
            "object_FOV_m": object_metadata.get("FOV_m"),
            "sequence_matrix_xy": acquisition.get("matrix_size"),
            "sequence_FOV_m": acquisition.get("FOV_m"),
            "slice_thickness_mm": preparation.get("slice_thickness_mm"),
        },
        "dephasing": {
            "native_policy": runtime.get("native_voxel_dephasing", runtime.get("voxel_dephasing")),
            "external_voxel_dephasing": runtime.get("external_voxel_dephasing", not bool(native_run)),
            "external_xyz": (None if runtime.get("external_voxel_dephasing") is False else
                runtime.get("voxel_xyz", runtime.get("inplane_voxel"))),
            "unresolved_coherence": runtime.get("multi_echo_coherence"),
            "B0_map_mode": preparation.get("B0_mode"),
            "B0_regularization": preparation.get("B0_regularization"),
            "B0_gradient_policy": runtime.get("B0_gradient_policy", runtime.get("inplane_voxel", {}).get(
                "through_plane_B0", preparation.get("through_plane_B0",
                preparation.get("calibration_pose", {}).get("through_plane_B0")))),
            "measured_B0_Hz": map_statistics(measured_B0),
            "used_B0_Hz": map_statistics(used_B0),
        },
        "execution": {
            "device": str(model.device), "model_dtype": str(model.dtype),
            "native_precision": runtime.get("native_precision"),
            "MRzero_version": runtime.get("mrzero_version"),
            "graph_options": graph_options,
            "native_blocks_simulated": runtime.get("native_evaluations", runtime.get("native_blocks_simulated")),
            "native_simulation_seconds": runtime.get("native_evaluation_seconds", runtime.get("native_cache_seconds")),
            "first_native_simulation": native_run,
        },
        "actual_scan": {key: scan.get(key) for key in (
            "TE_s", "TR_s", "header_flip_deg", "dwell_s", "image_shape", "FOV_m",
            "slice_thickness_mm", "receiver_count", "navigator_count")},
        "measured_tissue_maps": "B0 and multi-echo effective amplitude/relaxation; no measured T1, B1 or diffusion map",
        "baseline_kdata": {
            "shape": list(kdata.shape), "count": int(kdata.size),
            "finite_count": int(np.isfinite(kdata).sum()),
            "all_finite": bool(np.isfinite(kdata).all()),
            "magnitude": _statistics(np.abs(kdata)),
        },
        "baseline_vs_scan_magnitude": image_comparison,
        "comparison_gain_policy": "positive least-squares gain fitted for reporting only; no spatial registration",
    }

    unavailable = "not measured"
    measured_b0_stats = report["dephasing"]["measured_B0_Hz"]
    measured_shape = scan.get("image_shape")
    voxel_policy = runtime.get("native_voxel_dephasing", runtime.get("voxel_dephasing"))
    if voxel_policy is None:
        voxel_policy = runtime.get("inplane_voxel", {}).get("model")
    rows = [
        ("Stage", report["stage"], "uncorrected EPI"),
        ("RF flip [deg]", report["native_RF"]["flip_angle_deg"], _text(scan.get("header_flip_deg"))),
        ("TE [ms]", _text(1000 * model.TE_s), _text(None if scan.get("TE_s") is None else 1000 * scan["TE_s"])),
        ("ADC since RF, min/max [ms]", _text([1000 * elapsed.min(), 1000 * elapsed.max()]), "Pulseq sample centers"),
        ("Sequence TR [ms]", _text(1000 * acquisition["TR_s"]), _text(None if scan.get("TR_s") is None else 1000 * scan["TR_s"])),
        ("RF history / steady state", f"{report['native_RF']['RF_history_count']} independent / not simulated", "measured scan"),
        ("Effective PD, min/median/max", _range_text(parameters["PD"]), "echo-anchored effective amplitude"),
        ("Amplitude reference echo [ms]", _text(None if report["amplitude_reference_time_s"] is None else 1000 * report["amplitude_reference_time_s"]), "multi-echo fit"),
        ("B0 [Hz], min/median/max", _range_text(parameters["B0"]),
         _range_text(measured_b0_stats) if measured_b0_stats is not None else unavailable),
    ]
    for name, units in (("T1", "s"), ("T2", "s"), ("T2dash", "s"), ("D", "native value"), ("B1", "relative")):
        if name in parameters:
            actual = "effective multi-echo decay" if name == "T2" else unavailable
            rows.append((f"{name} [{units}], min/median/max", _range_text(parameters[name]), actual))
    rows.extend([
        ("Native object matrix [y,x]", _text(report["geometry"]["native_object_matrix_yx"]), "B0/multi-echo grid"),
        ("Acquisition matrix [x,y]", _text(acquisition.get("matrix_size")),
         _text(list(reversed(measured_shape)) if measured_shape is not None else None)),
        ("FOV [m]", _text(acquisition.get("FOV_m")), _text(scan.get("FOV_m"))),
        ("Slice thickness [mm]", _text(preparation.get("slice_thickness_mm")), _text(scan.get("slice_thickness_mm"))),
        ("MRzero voxels / z samples per parent", f"{report['geometry']['active_MRzero_voxels']} / {report['geometry']['z_samples_per_parent']}", "one acquired slice"),
        ("Native voxel size [m]", _text(voxel_size), "native object pixel × slice thickness"),
        ("Intravoxel model", _text(voxel_policy), "uniform voxel assumption"),
        ("Normal B0 derivative", _text(report["dephasing"]["B0_gradient_policy"]), "2D B0 map; normal derivative unmeasured"),
        ("Device / native precision", f"{model.device} / {runtime.get('native_precision', 'unavailable')}", "MRzero " + str(runtime.get("mrzero_version", "unavailable"))),
        ("PDG thresholds", _text(graph_options), "simulation settings"),
        ("Finite k-space samples", f"{np.isfinite(kdata).sum()} / {kdata.size}", "first baseline"),
        ("Magnitude gain / NRMSE [%]", f"{image_comparison['gain']:.6g} / {image_comparison['NRMSE_percent']:.4g}", "simulated baseline → measured EPI"),
    ])
    return _json_safe(report), _json_safe(rows)


__all__ = ["first_simulation_report"]
