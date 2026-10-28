"""Verified, persistent ACDC commands for manual experimental feedback.

Only data files are produced. Nothing in this module loads a controller or
changes scanner currents. Historical scan/command association must separately
be confirmed by the caller: filenames cannot establish that association.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "Experiment"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from export_acdc_waveform import (export_waveform, export_zero_time_copy, quantize_currents,
                                  validate_waveform_settings)
from acdc_hardware import HardwareConstraints
from acdc_gpu import ACDCModel
from shim_export_timing import sequence_trigger_timing, waveform_edges_for_origin

_VERSION = 1
_ARRAY_KEYS = ("line_start_s", "line_end_s", "max_current_A", "tau_s",
               "field_basis_Hz_per_A", "csv_edges_s")
_DAC_SCALE = 32767.0 / 5.0


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _json(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(value, name, shape=None):
    if np.iscomplexobj(value) or np.asarray(value).dtype.kind == "b":
        raise ValueError(f"{name} must contain finite real numbers.")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} has invalid shape or nonfinite values; expected {shape}.")
    return result.copy()


def _scalar(value, name, *, positive=False):
    array = _array(value, name, ())
    result = float(array)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _same(actual, expected, name, *, atol=1e-12):
    a, b = _array(actual, name), _array(expected, name)
    if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=atol):
        raise ValueError(f"Historical {name} does not match the confirmed command context.")


def _context_digest(context):
    digest = hashlib.sha256(_json(context["metadata"]).encode())
    for name in _ARRAY_KEYS:
        array = np.ascontiguousarray(context[name], dtype="<f8")
        digest.update(name.encode())
        digest.update(_json(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_context(context):
    if not isinstance(context, dict) or context.get("version") != _VERSION:
        raise ValueError("Unsupported or missing closed-loop command context version.")
    if context.get("fingerprint") != _context_digest(context):
        raise ValueError("Command context has changed or failed its integrity fingerprint.")
    return context


def build_command_context(model, inputs, prepared, *, waveform_parameters,
                          csv_time_origin, trigger_channel,
                          total_current_limit_A=None, field_limit_Hz):
    """Freeze the exact model constraint basis, mappings and verified sequence.

    The resulting dictionary contains JSON metadata and NumPy arrays. Save it
    with :func:`save_command_context` to use later rounds without a live model.
    Optional electrical limits remain disabled exactly as in the source model;
    no absent voltage, resistance or thermal parameter is invented.
    """
    nc, states = int(model.nc), int(model.nstates)
    settings = validate_waveform_settings(dict(waveform_parameters), nc)
    if "min_dac_interval_s" not in settings:
        raise ValueError("Explicit min_dac_interval_s is required for closed-loop export.")
    total = _scalar(settings["total_current_limit_A"] if total_current_limit_A is None
                    else total_current_limit_A, "total_current_limit_A", positive=True)
    if total != settings["total_current_limit_A"]:
        raise ValueError("CSV and hardware total_current_limit_A disagree.")
    field = _scalar(field_limit_Hz, "field_limit_Hz", positive=True)
    response = _jsonable(model.current_response)
    if response.get("enabled") is not True or "tau_s" not in response:
        raise ValueError("An explicit enabled first-order current response is required.")
    config = _jsonable(model.hardware.config)
    if config.get("cycle_duration_s") is None:
        raise ValueError("Complete acquisition cycle_duration_s is required in hardware context.")
    minimum = _scalar(config.get("min_update_interval_s", model.hardware.min_update_interval_s),
                      "min_update_interval_s", positive=True)
    if minimum != settings["min_dac_interval_s"]:
        raise ValueError("CSV and physical hardware minimum update intervals disagree.")
    trigger = sequence_trigger_timing(inputs["paths"]["scan_sequence"], inputs["timing"],
                                      channel=trigger_channel)
    starts = _array(model.line_start_s, "line_start_s", (states,))
    ends = _array(model.line_end_s, "line_end_s", (states,))
    _same(starts, inputs["timing"]["state_start_s"], "sequence command starts")
    if "state_end_s" in inputs["timing"]:
        _same(ends, inputs["timing"]["state_end_s"], "sequence command ends")
    edges, timing = waveform_edges_for_origin(starts, ends, origin=csv_time_origin,
        trigger_rf_s=trigger["trigger_rf_s"], min_update_interval_s=minimum)
    limits = _array(model.numpy(model.max_current_A), "max_current_A", (nc,))
    _same(settings["max_current_A"], limits, "per-channel current limits", atol=0)
    basis = _array(model.numpy(model.B_quality), "field_basis_Hz_per_A")
    if basis.ndim != 2 or basis.shape[1] != nc or not basis.shape[0]:
        raise ValueError("Complete finite field-constraint basis is required.")
    tau = _array(model.hardware.tau_s, "tau_s", (nc,))
    if np.any(tau <= 0):
        raise ValueError("Current-response time constants must be positive.")
    sources = _jsonable(inputs.get("source_sha256", {}))
    if sources.get("scan_sequence") != trigger["sequence_sha256"]:
        raise ValueError("Missing or inconsistent confirmed EPI sequence source SHA256.")
    calibration_hash = prepared.get("report", {}).get("calibration_source_sha256")
    if calibration_hash is None:
        calibration_hash = prepared.get("calibration", {}).get("metadata", {}).get("source_sha256")
    if not isinstance(calibration_hash, str) or len(calibration_hash) != 64:
        raise ValueError("Confirmed calibration_source_sha256 is required.")
    sources["calibration"] = calibration_hash
    context = dict(version=_VERSION, line_start_s=starts, line_end_s=ends,
        max_current_A=limits, tau_s=tau, field_basis_Hz_per_A=basis, csv_edges_s=edges,
        metadata=_jsonable(dict(n_channels=nc, n_states=states, settings=settings,
            total_current_limit_A=total, field_limit_Hz=field, current_response=response,
            hardware_configuration=config, trigger=trigger, CSV_timing=timing,
            source_sha256=sources, quantization="export_acdc_waveform.quantize_currents: signed16 truncation, full scale 5 A",
            no_hardware_action_performed=True,
            assumptions=["Nominal DAC commands; actual amplifier output is not telemetry.",
                         "Physical checks reuse the source model's first-order response and complete field-constraint basis."])))
    context["fingerprint"] = _context_digest(context)
    _hardware(context)  # Validate that the complete snapshot can stand alone.
    return context


def save_command_context(context, path):
    """Write an integrity-checked, pickle-free snapshot; never overwrite it."""
    _validate_context(context)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **{key: context[key] for key in _ARRAY_KEYS},
            metadata_json=np.asarray(_json(context["metadata"])),
            fingerprint=np.asarray(context["fingerprint"]), version=np.asarray(_VERSION))
    return str(path)


def load_command_context(path):
    with np.load(path, allow_pickle=False) as archive:
        context = {key: archive[key].copy() for key in _ARRAY_KEYS}
        context.update(version=int(archive["version"]),
                       metadata=json.loads(str(archive["metadata_json"].item())),
                       fingerprint=str(archive["fingerprint"].item()))
    _validate_context(context)
    _hardware(context)
    return context


def _hardware(context):
    _validate_context(context)
    metadata = context["metadata"]
    model = SimpleNamespace(nc=metadata["n_channels"], nstates=metadata["n_states"],
        current_response={"enabled": True, "tau_s": context["tau_s"]},
        line_start_s=context["line_start_s"], line_end_s=context["line_end_s"],
        tensor=lambda value: torch.as_tensor(value, dtype=torch.float64),
        numpy=lambda value: value.detach().cpu().numpy())
    model.max_current_A = model.tensor(context["max_current_A"])
    model.B_quality = model.tensor(context["field_basis_Hz_per_A"])
    def check(value):
        return model.tensor(_array(value, "currents_A", (model.nc, model.nstates)))
    model._check_currents = check
    return HardwareConstraints(model, metadata["hardware_configuration"])


def _diagnostics(context, currents, hardware=None):
    metadata = context["metadata"]
    currents = _array(currents, "currents_A", (metadata["n_channels"], metadata["n_states"]))
    result = (hardware or _hardware(context)).diagnostics(currents,
        metadata["total_current_limit_A"], metadata["field_limit_Hz"])
    # The original exporter intentionally demands a strict total-current margin.
    result["checks"]["CSV_strict_total_margin"] = bool(np.all(
        np.abs(currents).sum(0) < metadata["settings"]["total_current_limit_A"]))
    result["all_limits_met"] = all(result["checks"].values())
    return result


def _codes_from_applied(currents):
    currents = _array(currents, "quantized_currents_A")
    codes = np.rint(currents * _DAC_SCALE).astype(np.int64)
    reconstructed = (codes.astype(float) / 32767.) * 5.
    if (np.any(np.abs(currents) > 3) or not np.allclose(currents, reconstructed, rtol=0, atol=2e-15)):
        raise ValueError("Historical quantized_currents_A are not valid nominal signed16 DAC values.")
    return codes


def _fingerprint(context, codes):
    digest = hashlib.sha256(context["fingerprint"].encode())
    digest.update(np.ascontiguousarray(codes, dtype="<i8").tobytes())
    return digest.hexdigest()


def _canonical_commands(codes):
    """Stay within a few floating ULPs of the DAC value, inside its truncation cell.

    Re-quantizing DAC ampere values can lose one code through floating-point
    rounding. A minimal outward ULP adjustment makes CSV->DAC reproducible,
    including alpha=1, without adding a half-LSB offset to every channel.
    """
    applied = (codes.astype(float) / 32767.) * 5.
    commands = applied.copy()
    for _ in range(16):
        found = np.trunc((commands / 5.) * 32767.).astype(np.int64)
        wrong = found != codes
        if not wrong.any():
            return commands, applied
        commands[wrong] = np.nextafter(commands[wrong], np.copysign(np.inf, codes[wrong]))
    raise ValueError("Cannot encode the requested DAC codes faithfully in the original CSV schema.")


def _check_historical_metadata(context, metadata):
    current = context["metadata"]
    try:
        exported = metadata["waveform_export"]
        settings = validate_waveform_settings(exported["settings"], current["n_channels"])
        timing, trigger = metadata["CSV_timing"], metadata["trigger"]
        sources, response = metadata["source_sha256"], metadata["current_response"]
        state_edges, state_count = exported["state_edges_s"], exported["states"]
        trigger_rf, origin_rf = trigger["trigger_rf_s"], timing["origin_rf_s"]
        physical_count, final_zero = exported["physical_channels"], exported["final_zero_s"]
    except (KeyError, TypeError) as error:
        raise ValueError("Historical export metadata are incomplete: require waveform_export.settings, "
                         "state_edges_s, CSV_timing, trigger, source_sha256 and current_response; "
                         "supply metadata_path or metadata explicitly.") from error
    # Older persisted contexts predate optional disabled_channels metadata.
    # Compare normalized copies without changing their integrity fingerprint.
    current_settings = validate_waveform_settings(current["settings"], current["n_channels"])
    if settings != current_settings:
        raise ValueError("Historical channel mapping, polarity, limits or DAC update interval disagree.")
    _same(state_edges, context["csv_edges_s"], "state edges")
    if state_count != current["n_states"]:
        raise ValueError("Historical number of command states differs.")
    physical = ((max(settings["channel_map"]) + 7) // 8) * 8
    if physical_count != physical:
        raise ValueError("Historical number of physical CSV outputs differs.")
    _same(final_zero, context["csv_edges_s"][-1], "final switch-off time")
    for name in ("scan_sequence", "calibration"):
        if sources.get(name) != current["source_sha256"].get(name):
            raise ValueError(f"Historical {name} SHA256 differs or is missing.")
    if (trigger.get("verified_first_imaging_trigger") is not True
            or trigger.get("channel") != current["trigger"]["channel"]
            or trigger.get("sequence_sha256") != current["trigger"]["sequence_sha256"]):
        raise ValueError("Historical trigger channel or verified sequence identity differs/missing.")
    _same(trigger_rf, current["trigger"]["trigger_rf_s"], "trigger RF time")
    if (timing.get("origin") != current["CSV_timing"]["origin"]
            or timing.get("zero_time_column_must_be_false") is not True
            or timing.get("final_zero_required") is not True):
        raise ValueError("Historical CSV origin, nonzero timing or final-zero requirement differs/missing.")
    _same(origin_rf, current["CSV_timing"]["origin_rf_s"], "CSV origin RF time")
    for name in ("first_command_time_s", "final_zero_time_s", "command_state_count",
                 "csv_rows_with_final_zero_and_preamble"):
        if name not in timing:
            raise ValueError(f"Historical CSV_timing.{name} is missing; exact timing is required.")
        _same(timing[name], current["CSV_timing"][name], f"CSV timing {name}")
    if timing.get("includes_zero_preamble") is not current["CSV_timing"]["includes_zero_preamble"]:
        raise ValueError("Historical zero preamble differs from the confirmed context.")
    if response.get("enabled") is not True or "tau_s" not in response:
        raise ValueError("Historical current response is absent or disabled.")
    tau = np.asarray(response["tau_s"], dtype=float)
    tau = np.full(current["n_channels"], tau.item()) if tau.size == 1 else tau.reshape(-1)
    _same(tau, context["tau_s"], "current-response tau", atol=1e-15)
    _csv_layout(context, exported)


def _csv_layout(context, exported=None):
    """Validate optional CSV rebasing against the frozen original state edges."""
    edges = context["csv_edges_s"]
    states = context["metadata"]["n_states"]
    if exported is None or "start_at_first_nonzero" not in exported:
        preamble = int(edges[0] > 0)
        return dict(edges=edges, states=states, preamble=preamble,
                    leading=0, first_nonzero_found=None, rebased=False)
    try:
        rebased = exported["start_at_first_nonzero"]
        leading = exported["leading_zero_states"]
        first_found = exported["first_nonzero_found"]
        csv_states = exported["csv_states"]
        csv_preamble = exported["csv_includes_zero_preamble"]
        shift = exported["time_origin_shift_s"]
        csv_edges = exported["csv_state_edges_s"]
        csv_final = exported["csv_final_zero_s"]
    except (KeyError, TypeError) as error:
        raise ValueError("Historical CSV rebasing metadata are incomplete.") from error
    if (not isinstance(rebased, bool) or not isinstance(first_found, bool)
            or not isinstance(csv_preamble, bool)
            or isinstance(leading, bool) or not isinstance(leading, (int, np.integer))
            or isinstance(csv_states, bool) or not isinstance(csv_states, (int, np.integer))
            or not 0 <= leading < states):
        raise ValueError("Historical CSV rebasing flags/state counts are invalid.")
    if (not rebased or not first_found) and leading != 0:
        raise ValueError("Historical CSV cannot discard states without a first nonzero command.")
    expected_shift = float(edges[leading]) if rebased else 0.
    expected_edges = edges[leading:] - expected_shift
    expected_preamble = not rebased and edges[0] > 0
    if csv_states != states - leading or csv_preamble != expected_preamble:
        raise ValueError("Historical CSV rebasing row/state layout differs from its original context.")
    _same(exported["state_edges_s"], edges, "original CSV state edges")
    _same(exported["states"], states, "original CSV state count")
    _same(exported["final_zero_s"], edges[-1], "original CSV final-zero time")
    _same(shift, expected_shift, "CSV time-origin shift")
    _same(csv_edges, expected_edges, "rebased CSV state edges")
    _same(csv_final, expected_edges[-1], "rebased CSV final-zero time")
    return dict(edges=expected_edges, states=csv_states, preamble=int(expected_preamble),
                leading=leading, first_nonzero_found=first_found, rebased=rebased)


def _read_csv(context, path, *, export_metadata=None):
    settings = context["metadata"]["settings"]
    mapping = np.asarray(settings["channel_map"], int) - 1
    physical = ((int(mapping.max()) + 8) // 8) * 8
    path = Path(path)
    with path.open() as stream:
        first = stream.readline().strip()
    expected_header = "time_s," + ",".join(f"ch{k}_A" for k in range(physical))
    if first != "# " + expected_header:
        raise ValueError("CSV must use the original commented time_s,ch0_A,... ampere schema.")
    table = _array(np.loadtxt(path, delimiter=",", ndmin=2), "historical CSV")
    layout = _csv_layout(context, export_metadata)
    edges, preamble = layout["edges"], layout["preamble"]
    times = np.r_[0., edges] if preamble else edges
    if table.shape != (len(times), physical + 1):
        raise ValueError("CSV row/output count differs from its verified state/preamble/final-zero layout.")
    _same(table[:, 0], times, "CSV timestamps")
    if np.any(np.diff(table[:, 0]) < settings["min_dac_interval_s"]):
        raise ValueError("CSV timing violates the configured DAC update interval.")
    if np.any(table[-1, 1:] != 0) or (preamble and np.any(table[0, 1:] != 0)):
        raise ValueError("CSV must contain exact zero preamble/final switch-off rows.")
    unused = np.ones(physical, bool)
    unused[mapping] = False
    if np.any(table[:, 1:][:, unused] != 0):
        raise ValueError("CSV commands an uncalibrated/unmapped physical output.")
    commands = (table[preamble:preamble + layout["states"], 1:][:, mapping]
                * np.asarray(settings["channel_polarity"])).T.copy()
    found = bool(np.any(commands != 0))
    if (layout["first_nonzero_found"] is not None
            and found != layout["first_nonzero_found"]):
        raise ValueError("CSV nonzero commands disagree with its rebasing metadata.")
    if layout["rebased"] and found and not np.any(commands[:, 0] != 0):
        raise ValueError("Rebased CSV must begin with its first nonzero command.")
    original = np.zeros((context["metadata"]["n_channels"], context["metadata"]["n_states"]))
    original[:, layout["leading"]:] = commands
    return original


def load_historical_commands(context, *, csv_path=None, npz_path=None,
                             metadata_path=None, metadata=None):
    """Invert only a verified original export; prefer its stored DAC waveform.

    A same-stem JSON is discovered for an explicit NPZ. CSV-only imports need
    an explicit metadata sidecar/dictionary; channel identities and trigger
    origins are never guessed. Both supplied files must agree if both exist.
    The returned DAC waveform estimates applied commands, not measured currents.
    """
    _validate_context(context)
    if csv_path is None and npz_path is None:
        raise ValueError("Supply an explicit historical CSV and/or optimization NPZ.")
    if metadata is not None and metadata_path is not None:
        raise ValueError("Supply metadata or metadata_path, not both.")
    if metadata is None:
        if metadata_path is None and npz_path is not None:
            metadata_path = Path(npz_path).with_suffix(".json")
        if metadata_path is None or not Path(metadata_path).is_file():
            raise ValueError("Historical export metadata missing; supply metadata_path/metadata with mapping, trigger and timing.")
        metadata = json.loads(Path(metadata_path).read_text())
    _check_historical_metadata(context, metadata)
    nominal, stored_q = None, None
    if npz_path is not None:
        with np.load(npz_path, allow_pickle=False) as archive:
            if "currents_A" in archive:
                nominal = _array(archive["currents_A"], "currents_A")
            if "quantized_currents_A" in archive:
                stored_q = _array(archive["quantized_currents_A"], "quantized_currents_A")
        if nominal is None and stored_q is None:
            raise ValueError("Historical NPZ has neither currents_A nor quantized_currents_A.")
    if csv_path is not None:
        from_csv = _read_csv(context, csv_path, export_metadata=metadata["waveform_export"])
        if nominal is not None:
            _same(from_csv, nominal, "CSV versus NPZ nominal commands", atol=2e-14)
        nominal = from_csv
    if nominal is None:
        codes = _codes_from_applied(stored_q)
        nominal, _ = _canonical_commands(codes)
    predicted_q = quantize_currents(nominal)
    if stored_q is not None:
        _same(stored_q, predicted_q, "NPZ DAC waveform versus exported commands", atol=2e-15)
        applied = stored_q
    else:
        applied = predicted_q
    shape = (context["metadata"]["n_channels"], context["metadata"]["n_states"])
    nominal, applied = _array(nominal, "currents_A", shape), _array(applied, "quantized_currents_A", shape)
    codes = _codes_from_applied(applied)
    hardware = _hardware(context)
    before, after = _diagnostics(context, nominal, hardware), _diagnostics(context, applied, hardware)
    if not before["all_limits_met"] or not after["all_limits_met"]:
        raise ValueError("Historical commands fail current hardware/field constraints before or after DAC; "
                         "the closed loop cannot safely reuse this waveform.")
    hashes = {name: _sha_file(path) for name, path in
              (("csv", csv_path), ("npz", npz_path), ("metadata", metadata_path)) if path is not None}
    return dict(nominal_currents_A=nominal, quantized_currents_A=applied, dac_codes=codes,
        fingerprint=_fingerprint(context, codes), report=dict(source_sha256=hashes,
            stored_DAC_waveform_used=stored_q is not None, metadata_verified=True,
            scan_current_association_verified=False,
            hardware_before_DAC=before, hardware_after_DAC=after,
            interpretation="Nominal downstream DAC conversion; application to a particular scan requires separate explicit confirmation."))


def evaluate_candidate(context, old_quantized_currents_A, alpha):
    """Scale the verified applied waveform, quantize, then check every limit.

    No projection/clipping changes the one-dimensional search. Failed
    candidates are returned with valid=False and cannot be exported.
    """
    _validate_context(context)
    metadata = context["metadata"]
    old = _array(old_quantized_currents_A, "old_quantized_currents_A",
                 (metadata["n_channels"], metadata["n_states"]))
    old_codes = _codes_from_applied(old)
    alpha = _scalar(alpha, "alpha")
    if not 0 <= alpha <= 1:
        raise ValueError("Closed-loop alpha must lie in [0, 1]; no extrapolation is allowed.")
    target = alpha * old
    codes = old_codes if alpha == 1 else np.trunc((target / 5.) * 32767.).astype(np.int64)
    nominal, applied = _canonical_commands(codes)
    hardware = _hardware(context)
    target_d = _diagnostics(context, target, hardware)
    before = _diagnostics(context, nominal, hardware)
    after = _diagnostics(context, applied, hardware)
    valid = target_d["all_limits_met"] and before["all_limits_met"] and after["all_limits_met"]
    return dict(alpha=alpha, target_currents_A=target, command_currents_A=nominal,
        quantized_currents_A=applied, dac_codes=codes,
        fingerprint=_fingerprint(context, codes), context_fingerprint=context["fingerprint"],
        hardware_target=target_d, hardware_before_DAC=before, hardware_after_DAC=after,
        maximum_quantization_error_A=float(np.max(np.abs(target - applied))),
        maximum_CSV_encoding_offset_A=float(np.max(np.abs(nominal - applied))), valid=bool(valid))


def sample_command_waveform(context, quantized_currents_A, times_s=None):
    """Sample the original exact amplifier response from a persisted snapshot.

    Times are RF-relative seconds, matching ``ACDCModel.sample_waveform``.
    With no supplied grid, the original transition/settling grid is extended to
    the complete acquisition cycle. Physical outputs are still in calibrated
    coil order; channel_map/polarity describe the CSV wiring explicitly.
    """
    _validate_context(context)
    metadata = context["metadata"]
    applied = _array(quantized_currents_A, "quantized_currents_A",
                     (metadata["n_channels"], metadata["n_states"]))
    _codes_from_applied(applied)
    model = _hardware(context).model
    sampled = ACDCModel.sample_waveform(model, applied, times_s)
    if times_s is None:
        cycle = metadata["hardware_configuration"]["cycle_duration_s"]
        grid = sampled["time_s"]
        grid = np.unique(np.r_[grid[grid <= cycle], cycle])
        sampled = ACDCModel.sample_waveform(model, applied, grid)
    sampled.update(CSV_time_s=sampled["time_s"] - metadata["CSV_timing"]["origin_rf_s"],
        channel_map=list(metadata["settings"]["channel_map"]),
        channel_polarity=list(metadata["settings"]["channel_polarity"]),
        max_current_A=context["max_current_A"].copy(),
        total_current_limit_A=metadata["total_current_limit_A"],
        no_hardware_action_performed=True)
    return sampled


def export_candidate(context, candidate, output_dir):
    """Check afresh and atomically publish a new, complete export directory.

    Uses the original exporter and verifies its inverse before publication.
    CSV timing starts at the first nonzero command; removed leading zero
    states and the original timing remain recorded for exact reconstruction.
    A companion with an all-zero time column is published alongside it. The
    timestamped CSV remains the canonical timing and historical command source.
    ``output_dir`` must not already exist. A failed check leaves no published
    CSV. Data are never sent to scanner/controller hardware.
    """
    _validate_context(context)
    if candidate.get("context_fingerprint") != context["fingerprint"]:
        raise ValueError("Candidate belongs to a different command context.")
    commands = _array(candidate["command_currents_A"], "command_currents_A")
    applied = quantize_currents(commands)
    _same(applied, candidate["quantized_currents_A"], "candidate DAC waveform", atol=0)
    codes = _codes_from_applied(applied)
    if (not np.array_equal(codes, candidate["dac_codes"])
            or candidate.get("fingerprint") != _fingerprint(context, codes)):
        raise ValueError("Candidate DAC codes/fingerprint have changed.")
    hardware = _hardware(context)
    before, after = _diagnostics(context, commands, hardware), _diagnostics(context, applied, hardware)
    if not candidate.get("valid") or not before["all_limits_met"] or not after["all_limits_met"]:
        raise ValueError("Refusing CSV export: candidate fails hardware constraints before/after DAC.")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Candidate export directory already exists; existing commands are never overwritten.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".closed_loop_export_", dir=output.parent))
    try:
        exported = export_waveform(commands, context["csv_edges_s"], context["metadata"]["settings"], staging,
            include_header=True, include_zero_preamble=True, include_final_zero_row=True,
            zero_time_column=False, numbered_filename=True, start_at_first_nonzero=True)
        inverse = _read_csv(context, exported["csv_file"], export_metadata=exported)
        _same(inverse, commands, "exported CSV command round-trip", atol=0)
        _same(quantize_currents(inverse), applied, "exported CSV DAC round-trip", atol=0)
        csv_name = Path(exported["csv_file"]).name
        csv_hash = _sha_file(exported["csv_file"])
        zero_time = export_zero_time_copy(exported["csv_file"])
        zero_time_name = Path(zero_time["csv_file"]).name
        exported["csv_file"] = str(output / csv_name)
        zero_time["source_csv_file"] = exported["csv_file"]
        zero_time["csv_file"] = str(output / zero_time_name)
        exported["csv_zero_time_file"] = zero_time["csv_file"]
        exported["zero_time_export"] = zero_time
        metadata = dict(context["metadata"], waveform_export=exported,
            context_fingerprint=context["fingerprint"], command_fingerprint=candidate["fingerprint"],
            alpha=float(candidate["alpha"]), hardware_before_DAC=before, hardware_after_DAC=after,
            no_hardware_action_performed=True)
        np.savez_compressed(staging / "commands.npz", currents_A=commands,
                            quantized_currents_A=applied, dac_codes=codes)
        (staging / "commands.json").write_text(json.dumps(_jsonable(metadata), indent=2, allow_nan=False))
        hashes = dict(csv=csv_hash, csv_zero_time=zero_time["csv_sha256"],
                      npz=_sha_file(staging / "commands.npz"),
                      metadata=_sha_file(staging / "commands.json"))
        # rename() is atomic on the same filesystem. Never replace a nonempty
        # directory; root session locking serializes legitimate notebook calls.
        if output.exists():
            raise FileExistsError("Candidate export directory was created concurrently.")
        os.rename(staging, output)
        return dict(exported, npz_file=str(output / "commands.npz"),
            metadata_file=str(output / "commands.json"), source_sha256=hashes,
            fingerprint=candidate["fingerprint"], context_fingerprint=context["fingerprint"],
            hardware_before_DAC=before, hardware_after_DAC=after,
            no_hardware_action_performed=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = ["build_command_context", "save_command_context", "load_command_context",
           "load_historical_commands", "evaluate_candidate", "export_candidate",
           "sample_command_waveform"]
