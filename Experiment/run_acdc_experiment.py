"""GRE/FLASH reference + B0 + scan and Pulseq files -> MATLAB -> waveform.csv.

The run command delivers only waveform.csv; intermediate files are temporary.
The prepare and export commands are retained for inspecting older sessions.
See TUTORIEL_SHIMMING_DYNAMIQUE.md for the input contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def template():
    """Unknown installation parameters stay empty; these are not rig defaults."""
    return {
        "version": 1,
        "files": {"gre": "gre_reference.dat", "b0": "b0_map.dat", "scan": "scan.dat",
                  "sequence": "sequences/scan.seq",
                  "reference_sequence": "sequences/reference_gre.seq",
                  "b0_sequence": "sequences/b0_map.seq", "calibration": "ACDC_3T.npz"},
        "preparation": {"reference_threshold_fraction": 0.03, "b0_units": "Hz",
                        "b0_sign": 1, "geometry_confirmed": False,
                        "b0_sign_confirmed": False, "b0_on_gre_grid": False},
        "calibration": {"units": "Hz/A", "registration_confirmed": False,
                        "patient_to_source_m": None},
        "optimization": {"field_limit_Hz": 100, "max_iterations": 600, "target_mse": None, "workers": 1,
                         "threads_per_worker": 1, "device": "cpu", "scratch_dir": "/tmp",
                         "matlab_command": "matlab"},
        "hardware": {"channel_map": None, "channel_polarity": None, "max_current_A": None,
                     "total_current_limit_A": None, "min_dac_interval_s": None,
                     "settling_time_s": None, "trigger_period_s": None,
                     "rf_center_after_trigger_s": None},
    }


def initialize(input_dir):
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / "config.json"
    if path.exists():
        raise ValueError(f"Configuration already exists: {path}")
    write_json(path, template())
    (input_dir / "sequences").mkdir(exist_ok=True)
    (input_dir / "INPUTS.md").write_text(
        "# Entrée ACDC\n\n"
        "Déposer ici gre_reference.dat, b0_map.dat, scan.dat et ACDC_3T.npz.\n"
        "Placer les séquences dans sequences/scan.seq et sequences/reference_gre.seq.\n"
        "La séquence multi-écho du Twix B0 va dans sequences/b0_map.seq.\n"
        "La B0 en Hz et le MAT de calibration sont préparés automatiquement.\n"
        "Référence : Twix GRE/FLASH cartésienne 2D mono-écho, avec sa propre séquence .seq.\n"
        "Scan : GRE cartésienne 2D mono-écho ou EPI prise en charge par le lecteur.\n"
        "Les matrices, FOV et temps sont lus automatiquement, sans taille à saisir.\n"
        "Renseigner config.json avec les paramètres mesurés de votre installation.\n"
        "Les chemins peuvent être absolus ou relatifs à ce dossier.\n"
        "Le tutoriel Experiment/TUTORIEL_SHIMMING_DYNAMIQUE.md décrit les entrées Twix.\n"
        "Aucun fichier de calibration ni paramètre matériel fictif n'est créé.\n", encoding="utf-8")
    print(f"Configuration to complete: {path}")


def read_config(input_dir):
    config = json.loads((input_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("config.json: version must be 1.")
    for section in ("files", "preparation", "optimization", "hardware"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"config.json: missing object {section}.")
    return config


def input_paths(input_dir, config):
    paths = {}
    for key in ("gre", "b0", "scan", "sequence", "calibration"):
        value = config["files"].get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"files.{key} must be a file path.")
        paths[key] = (input_dir / value).resolve()
        if not paths[key].is_file():
            raise ValueError(f"Missing {key}: {paths[key]}")
    if config["preparation"].get("roi_mask"):
        paths["roi_mask"] = (input_dir / config["preparation"]["roi_mask"]).resolve()
        if not paths["roi_mask"].is_file():
            raise ValueError(f"Missing ROI mask: {paths['roi_mask']}")
    for key in ("reference_sequence", "b0_sequence"):
        if config["files"].get(key):
            paths[key] = (input_dir / config["files"][key]).resolve()
            if not paths[key].is_file():
                raise ValueError(f"Missing {key}: {paths[key]}")
    return paths


def check_calibration(calibration, experiment):
    """Check the physical calibration independently of the old virtual workflow."""
    if np.asarray(calibration.get("registration_confirmed", False)).item() != 1:
        raise ValueError("Calibration requires registration_confirmed=true for the actual coil placement.")
    if str(calibration.get("units", "")) != "Hz/A":
        raise ValueError("Calibration units must be Hz/A.")
    maps = np.asarray(calibration.get("acdc_maps"), dtype=float)
    object_shape = np.asarray(experiment["m0_proxy"]).shape
    if len(object_shape) != 2 or maps.shape != (*object_shape, 32):
        raise ValueError("Calibration acdc_maps must match the reference grid, with 32 channels.")
    for axis in ("X_m", "Y_m"):
        grid = np.asarray(calibration.get(axis), dtype=float)
        if grid.shape != object_shape or not np.all(np.isfinite(grid)) or not np.allclose(
                grid, experiment["geometry"][axis], atol=1e-9, rtol=0):
            raise ValueError(f"Calibration {axis} does not match the GRE grid; register maps first.")
    support = np.asarray(experiment["field"]["simulation_mask"], dtype=bool)
    valid = np.all(np.isfinite(maps), axis=2)
    if "spatial_valid_mask" in calibration:
        mask = np.asarray(calibration["spatial_valid_mask"])
        if mask.shape != object_shape or not np.all(np.isin(mask, [0, 1])):
            raise ValueError("Calibration spatial_valid_mask must be binary and match the reference grid.")
        valid &= mask.astype(bool)
    if not np.all(valid[support]):
        raise ValueError("Some GRE support points have no measured coil calibration; no extrapolation is allowed.")
    if not np.all(np.any(np.abs(maps[support]) > 0, axis=0)):
        raise ValueError("Every calibration channel needs a nonzero response on the selected support.")
    return maps, support


def calibration_on_reference(calibration, experiment):
    """Sample the already aligned measured maps on this reference's grid."""
    from resample_acdc_maps import resample_to_grid
    if str(calibration.get("units", "")) != "Hz/A":
        raise ValueError("Calibration units must be Hz/A.")
    maps = np.asarray(calibration.get("acdc_maps"))
    if maps.ndim != 3 or maps.shape[2] != 32 or np.iscomplexobj(maps):
        raise ValueError("Calibration needs a real 2D grid of 32 measured channel maps.")
    source_mask = calibration.get("spatial_valid_mask", np.isfinite(maps).all(axis=2))
    geometry = experiment["geometry"]
    sampled, valid = resample_to_grid(maps, calibration["X_m"], calibration["Y_m"],
                                     source_mask, geometry["X_m"], geometry["Y_m"])
    result = {key: value for key, value in calibration.items() if not key.startswith("__")}
    result.update(acdc_maps=sampled, spatial_valid_mask=valid,
                  X_m=geometry["X_m"], Y_m=geometry["Y_m"])
    # Source geometry and original file hash remain in the session provenance.
    result["resampling"] = {"source_grid_shape": np.asarray(maps.shape[:2]),
                            "target_grid_shape": np.asarray(sampled.shape[:2]),
                            "method": "Linear on registered coordinates; no extrapolation"}
    check_calibration(result, experiment)
    return result


def check_optimization(config):
    opt = config["optimization"]
    for key, minimum, maximum in (("workers", 1, 1), ("threads_per_worker", 1, 6),
                                  ("max_iterations", 0, 1000000)):
        value = opt.get(key)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"optimization.{key} must be an integer in [{minimum}, {maximum}].")
    field = opt.get("field_limit_Hz")
    if not isinstance(field, (int, float)) or not np.isfinite(field) or field <= 0:
        raise ValueError("optimization.field_limit_Hz must be positive finite Hz.")
    target = opt.get("target_mse")
    if target is not None and (type(target) not in (int, float) or not np.isfinite(target) or target <= 0):
        raise ValueError("optimization.target_mse must be positive and finite, or null to disable it.")
    if opt.get("device") not in ("cpu", "gpu"):
        raise ValueError("optimization.device must be cpu or gpu.")
    if not Path(opt.get("scratch_dir", "")).is_dir():
        raise ValueError("optimization.scratch_dir must be an existing writable directory.")
    if not isinstance(opt.get("matlab_command"), str) or not opt["matlab_command"]:
        raise ValueError("optimization.matlab_command must identify the MATLAB executable.")


def check_optimization_result(config, info):
    """Accept solver convergence or a verified MSE stop; return whether the target stopped it.

    Physical current and field limits must also be recomputed from the saved
    currents before export. This check only validates the solver's stop reason.
    """
    def scalar_number(key):
        value = np.asarray(info.get(key))
        if value.ndim != 0 or value.dtype.kind not in "iuf" or not np.isfinite(value):
            raise ValueError(f"Invalid optimization_info.{key}; no export.")
        return float(value)

    def scalar_flag(key):
        value = np.asarray(info.get(key, False))
        if value.ndim != 0 or value.dtype.kind not in "biuf" or value.item() not in (0, 1):
            raise ValueError(f"Invalid optimization_info.{key}; no export.")
        return bool(value)

    if not all(scalar_flag(key) for key in ("current_limits_met", "field_limits_met")):
        raise ValueError("Solution is not converged and feasible. Inspect optimization/solver_*.txt; no export.")
    exitflag = scalar_number("exitflag")
    if exitflag > 0 and scalar_flag("converged"):
        return False
    target = config["optimization"].get("target_mse")
    if exitflag != -1 or target is None:
        raise ValueError("Solution is not converged and has no verified MSE target stop; no export.")
    fval = scalar_number("fval")
    if (scalar_flag("converged") or not scalar_flag("target_reached")
            or scalar_number("target_mse") != target or not 0 <= fval < target
            or ("accepted" in info and not scalar_flag("accepted"))):
        raise ValueError("MSE target stop does not match the configured target or saved objective; no export.")
    return True


def current_windows(report):
    """One measured-time interval per optimized current state, in RF-relative s."""
    if "state_start_s" in report:
        starts = np.asarray(report["state_start_s"], dtype=float)
        ends = np.asarray(report["state_end_s"], dtype=float)
    else:  # Legacy EPI report.
        edges = np.asarray(report["state_edges_s"], dtype=float)
        starts, ends = edges[:-1], edges[1:]
    if (starts.ndim != 1 or ends.shape != starts.shape or not starts.size
            or not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or np.any(ends <= starts) or np.any(starts[1:] < ends[:-1] - 1e-12)):
        raise ValueError("Invalid or overlapping current state windows.")
    return starts, ends


def hardware_edges(config, report):
    """Place transitions after the selected CSV time origin.

    An offset of zero uses the first RF center as time zero. A measured
    RF-after-trigger offset instead expresses transitions after that trigger.
    """
    hw = config["hardware"]
    offset = hw.get("rf_center_after_trigger_s")
    if not isinstance(offset, (int, float)) or not np.isfinite(offset) or offset < 0:
        raise ValueError("rf_center_after_trigger_s must be finite and nonnegative.")
    # These measured values add timing checks when supplied. They do not
    # configure a controller and are not required to write the current CSV.
    settle = hw.get("settling_time_s")
    if settle is not None:
        if not isinstance(settle, (int, float)) or not np.isfinite(settle) or settle <= 0:
            raise ValueError("settling_time_s must be positive and measured.")
        if np.any(settle >= np.asarray(report["first_adc_delay_s"])):
            raise ValueError("Coil settling reaches the scan ADC window; this sequence/model cannot be exported.")
    period = hw.get("trigger_period_s")
    if period is not None:
        end_sequence = offset - report["rf_center_s"] + report["sequence_duration_s"]
        if not isinstance(period, (int, float)) or not np.isfinite(period) or period < end_sequence - 1e-9:
            raise ValueError("trigger_period_s is shorter than the sequence ending after this trigger.")
    starts, ends = current_windows(report)
    if report.get("sequence_kind") == "gre":
        # Each GRE state is followed by zero before the next excitation.
        rf_starts = np.asarray(report["rf_start_s"], dtype=float)
        rf_ends = np.asarray(report["rf_end_s"], dtype=float)
        for start, end in zip(starts, ends):
            if np.any((rf_starts < end - 1e-12) & (rf_ends > start + 1e-12)):
                raise ValueError("A GRE current window overlaps an RF pulse.")
            later_rf = rf_starts[rf_starts >= end - 1e-12]
            if settle is not None and later_rf.size and end + settle >= later_rf.min():
                raise ValueError("Return to zero has not settled before the next GRE RF pulse.")
        return np.column_stack((starts, ends)).ravel() + offset
    return np.r_[starts, ends[-1]] + offset


def hardware_currents(currents, report):
    """Insert the zero intervals between GRE lines; retain EPI states unchanged."""
    if report.get("sequence_kind") != "gre":
        return currents
    expanded = np.zeros((currents.shape[0], 2 * currents.shape[1] - 1))
    expanded[:, ::2] = currents
    return expanded


def prepare(input_dir, output_dir):
    from prepare_acdc_inputs import prepare_inputs
    from read_acdc_calibration import read_calibration
    config = read_config(input_dir)
    paths = input_paths(input_dir, config)
    if output_dir.exists():
        raise ValueError(f"Output folder already exists: {output_dir}")
    # Preserve any failed preparation for diagnosis; it can never be exported
    # without the complete provenance manifest written below.
    output_dir.mkdir(parents=True)
    report = prepare_inputs(input_dir, output_dir / "prepared", config)
    input_mat = output_dir / "prepared" / "experiment_input.mat"
    experiment = loadmat(input_mat, simplify_cells=True)["experiment"]
    calibration = calibration_on_reference(
        read_calibration(paths["calibration"], experiment, config.get("calibration")), experiment)
    calibration_mat = output_dir / "prepared" / "calibration.mat"
    savemat(calibration_mat, calibration, do_compression=True, long_field_names=True)
    write_json(output_dir / "config.json", config)
    # Hash source code as well as inputs: a resumed run must use the same model.
    code_paths = sorted(ROOT.glob("*.m")) + [ROOT / name for name in (
        "run_acdc_experiment.py", "prepare_acdc_inputs.py", "read_acdc_sequence.py",
        "resample_acdc_maps.py", "export_acdc_waveform.py", "read_acdc_b0.py",
        "read_acdc_calibration.py", "prepare_acdc_calibration.py")]
    prepared_files = sorted((output_dir / "prepared").iterdir()) + [output_dir / "config.json"]
    provenance = {
        "input_dir": str(input_dir),
        "sources": {str(path): sha256(path) for path in [input_dir / "config.json", *paths.values()]},
        "code": {str(path): sha256(path) for path in code_paths},
        "prepared": {str(path.relative_to(output_dir)): sha256(path) for path in prepared_files if path.is_file()},
    }
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "report.json", {"status": "prepared", "waveform_exported": False})
    print(f"Prepared inputs: {output_dir / 'prepared'}")
    return config, report


def check_snapshot(output_dir):
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    for group in ("sources", "code", "prepared"):
        for name, expected in provenance[group].items():
            path = output_dir / name if group == "prepared" else Path(name)
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"File changed since preparation: {path}. Prepare a new output folder.")
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "prepared" / "preparation_report.json").read_text(encoding="utf-8"))
    return config, report, provenance


def export(output_dir, *, csv_output_dir=None):
    from export_acdc_waveform import export_waveform, quantize_currents
    config, report, provenance = check_snapshot(output_dir)
    check_optimization(config)
    edges = hardware_edges(config, report)
    solutions = list((output_dir / "optimization").glob("solution_*Hz.mat"))
    if len(solutions) != 1:
        raise ValueError("Exactly one solution is required; run the optimization first.")
    solution = loadmat(solutions[0], simplify_cells=True)
    info = solution["optimization_info"]
    target_stop = check_optimization_result(config, info)
    currents = np.asarray(solution["currents_A"], dtype=float)
    starts, ends = current_windows(report)
    if currents.shape != (32, len(starts)) or not np.all(np.isfinite(currents)):
        raise ValueError("Expected finite currents_A[32, number of acquired lines].")
    experiment = loadmat(output_dir / "prepared" / "experiment_input.mat", simplify_cells=True)["experiment"]
    calibration = loadmat(output_dir / "prepared" / "calibration.mat", simplify_cells=True)
    maps, support = check_calibration(calibration, experiment)
    if str(solution.get("calibration_mode")) != "registered" or not bool(solution.get("registration_confirmed")):
        raise ValueError("Virtual/unregistered solutions cannot be exported for an experiment.")
    if not np.allclose(np.asarray(solution["line_start_s"]).ravel(), starts, atol=1e-12, rtol=0):
        raise ValueError("Solution line timing does not match the prepared sequence.")
    if report.get("sequence_kind") == "gre" and ("line_end_s" not in solution or not np.allclose(
            np.asarray(solution["line_end_s"]).ravel(), ends, atol=1e-12, rtol=0)):
        raise ValueError("Solution GRE current end times do not match the prepared sequence.")
    field = float(np.max(np.abs(maps[support] @ currents)))
    limit = float(config["optimization"]["field_limit_Hz"])
    if float(solution["field_limit_Hz"]) != limit or field > limit + 1e-8:
        raise ValueError("Recomputed ACDC field exceeds the requested field constraint.")
    quantized_field = float(np.max(np.abs(maps[support] @ quantize_currents(currents))))
    if quantized_field > limit + 1e-8:
        raise ValueError("Nominal DAC quantization exceeds the field constraint; optimize with a margin.")
    csv_output_dir = output_dir / "waveform" if csv_output_dir is None else Path(csv_output_dir)
    waveform = export_waveform(hardware_currents(currents, report), edges,
                               config["hardware"], csv_output_dir)
    result = {"status": "exported", "waveform_exported": True,
              "solution_sha256": sha256(solutions[0]),
              "sequence_sha256": report["sequence_sha256"], "maximum_field_Hz": field,
              "maximum_nominal_quantized_field_Hz": quantized_field,
              "sequence_kind": report.get("sequence_kind", "epi"),
              "current_states": list(currents.shape),
              "reference_sequence": report.get("reference_sequence_validation"),
              "optimization_info": {key: float(info[key]) for key in ("exitflag", "fval")},
              "input_dir": provenance["input_dir"], "waveform": waveform,
              "limitations": ["Instantaneous piecewise-constant current model; physical transitions are not simulated.",
                              "GRE magnitude is an amplitude proxy, uniform receiver, no measured slice/RF model.",
                              "Generated files have not been validated on the physical amplifier/scanner."]}
    if config["optimization"].get("target_mse") is not None:
        result["optimization_info"].update(target_mse=config["optimization"]["target_mse"],
                                           target_reached=target_stop)
    print(f"Waveform CSV: {waveform['csv_file']}")
    return result


def run(input_dir, output_dir):
    """Deliver one CSV and remove the temporary numerical workspace."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output folder must be empty; existing files are never replaced.")
    with tempfile.TemporaryDirectory(prefix="acdc_csv_session_") as temporary:
        session_dir = Path(temporary) / "session"
        return run_session(input_dir, session_dir, csv_output_dir=output_dir)


def run_session(input_dir, output_dir, *, csv_output_dir):
    """Use the existing MATLAB preparation and optimization inside a workspace."""
    from export_acdc_waveform import export_waveform
    if output_dir.exists():
        config, report, provenance = check_snapshot(output_dir)
        if Path(provenance["input_dir"]) != input_dir:
            raise ValueError("This output folder belongs to another input folder.")
    else:
        config, report = prepare(input_dir, output_dir)
    check_optimization(config)
    edges = hardware_edges(config, report)
    # Validate the export's timing/mapping/limits before spending time in MATLAB.
    # The validation directory is temporary and contains only zero currents.
    with tempfile.TemporaryDirectory(prefix="acdc_export_check_") as temp:
        starts, _ = current_windows(report)
        empty_currents = hardware_currents(np.zeros((32, len(starts))), report)
        export_waveform(empty_currents, edges, config["hardware"], Path(temp) / "check")
    opt = config["optimization"]
    if (output_dir / "optimization").exists():
        raise ValueError("Optimization already exists. Use export, or choose a new output folder for another run.")
    request = {key: opt[key] for key in ("field_limit_Hz", "max_iterations", "workers",
               "threads_per_worker", "device", "scratch_dir")}
    for key in ("field_limit_Hz", "max_iterations", "workers", "threads_per_worker"):
        request[key] = float(request[key])
    request.update(input_file=str(output_dir / "prepared" / "experiment_input.mat"),
                   calibration_file=str(output_dir / "prepared" / "calibration.mat"),
                   output_dir=str(output_dir / "optimization"),
                   max_current_A=np.asarray(config["hardware"]["max_current_A"], dtype=float),
                   total_current_limit_A=float(config["hardware"]["total_current_limit_A"]))
    if opt.get("target_mse") is not None:
        request["target_mse"] = float(opt["target_mse"])
    request_file = output_dir / "optimizer_request.mat"
    savemat(request_file, request)
    # MATLAB string escaping, without a shell or user-provided code execution.
    quote = lambda path: str(path).replace("'", "''")
    command = f"addpath('{quote(ROOT)}'); run_acdc_folder('{quote(request_file)}');"
    print("MATLAB optimization in progress...", flush=True)
    completed = subprocess.run([opt["matlab_command"], "-batch", command], check=False)
    if completed.returncode:
        write_json(output_dir / "report.json", {"status": "optimization_failed", "waveform_exported": False,
                                               "matlab_returncode": completed.returncode})
        raise ValueError("MATLAB failed or the solution was not accepted; no waveform CSV was exported.")
    return export(output_dir, csv_output_dir=csv_output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "prepare", "run", "export"):
        sub = commands.add_parser(name)
        if name != "export":
            sub.add_argument("--input", type=Path, required=True, help="Input folder (config.json and images)")
        if name != "init":
            sub.add_argument("--output", type=Path, required=True,
                             help="CSV output folder" if name == "run" else "Numerical session folder")
    args = parser.parse_args(argv)
    try:
        input_dir = args.input.resolve() if hasattr(args, "input") else None
        output_dir = args.output.resolve() if hasattr(args, "output") else None
        if args.command == "init":
            initialize(input_dir)
        elif args.command == "prepare":
            prepare(input_dir, output_dir)
        elif args.command == "run":
            run(input_dir, output_dir)
        else:
            export(output_dir)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"ACDC : {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
