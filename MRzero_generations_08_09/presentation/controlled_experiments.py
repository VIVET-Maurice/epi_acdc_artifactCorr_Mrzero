"""Reproducible numerical controls for the weekly 08/09 presentation.

Production physics modules are imported, never patched. Uniform objects and
rectangular-ADC-aperture calculations are explicitly separate diagnostic cases.
All timing excludes file I/O and uses the same prepared object/device/precision.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import pickle
import platform
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_erosion
import torch

HELPER_DIR = Path(__file__).resolve().parent.parent
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))
from inplane_geometry import FrameReconstruction
from measured_image_target import configure_measured_gre_target
from native_b0_model import NativeB0ACDCModel
from reconstruction_08 import FrameDFTReconstruction
from voxel_3d import FiniteVoxel3DACDCModel


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_result(output_dir, name, arrays, metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path, metrics_path = output_dir / f"{name}.npz", output_dir / f"{name}.json"
    np.savez_compressed(arrays_path, **arrays)
    metrics_path.write_text(json.dumps(json_safe(metrics), indent=2), encoding="utf-8")
    return dict(arrays=str(arrays_path), metrics=str(metrics_path))


def load_context(path):
    """Load the local, trusted prepared-object cache made by this notebook."""
    path = Path(path)
    with path.open("rb") as stream:
        context = pickle.load(stream)
    context = dict(context)
    context["context_cache"] = dict(path=str(path.resolve()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return context


def provenance(context):
    cpu_name = platform.processor()
    try:
        cpu_name = next(line.split(":", 1)[1].strip() for line in
            Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return dict(timestamp_UTC=datetime.now(timezone.utc).isoformat(),
        python=sys.executable, python_version=platform.python_version(),
        torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
        cpu_model=cpu_name, machine=platform.machine(),
        torch_threads=torch.get_num_threads(), model_options=context.get("model_options"),
        context_cache=context.get("context_cache"), configuration=context.get("configuration"),
        prepared_provenance=context.get("provenance"), input_sha256=context.get("input_sha256"),
        fit_policy=context.get("fit_policy"), fit_iteration_override=context.get("fit_iteration_override"),
        source_inputs=context["inputs"].get("paths"),
        runner_path=str(Path(__file__).resolve()),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def _sync(model):
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)


def build_model(context, generation="08", *, experiment=None):
    """Build one production model with a fixed, generation-specific operator."""
    import continuous_reconstruction
    # Explicit factories select the model operator. Global switches belong to
    # raw-data readers and must not leak into this controlled model comparison.
    previous = (continuous_reconstruction.FORCE_DIRECT_DFT,
                continuous_reconstruction.MATRIX_DFT_ENABLED)
    continuous_reconstruction.FORCE_DIRECT_DFT = False
    continuous_reconstruction.MATRIX_DFT_ENABLED = False
    try:
        prepared = context["prepared"]
        exp = deepcopy(prepared["expanded_experiment"] if experiment is None else experiment)
        options = dict(context.get("model_options", {}))
        options.pop("reconstruction", None)
        options.pop("reconstruction_factory", None)
        frame = np.asarray(context["frame_matrix"])
        reconstruction_class = FrameDFTReconstruction if generation == "08V2" else FrameReconstruction
        options["reconstruction_factory"] = lambda acq, device, dtype: reconstruction_class(
            acq, device, dtype, frame_matrix=frame,
            sample_chunk=int(context.get("reconstruction_chunk", 256)))
        model_class = NativeB0ACDCModel if generation == "09" else FiniteVoxel3DACDCModel
        return model_class(exp, prepared["expanded_calibration"],
            max_current_A=context.get("max_current_A", 3.), **options)
    finally:
        continuous_reconstruction.FORCE_DIRECT_DFT, continuous_reconstruction.MATRIX_DFT_ENABLED = previous


def controlled_currents(context, model):
    """Deterministic smooth diagnostic currents, projected by production limits."""
    states = np.arange(model.nstates, dtype=float)[None] / max(1, model.nstates - 1)
    channels = np.arange(model.nc, dtype=float)[:, None]
    requested = (.04 * np.sin(2 * np.pi * (1 + channels % 4) * states + .29 * channels)
                 + .015 * np.cos(2 * np.pi * .7 * states - .17 * channels))
    currents = model.project_currents(requested,
        float(context.get("total_current_limit_A", 30.)),
        float(context.get("field_limit_Hz", 1000.)))
    hardware = model.hardware.diagnostics(currents,
        float(context.get("total_current_limit_A", 30.)),
        float(context.get("field_limit_Hz", 1000.)))
    if not hardware["all_limits_met"]:
        raise RuntimeError("Projected controlled diagnostic currents failed production hardware checks.")
    return currents, dict(description="deterministic smooth currents; not an optimized waveform",
        requested_peak_A=float(np.max(np.abs(requested))),
        projected_peak_A=float(np.max(np.abs(currents))),
        maximum_total_absolute_current_A=float(np.max(np.abs(currents).sum(axis=0))),
        disabled_channel_indices_zero_based=np.flatnonzero(np.asarray(model.numpy(model.max_current_A)) == 0),
        hardware=hardware)


def relative_error(actual, reference, mask=None):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if mask is not None:
        actual, reference = actual[mask], reference[mask]
    denominator = np.linalg.norm(reference.ravel())
    return float(np.linalg.norm((actual - reference).ravel()) / denominator) if denominator else None


def run_equivalence(context, output_dir, *, models=None):
    """Compare every ADC and matched image for zero and dynamic currents."""
    model08 = models["08"] if models is not None else build_model(context, "08")
    model09 = models["09"] if models is not None else build_model(context, "09")
    dynamic, current_report = controlled_currents(context, model08)
    arrays = dict(time_s=model08.time_s, chronological_ADC_order=np.argsort(model08.time_s.ravel()),
        X_m=model08.reconstruction.X_m, Y_m=model08.reconstruction.Y_m,
        dynamic_currents_A=dynamic, currents_A=dynamic, adc_time_s=model08.time_s,
        ideal_image08=model08.ideal_image, ideal_kdata08=model08.ideal_kdata)
    rows = []
    for label, currents in (("zero", np.zeros_like(dynamic)), ("dynamic", dynamic)):
        print(f"Controlled equivalence: {label} currents", flush=True)
        image08, signal08 = model08.forward(currents)
        image09, signal09 = model09.forward(currents)
        residual = signal09 - signal08
        peak = float(np.max(np.abs(signal08)))
        reliable = np.abs(signal08) > 1e-6 * peak
        point_relative = np.full(signal08.shape, np.nan)
        point_relative[reliable] = np.abs(residual[reliable]) / np.abs(signal08[reliable])
        arrays.update({f"{label}_signal08": signal08, f"{label}_signal09": signal09,
            f"{label}_signal_residual": residual,
            f"{label}_pointwise_relative_residual": point_relative,
            f"{label}_pointwise_relative_valid_mask": reliable,
            f"{label}_image08": image08, f"{label}_image09": image09,
            f"{label}_image_residual": image09 - image08})
        canonical = "baseline" if label == "zero" else "dynamic"
        arrays.update({f"{canonical}_image08": image08, f"{canonical}_image09": image09,
            f"{canonical}_kdata08": signal08, f"{canonical}_kdata09": signal09})
        rows.append(dict(case=label, signal_complex_relative_L2=relative_error(signal09, signal08),
            image_complex_relative_L2=relative_error(image09, image08),
            image_magnitude_NRMSE_percent=100 * relative_error(np.abs(image09), np.abs(image08)),
            maximum_absolute_signal_residual=float(np.max(np.abs(residual))),
            maximum_signal_residual_fraction_of_peak=float(np.max(np.abs(residual)) / peak),
            pointwise_relative_threshold_fraction_of_peak=1e-6,
            pointwise_relative_valid_count=int(reliable.sum()),
            reconstruction08=model08.reconstruction.report, reconstruction09=model09.reconstruction.report))
    metrics = dict(experiment="08 versus 09 implementation equivalence", rows=rows,
        current_configuration=current_report, provenance=provenance(context),
        common_settings=dict(device=str(model08.device), dtype=str(model08.dtype),
            native_precision="float32", voxel_count=int(model08.support.sum()),
            ADC_sample_count=model08.nsamples, ACDC_channel_count=model08.nc,
            sample_chunk08=model08.sample_chunk, sample_chunk09=model09.sample_chunk,
            B0_enabled=True, receiver_model="one uniform receive channel in both paths",
            external_phase_and_sinc_in08=True, local_native_executor_in09=True),
        interpretation="Implementation agreement does not establish scanner agreement.")
    artifacts = _write_result(output_dir, "equivalence", arrays, metrics)
    return dict(artifacts=artifacts, metrics=metrics)


def _benchmark_callable(function, model, warmup, repeats):
    for _ in range(warmup):
        function()
    durations = []
    peak_bytes = None
    for _ in range(repeats):
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        _sync(model)
        started = perf_counter()
        function()
        _sync(model)
        durations.append(perf_counter() - started)
        if model.device.type == "cuda":
            observed = int(torch.cuda.max_memory_allocated(model.device))
            peak_bytes = observed if peak_bytes is None else max(peak_bytes, observed)
    return _timing_summary(durations, warmup=warmup, peak_gpu_allocated_bytes=peak_bytes)


def _timing_summary(durations, *, warmup=0, peak_gpu_allocated_bytes=None):
    q25, median, q75 = np.percentile(durations, [25, 50, 75])
    return dict(raw_seconds=list(durations), median_seconds=float(median),
        q25_seconds=float(q25), q75_seconds=float(q75), IQR_seconds=float(q75 - q25),
        warmup_count=int(warmup), repetitions=len(durations),
        peak_gpu_allocated_bytes=peak_gpu_allocated_bytes)


def _install_common_target(model, reference_image, mask):
    reference = torch.as_tensor(reference_image, device=model.device, dtype=model.complex_dtype)
    weights = torch.as_tensor(mask, device=model.device, dtype=model.dtype)
    model._reference = reference
    model._reference_amplitude = reference.abs()
    model._reference_energy = (reference.abs().square() * weights).sum()
    model._image_target_weights = weights


def run_runtime_benchmarks(context, output_dir, *, warmup=1, repeats=3, setup_repeats=3):
    """Time fresh setup separately from warm forward and loss+gradient calls.

    A fresh constructor includes initialization, native baseline/ideal work and
    reconstruction preparation. Baseline availability additionally includes the
    first B0-enabled zero-current forward. Generation 09 has no baseline cache.
    Setup repetitions always construct new model instances; disk I/O is excluded.
    """
    if warmup < 0 or repeats < 1 or setup_repeats < 1:
        raise ValueError("Require warmup >= 0 and positive repetition counts.")
    records, models, baseline_outputs = [], {}, {}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    def checkpoint():
        # Outside every timed interval. Preserve finished operations if a later
        # expensive diagnostic fails or the run is interrupted.
        (Path(output_dir) / "runtime_benchmarks_partial.json").write_text(
            json.dumps(json_safe(dict(status="running", records=records,
                provenance=provenance(context))), indent=2), encoding="utf-8")
    for generation in ("08", "08V2", "09"):
        constructor_times, first_forward_times, available_times = [], [], []
        for repeat in range(setup_repeats):
            print(f"Benchmark fresh setup {generation}: {repeat + 1}/{setup_repeats}", flush=True)
            started = perf_counter()
            candidate = build_model(context, generation)
            _sync(candidate)
            constructor_s = perf_counter() - started
            started = perf_counter()
            baseline = candidate.forward(np.zeros((candidate.nc, candidate.nstates)))
            _sync(candidate)
            first_forward_s = perf_counter() - started
            constructor_times.append(constructor_s)
            first_forward_times.append(first_forward_s)
            available_times.append(constructor_s + first_forward_s)
            if repeat == 0:
                models[generation], baseline_outputs[generation] = candidate, baseline
            else:
                del candidate
                gc.collect()
        for operation, times in (("fresh_constructor", constructor_times),
                                 ("first_zero_current_forward", first_forward_times),
                                 ("cold_baseline_available", available_times)):
            records.append(dict(generation=generation, operation=operation,
                **_timing_summary(times), includes_file_IO=False,
                includes_model_construction=operation != "first_zero_current_forward"))
        checkpoint()

    target = configure_measured_gre_target(models["08"], context["inputs"]["gre"],
        baseline_image=baseline_outputs["08"][0],
        measured_baseline_image=context["inputs"]["scan"],
        frame_matrix=context["frame_matrix"], threshold_fraction=0.)
    reference = models["08"].numpy(models["08"]._reference_amplitude)
    mask = models["08"].numpy(models["08"]._image_target_weights).astype(bool)
    dynamic, current_report = controlled_currents(context, models["08"])
    for generation, model in models.items():
        _install_common_target(model, reference, mask)
        operations = {
            "warm_forward_and_reconstruction": lambda model=model: model.forward(dynamic),
            "warm_loss_and_gradient": lambda model=model: model.objective_and_grad(dynamic),
            "reconstruction_only_production": lambda model=model: model.reconstruction.image(
                torch.as_tensor(baseline_outputs["08"][1], device=model.device, dtype=model.complex_dtype)),
        }
        for operation, function in operations.items():
            print(f"Benchmark {generation}: {operation}", flush=True)
            timing = _benchmark_callable(function, model, warmup, repeats)
            records.append(dict(generation=generation, operation=operation, **timing,
                includes_file_IO=False, includes_model_construction=False))
            checkpoint()
    metrics = dict(experiment="production model runtime", records=records,
        provenance=provenance(context), common_target=target["report"],
        current_configuration=current_report,
        common_settings=dict(device=str(models["08"].device), dtype=str(models["08"].dtype),
            native_MRzero_precision="float32", full_ADC_count=models["08"].nsamples,
            full_voxel_count=int(models["08"].support.sum()),
            model_sample_chunk=models["08"].sample_chunk, file_IO_included=False,
            setup_warmup_count=0, per_operation_warmup_count=warmup,
            GPU_memory="unavailable on CPU" if models["08"].device.type == "cpu" else "PyTorch peak allocated bytes",
            current_projection_and_field_guidance_included=False,
            loss="fixed measured GRE magnitude MSE; identical target/mask/gain copied from 08",
            reconstruction08=models["08"].reconstruction.report,
            reconstruction08V2=models["08V2"].reconstruction.report,
            reconstruction09=models["09"].reconstruction.report),
        interpretation="08V2 is the production matrix least-squares inverse; it is distinct from the separately forced direct Fourier-adjoint comparator.")
    artifacts = _write_result(output_dir, "runtime_benchmarks", dict(currents_A=dynamic,
        common_reference_amplitude=reference, common_reference_mask=mask), metrics)
    csv_path = Path(output_dir) / "runtime_benchmarks.csv"
    with csv_path.open("w", newline="") as stream:
        fields = ["generation", "operation", "median_seconds", "q25_seconds", "q75_seconds", "IQR_seconds", "warmup_count", "repetitions", "peak_gpu_allocated_bytes"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    artifacts["csv"] = str(csv_path)
    (Path(output_dir) / "runtime_benchmarks_partial.json").write_text(
        json.dumps(dict(status="complete", full_metrics=artifacts["metrics"]), indent=2), encoding="utf-8")
    return dict(artifacts=artifacts, metrics=metrics, models=models)


def _uniform_metric(image, roi):
    values = np.abs(image)[roi]
    mean = float(values.mean())
    return dict(interior_pixels=int(roi.sum()), mean=mean, std=float(values.std()),
        CV=float(values.std() / mean), CV_percent=float(100 * values.std() / mean),
        relative_spatial_error_from_constant_one=float(np.linalg.norm(values - 1) / np.sqrt(len(values))))


def run_uniform_phantom(context, output_dir):
    """One finite-support, constant-amplitude object; isolate measured B0 effects.

    Explicit diagnostic interventions: all retained voxels have equal amplitude,
    T2=inf, no unresolved fitted distribution, ACDC=0, uniform receive sensitivity.
    Native finite voxel dimensions and measured B0/known gradients are retained.
    """
    experiment = deepcopy(context["prepared"]["expanded_experiment"])
    supported = np.asarray(experiment["field"]["simulation_mask"], bool)
    experiment["m0_proxy"] = supported.astype(float)
    experiment["relaxation"]["T2star_s"] = np.full(supported.shape, np.inf)
    experiment.pop("multi_echo_coherence", None)
    print("Uniform phantom: fresh native 08 model", flush=True)
    model = build_model(context, "08", experiment=experiment)
    zero_image, zero_signal = model.ideal_image, model.ideal_kdata
    b0_image, b0_signal = model.forward(np.zeros((model.nc, model.nstates)))
    if tuple(model.parent_shape) != tuple(model.image_size):
        raise ValueError("Uniform diagnostic support mask requires matching native object/output grids.")
    support_mask = np.zeros(model.image_size, bool)
    support_mask.ravel()[model.parent_flat_index] = np.any(model.support, axis=1)
    roi = binary_erosion(support_mask, iterations=3, border_value=0)
    if not roi.any():
        raise ValueError("Three-pixel erosion leaves no uniform-phantom interior.")
    gain = 1 / float(np.abs(zero_image)[roi].mean())
    zero_image, b0_image = zero_image * gain, b0_image * gain
    arrays = dict(B0_zero_image=zero_image, measured_B0_image=b0_image,
        b0_zero_image=zero_image, b0_measured_image=b0_image,
        B0_zero_signal=zero_signal, measured_B0_signal=b0_signal,
        support_mask=support_mask, interior_mask=roi,
        magnitude_residual=np.abs(b0_image) - np.abs(zero_image),
        X_m=model.reconstruction.X_m, Y_m=model.reconstruction.Y_m,
        time_s=model.time_s)
    metrics = dict(experiment="constant-amplitude finite-support phantom", provenance=provenance(context),
        normalization=dict(gain=gain, policy="one gain from B0=0 interior mean; shared by both images"),
        interior_mask=dict(policy="three iterations of scipy binary erosion, 4-connected default structure",
            support_pixels=int(support_mask.sum()), interior_pixels=int(roi.sum())),
        B0_zero=_uniform_metric(zero_image, roi), measured_B0=_uniform_metric(b0_image, roi),
        magnitude_change_NRMSE_percent=100 * relative_error(np.abs(b0_image), np.abs(zero_image), roi),
        controlled_changes=dict(amplitude_per_retained_voxel=1., T2_s="infinity", unresolved_coherence="removed",
            currents_A=0., receiver_sensitivity="one uniform receive channel", native_voxel_box="retained",
            B0_normal_gradient="assumed zero in both cases"),
        interpretation="CV quantifies spatial intensity variation. Finite support, truncated k-space and voxel windows can produce nonzero CV even at B0=0; this does not identify the scanner-pattern mechanism.")
    artifacts = _write_result(output_dir, "uniform_phantom", arrays, metrics)
    del model
    gc.collect()
    return dict(artifacts=artifacts, metrics=metrics)


def run_archived_optimization_replay(context, output_dir, *, model08=None, archive_path=None):
    """Replay saved commands on the fresh object; do not rerun optimization.

    Command values are never projected or modified. Hardware checks describe
    the original saved values. This is an offline prediction, with no verified
    association between the archived waveform and a measured ACDC-ON scan.
    """
    archive_path = Path(archive_path or HELPER_DIR / "results" /
        "mrzero08_20261110_115401_682397" / "mrzero08_optimization.npz")
    if not archive_path.exists():
        return dict(status="pending", reason="archived optimization commands unavailable",
            archive_path=str(archive_path))
    model = model08 if model08 is not None else build_model(context, "08")
    zeros = np.zeros((model.nc, model.nstates))
    baseline_image, baseline_signal = model.forward(zeros)
    if not hasattr(model, "image_target"):
        configure_measured_gre_target(model, context["inputs"]["gre"],
            baseline_image=baseline_image, measured_baseline_image=context["inputs"]["scan"],
            frame_matrix=context["frame_matrix"], threshold_fraction=0.)
    target = model.image_target
    mask = np.asarray(target["mask"], bool)
    reference = np.asarray(target["measured_reference_magnitude"])
    gains = target["report"]
    gre_gain = float(gains["fixed_simulator_to_GRE_gain"])
    off_gain = float(gains["simulation_to_baseline_EPI_gain"])
    measured_off = np.asarray(context["inputs"]["scan"])
    arrays = dict(baseline_image=baseline_image, baseline_kdata=baseline_signal,
        measured_GRE=reference, measured_OFF=measured_off, mask=mask,
        X_m=model.reconstruction.X_m, Y_m=model.reconstruction.Y_m)
    rows = []
    with np.load(archive_path, allow_pickle=False) as archive:
        for label, key in (("archived", "currents_A"), ("quantized", "quantized_currents_A")):
            if key not in archive:
                continue
            currents = np.asarray(archive[key])
            if currents.shape != zeros.shape or not np.isfinite(currents).all():
                raise ValueError(f"Archived {key} does not match the fresh model's current shape.")
            print(f"Fresh-context archived command replay: {label}", flush=True)
            image, signal = model.forward(currents)
            hardware = model.hardware.diagnostics(currents,
                float(context.get("total_current_limit_A", 30.)),
                float(context.get("field_limit_Hz", 1000.)))
            arrays.update({f"{label}_currents_A": currents, f"{label}_image": image,
                f"{label}_kdata": signal})
            rows.append(dict(case=label,
                simulated_prediction_vs_measured_GRE_NRMSE_percent=100 * relative_error(
                    gre_gain * np.abs(image), reference, mask),
                prediction_difference_vs_measured_OFF_percent=100 * relative_error(
                    off_gain * np.abs(image), measured_off, mask), hardware=hardware))
    metrics = dict(experiment="ARCHIVED COMMAND REPLAY on fresh 32-echo object", rows=rows,
        provenance=provenance(context), archive_path=str(archive_path.resolve()),
        archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        target_report=target["report"], fixed_simulator_to_GRE_gain=gre_gain,
        fixed_simulator_to_OFF_gain=off_gain, mask_pixels=int(mask.sum()),
        baseline_vs_measured_GRE_NRMSE_percent=100 * relative_error(
            gre_gain * np.abs(baseline_image), reference, mask),
        baseline_vs_measured_OFF_NRMSE_percent=100 * relative_error(
            off_gain * np.abs(baseline_image), measured_off, mask),
        new_optimization_performed=False, archived_commands_modified=False,
        measured_ACDC_ON_waveform_link_verified=False,
        interpretation="Fresh offline replay of earlier commands; not a new optimization and not physical ACDC-ON validation. Prediction differences from measured OFF are not an ON-scan fidelity metric.")
    artifacts = _write_result(output_dir, "archived_optimization_replay", arrays, metrics)
    return dict(artifacts=artifacts, metrics=metrics)


def _pulseq_dwell_offsets(context, model, offsets_s):
    """Evaluate exact nominal gradient integrals across each ADC aperture."""
    import pypulseq as pp
    inputs = context["inputs"]
    sequence_path = Path(inputs["paths"]["scan_sequence"])
    sequence = pp.Sequence()
    sequence.read(str(sequence_path))
    absolute_rf = float(inputs["timing"]["rf_center_s"])
    centers = model.time_s.ravel() + absolute_rf
    gradient_polynomials = sequence.get_gradients()
    frame = np.asarray(context["frame_matrix"])
    result = []
    for offset in offsets_s:
        delta = np.zeros((3, len(centers)))
        for axis, gradient in enumerate(gradient_polynomials):
            if gradient is not None:
                primitive = gradient.antiderivative()
                delta[axis] = primitive(centers + offset) - primitive(centers)
        if not np.isfinite(delta).all():
            raise ValueError("Pulseq gradient polynomial is undefined inside an ADC dwell.")
        if np.max(np.abs(delta[2])) > 1e-9:
            raise ValueError("ADC aperture crosses nonzero slice encoding; this 2D diagnostic does not support it.")
        result.append(frame @ delta[:2])
    return np.asarray(result), sequence_path


@torch.no_grad()
def _analytic_zero_current_samples(model, time_offset_s, delta_k):
    """Independent ideal-RF oracle for the stated local box model, not a production replacement."""
    signal = np.empty(model.nsamples, complex)
    x, y, b0 = model._x, model._y, model._b0
    amplitude, rate = model._amplitude, model._rate
    model._ensure_voxel_tensors()
    for first in range(0, model.nsamples, model.sample_chunk):
        selected = slice(first, min(first + model.sample_chunk, model.nsamples))
        elapsed = model._evolution[selected][None] + float(time_offset_s)
        kx = model._kx[selected] + model.tensor(delta_k[0, selected])
        ky = model._ky[selected] + model.tensor(delta_k[1, selected])
        cycles = x * kx + y * ky + b0 * elapsed
        point = amplitude * torch.exp(-rate * (elapsed - model.amplitude_reference_time_s))
        point = point * torch.exp(-2j * torch.pi * cycles + 1j * model._phase[selected])
        factor = torch.ones_like(cycles)
        for axis in range(3):
            edge = model._voxel_edges[axis]
            q = edge[0] * kx + edge[1] * ky + model._voxel_b0_gradient[:, axis, None] * elapsed
            factor *= torch.sinc(q)
        if model._coherence_numpy is not None:
            width, kind = model._coherence_numpy
            width = model.tensor(width[model.support])[:, None]
            kind = torch.as_tensor(kind[model.support], device=model.device)[:, None]
            q = width * elapsed
            factor *= torch.where(kind == 1, torch.sinc(q),
                torch.where(kind == 2, torch.exp(-2 * torch.pi**2 * q.square()), torch.ones_like(q)))
        signal[selected] = model.numpy((point * factor).sum(dim=0))
    return signal.reshape(model.sampling_shape)


def run_adc_window(context, output_dir, *, model08=None, orders=(3, 5)):
    """Test a hypothetical rectangular ADC aperture using exact Pulseq gradients.

    This uses a separately validated analytic ideal-RF oracle at zero ACDC.
    The scanner receive-filter impulse response is not available; the result
    tests the rectangular-window hypothesis, not an identified scanner filter.
    """
    model = model08 if model08 is not None else build_model(context, "08")
    dwell = float(context["prepared"]["expanded_experiment"]["acquisition"]["dwell_s"])
    zero = np.zeros((model.nc, model.nstates))
    center_image, native_center = model.forward(zero)
    oracle_center = _analytic_zero_current_samples(model, 0., np.zeros((2, model.nsamples)))
    validation_error = relative_error(oracle_center, native_center)
    if validation_error is None or validation_error > 5e-5:
        raise RuntimeError(f"ADC oracle center mismatch {validation_error}; aperture comparison is not valid.")
    arrays = dict(center_native_signal=native_center, center_oracle_signal=oracle_center,
        center_native_image=center_image, X_m=model.reconstruction.X_m, Y_m=model.reconstruction.Y_m,
        time_s=model.time_s)
    rows, signals = [], {}
    for order in orders:
        nodes, weights = np.polynomial.legendre.leggauss(int(order))
        offsets = nodes * dwell / 2
        deltas, sequence_path = _pulseq_dwell_offsets(context, model, offsets)
        started = perf_counter()
        averaged = np.zeros_like(oracle_center)
        for node, offset in enumerate(offsets):
            print(f"ADC aperture quadrature order {order}, node {node + 1}/{order}", flush=True)
            averaged += (weights[node] / 2) * _analytic_zero_current_samples(model, offset, deltas[node])
        image = model.numpy(model.reconstruction.image(torch.as_tensor(
            averaged, device=model.device, dtype=model.complex_dtype)))
        signals[order] = averaged
        arrays.update({f"order{order}_window_signal": averaged, f"order{order}_window_image": image,
            f"order{order}_nodes_s": offsets, f"order{order}_quadrature_weights": weights / 2,
            f"order{order}_delta_k_mInv": deltas,
            f"window{order}_image": image, f"window{order}_kdata": averaged})
        rows.append(dict(quadrature_order=int(order), computation_seconds=perf_counter() - started,
            complex_signal_relative_change=relative_error(averaged, oracle_center),
            complex_image_relative_change=relative_error(image, center_image),
            magnitude_image_NRMSE_percent=100 * relative_error(np.abs(image), np.abs(center_image))))
    convergence = relative_error(signals[orders[-1]], signals[orders[-2]]) if len(orders) > 1 else None
    metrics = dict(experiment="hypothetical finite rectangular ADC aperture", provenance=provenance(context),
        dwell_s=dwell, quadrature_rows=rows, quadrature_convergence_relative_L2=convergence,
        center_oracle_vs_MRzero_relative_L2=validation_error,
        sequence_path=sequence_path, sequence_sha256=hashlib.sha256(sequence_path.read_bytes()).hexdigest(),
        gradient_source="Pulseq get_gradients() piecewise polynomial antiderivatives in Hz/m",
        sampling="actual sorted/reversed ADC timestamps; exact nominal k offsets at aperture quadrature nodes",
        currents_A=0., gain_policy="no fitted gains; identical simulation scale",
        production_forward_model_changed=False,
        receiver_filter_status="unknown; rectangular aperture is a diagnostic hypothesis",
        interpretation="A finite-window difference does not establish the scanner's actual digital/analog receive filtering.")
    artifacts = _write_result(output_dir, "adc_window", arrays, metrics)
    return dict(artifacts=artifacts, metrics=metrics)


def export_local_q_maps(context, output_dir):
    """Map local cycles across each voxel at three explicitly selected ADCs."""
    from acdc_gpu import response_matrices
    output_dir = Path(output_dir)
    with np.load(output_dir / "equivalence.npz", allow_pickle=False) as archive:
        currents = np.asarray(archive["currents_A"])
        X_m, Y_m = np.asarray(archive["X_m"]), np.asarray(archive["Y_m"])
    prepared = context["prepared"]
    experiment = prepared["expanded_experiment"]
    acquisition = experiment["acquisition"]
    times = np.asarray(acquisition["time_s"]).ravel()
    rf = np.asarray(acquisition.get("rf_time_s", np.zeros_like(times))).ravel()
    k = np.stack((np.asarray(acquisition["kx_mInv"]).ravel(),
                  np.asarray(acquisition["ky_mInv"]).ravel()))
    selected = np.array([np.argmin(times), np.argmin(np.sum(k*k, axis=0)), np.argmax(times)])
    starts, ends = (np.asarray(acquisition[key]).ravel() for key in ("line_start_s", "line_end_s"))
    response = experiment.get("current_response", {})
    if response.get("enabled", False):
        tau = np.broadcast_to(np.asarray(response["tau_s"]), (currents.shape[0],))
        H, _ = response_matrices(starts, ends, times[selected], tau, rf[selected])
        integrated = np.einsum("cs,cns->cn", currents, H)
    else:
        H = np.maximum(0., np.minimum(times[selected, None], ends[None, :])
                       - np.maximum(rf[selected, None], starts[None, :]))
        integrated = currents @ H.T
    metadata = experiment["inplane_voxel"]
    edges = np.asarray(metadata["edge_vectors_m"])
    gB0 = np.asarray(metadata["b0_gradient_Hz_per_pixel"])[:, 0]
    gACDC = np.asarray(metadata["field_gradient_Hz_per_A_per_pixel"])[:, 0]
    q0 = ((edges @ k[:, selected])[None, :, :]
          + gB0[:, :, None] * (times - rf)[selected][None, None, :])
    qdynamic = q0 + np.einsum("vac,cn->van", gACDC, integrated)
    parent_shape = tuple(prepared["parent_shape"])
    parent_indices = np.asarray(prepared["parent_flat_index"])
    def parent_maps(value):
        output = np.full((3, len(selected), int(np.prod(parent_shape))), np.nan)
        output[..., parent_indices] = value.transpose(1, 2, 0)
        return output.reshape((3, len(selected)) + parent_shape)
    metrics = dict(experiment="local phase-slope q maps", provenance=provenance(context),
        units="cycles across one full native voxel",
        formula="q[a,v,n] = k[n].e[a] + gB0[v,a]*elapsed[n] + sum_j gACDC[v,a,j]*J[j,n]",
        axes=list(metadata["axes"]), sample_labels=["first imaging ADC", "ADC nearest k=0", "last imaging ADC"],
        sample_flat_indices=selected, ADC_time_s=times[selected], elapsed_s=(times-rf)[selected],
        through_plane_B0="unmeasured; normal gradient assumed zero",
        normal_ACDC_slope="calibrated 3D response endpoint difference across the full slice",
        current_role="deterministic projected diagnostic currents, not optimized commands")
    artifacts = _write_result(output_dir, "local_q", dict(q_baseline=parent_maps(q0),
        q_dynamic=parent_maps(qdynamic), sample_indices=selected,
        sample_time_s=times[selected], integrated_currents_As=integrated, currents_A=currents,
        X_m=X_m, Y_m=Y_m), metrics)
    return dict(artifacts=artifacts, metrics=metrics)


def run_reconstruction_comparators(context, output_dir, *, warmup=1, repeats=3):
    """Time actual-k direct Fourier adjoint and forced nominal-grid IFFT.

    Both consume the same stored complex 08 OFF signal. This comparison does
    not relabel the production 08V2 matrix least-squares inverse as an adjoint.
    Operator construction and input tensor preparation are outside the timers.
    """
    import continuous_reconstruction
    from acdc_gpu import select_device
    output_dir = Path(output_dir)
    with np.load(output_dir / "equivalence.npz", allow_pickle=False) as archive:
        baseline_kdata = np.asarray(archive["baseline_kdata08"])
    options = context["model_options"]
    device, dtype, _ = select_device(options.get("profile", "cpu"),
        options.get("precision", "float64"), require_cuda=options.get("require_cuda", False))
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    signal = torch.as_tensor(baseline_kdata, device=device, dtype=complex_dtype)
    acquisition = dict(context["prepared"]["expanded_experiment"]["acquisition"], _skip_matrix_dft=True)
    previous = (continuous_reconstruction.FORCE_DIRECT_DFT, continuous_reconstruction.MATRIX_DFT_ENABLED)
    continuous_reconstruction.FORCE_DIRECT_DFT = False
    continuous_reconstruction.MATRIX_DFT_ENABLED = False
    try:
        automatic = FrameReconstruction(acquisition, device, dtype,
            frame_matrix=context["frame_matrix"], sample_chunk=int(context.get("reconstruction_chunk", 256)))
        direct = FrameReconstruction(acquisition, device, dtype,
            frame_matrix=context["frame_matrix"], sample_chunk=int(context.get("reconstruction_chunk", 256)))
    finally:
        continuous_reconstruction.FORCE_DIRECT_DFT, continuous_reconstruction.MATRIX_DFT_ENABLED = previous
    direct.base.method = "direct_chunked"
    direct.base._offset_phase = direct.base._row_x = direct.base._row_y = None
    direct.base.report.update(method="direct_chunked", evaluation="forced positive-sign Fourier adjoint at actual ADC k")
    functions = {
        "actual_k_direct_adjoint": lambda: direct.image(signal),
        "forced_nominal_grid_IFFT": lambda: torch.fft.fftshift(torch.fft.ifft2(
            torch.fft.ifftshift(signal, dim=(-2, -1))), dim=(-2, -1)),
    }
    records, arrays = [], dict(kdata=baseline_kdata, X_m=automatic.X_m, Y_m=automatic.Y_m)
    dummy = SimpleNamespace(device=device)
    for label, function in functions.items():
        print(f"Reconstruction-only comparator benchmark: {label}", flush=True)
        timing = _benchmark_callable(function, dummy, warmup, repeats)
        records.append(dict(generation=f"comparator_{label}",
            operation=f"reconstruction_only_{label}", **timing,
            includes_file_IO=False, includes_model_construction=False))
        arrays[f"{label}_image"] = function().detach().cpu().numpy()
    automatic_image = automatic.image(signal).detach().cpu().numpy()
    arrays["automatic_actual_k_image"] = automatic_image
    metrics = dict(experiment="FFT versus actual-k direct-adjoint timing", records=records,
        provenance=provenance(context), same_complex_kspace=True, same_physical_output_grid=True,
        automatic_method=automatic.report, direct_method=direct.base.report,
        automatic_vs_direct_complex_relative_L2=relative_error(
            arrays["actual_k_direct_adjoint_image"], automatic_image),
        forced_grid_vs_actual_direct_complex_relative_L2=relative_error(
            arrays["forced_nominal_grid_IFFT_image"], arrays["actual_k_direct_adjoint_image"]),
        forced_grid_vs_actual_direct_magnitude_NRMSE_percent=100 * relative_error(
            np.abs(arrays["forced_nominal_grid_IFFT_image"]), np.abs(arrays["actual_k_direct_adjoint_image"])),
        normalization="1/N_ADC in both operators", input_tensor_preparation_timed=False,
        forced_grid_policy="ignore actual coordinate offsets; assign nominal centered Cartesian bins",
        production_08V2_role="separate matrix least-squares inverse, not this direct-adjoint comparator")
    artifacts = _write_result(output_dir, "reconstruction_comparators", arrays, metrics)
    runtime_path = output_dir / "runtime_benchmarks.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        runtime["records"] = [row for row in runtime["records"]
            if not str(row["generation"]).startswith("comparator_")] + records
        runtime["reconstruction_comparators"] = metrics
        runtime_path.write_text(json.dumps(json_safe(runtime), indent=2), encoding="utf-8")
        with (output_dir / "runtime_benchmarks.csv").open("w", newline="") as stream:
            fields = ["generation", "operation", "median_seconds", "q25_seconds", "q75_seconds", "IQR_seconds", "warmup_count", "repetitions", "peak_gpu_allocated_bytes"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(runtime["records"])
    return dict(artifacts=artifacts, metrics=metrics)


def run_controlled_experiments(context_or_path, output_dir, *, run_expensive=True,
                               warmup=1, repeats=3, setup_repeats=3):
    context = load_context(context_or_path) if isinstance(context_or_path, (str, Path)) else context_or_path
    output_dir = Path(output_dir)
    results, models = {}, None
    if run_expensive:
        benchmark = run_runtime_benchmarks(context, output_dir, warmup=warmup,
            repeats=repeats, setup_repeats=setup_repeats)
        models = benchmark.pop("models")
        results["runtime"] = benchmark
    results["equivalence"] = run_equivalence(context, output_dir, models=models)
    results["local_q"] = export_local_q_maps(context, output_dir)
    if run_expensive:
        results["reconstruction_comparators"] = run_reconstruction_comparators(
            context, output_dir, warmup=warmup, repeats=repeats)
    if run_expensive:
        results["archived_replay"] = run_archived_optimization_replay(context, output_dir,
            model08=None if models is None else models["08"])
    results["uniform"] = run_uniform_phantom(context, output_dir)
    if run_expensive:
        results["adc_window"] = run_adc_window(context, output_dir,
            model08=None if models is None else models["08"])
    else:
        results["adc_window"] = dict(status="pending", reason="RUN_EXPENSIVE=False",
            generation_command="run_adc_window(context, output_dir)")
    (output_dir / "controlled_experiments_summary.json").write_text(
        json.dumps(json_safe(results), indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("context", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--setup-repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2,
        help="PyTorch CPU thread count; recorded in provenance (default: 2).")
    parser.add_argument("--supplement-only", action="store_true",
        help="Add q maps and reconstruction comparators to an existing controlled run.")
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    if arguments.supplement_only:
        context = load_context(arguments.context)
        q_result = export_local_q_maps(context, arguments.output_dir)
        comparison_result = run_reconstruction_comparators(context, arguments.output_dir,
            warmup=arguments.warmup, repeats=arguments.repeats)
        summary_path = arguments.output_dir / "controlled_experiments_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summary.update(local_q=q_result, reconstruction_comparators=comparison_result)
        runtime_path = arguments.output_dir / "runtime_benchmarks.json"
        if runtime_path.exists() and "runtime" in summary:
            summary["runtime"]["metrics"] = json.loads(runtime_path.read_text())
        summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
        (arguments.output_dir / "runtime_benchmarks_partial.json").write_text(
            json.dumps(dict(status="complete", full_metrics=str(runtime_path)), indent=2), encoding="utf-8")
    else:
        run_controlled_experiments(arguments.context, arguments.output_dir,
            run_expensive=not arguments.fast, warmup=arguments.warmup,
            repeats=arguments.repeats, setup_repeats=arguments.setup_repeats)
