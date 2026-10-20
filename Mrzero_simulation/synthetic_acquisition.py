"""Generate raw MRzero ADCs independently of the NativeEPITwin tested by V4.

Pulseq supplies actual ADC positions, center times and RF/receiver phases.
Model: stationary 2D point spins, independent ideal 90-degree excitations,
mono-exponential R2*, static B0, no ACDC. No slice profile or T1 steady state.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
import torch

try:
    from .synthetic_phantom import Phantom2D, make_phantom, save_ground_truth
    from .synthetic_twix import write_synthetic_twix
except ImportError:
    from synthetic_phantom import Phantom2D, make_phantom, save_ground_truth
    from synthetic_twix import write_synthetic_twix


@dataclass(frozen=True)
class SimulationConfig:
    """Explicit settings, with units in parameter names."""
    object_matrix: int = 128
    receiver_count: int = 1
    b0_scale: float = 1.0
    noise_std_image: float = 0.0005  # Std of EACH real/imaginary image component.
    seed: int = 15
    device: str = "auto"
    adc_chunk: int = 256
    copy_calibration: bool = True


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _one(folder, pattern):
    paths = list(folder.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} in {folder}; found {len(paths)}.")
    return paths[0]


def read_adc_samples(path):
    """Read ALL ADCs in chronology, including navigators and B0 echoes.

    k_mInv: [ADC, readout, xyz], elapsed_s/phase_rad: [ADC, readout].
    rf_index identifies the most recent excitation, including leading dummies.
    """
    import pypulseq as pp
    sequence = pp.Sequence()
    sequence.read(str(path))
    k, _, rf_times, refocusing, adc_times = sequence.calculate_kspace()
    if len(refocusing):
        raise ValueError("Only excitation RF histories are supported.")
    rf_times = np.asarray(rf_times, dtype=float)
    rf_phases, adc_phases, adc_lengths, adc_dwells = [], [], [], []
    for number in sequence.block_events:
        block = sequence.get_block(number)
        if block.rf is not None:
            rf_phases.append(float(block.rf.phase_offset))
        if block.adc is not None:
            if abs(float(block.adc.freq_offset)) > 1e-9:
                raise ValueError("ADC frequency offsets need an additional receiver model.")
            adc_phases.append(float(block.adc.phase_offset))
            adc_lengths.append(int(block.adc.num_samples))
            adc_dwells.append(float(block.adc.dwell))
    if not adc_lengths or len(set(adc_lengths)) != 1 or len(rf_phases) != len(rf_times):
        raise ValueError("Expected excitation RF and constant readout length.")
    nx = adc_lengths[0]
    times = np.asarray(adc_times).reshape(-1, nx)
    indices = np.searchsorted(rf_times, times[:, 0], side="right") - 1
    last = np.searchsorted(rf_times, times[:, -1], side="right") - 1
    if np.any(indices < 0) or not np.array_equal(indices, last):
        raise ValueError("Each ADC must belong to one preceding excitation.")
    elapsed = times - rf_times[indices, None]
    trajectory = np.moveaxis(np.asarray(k).reshape(3, -1, nx), 0, -1)
    if not np.isfinite(trajectory).all() or np.max(abs(trajectory[..., 2])) > 1e-3:
        raise ValueError("Expected a finite 2D trajectory with kz=0.")
    if np.any(elapsed <= 0):
        raise ValueError("Every ADC sample must follow its RF center.")
    phases = np.asarray(rf_phases)[indices] - np.asarray(adc_phases)
    return dict(k_mInv=trajectory, elapsed_s=elapsed, rf_index=indices,
        phase_rad=np.repeat(phases[:, None], nx, axis=1),
        FOV_m=np.asarray(sequence.definitions["FOV"], float)[:2],
        matrix_size=np.asarray(sequence.definitions["Matrix"], int),
        TE_s=np.atleast_1d(sequence.definitions["TE"]).astype(float),
        TR_s=float(sequence.definitions["TR"]),
        navigator_count=int(sequence.definitions.get("NavigatorEchoes", 0)),
        dwell_s=float(adc_dwells[0]),
        sequence_sha256=sha256(path))


def simulate_adc(phantom, samples, *, device="auto", adc_chunk=256):
    """Native MRzero output [receiver, ADC, readout], bounded in memory.

    A block jumps from one ideal RF to its first requested sample. Splitting
    evaluates the SAME single-RF history at requested times; relaxation and
    accumulated B0 phase are never restarted at an echo or chunk boundary.
    """
    import MRzeroCore as mr0
    if isinstance(adc_chunk, bool) or int(adc_chunk) != adc_chunk or adc_chunk < 1:
        raise ValueError("adc_chunk must be a positive integer.")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    version = tuple(int(part) for part in mr0.__version__.split(".")[:2])
    if version < (1, 0):
        raise RuntimeError("Requires MRzeroCore >=1.0: negative Fourier/B0 phase.")
    mask = phantom.support_mask
    positions = np.column_stack((phantom.X_m[mask], phantom.Y_m[mask], np.zeros(mask.sum())))
    image_area = np.prod(samples["FOV_m"]) / np.prod(samples["matrix_size"])
    object_area = np.prod(phantom.FOV_m) / phantom.proton_density.size
    area_weight = float(object_area / image_area)
    pd = phantom.proton_density[mask] * area_weight
    rates = phantom.R2star_sInv[mask]
    t2 = np.divide(1., rates, out=np.full_like(rates, np.inf), where=rates > 0)
    data = mr0.CustomVoxelPhantom(positions.astype(np.float32), PD=pd.astype(np.float32),
        B0=phantom.B0_Hz[mask].astype(np.float32), B1=1., T1=np.inf,
        T2=t2.astype(np.float32), T2dash=np.inf, D=0., voxel_size=0., voxel_shape="box").build()
    if device.type == "cuda":
        with torch.cuda.device(device):
            data = data.cuda()
    data.D = torch.zeros_like(data.D)  # SimData otherwise clamps D above zero.
    data.dephasing_func = lambda k, _: torch.ones(k.shape[0], device=k.device, dtype=k.dtype)
    data.coil_sens = torch.as_tensor(phantom.receive_weights[:, mask], device=device, dtype=torch.complex64)
    ncoils = data.coil_sens.shape[0]
    shape = samples["elapsed_s"].shape
    total = int(np.prod(shape))
    output = np.empty((total, ncoils), dtype=np.complex64)
    k = samples["k_mInv"].reshape(-1, 3)
    elapsed, phase = samples["elapsed_s"].ravel(), samples["phase_rad"].ravel()
    histories = np.repeat(samples["rf_index"], shape[1])
    started, block_count = perf_counter(), 0
    for history in np.unique(histories):
        chronological = np.flatnonzero(histories == history)
        chronological = chronological[np.argsort(elapsed[chronological], kind="stable")]
        for first in range(0, len(chronological), int(adc_chunk)):
            selected = chronological[first:first + int(adc_chunk)]
            durations = np.diff(np.r_[0., elapsed[selected]])
            if np.any(durations <= 0):
                raise ValueError("Distinct ADC times are required within each RF history.")
            repetition = mr0.Repetition.zero(len(selected))
            repetition.pulse.angle = torch.tensor(np.pi / 2, dtype=torch.float32)
            repetition.pulse.phase = torch.tensor(np.pi / 2, dtype=torch.float32)
            repetition.pulse.usage = mr0.PulseUsage.EXCIT
            repetition.pulse.selective = False
            repetition.event_time[:] = torch.from_numpy(durations.astype(np.float32))
            moments = np.diff(np.vstack((np.zeros(3), k[selected])), axis=0)
            repetition.gradm[:] = torch.from_numpy(moments.astype(np.float32))
            repetition.adc_usage[:] = 1
            repetition.adc_phase[:] = torch.from_numpy(phase[selected].astype(np.float32))
            sequence = mr0.Sequence([repetition], normalized_grads=False)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    sequence = sequence.cuda()
            with redirect_stdout(io.StringIO()), torch.no_grad():
                graph = mr0.compute_graph(sequence, data, max_state_count=16, min_state_mag=0.)
                signal = mr0.execute_graph(graph, sequence, data, min_emitted_signal=0.,
                    min_latent_signal=0., print_progress=False)
            output[selected] = signal.detach().cpu().numpy()
            block_count += 1
    if not np.isfinite(output).all():
        raise RuntimeError("MRzero returned nonfinite samples.")
    # Independent double-precision equation on 13 samples, BEFORE adding noise.
    centers = np.argsort(np.sum(k * k, axis=1))[:3]
    selected = np.unique(np.r_[np.linspace(0, total - 1, 13, dtype=int), centers])
    exponent = (-rates[:, None] * elapsed[selected][None, :]
        - 2j * np.pi * (positions @ k[selected].T
            + phantom.B0_Hz[mask][:, None] * elapsed[selected][None, :]))
    expected = phantom.receive_weights[:, mask] @ (pd[:, None] * np.exp(exponent))
    expected *= np.exp(1j * phase[selected])[None, :]
    error = float(np.linalg.norm(output[selected].T - expected) / max(np.linalg.norm(expected), 1e-30))
    if not np.isfinite(error) or error > 2e-4:
        raise RuntimeError(f"Independent point-spin check failed ({error:.3g}).")
    report = dict(simulator="MRzeroCore", mrzero_version=mr0.__version__, device=str(device),
        native_precision="float32", active_spins=int(mask.sum()), receiver_count=int(ncoils),
        adc_count=int(shape[0]), readout_samples=int(shape[1]), blocks=block_count,
        adc_chunk=int(adc_chunk), area_weight=area_weight,
        analytic_check_samples=int(len(selected)), analytic_relative_error=error,
        elapsed_s=perf_counter() - started)
    return output.T.reshape((ncoils,) + shape), report


def ordered_kspace(signal, samples, kind):
    """Reflect negative readouts, then sort phase rows exactly once."""
    result, k = signal.copy(), samples["k_mInv"].copy()
    negative = np.mean(np.diff(k[..., 0], axis=1), axis=1) < 0
    result[:, negative] = result[:, negative, ::-1]
    k[negative] = k[negative, ::-1]
    nav_count = samples["navigator_count"] if kind == "scan" else 0
    navigators = np.moveaxis(result[:, :nav_count], 0, 1)
    result, k = result[:, nav_count:], k[nav_count:]
    nx, ny = samples["matrix_size"]
    if kind == "b0":
        ne = len(samples["TE_s"])
        result = result.reshape(result.shape[0], ny, ne, nx)
        k = k.reshape(ny, ne, nx, 3)
        order = np.argsort(k[:, 0, :, 1].mean(axis=1))
        return np.transpose(result[:, order], (2, 0, 1, 3)), navigators
    if result.shape[1:] != (ny, nx):
        raise ValueError(f"Unexpected {kind} ADC count after removing navigators.")
    order = np.argsort(k[..., 1].mean(axis=1))
    return np.moveaxis(result[:, order], 0, 1), navigators


def verify_acquisition_files(folder, manifest):
    """Check the three native DAT files and all declared checksums before reuse."""
    folder = Path(folder).resolve()
    if manifest.get("format") != "mrzero_synthetic_twix" or manifest.get("schema_version") != 2:
        raise ValueError("Expected a synthetic Twix acquisition (schema 2); choose a new dataset name.")
    for name, relative in manifest["files"].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError(f"Missing file or path outside acquisition: {relative}")
        if name in ("reference", "b0", "scan") and path.suffix.lower() != ".dat":
            raise ValueError(f"The acquired {name} signal must be a native Twix .dat file.")
        if sha256(path) != manifest["files_sha256"][name]:
            raise ValueError(f"SHA-256 mismatch for {relative}")
    return True


def generate_acquisition(source_folder, output_folder, config=None):
    """Create a NEW acquisition with three native Siemens VD/VE Twix DATs.

    Only .seq files and optionally ACDC_3T.npz are copied from the example.
    All ADC samples and all Twix headers are synthetic. Ground truth is stored
    separately in simulation_metadata/<dataset>/, outside the acquisition.
    """
    config = SimulationConfig() if config is None else config
    if not np.isfinite(config.noise_std_image) or config.noise_std_image <= 0:
        raise ValueError("Positive noise_std_image required by V4's corner noise estimator.")
    source, output = Path(source_folder).resolve(), Path(output_folder).resolve()
    metadata_dir = output.parent / "simulation_metadata" / output.name
    if output.exists() or metadata_dir.exists():
        raise FileExistsError(f"Choose a new dataset name: {output} or its metadata already exists.")
    sequence_paths = dict(scan_sequence=_one(source, "sequences/*epi*.seq"),
        reference_sequence=_one(source, "sequences/gre_32*.seq"),
        b0_sequence=source / "sequences/b0_map.seq")
    samples = {name: read_adc_samples(path) for name, path in sequence_paths.items()}
    fov = samples["scan_sequence"]["FOV_m"]
    if not np.allclose(fov, fov[0]) or any(not np.allclose(g["FOV_m"], fov) for g in samples.values()):
        raise ValueError("Requires the same square FOV for all three acquisitions.")
    phantom = make_phantom(config.object_matrix, float(fov[0]),
                           b0_scale=config.b0_scale, ncoils=config.receiver_count)
    calibration = source / "ACDC_3T.npz"
    if config.copy_calibration and not calibration.is_file():
        raise FileNotFoundError(f"Missing calibration: {calibration}")
    output.mkdir(parents=True)
    (output / "sequences").mkdir()
    metadata_dir.mkdir(parents=True)
    files = {}
    for name, path in sequence_paths.items():
        target = output / "sequences" / path.name
        shutil.copy2(path, target)
        files[name] = target.relative_to(output).as_posix()
    save_ground_truth(metadata_dir / "ground_truth.npz", phantom)
    rng, reports = np.random.default_rng(config.seed), {}
    for kind in ("reference", "b0", "scan"):
        print(f"MRzero -> Twix DAT: {kind}", flush=True)
        sample_key = {"reference": "reference_sequence", "b0": "b0_sequence", "scan": "scan_sequence"}[kind]
        geom = samples[sample_key]
        signal, report = simulate_adc(phantom, geom, device=config.device, adc_chunk=config.adc_chunk)
        # Set a consistent real/imaginary image noise std across acquired matrices.
        noise_std_kspace = config.noise_std_image * np.sqrt(np.prod(geom["matrix_size"]))
        noise = rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape)
        signal += (noise_std_kspace * noise).astype(np.complex64)
        filename = {"reference": "reference.dat", "b0": "b0_map.dat", "scan": "scan.dat"}[kind]
        # Native ADC chronology goes into MDHs. REFLECT reordering belongs to the reader.
        twix_report = write_synthetic_twix(output / filename, signal, geom, kind)
        files[kind] = filename
        report.update(noise_std_real_imag_kspace=float(noise_std_kspace),
            target_noise_std_real_imag_image=config.noise_std_image,
            TE_s=geom["TE_s"].tolist(), TR_s=geom["TR_s"], FOV_m=geom["FOV_m"].tolist(),
            stored_shape=list(signal.shape), navigator_count=int(geom["navigator_count"]),
            output_format="Siemens VD/VE Twix", twix_export=twix_report)
        reports[kind] = report
        print(f"  {filename}: {signal.shape}, {report['elapsed_s']:.1f} s; analytic error {report['analytic_relative_error']:.2g}", flush=True)
    if config.copy_calibration:
        shutil.copy2(calibration, output / calibration.name)
        files["calibration"] = calibration.name
    manifest = dict(format="mrzero_synthetic_twix", schema_version=2,
        created_utc=datetime.now(timezone.utc).isoformat(), files=files,
        files_sha256={name: sha256(output / relative) for name, relative in files.items()},
        source_example=str(source), source_sequence_sha256={name: sha256(path) for name,path in sequence_paths.items()},
        config=asdict(config), python_executable=sys.executable,
        expected_environment=".venv-gpu-win/Scripts/python.exe", reports=reports,
        metadata_directory=f"../simulation_metadata/{output.name}",
        stored_axes="Each Twix MDB stores [receiver, readout] in native ADC chronology.",
        model=dict(RF="independent ideal 90-degree, real initial transverse magnetization",
            amplitude="effective amplitude at RF; pixel-area weighted per acquisition",
            phase="exp(-2*pi*i*(k dot r+B0*time_since_RF))*exp(+i*(RFphase-ADCphase))",
            relaxation="mono-exponential R2star; same maps in all acquisitions",
            slice="2D integrated 5-mm slice; no through-slice profile", ACDC_currents_A=0,
            navigator_correction="Native Twix reader applies its usual three-navigator estimate; no odd/even mismatch injected.",
            omitted=["RF waveform and nominal flip contrast", "T1 saturation/steady state",
                "motion", "gradient delay/eddy currents", "diffusion", "through-slice B0 dispersion"]),
        ground_truth_used_by_pipeline=False, calibration_used_for_simulation=False,
        synthetic_headers=True, real_Twix_data_or_headers_copied=False)
    (output / "acquisition.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    verify_acquisition_files(output, manifest)
    return output, manifest
