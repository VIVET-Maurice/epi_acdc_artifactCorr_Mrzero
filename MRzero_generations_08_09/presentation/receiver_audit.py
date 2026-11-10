"""Read-only receiver audit and data-derived legacy diagnostic figures.

This module never adds a receive model to an existing EPI simulation. Measured
receiver images and fitted complex echo amplitudes are inspected separately
from the receiver encoding actually used by the EPI forward model.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
from scipy import ndimage


LEGACY_RUN = "mrzero06_20261103_121718_079562"


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root(project_root=None):
    return Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]


def _write_report(output_dir, section, report):
    path = Path(output_dir) / "legacy_and_receiver_audit.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[section] = _safe(report)
    path.write_text(json.dumps(existing, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _save_figure(figure, stem):
    paths = []
    for extension in ("png", "svg", "pdf"):
        path = Path(stem).with_suffix("." + extension)
        figure.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path.resolve()))
    return paths


def intensity_pattern_metrics(image, mask):
    """Describe spatial variation without identifying it as noise or a cause.

    CV = population standard deviation / mean. Adjacent roughness is the RMS
    difference of horizontal/vertical adjacent pixels whose BOTH endpoints
    lie inside the fixed mask, divided by the same within-mask mean. No
    spatial smoothing or detrending is applied. Contrast also affects both.
    """
    image = np.abs(np.asarray(image, dtype=np.complex128)).astype(float)
    mask = np.asarray(mask, dtype=bool)
    if image.shape != mask.shape or not mask.any() or not np.isfinite(image[mask]).all():
        raise ValueError("Pattern metrics require a nonempty fixed mask and finite matching image.")
    mean = float(image[mask].mean())
    if mean <= 0:
        raise ValueError("Pattern metrics require positive within-mask mean magnitude.")
    differences = []
    for axis in (0, 1):
        lower = [slice(None), slice(None)]
        upper = [slice(None), slice(None)]
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        lower, upper = tuple(lower), tuple(upper)
        both = mask[lower] & mask[upper]
        differences.append((image[upper] - image[lower])[both])
    differences = np.concatenate(differences)
    if not differences.size:
        raise ValueError("Pattern metrics require at least one pair of adjacent masked pixels.")
    return dict(pixels=int(mask.sum()), mean_magnitude=mean,
        std_magnitude=float(image[mask].std()), CV=float(image[mask].std() / mean),
        normalized_adjacent_RMS=float(np.sqrt(np.mean(differences**2)) / mean),
        adjacent_pairs=int(differences.size),
        interpretation="Descriptive spatial variation; includes object contrast and remaining boundary/window effects; no noise or causal model.")


def generate_legacy_pattern_figure(project_root=None, output_dir=None):
    """Regenerate slide 2 from exact legacy06V3 numerical output arrays."""
    root = _root(project_root)
    output = Path(output_dir or root / "MRzero_generations_08_09/presentation").resolve()
    figures, data = output / "figures", output / "data"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    run = root / "MRzero_validation/results_experiment_echo32" / LEGACY_RUN
    source = run / "mrzero06_optimization.npz"
    summary_path = run / "mrzero06_optimization.json"
    summary = json.loads(summary_path.read_text())
    with np.load(source, allow_pickle=False) as archive:
        baseline = np.asarray(archive["uncalibrated_baseline_image"], dtype=np.complex128)
        zero = np.asarray(archive["uncalibrated_ideal_image"], dtype=np.complex128)
        measured = np.asarray(archive["measured_image"], dtype=float)
        X, Y = np.asarray(archive["X_image_m"]), np.asarray(archive["Y_image_m"])
        support = np.asarray(archive["parent_valid_mask"], dtype=bool)
        kdata = np.asarray(archive["uncalibrated_baseline_kdata"])
    full_mask = np.isfinite(baseline) & np.isfinite(zero) & np.isfinite(measured)
    gain = float(np.sum(np.abs(baseline[full_mask]) * measured[full_mask])
                 / np.sum(np.abs(baseline[full_mask])**2))
    images = np.stack((gain * abs(zero), gain * abs(baseline), measured))
    pattern_mask = ndimage.binary_erosion(support, structure=np.ones((3, 3)), iterations=2)
    pattern_mask &= full_mask
    image_names = ["legacy_zero_B0", "legacy_measured_B0", "measured_OFF_RSS"]
    patterns = {name: intensity_pattern_metrics(image, pattern_mask)
                for name, image in zip(image_names, images)}
    nrmse = float(100 * np.linalg.norm((images[1] - measured)[full_mask])
                  / np.linalg.norm(measured[full_mask]))
    hashes = {"source_archive": _sha256(source), "source_summary": _sha256(summary_path)}
    checks = {}
    for name, expected in summary.get("source_sha256", {}).items():
        candidate = (root / "data/ACDC_3T.npz" if name == "calibration"
                     else Path(summary["input_selection"][name]))
        if not candidate.exists() and name != "calibration":
            # Preserve the archived identity when the whole project has moved;
            # acceptance still requires the exact recorded content hash.
            relative_folder = Path(summary["input_dir"]).name
            candidate = root / relative_folder / ("sequences" if "sequence" in name else "") / candidate.name
        if not candidate.exists():
            checks[name] = dict(path=str(candidate), exists=False, matches_saved_hash=False)
        else:
            actual = _sha256(candidate)
            checks[name] = dict(path=str(candidate), exists=True, actual_sha256=actual,
                                saved_sha256=expected, matches_saved_hash=actual == expected)
    notebook = root / "MRzero_validation/06_acdc_echo32_b0_s0_t2star_MRzeroV3.ipynb"
    notebook_data = json.loads(notebook.read_text())
    text_run_cells = []
    for index, cell in enumerate(notebook_data["cells"]):
        for result in cell.get("outputs", []):
            text = "".join(result.get("text", []))
            text += "".join(result.get("data", {}).get("text/plain", []))
            if LEGACY_RUN in text:
                text_run_cells.append(index)
    report = dict(stage="archived uncalibrated 06V3 baseline; not current 08/09",
        notebook=str(notebook), notebook_text_output_run_cells=sorted(set(text_run_cells)),
        numerical_source=str(source), source_summary=str(summary_path), source_sha256=hashes,
        input_hash_verification=checks, archived_runtime=summary.get("runtime"),
        inputs=summary.get("input_selection"), orientation=summary.get("orientation"),
        archived_helper_code_hashes_available=False,
        provenance_qualification="The current 06V3 notebook's textual outputs identify this archived run. Input-file hashes match. Its original simulation helper hashes were not recorded, so the archived simulation is not claimed to be a fresh execution of today's helpers.",
        gain=gain, gain_definition="Least squares scalar fitted once: sum(abs(baseline)*measured)/sum(abs(baseline)^2), all finite FOV pixels; same gain applied to B0=0 image.",
        metric_precision="Float64 magnitude metrics recomputed from archived complex64 simulation and float32 measured arrays.",
        measured_OFF_NRMSE_percent=nrmse,
        NRMSE_mask="All finite output FOV pixels, fixed across all images.",
        NRMSE_pixels=int(full_mask.sum()),
        pattern_mask="Legacy retained parent support eroded by two 3x3 binary erosions; same fixed mask for all panels.",
        pattern_metrics=patterns, through_plane_children=summary["preparation"]["children_per_parent"],
        B0_zero_policy="Remove measured macroscopic B0 centers and known in-plane slopes; retain fitted residual relaxation and unresolved coherence.",
        causal_status="The PI observation is a hypothesis/visual description. These descriptive metrics do not identify a mechanism or establish that B0=0 removes all spatial variation.")
    archive_path = data / "legacy06_diagnostic.npz"
    np.savez_compressed(archive_path, images=images, labels=np.array(image_names),
        X_m=X, Y_m=Y, pattern_mask=pattern_mask, full_FOV_mask=full_mask,
        baseline_complex=baseline, zero_B0_complex=zero, baseline_kdata=kdata,
        gain=np.asarray(gain))
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.65), sharex=True, sharey=True,
                             layout="constrained")
    vmax = float(images.max())
    titles = ["Legacy 06 V3: B0 = 0", "Legacy 06 V3: measured B0", "Measured EPI OFF: receiver RSS"]
    for ax, image, title, name in zip(axes, images, titles, image_names):
        artist = ax.pcolormesh(X * 1000, Y * 1000, image, shading="auto", cmap="gray", vmin=0, vmax=vmax)
        ax.contour(X * 1000, Y * 1000, pattern_mask.astype(float), levels=[.5],
                   colors="#2478b7", linewidths=.55)
        ax.set(title=title, xlabel="Common-frame x [mm]", aspect="equal")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        stats = patterns[name]
        ax.text(.02, .02, f"CV {stats['CV']:.3f}\nAdjacent RMS/mean {stats['normalized_adjacent_RMS']:.3f}",
                transform=ax.transAxes, fontsize=9, color="white", va="bottom",
                bbox=dict(facecolor="black", alpha=.65, edgecolor="none", pad=2))
    axes[0].set_ylabel("Common-frame y [mm]")
    fig.colorbar(artist, ax=axes, shrink=.8, label="Magnitude [measured EPI a.u.; fixed gain]")
    fig.suptitle("Archived 06 V3, not current 08/09: B0 diagnostic and measured OFF", fontsize=13)
    fig.supxlabel(f"Acquisition 029; same physical axes/scale; fixed baseline gain {gain:.5g}; OFF NRMSE {nrmse:.2f}% over full FOV.\nBlue contour: fixed eroded support. CV/adjacent differences include anatomical contrast; no smoothing or noise claim.", fontsize=9)
    artifacts = _save_figure(fig, figures / "slide02_legacy_pattern")
    plt.close(fig)
    report["figure_files"] = artifacts
    report["data_archive"] = str(archive_path)
    _write_report(output, "legacy06", report)
    metrics = [dict(experiment="legacy06", metric="OFF_NRMSE_percent", value=nrmse, units="%")]
    for name, values in patterns.items():
        for key in ("CV", "normalized_adjacent_RMS"):
            metrics.append(dict(experiment=name, metric=key, value=values[key], units="dimensionless"))
    return dict(report=_safe(report), artifacts=artifacts, metrics=metrics)


def _source_evidence(root):
    specifications = [
        ("MRzero_generations_08_09/complex_echo_fit.py", "envelope = coil_amplitude.abs().square().sum", "The multi-echo fit computes an effective RSS envelope from separate complex receiver amplitudes."),
        ("MRzero_generations_08_09/echo_shim_inputs.py", 'amplitude = complex_maps["amplitude_envelope_at_reference"]', "EPI object preparation passes the scalar envelope, not complex receiver directions."),
        ("MRzero_generations_08_09/experiment_echo.py", "phantom = self._mr0.CustomVoxelPhantom(", "The EPI phantom constructor provides no measured receive maps or coil_sens argument."),
        ("MRzero_generations_08_09/experiment_mrzero.py", "magnetization = pathways[0][0].detach() * phantom.coil_sens[0]", "Generation 08 uses receiver index zero when forming native center contributions."),
        ("MRzero_generations_08_09/native_b0_model.py", "magnetization = pathways[0][0] * phantom.coil_sens[0]", "Generation 09 also uses receiver index zero."),
        ("MRzero_generations_08_09/acdc_gpu.py", "signal = torch.empty(self.nsamples", "The inherited EPI forward allocates a single ADC vector, then sums voxel contributions."),
        ("MRzero_generations_08_09/measured_continuous.py", "image = np.sqrt(np.sum(np.abs(coils)**2, axis=1))", "Measured data are reconstructed per receiver before RSS combination."),
        ("MRzero_generations_08_09/read_acdc_b0.py", "channel_ids = receiver_channel_ids", "The B0 reader explicitly validates physical receiver IDs."),
        ("MRzero_validation/spiral_receive.py", 'if inputs["acquisition"].get("sequence_kind") != "spiral":', "A separate receiver-transfer implementation rejects EPI inputs."),
        ("MRzero_validation/spiral_receive.py", "parent_weights = selected / norm[None] * phase[None]", "Spiral receiver weights have unit RSS norm and restore the echo-reference phase."),
        ("MRzero_validation/experiment_spiral.py", "signal[rx, selected] = self._receive_weights[rx] @ contribution", "The spiral-specific forward actually emits separate receiver ADC signals."),
    ]
    evidence = []
    for relative, needle, conclusion in specifications:
        path = root / relative
        lines = path.read_text().splitlines()
        matches = [i + 1 for i, line in enumerate(lines) if needle in line]
        evidence.append(dict(path=str(path), lines=matches, searched_text=needle,
                             sha256=_sha256(path), conclusion=conclusion, found=bool(matches)))
    return evidence


def _raw_receiver_ids(path, label):
    """Inspect all image/navigator ChannelId headers without modifying readers."""
    import twixtools
    from receiver_channels import receiver_channel_ids
    with contextlib.redirect_stdout(io.StringIO()):
        scan = twixtools.read_twix(str(path), verbose=False)[-1]
    blocks = [block for block in scan["mdb"]
              if block.is_image_scan() or block.is_flag_set("PHASCOR")]
    if not blocks:
        raise ValueError(f"No receiver blocks found in {label}.")
    counts = {np.asarray(block.data).shape[0] for block in blocks}
    if len(counts) != 1:
        raise ValueError(f"Receiver dimension changes in {label}.")
    count = counts.pop()
    ids = receiver_channel_ids(blocks, count, label=label)
    return dict(count=int(count), physical_IDs=ids, inspected_blocks=len(blocks),
        IDs_source="Read-only audit of Twix ChannelId headers in native receiver-row order", path=str(path))


def run_receiver_audit(context, output_dir, *, project_root=None, read_twix_ids=True):
    """Audit the current EPI receiver path and plot measured receiver data.

    Required context: inputs, fit and prepared dictionaries. A model08/model
    object is optional for directly inspecting its native phantom. No new
    receiver-sensitive EPI simulation is implemented or executed here.
    """
    root = _root(project_root)
    helpers = root / "MRzero_generations_08_09"
    if str(helpers) not in sys.path:
        sys.path.insert(0, str(helpers))
    output = Path(output_dir).resolve()
    figures, data = output / "figures", output / "data"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    inputs, fit, prepared = context["inputs"], context["fit"], context["prepared"]
    channels = {}
    for label, key in (("EPI_OFF", "scan"), ("GRE_reference", "reference"), ("MEGRE", "b0")):
        if read_twix_ids:
            channels[label] = _raw_receiver_ids(inputs["paths"][key], label)
        else:
            metadata = (inputs["scan_metadata"] if key == "scan" else
                        inputs["b0_raw"]["metadata"] if key == "b0" else inputs["gre"].get("twix_metadata", {}))
            channels[label] = dict(count=metadata.get("receiver_count"), physical_IDs=metadata.get("receiver_channel_ids"),
                IDs_source="Current reader metadata; missing IDs remain unknown", path=str(inputs["paths"][key]))
    experiment = prepared["expanded_experiment"]
    from MRzeroCore import CustomVoxelPhantom
    # Verify the exact constructor's native default without creating a new
    # receive model: the EPI constructors omit coil_sens and therefore use this.
    default_phantom = CustomVoxelPhantom(np.array([[0., 0., 0.]], np.float32), PD=1.,
        B0=0., B1=1., T1=np.inf, T2=np.inf, T2dash=np.inf, D=0., voxel_size=0., voxel_shape="box").build()
    default_sens = default_phantom.coil_sens.detach().cpu().numpy()
    model = context.get("model08", context.get("model"))
    actual_sens = None
    if model is not None:
        native = getattr(model, "first_native_phantom", None)
        if native is None:
            old = dict(model._native_phantoms)
            native = model._phantom(True)
            model._native_phantoms.clear()
            model._native_phantoms.update(old)
        actual_sens = native.coil_sens.detach().cpu().numpy()
    evidence = _source_evidence(root)
    if not all(item["found"] for item in evidence):
        raise RuntimeError("A receiver audit source assertion changed; review the current implementation before reporting its status.")
    notebook_selection = {}
    for name, expected in (("08_acdc_echo32_b0_s0_t2star_MRzero.ipynb", "FiniteVoxel3DACDCModel"),
                           ("08_V2_acdc_echo32_b0_s0_t2star_MRzero.ipynb", "FiniteVoxel3DACDCModel"),
                           ("09_acdc_echo32_b0_s0_t2star_MRzero.ipynb", "NativeB0ACDCModel")):
        path = helpers / name
        notebook = json.loads(path.read_text())
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        selected = [line.strip() for line in code.splitlines() if line.strip().startswith("model_class=")]
        expected_assignment = "model_class=" + expected
        if selected != [expected_assignment] or any(token in code for token in ("receive_weights", "receiver_forward", "SpiralReceiver")):
            raise RuntimeError(f"{name} forward selection changed; receiver status needs a new forensic review.")
        notebook_selection[name] = dict(model_class=expected, sha256=_sha256(path),
            explicit_receiver_forward_calls=False)
    forward_has_receivers = "receive_weights" in experiment or hasattr(model, "receiver_forward")
    if forward_has_receivers:
        raise RuntimeError("Receiver-sensitive EPI code was detected; rerun the forensic review before reporting a controlled comparison.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    report = dict(git_SHA=revision, inputs={k: str(v) for k, v in inputs["paths"].items()},
        measured_channels=channels, ACDC_shim_channels=int(prepared["expanded_calibration"]["acdc_maps"].shape[-1]),
        ACDC_physical_output_map=context.get("configuration", {}).get("WAVEFORM_PARAMETERS", {}).get("channel_map"),
        ACDC_disabled_outputs=context.get("configuration", {}).get("DISABLED_CHANNELS"),
        current_EPI_forward_receiver_count=1,
        default_native_coil_sens_shape=list(default_sens.shape),
        default_native_coil_sens_all_one=bool(np.array_equal(default_sens, np.ones_like(default_sens))),
        actual_native_coil_sens_shape=None if actual_sens is None else list(actual_sens.shape),
        actual_native_coil_sens_all_one=None if actual_sens is None else bool(np.all(actual_sens == 1)),
        EPI_receive_sensitivity_estimation=False, EPI_per_receiver_forward=False,
        EPI_simulated_receiver_RSS=False, measured_receiver_RSS=True,
        current_reader_records_physical_IDs=dict(
            EPI_OFF="receiver_channel_ids" in inputs["scan_metadata"],
            GRE_reference="receiver_channel_ids" in inputs["gre"].get("twix_metadata", {}),
            MEGRE="receiver_channel_ids" in inputs["b0_raw"]["metadata"]),
        raw_receiver_ID_orders_equal=(channels["EPI_OFF"]["physical_IDs"] is not None
            and channels["EPI_OFF"]["physical_IDs"] == channels["GRE_reference"]["physical_IDs"]
            == channels["MEGRE"]["physical_IDs"]),
        B1_interpretation="The native B1=1 parameter controls transmit excitation (B1+); it is not a measured receive sensitivity (B1-).",
        current_model="One complex uniform receiver applied to a receive-weighted scalar effective object amplitude.",
        measured_echo_fit="Independent complex receiver amplitudes are fitted; their RSS envelope enters EPI. Fitted receiver directions/phases are not propagated to EPI.",
        sensitivity_identifiability="Complex echo amplitudes include effective tissue amplitude, reception and initial transverse phase; they are not isolated quantitative C_c maps.",
        controlled_EPI_receiver_comparison=dict(status="not implemented; pending", executed=False,
            reason="The current 08/08V2/09 EPI forward has no per-receiver encoding path. No new physical model was added to justify a comparison.",
            required_changes=["Verify and align physical receiver IDs across MEGRE and EPI.",
                "Transfer justified complex receive encoding with explicit cross-acquisition phase assumptions.",
                "Normalize receive vectors to unit RSS when multiplying an already RSS-weighted object, or estimate a separate object amplitude.",
                "Emit receiver-indexed ADC signals, reconstruct each receiver, then combine RSS; propagate its gradient consistently.",
                "Run matched OFF before/after controls using identical objects, support, trajectories and comparison gains."]),
        separate_spiral_feature=dict(implemented=True, imported_by_EPI_notebooks=False,
            files=[str(root / "MRzero_validation/spiral_receive.py"), str(root / "MRzero_validation/experiment_spiral.py")],
            interpretation="Effective unit-RSS receiver weights and coil-by-coil signal/RSS exist for the separate spiral pipeline only; not a demonstrated EPI improvement."),
        notebook_forward_selection=notebook_selection, source_evidence=evidence)
    arrays = {}
    if isinstance(fit.get("multi_echo"), dict) and "complex_receiver_amplitude" in fit["multi_echo"]:
        amplitudes = np.asarray(fit["multi_echo"]["complex_receiver_amplitude"])
        support = np.asarray(fit["multi_echo"]["support_mask"], dtype=bool)
        envelope = np.asarray(fit["multi_echo"]["amplitude_envelope_at_reference"])
        rss = np.sqrt(np.sum(abs(amplitudes)**2, axis=0))
        valid = support & np.isfinite(rss) & (rss > 0) & np.isfinite(envelope)
        envelope_error = float(np.linalg.norm((rss - envelope)[valid]) / np.linalg.norm(envelope[valid]))
        direction = amplitudes[:, valid] / rss[valid][None]
        unit_error = float(np.max(np.abs(np.sum(abs(direction)**2, axis=0) - 1)))
        report["amplitude_accounting"] = dict(
            verification_voxels=int(valid.sum()),
            verification_mask="Finite positive fitted multi-echo support; not a scanner-agreement ROI.",
            observed_RSS_complex_amplitude_vs_scalar_envelope_relative_error=envelope_error,
            unit_RSS_normalized_direction_max_error=unit_error,
            normalized_direction_used_in_EPI_forward=False,
            scalar_object_amplitude="Envelope at reference TE times native/EPI pixel area ratio times child volume fraction; PD restores only exponential reference-time decay.",
            current_receive_factor="Unit native coil_sens; scalar receive envelope enters once.",
            double_counting_warning="Multiplying this scalar envelope by unnormalized fitted A_c would apply the receive envelope twice. Unit-RSS normalization is required for a proposed encoding transfer; this audit does not implement that transfer.")
        arrays.update(fitted_complex_receiver_amplitudes=amplitudes, fitted_scalar_envelope=envelope,
                      fitted_RSS_envelope=rss, fit_support_mask=support)
    raw = inputs["b0_raw"]
    coils = np.asarray(raw["coil_images"])[0]
    ids = channels["MEGRE"]["physical_IDs"]
    shown = np.arange(min(4, len(coils)))
    arrays.update(measured_MEGRE_first_echo_coils=coils[shown], displayed_receiver_rows=shown,
        measured_MEGRE_first_echo_RSS=np.sqrt(np.sum(abs(coils)**2, axis=0)),
        X_m=np.asarray(raw["X_m"]), Y_m=np.asarray(raw["Y_m"]))
    archive = data / "receiver_diagnostics.npz"
    np.savez_compressed(archive, **arrays)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(shown), figsize=(11.5, 4.2), sharex=True, sharey=True,
                             layout="constrained", squeeze=False)
    vmax = float(np.max(abs(coils[shown])))
    for ax, row in zip(axes[0], shown):
        artist = ax.pcolormesh(arrays["X_m"] * 1000, arrays["Y_m"] * 1000,
                              abs(coils[row]), shading="auto", cmap="gray", vmin=0, vmax=vmax)
        channel = f"ID {ids[row]}" if ids is not None else f"row {row}; ID unavailable"
        ax.set(title=f"Receiver {channel}", xlabel="Native x [mm]", aspect="equal")
    axes[0, 0].set_ylabel("Native y [mm]")
    fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=.8, label="Measured coil magnitude [a.u.; common scale]")
    te_ms = float(inputs["echo_times_s"][0]) * 1000
    fig.suptitle(f"Measured first-echo MEGRE: four of {len(coils)} receivers, TE {te_ms:.3f} ms", fontsize=12)
    fig.supxlabel("Common magnitude scale; MRI receivers are distinct from the 32 ACDC shim channels.\nObject contrast and reception remain combined; these are not isolated sensitivity maps.", fontsize=9)
    artifacts = _save_figure(fig, figures / "slide04_receiver_channels")
    plt.close(fig)
    report["figure_files"], report["data_archive"] = artifacts, str(archive)
    _write_report(output, "receivers", report)
    (output / "receiver_audit.json").write_text(json.dumps(_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    metrics = [dict(experiment="receiver_audit", metric="EPI_simulated_receive_channels", value=1, units="channels"),
               dict(experiment="receiver_audit", metric="measured_EPI_receive_channels", value=channels["EPI_OFF"]["count"], units="channels")]
    if "amplitude_accounting" in report:
        for key in ("observed_RSS_complex_amplitude_vs_scalar_envelope_relative_error",
                    "unit_RSS_normalized_direction_max_error"):
            metrics.append(dict(experiment="receiver_audit", metric=key,
                value=report["amplitude_accounting"][key], units="dimensionless"))
    return dict(report=_safe(report), artifacts=artifacts, metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, help="Locally generated trusted prepared_context.pkl")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--legacy-only", action="store_true")
    arguments = parser.parse_args()
    generate_legacy_pattern_figure(output_dir=arguments.output)
    if arguments.context and not arguments.legacy_only:
        with arguments.context.open("rb") as stream:
            run_receiver_audit(pickle.load(stream), arguments.output)
