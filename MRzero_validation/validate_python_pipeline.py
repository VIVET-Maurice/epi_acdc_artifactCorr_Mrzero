"""Check signal parity, automatic gradients, and phase integration timing."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import loadmat
import torch

from model import MRzeroACDCModel, device_settings, magnitude_mse
from mrzero_dynamic import analytical_adc_signal
from sequence import make_cartesian_epi_sequence, audit_sequence


def synchronized_time(function, device, repetitions=5):
    """Time steady-state work, including synchronization on CUDA."""
    function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    for _ in range(repetitions):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (perf_counter() - start) / repetitions


def validate(device="auto", profile="auto"):
    root = Path(__file__).resolve().parent
    inputs = dict(np.load(root / "subject006_slice120_inputs.npz"))
    saved = loadmat(root.parent / "sol_BASELINE.mat", simplify_cells=True)
    params = saved["params"]
    sequence = make_cartesian_epi_sequence(params)
    for key in saved["seq"]:
        assert np.array_equal(sequence[key], saved["seq"][key]), key
    settings = device_settings(profile, device)
    device = settings["device"]
    torch.set_num_threads(4)
    model = MRzeroACDCModel(inputs, params, sequence, device=device, dtype=torch.float64)
    model.phase_method = "ordered"
    currents = torch.tensor(saved["Ishim"], device=device, dtype=torch.float64)
    image = model(currents).detach().cpu().numpy()
    image_error = np.linalg.norm(image - saved["img_acdc"]) / np.linalg.norm(saved["img_acdc"])
    assert image_error < 5e-5

    # Independent original timestep recurrence checks the cumulative phase history.
    independent = analytical_adc_signal(inputs["img"], inputs["b0_Hz"],
        inputs["shim_Hz_per_A"], saved["Ishim"], params, sequence)
    signal = model.signal(currents).detach().cpu().numpy()
    signal_error = np.linalg.norm(signal - independent) / np.linalg.norm(independent)
    assert signal_error < 5e-5

    reference = torch.as_tensor(saved["img_ref"], device=device)
    generator = np.random.default_rng(17)
    direction = torch.as_tensor(generator.normal(size=currents.shape), device=device)
    direction = direction / direction.norm()
    x = currents.detach().clone().requires_grad_(True)
    loss = magnitude_mse(model(x), reference)
    gradient = torch.autograd.grad(loss, x)[0]
    automatic = (gradient * direction).sum().item()
    epsilon = 1e-4
    with torch.no_grad():
        finite = (magnitude_mse(model(x + epsilon * direction), reference)
                  - magnitude_mse(model(x - epsilon * direction), reference)).item() / (2 * epsilon)
    gradient_error = abs(automatic - finite) / max(abs(automatic), abs(finite), 1e-10)
    assert gradient_error < 1e-5

    events = np.asarray(sequence["eventOfTime"], dtype=int)
    exposure = np.cumsum(events[:, None] == np.arange(1, model.n_lines + 1), axis=0)
    exposure = torch.as_tensor(exposure[sequence["adc"]] * sequence["dt"],
                                dtype=model.dtype, device=device)

    def dense_signal(x):
        cycles = (exposure @ x.T) @ model.basis.T
        phase = torch.polar(torch.ones_like(cycles), -2 * torch.pi * cycles)
        return (model.magnetization * phase).sum(dim=1)

    difference = (dense_signal(currents) - model.signal(currents)).norm() / dense_signal(currents).norm()
    assert difference.item() < 1e-10

    def forward_backward(signal_function):
        x = currents.detach().clone().requires_grad_(True)
        loss = magnitude_mse(model.reconstruct(signal_function(x)), reference)
        torch.autograd.grad(loss, x)

    dense_seconds = synchronized_time(lambda: forward_backward(dense_signal), device)
    sparse_seconds = synchronized_time(lambda: forward_backward(model.signal), device)
    report = {
        "device": str(device), "gpu_name": settings["name"], "torch_version": torch.__version__,
        "sequence_bitwise_equal_to_saved_matlab": True,
        "trajectory": audit_sequence(sequence, model.n_lines),
        "native_mrzero_vs_saved_matlab_image_relative_error": float(image_error),
        "native_mrzero_vs_independent_recurrence_signal_relative_error": float(signal_error),
        "autograd_directional_relative_error": float(gradient_error),
        "ordered_event_vs_dense_phase_relative_error": difference.item(),
        "forward_backward_seconds_float64": {"dense_event_matrix": dense_seconds,
                                                "ordered_events": sparse_seconds},
        "phase_integration_speedup": dense_seconds / sparse_seconds,
        "gpu_tested": device.type == "cuda",
    }
    if device.type == "cuda":
        report["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.device, args.profile)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
