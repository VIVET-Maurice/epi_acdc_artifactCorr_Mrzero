"""Differentiable MRzero forward model for the original single-RF ACDC method."""

from time import perf_counter

import numpy as np
import torch


def device_settings(profile="auto", device="auto"):
    """Select a device and readable presets for V100, GTX 1660 Ti, or CPU."""
    if profile not in {"auto", "v100", "gtx1660ti", "cpu"}:
        raise ValueError("profile must be auto, v100, gtx1660ti, or cpu.")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() and profile != "cpu" else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select CPU or install CUDA-enabled PyTorch.")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("This notebook supports CPU and NVIDIA CUDA devices.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        memory_gb = properties.total_memory / 2**30
        name = properties.name
        if profile == "auto":
            profile = "v100" if memory_gb >= 16 else "gtx1660ti"
    else:
        profile, name, memory_gb = "cpu", "CPU", None
    block_size = {"v100": 1024, "gtx1660ti": 256, "cpu": 1024}[profile]
    return dict(device=device, profile=profile, name=name,
                memory_gb=memory_gb, adc_block_size=block_size)


def _native_magnetization(inputs, params, seq, b0_hz, device):
    """Run native MRzero once and keep ADC magnetization as Torch tensors."""
    import MRzeroCore as mr0

    dt = float(seq["dt"])
    adc = np.asarray(seq["adc"], dtype=bool)
    adc_steps = np.flatnonzero(adc)
    endpoints = np.unique(np.r_[adc_steps, len(adc) - 1])
    time = np.arange(1, len(adc) + 1) * dt
    k = float(params["gammaHz"]) * np.cumsum(
        np.column_stack([seq["Gx"], seq["Gy"], np.zeros(len(adc))]), axis=0
    ) * dt
    durations = np.diff(np.r_[0.0, time[endpoints]])
    moments = np.diff(np.vstack([np.zeros(3), k[endpoints]]), axis=0)

    repetition = mr0.Repetition.zero(len(endpoints))
    repetition.pulse.angle = torch.tensor(np.pi / 2)
    repetition.pulse.phase = torch.tensor(np.pi / 2)  # Mxy starts real, as in MATLAB.
    repetition.pulse.usage = mr0.PulseUsage.EXCIT
    repetition.pulse.selective = False
    repetition.event_time[:] = torch.from_numpy(durations.astype(np.float32))
    repetition.gradm[:] = torch.from_numpy(moments.astype(np.float32))
    repetition.adc_usage[:] = torch.from_numpy(adc[endpoints].astype(np.int32))
    sequence = mr0.Sequence([repetition], normalized_grads=False)

    coordinates = np.column_stack([inputs["X_m"].ravel(), inputs["Y_m"].ravel(),
                                   np.zeros(inputs["img"].size)]).astype(np.float32)
    phantom = mr0.CustomVoxelPhantom(
        coordinates, PD=inputs["img"].astype(np.float32).ravel(),
        B0=np.asarray(b0_hz, dtype=np.float32).ravel(), B1=1.0,
        T1=float(params["T1"]), T2=float(params["T2"]), T2dash=np.inf,
        D=0.0, voxel_size=0.0, voxel_shape="box",
    ).build()
    if device.type == "cuda":
        sequence, phantom = sequence.cuda(), phantom.cuda()
    # Match stationary point spins: no diffusion or finite-voxel attenuation.
    phantom.D = torch.zeros_like(phantom.D)
    phantom.dephasing_func = lambda k, _: torch.ones(k.shape[0], device=k.device)
    graph = mr0.compute_graph(sequence, phantom, max_state_count=16, min_state_mag=0.0)
    with torch.no_grad():
        signal, pathways = mr0.execute_graph(
            graph, sequence, phantom, min_emitted_signal=0.0,
            min_latent_signal=0.0, print_progress=False, return_mag_adc=True,
        )
    if len(pathways) != 1 or len(pathways[0]) != 1:
        raise RuntimeError("This adapter requires one RF and one transverse pathway.")
    magnetization = pathways[0][0].detach()
    receive = phantom.coil_sens[0].detach()
    return magnetization * receive[None, :], signal[:, 0].detach()


class MRzeroACDCModel:
    """Cache MRzero and differentiate the integrated ACDC phase and image FFT.

    Valid for the original ideal single-RF, stationary point-spin sequence.
    The sequence, coil basis, magnitude MSE, and reconstruction are unchanged.
    No simulator is rerun during optimization. Currents remain [channel, line].
    """

    def __init__(self, inputs, params, seq, b0_hz=None, *, device="cpu",
                 dtype=torch.float32, adc_block_size=1024, phase_method="auto"):
        self.device = torch.device(device)
        self.dtype = dtype
        self.n_lines = int(params["matrix_size"])
        self.n_channels = inputs["shim_Hz_per_A"].shape[-1]
        self.adc_block_size = int(adc_block_size)
        if phase_method not in {"auto", "dense", "ordered"}:
            raise ValueError("phase_method must be auto, dense, or ordered.")
        self.phase_method = ("ordered" if self.device.type == "cuda" else "dense") if phase_method == "auto" else phase_method
        if self.adc_block_size < 1:
            raise ValueError("adc_block_size must be positive.")
        started = perf_counter()
        if b0_hz is None:
            b0_hz = inputs["b0_Hz"]
        magnetization, native_signal = _native_magnetization(
            inputs, params, seq, b0_hz, self.device
        )
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.magnetization = magnetization.to(complex_dtype)
        self.native_signal = native_signal.to(complex_dtype)
        self.basis = torch.as_tensor(inputs["shim_Hz_per_A"].reshape(-1, self.n_channels),
                                     dtype=dtype, device=self.device)

        # The original events occur consecutively: 0, 1, ..., Nlines.
        # This allows exact phase integration without a dense ADC-by-event product.
        events = np.asarray(seq["eventOfTime"], dtype=int)
        adc = np.asarray(seq["adc"], dtype=bool)
        if np.any(np.diff(events) < 0) or events.max() != self.n_lines:
            raise ValueError("The fast adapter requires consecutive ordered EPI events.")
        if np.any(events[adc] == 0):
            raise ValueError("Original EPI ADC samples must belong to a shim event.")
        durations = np.bincount(events, minlength=self.n_lines + 1)[1:] * seq["dt"]
        exposure = np.cumsum(events[:, None] == np.arange(1, self.n_lines + 1), axis=0)
        exposure = exposure[adc] * seq["dt"]
        self.exposure = torch.as_tensor(exposure, dtype=dtype, device=self.device)
        adc_lines = events[adc] - 1
        elapsed = exposure[np.arange(len(adc_lines)), adc_lines]
        self.event_durations = torch.as_tensor(durations, dtype=dtype, device=self.device)
        self.adc_lines = torch.as_tensor(adc_lines, device=self.device)
        self.elapsed_in_event = torch.as_tensor(elapsed, dtype=dtype, device=self.device)

        # Preserve MATLAB's round-away-from-zero and last-write behavior exactly.
        scaled = np.column_stack([seq["kx_adc"], seq["ky_adc"]]) / seq["dk"]
        bins = (np.sign(scaled) * np.floor(np.abs(scaled) + .5)).astype(int) + self.n_lines//2
        valid = np.all((bins >= 0) & (bins < self.n_lines), axis=1)
        sources = np.flatnonzero(valid)
        flat_bins = bins[valid, 0] * self.n_lines + bins[valid, 1]
        destinations, reverse_indices = np.unique(flat_bins[::-1], return_index=True)
        self.sources = torch.as_tensor(sources[::-1][reverse_indices].copy(), device=self.device)
        self.destinations = torch.as_tensor(destinations, device=self.device)
        self.cache_time_s = perf_counter() - started

    def signal(self, currents):
        """Return all chronological ADC samples, including subsequently dropped ADC."""
        currents = currents.to(device=self.device, dtype=self.dtype)
        if currents.shape != (self.n_channels, self.n_lines):
            raise ValueError("currents must have shape [channel, line].")
        if self.phase_method == "ordered":
            # Rows represent lines so gathered ADC phases stay contiguous in memory.
            fields = (self.basis @ currents).T.contiguous()  # [line, voxel], Hz.
            completed_phase = torch.cumsum(fields * self.event_durations[:, None], dim=0)
            prior_phase = torch.cat([torch.zeros_like(completed_phase[:1]),
                                     completed_phase[:-1]], dim=0)
        signals = []
        for start in range(0, len(self.adc_lines), self.adc_block_size):
            stop = start + self.adc_block_size
            lines = self.adc_lines[start:stop]
            if self.phase_method == "ordered":
                cycles = prior_phase[lines] + fields[lines] * self.elapsed_in_event[start:stop, None]
            else:
                cycles = (self.exposure[start:stop] @ currents.T) @ self.basis.T
            phase = torch.polar(torch.ones_like(cycles), -2 * torch.pi * cycles)
            signals.append((self.magnetization[start:stop] * phase).sum(dim=1))
        return torch.cat(signals)

    def reconstruct(self, signal):
        """Use the same rounded Cartesian grid and centered IFFT as MATLAB."""
        kspace = torch.zeros(self.n_lines**2, dtype=signal.dtype, device=self.device)
        kspace = kspace.index_copy(0, self.destinations, signal[self.sources])
        kspace = kspace.reshape(self.n_lines, self.n_lines)
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace)))

    def __call__(self, currents):
        return self.reconstruct(self.signal(currents))


def magnitude_mse(image, reference):
    """Original whole-image objective; no image normalization or phase term."""
    return (image.abs() - reference.abs()).square().mean()


def magnitude_nrmse_percent(image, reference):
    """Same magnitude NRMSE as plot_lcurve.m."""
    return 100 * ((image.abs() - reference.abs()).square().mean()
                  / reference.abs().square().mean()).sqrt().item()
