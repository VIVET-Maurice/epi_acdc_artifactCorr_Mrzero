"""Readable PyTorch implementation of the experimental ACDC image model.

All Fourier phases and current integrals follow the existing experiment model.
The optimizer runs on one selected CUDA device, or on the CPU for validation.
Its variables are DAC commands, not the filtered physical currents.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import torch


def select_device(profile="auto", precision="auto", *, require_cuda=False):
    """Return a device, real dtype and a small, printable settings report.

    A profile selects the computer, not a different physical model. Float32 is
    the fast GPU default; float64 is available on every profile for comparison.
    Auto falls back to CPU if the CUDA execution test fails, with a warning.
    ``require_cuda=True`` keeps automatic GPU selection but forbids CPU fallback.
    Explicit GPU profiles raise an error when CUDA cannot execute a small test.
    """
    profiles = {"auto", "v100", "gtx1660ti", "blackwell", "cpu"}
    if profile not in profiles:
        raise ValueError(f"profile must be one of {sorted(profiles)}.")
    if precision not in {"auto", "float32", "float64"}:
        raise ValueError("precision must be auto, float32 or float64.")
    if not isinstance(require_cuda, bool):
        raise ValueError("require_cuda must be boolean.")
    if require_cuda and profile == "cpu":
        raise ValueError("profile='cpu' conflicts with require_cuda=True.")
    use_cuda = profile != "cpu" and torch.cuda.is_available()
    if (require_cuda or profile not in {"auto", "cpu"}) and not use_cuda:
        raise RuntimeError("CUDA is unavailable; this run requires a GPU. Check the notebook's Python environment, "
                           "GPU allocation and NVIDIA driver; install a compatible PyTorch/CUDA build "
                           "(see docs/SERVER_SETUP.md), then restart the kernel.")
    fallback_reason = None
    detected_gpu, capability, architectures = None, None, []
    if use_cuda:
        # Keep full float32 mantissas for phase accumulation; no TF32/FP16.
        torch.backends.cuda.matmul.allow_tf32 = False
        properties = torch.cuda.get_device_properties(torch.device("cuda"))
        detected_gpu = properties.name
        capability = f"sm_{properties.major}{properties.minor}"
        architectures = torch.cuda.get_arch_list()
        probe_dtype = torch.float32 if precision == "auto" else getattr(torch, precision)
        try:
            # CUDA availability alone does not establish kernel compatibility.
            probe = torch.ones((2, 2), device="cuda", dtype=probe_dtype)
            _ = probe @ probe
            torch.cuda.synchronize()
        except RuntimeError as error:
            fallback_reason = (
                f"CUDA execution test failed on {detected_gpu} ({capability}) with "
                f"PyTorch {torch.__version__}, CUDA runtime {torch.version.cuda}; "
                f"build architectures: {', '.join(architectures) or 'unknown'}. "
                f"Original error: {error}"
            )
            guidance = ("Install a PyTorch CUDA build compatible with this GPU in the notebook's "
                        "Python environment (see docs/SERVER_SETUP.md), then restart the kernel. "
                        "GPU profile names do not change the installed build.")
            if require_cuda or profile != "auto":
                raise RuntimeError(f"{fallback_reason}\n{guidance} "
                                   "For an explicit CPU run, set require_cuda=False and profile='cpu'.") from error
            warnings.warn(f"{fallback_reason}\nFalling back to CPU because profile='auto'; "
                          f"this may be slower. {guidance}", RuntimeWarning, stacklevel=2)
            use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    precision = ("float32" if use_cuda else "float64") if precision == "auto" else precision
    dtype = getattr(torch, precision)
    name = detected_gpu if use_cuda else "CPU"
    memory_gib = properties.total_memory / 1024**3 if use_cuda else None
    info = {"profile": profile, "device": str(device), "gpu_name": name,
            "precision": precision, "memory_GiB": memory_gib,
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "detected_gpu_name": detected_gpu, "gpu_capability": capability,
            "cuda_build_architectures": architectures, "cpu_fallback_reason": fallback_reason,
            "require_cuda": require_cuda, "tf32": False, "mixed_precision": False}
    return device, dtype, info


def _real_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite.")
    return result


def _windows(starts_s, ends_s):
    starts = _real_array(starts_s, "line_start_s").reshape(-1)
    ends = np.asarray(ends_s, dtype=np.float64).reshape(-1)
    if (not starts.size or starts.shape != ends.shape or np.any(starts < 0)
            or np.any(np.diff(starts) <= 0) or np.isnan(ends).any()
            or np.any(ends <= starts) or not np.isfinite(ends[:-1]).all()
            or np.any(ends[:-1] > starts[1:])):
        raise ValueError("Current windows need ordered, nonoverlapping positive durations; only the final end may be infinite.")
    return starts, ends


def _step_integral(elapsed, tau):
    """Integral of a unit first-order step, stable for very short times."""
    z = elapsed / tau
    result = elapsed + tau * np.expm1(-z)
    small = z < 1e-4
    polynomial = tau * z**2 * (0.5 - z / 6 + z**2 / 24 - z**3 / 120)
    return np.where(small, polynomial, result)


def response_matrices(starts_s, ends_s, times_s, tau_s, rf_time_s=None):
    """Exact rectangular-command integrals H and physical-current weights R.

    Both outputs have shape [channel, sample, state]. H has units of seconds.
    Subtracting the integral at each RF center resets transverse phase for GRE
    without resetting the physical current. The notebook enables response for
    EPI only, consistent with the measured-response assumptions of v3.
    """
    starts, ends = _windows(starts_s, ends_s)
    times = _real_array(times_s, "times_s").reshape(-1)
    tau = _real_array(tau_s, "tau_s").reshape(-1)
    rf = np.zeros_like(times) if rf_time_s is None else np.broadcast_to(
        _real_array(rf_time_s, "rf_time_s"), times.shape)
    if (np.any(times < 0) or np.any(rf < 0) or np.any(rf > times)
            or not tau.size or np.any(tau <= 0)):
        raise ValueError("Times must follow RF centers; tau_s must be positive.")
    a = np.maximum(0, times[:, None] - starts)
    b = np.maximum(0, times[:, None] - ends)
    ar = np.maximum(0, rf[:, None] - starts)
    br = np.maximum(0, rf[:, None] - ends)
    integrals, currents = [], []
    for value in tau:
        integrals.append(_step_integral(a, value) - _step_integral(b, value)
                         - _step_integral(ar, value) + _step_integral(br, value))
        currents.append(-np.expm1(-a / value) + np.expm1(-b / value))
    return np.stack(integrals), np.stack(currents)


@torch.no_grad()
def project_currents_tensor(currents, max_current_A, total_current_limit_A,
                            field_basis=None, field_limit_Hz=np.inf,
                            current_margin_A=1e-4, field_margin_Hz=0.05):
    """Enforce box/L1 bounds, then shrink each state's field if necessary.

    Bisection gives the Euclidean projection onto the channel box and total
    absolute-current ball. A subsequent radial contraction enforces the field
    bound while preserving those current bounds. That last step is feasible,
    but is not the Euclidean projection onto the full combined constraint set.
    Numerical margins keep strict CSV limits and float32 rounding inside bounds.
    """
    if currents.ndim != 2 or not currents.is_floating_point():
        raise ValueError("currents must be a real [channel, state] tensor.")
    total = float(total_current_limit_A)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("total_current_limit_A must be positive and finite.")
    if np.isnan(field_limit_Hz) or field_limit_Hz <= 0:
        raise ValueError("field_limit_Hz must be positive; infinity disables it.")
    limits = torch.as_tensor(max_current_A, device=currents.device, dtype=currents.dtype)
    if limits.numel() not in {1, currents.shape[0]}:
        raise ValueError("max_current_A needs one value or one per channel.")
    limits = limits.reshape(-1, 1)
    # A relative margin also supports unusually small user-defined limits.
    limits = torch.clamp(limits - torch.minimum(limits * 1e-3,
                         limits.new_tensor(current_margin_A)), min=0)
    budget = max(0.0, total - min(total * 1e-3, current_margin_A))
    magnitude = currents.abs()
    low = torch.zeros((1, currents.shape[1]), device=currents.device, dtype=currents.dtype)
    high = magnitude.amax(dim=0, keepdim=True)
    clipped = torch.minimum(magnitude, limits)
    needs_l1 = clipped.sum(dim=0, keepdim=True) > budget
    # Resolve the threshold to the selected floating-point precision.
    for _ in range(54 if currents.dtype == torch.float64 else 28):
        middle = (low + high) / 2
        candidate = torch.minimum(torch.clamp(magnitude - middle, min=0), limits)
        over_budget = candidate.sum(dim=0, keepdim=True) > budget
        low = torch.where(over_budget, middle, low)
        high = torch.where(over_budget, high, middle)
    threshold = torch.where(needs_l1, high, torch.zeros_like(high))
    projected = currents.sign() * torch.minimum(torch.clamp(magnitude - threshold, min=0), limits)
    if field_basis is not None and np.isfinite(field_limit_Hz):
        basis = torch.as_tensor(field_basis, device=currents.device, dtype=currents.dtype)
        peak = (basis @ projected).abs().amax(dim=0, keepdim=True)
        safe_limit = max(0.0, float(field_limit_Hz) - min(float(field_limit_Hz) * 1e-3, field_margin_Hz))
        scale = torch.clamp(safe_limit / peak.clamp_min(torch.finfo(currents.dtype).tiny), max=1)
        projected *= scale
    return projected


class ACDCModel:
    """Cache the physical model once; evaluate images and an analytic adjoint.

    The object uses [phase row, readout column], coordinates in metres, coil
    maps in Hz/A and acquisition k-space in cycles/m. Cartesian array order
    stays independent from chronological, one-based ``line_index`` values.
    """

    def __init__(self, experiment, calibration, max_current_A=3.0, *, profile="auto",
                 precision="auto", memory_fraction=0.5, sample_chunk=None, require_cuda=False):
        self.device, self.dtype, self.runtime_info = select_device(profile, precision, require_cuda=require_cuda)
        self.complex_dtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
        if not 0 < memory_fraction <= 0.8:
            raise ValueError("memory_fraction must be in (0, 0.8].")
        m0 = _real_array(experiment["m0_proxy"], "m0_proxy")
        if m0.ndim != 2 or np.any(m0 < 0):
            raise ValueError("m0_proxy must be a nonnegative 2D array.")
        self.object_size = m0.shape
        field, geometry = experiment["field"], experiment["geometry"]
        support = np.asarray(field["simulation_mask"], bool)
        quality = np.asarray(field["mask"], bool)
        if support.shape != m0.shape or quality.shape != m0.shape:
            raise ValueError("Field masks must match the object grid.")
        self.support = support & (m0 > 0)
        self.quality_mask = quality & self.support
        if not self.support.any() or not self.quality_mask.any():
            raise ValueError("Need nonempty object support and reliable field coverage.")
        b0 = _real_array(field["b0Hz"], "b0Hz")
        if b0.shape != m0.shape:
            raise ValueError("B0 must match the object grid.")
        if calibration.get("units") != "Hz/A":
            raise ValueError("Calibration units must be explicitly Hz/A.")
        if np.iscomplexobj(calibration["acdc_maps"]):
            raise ValueError("ACDC calibration maps must be real Hz/A responses.")
        maps = np.asarray(calibration["acdc_maps"], dtype=np.float64)
        if maps.ndim != 3 or maps.shape[:2] != m0.shape:
            raise ValueError("acdc_maps must have shape [object rows, object columns, channels].")
        self.nc = maps.shape[-1]
        valid = np.isfinite(maps).all(axis=-1)
        spatial_valid = np.asarray(calibration.get("spatial_valid_mask", valid), bool)
        if spatial_valid.shape != m0.shape:
            raise ValueError("Calibration validity mask must match the object grid.")
        valid &= spatial_valid
        if not valid[self.support].all() or not np.any(np.abs(maps[self.support]) > 0, axis=0).all():
            raise ValueError("Every object point and channel needs a valid, measured ACDC response.")
        self.maps_numpy = maps
        self.calibration_valid_mask = valid
        self.registration_confirmed = bool(calibration.get("registration_confirmed", False))
        for name in ("X_m", "Y_m"):
            coords = _real_array(geometry[name], name)
            calibration_coords = _real_array(calibration[name], "calibration " + name)
            if (coords.shape != m0.shape or calibration_coords.shape != m0.shape
                    or not np.allclose(coords, calibration_coords, rtol=0, atol=1e-9)):
                raise ValueError("Calibration coordinates must match the physical object grid within 1 nm.")
        acq = experiment["acquisition"]
        self.image_size = np.asarray(acq["kx_mInv"]).shape
        if len(self.image_size) != 2 or min(self.image_size) < 1:
            raise ValueError("Acquisition k-space must be a nonempty 2D array.")
        samples = {}
        for name in ("kx_mInv", "ky_mInv", "time_s", "line_index"):
            value = _real_array(acq[name], name)
            if value.shape != self.image_size:
                raise ValueError("All acquisition arrays must have the same 2D shape.")
            samples[name] = value.reshape(-1)
        self.nstates = self.image_size[0]
        self.nsamples = int(np.prod(self.image_size))
        self.sequence_kind = str(acq.get("sequence_kind", "epi")).lower()
        if self.sequence_kind not in {"epi", "gre"}:
            raise ValueError("sequence_kind must be epi or gre.")
        starts = _real_array(acq["line_start_s"], "line_start_s").reshape(-1)
        default_ends = np.r_[starts[1:], np.inf]
        if self.sequence_kind == "gre" and ("line_end_s" not in acq or "rf_time_s" not in acq):
            raise ValueError("GRE requires line_end_s and per-ADC rf_time_s.")
        self.line_start_s, self.line_end_s = _windows(starts, acq.get("line_end_s", default_ends))
        if starts.size != self.nstates:
            raise ValueError("Need one current state per Cartesian row.")
        indices = samples["line_index"]
        self.line_index = indices.reshape(self.image_size).astype(int)
        if (np.any(indices != np.floor(indices)) or not np.all(self.line_index == self.line_index[:, :1])
                or not np.array_equal(np.sort(self.line_index[:, 0]), np.arange(1, self.nstates + 1))):
            raise ValueError("line_index must label each row by a distinct chronological state, from 1.")
        times = samples["time_s"]
        state_index = indices.astype(int) - 1
        if (np.any(times < 0) or np.any(times < starts[state_index] - 1e-12)
                or np.any(times >= self.line_end_s[state_index])):
            raise ValueError("Each ADC time must belong to its command window.")
        rf = np.asarray(acq.get("rf_time_s", np.zeros(self.image_size)), dtype=np.float64)
        if rf.shape != self.image_size or not np.isfinite(rf).all() or np.any(rf < 0) or np.any(rf.reshape(-1) > times):
            raise ValueError("Each ADC needs an RF center no later than its acquisition.")
        rf = rf.reshape(-1)
        if self.sequence_kind == "epi":
            # Preserve the original single-RF convention, whose center is t=0.
            if np.any(np.abs(rf) > 1e-12):
                raise ValueError("EPI times must be relative to the single RF center at zero.")
        else:
            rf_by_state = []
            for state in range(self.nstates):
                local_rf = rf[state_index == state]
                if np.ptp(local_rf) > 1e-12:
                    raise ValueError("Each spoiled GRE line must have one RF center.")
                rf_by_state.append(local_rf[0])
            rf_by_state = np.asarray(rf_by_state)
            if (np.any(np.diff(rf_by_state) <= 0) or np.any(rf_by_state > starts)
                    or np.any(self.line_end_s[:-1] > rf_by_state[1:])):
                raise ValueError("GRE command windows must follow RF and finish before the next RF.")
        response = experiment.get("current_response", {})
        enabled = response.get("enabled", False)
        if not isinstance(enabled, (bool, np.bool_)) and enabled not in (0, 1):
            raise ValueError("current_response.enabled must be boolean.")
        tau = _real_array(response.get("tau_s", 133e-6 / np.log(100)), "tau_s")
        tau = np.full(self.nc, float(tau)) if tau.ndim == 0 or tau.size == 1 else tau.reshape(-1)
        if tau.shape != (self.nc,) or np.any(tau <= 0):
            raise ValueError("tau_s needs one positive value or one per channel.")
        self.current_response = {"enabled": bool(enabled), "tau_s": tau.copy()}
        if enabled and self.sequence_kind != "epi":
            raise ValueError("First-order response is currently enabled for EPI only; disable it for GRE.")
        limits = _real_array(max_current_A, "max_current_A")
        limits = np.full(self.nc, float(limits)) if limits.ndim == 0 else limits.reshape(-1)
        if limits.shape != (self.nc,) or np.any(limits <= 0):
            raise ValueError("max_current_A needs one positive value or one per channel.")
        self.max_current_A = self.tensor(limits)
        self.B = self.tensor(maps[self.support])
        self.B_quality = self.tensor(maps[self.quality_mask])
        self.b0_numpy = np.where(self.support, b0, 0)
        self._amplitude = self.tensor(m0[self.support])[:, None]
        self._x = self.tensor(np.asarray(geometry["X_m"])[self.support])[:, None]
        self._y = self.tensor(np.asarray(geometry["Y_m"])[self.support])[:, None]
        self._b0 = self.tensor(b0[self.support])[:, None]
        t2 = np.asarray(experiment["relaxation"]["T2star_s"], dtype=np.float64)
        if t2.ndim == 0:
            t2 = np.full(m0.shape, float(t2))
        if t2.shape != m0.shape or np.isnan(t2).any() or np.any(t2 <= 0):
            raise ValueError("T2star_s must be positive, scalar or matching the object; infinity is allowed.")
        self._rate = self.tensor(1 / t2[self.support])[:, None]
        self._evolution = self.tensor(times - rf)
        self._kx, self._ky = self.tensor(samples["kx_mInv"]), self.tensor(samples["ky_mInv"])
        phase = _real_array(acq.get("phase_offset_rad", np.zeros(self.image_size)), "phase_offset_rad")
        if phase.shape != self.image_size:
            raise ValueError("phase_offset_rad must match the acquisition grid.")
        self._phase = self.tensor(phase.reshape(-1))
        self.TE_s = float(acq["TE_s"])
        if not np.isfinite(self.TE_s) or self.TE_s < 0:
            raise ValueError("TE_s must be finite and nonnegative.")
        self.time_s = times.reshape(self.image_size)
        self.rf_time_s = rf.reshape(self.image_size)
        H = np.maximum(0, np.minimum(times[:, None], self.line_end_s)
                       - np.maximum(rf[:, None], starts))
        self.H_instant = self.tensor(H)
        self.common_tau = bool(np.all(tau == tau[0]))
        self.H_response = None
        if enabled:
            H_response, _ = response_matrices(starts, self.line_end_s, times,
                                              tau[:1] if self.common_tau else tau, rf)
            self.H_response = self.tensor(H_response[0] if self.common_tau else H_response)
        self._H = self.H_response if enabled else self.H_instant
        self._channel_response = bool(enabled and not self.common_tau)
        nvox = int(self.support.sum())
        if self.device.type == "cuda":
            available = torch.cuda.mem_get_info(self.device)[0]
            budget = int(available * memory_fraction)
        else:
            budget = int(1024**3 * memory_fraction)
        complex_bytes = torch.empty((), dtype=self.complex_dtype).element_size()
        self._full_cache_bytes = nvox * self.nsamples * complex_bytes
        self._cache_base = self._full_cache_bytes < budget // 3
        self._retain_contributions = self._full_cache_bytes < budget // 3
        actual_profile = profile
        if profile == "auto":
            name = self.runtime_info["gpu_name"].lower()
            actual_profile = ("cpu" if self.device.type == "cpu" else "v100" if "v100" in name
                              else "gtx1660ti" if "1660" in name else "blackwell" if "blackwell" in name
                              else "auto")
        # Conservative defaults for small cards; free memory can reduce them.
        profile_chunks = {"cpu": 256, "gtx1660ti": 256, "v100": 4096,
                          "blackwell": 4096, "auto": 1024}
        automatic_chunk = max(1, min(self.nsamples, profile_chunks[actual_profile],
                                     budget // max(1, nvox * complex_bytes * 8)))
        if sample_chunk is not None and (isinstance(sample_chunk, bool)
                or int(sample_chunk) != sample_chunk or sample_chunk < 1):
            raise ValueError("sample_chunk must be a positive integer or None.")
        # An explicit chunk overrides the profile cap, while retaining the
        # memory bound. It permits a short benchmark on the actual computer.
        self.sample_chunk = automatic_chunk if sample_chunk is None else min(
            self.nsamples, int(sample_chunk), max(1, budget // max(1, nvox * complex_bytes * 8)))
        self._base_cache = None
        if self._cache_base:
            self._base_cache = torch.empty((nvox, self.nsamples), device=self.device, dtype=self.complex_dtype)
            for first in range(0, self.nsamples, self.sample_chunk):
                selected = slice(first, min(first + self.sample_chunk, self.nsamples))
                self._base_cache[:, selected] = self._make_base(selected)
            if not bool(torch.isfinite(self._base_cache).all()):
                raise ValueError("Nonfinite base signal: inspect echo-relative T2star evolution.")
        self._reference, self._ideal_kdata = self._forward_tensor(
            self.tensor(np.zeros((self.nc, self.nstates))), b0_enabled=False)
        self._reference_amplitude = self._reference.abs()
        self._reference_energy = self._reference.abs().square().sum()
        if not bool(torch.isfinite(self._reference_energy)) or float(self._reference_energy) <= 0:
            raise ValueError("The ideal reference must have finite, nonzero energy.")
        self.reference = self.numpy(self._reference)
        self.ideal_image, self.ideal_kdata = self.reference, self.numpy(self._ideal_kdata)
        self.runtime_info.update({"selected_profile": actual_profile,
                                  "sample_chunk": self.sample_chunk, "samples": self.nsamples,
                                  "object_points": nvox, "base_cache": self._cache_base,
                                  "retain_contributions": self._retain_contributions,
                                  "cache_MiB": self._full_cache_bytes / 1024**2 if self._cache_base else 0,
                                  "memory_budget_MiB": budget / 1024**2,
                                  "response_enabled": bool(enabled), "common_tau": self.common_tau})

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    @staticmethod
    def numpy(value):
        return value.detach().cpu().numpy()

    def _check_currents(self, currents):
        value = _real_array(currents, "currents_A")
        if value.shape != (self.nc, self.nstates):
            raise ValueError("currents_A must have shape [channels, imaging states].")
        return self.tensor(value)

    def _make_base(self, selected, b0_enabled=True):
        t = self._evolution[selected][None, :]
        spatial = self._x * self._kx[selected] + self._y * self._ky[selected]
        cycles = spatial + self._b0 * t if b0_enabled else spatial
        phase = self._phase[selected] - (2 * np.pi) * cycles
        decay = torch.exp(-self._rate * (t - self.TE_s))
        return self._amplitude * decay * torch.polar(torch.ones_like(phase), phase)

    def _contribution(self, currents, selected, response_enabled=None, b0_enabled=True):
        use_response = self.current_response["enabled"] if response_enabled is None else response_enabled
        if use_response and self.H_response is None:
            raise ValueError("Response was disabled when preparing this model; prepare it again to enable it.")
        H = self.H_response if use_response else self.H_instant
        if use_response and not self.common_tau:
            integrated = torch.bmm(currents[:, None, :], H[:, selected, :].transpose(1, 2)).squeeze(1)
        else:
            integrated = currents @ H[selected].T
        phase = (-2 * np.pi) * (self.B @ integrated)
        base = self._base_cache[:, selected] if b0_enabled and self._base_cache is not None else self._make_base(selected, b0_enabled)
        return base * torch.polar(torch.ones_like(phase), phase)

    @staticmethod
    def _image(kdata):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kdata)))

    def _forward_tensor(self, currents, *, response_enabled=None, b0_enabled=True, retain=False):
        signal = torch.empty(self.nsamples, device=self.device, dtype=self.complex_dtype)
        saved = []
        for first in range(0, self.nsamples, self.sample_chunk):
            selected = slice(first, min(first + self.sample_chunk, self.nsamples))
            contribution = self._contribution(currents, selected, response_enabled, b0_enabled)
            signal[selected] = contribution.sum(dim=0)
            if retain:
                saved.append(contribution)
        kdata = signal.reshape(self.image_size)
        image = self._image(kdata)
        return (image, kdata, saved) if retain else (image, kdata)

    @torch.no_grad()
    def forward(self, currents_A, response_enabled=None, *, b0_enabled=True):
        image, kdata = self._forward_tensor(self._check_currents(currents_A),
                                           response_enabled=response_enabled, b0_enabled=b0_enabled)
        return self.numpy(image), self.numpy(kdata)

    @torch.no_grad()
    def _objective_tensor(self, currents, gradient=True):
        retain = gradient and self._retain_contributions
        output = self._forward_tensor(currents, retain=retain)
        image, kdata = output[:2]
        amplitude = image.abs()
        residual = amplitude - self._reference_amplitude
        value = residual.square().sum() / self._reference_energy
        if not gradient:
            return value, image, kdata
        # Generalized derivative zero at exactly zero image magnitude.
        derivative = 2 * residual * image / amplitude.clamp_min(torch.finfo(self.dtype).tiny)
        derivative /= self._reference_energy
        adjoint = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(derivative))) / self.nsamples
        adjoint = adjoint.reshape(-1)
        result = torch.zeros_like(currents)
        for block, first in enumerate(range(0, self.nsamples, self.sample_chunk)):
            selected = slice(first, min(first + self.sample_chunk, self.nsamples))
            contribution = output[2][block] if retain else self._contribution(currents, selected)
            sensitivity = (2 * np.pi) * (contribution * adjoint[selected].conj()).imag
            channel_sensitivity = self.B.T @ sensitivity
            if self._channel_response:
                result += torch.bmm(channel_sensitivity[:, None, :], self._H[:, selected, :]).squeeze(1)
            else:
                result += channel_sensitivity @ self._H[selected]
        return value, result, image, kdata

    @torch.no_grad()
    def objective_and_grad(self, currents_A):
        value, gradient, _, _ = self._objective_tensor(self._check_currents(currents_A))
        return float(value), self.numpy(gradient)

    def project_tensor(self, currents, total_current_limit_A, field_limit_Hz=np.inf):
        return project_currents_tensor(currents, self.max_current_A, total_current_limit_A,
                                       self.B_quality, field_limit_Hz)

    def project_currents(self, currents_A, total_current_limit_A, field_limit_Hz=np.inf):
        return self.numpy(self.project_tensor(self._check_currents(currents_A), total_current_limit_A, field_limit_Hz))

    def benchmark(self, repeats=3):
        """Time cached objective and gradient, including GPU synchronization."""
        if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 1:
            raise ValueError("repeats must be a positive integer.")
        zeros = self.tensor(np.zeros((self.nc, self.nstates)))
        self._objective_tensor(zeros)
        durations = []
        for _ in range(int(repeats)):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            self._objective_tensor(zeros)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
        return {"objective_gradient_median_s": float(np.median(durations)),
                "repeats": int(repeats), **self.runtime_info}

    def sample_waveform(self, currents_A, times_s=None, *, response_enabled=None):
        """Sample exact continuous current, including its final settling tail.

        Returns ``time_s``, ``actual_A`` and ``commanded_A`` (channels × times).
        Event recurrence includes all preceding transitions, even for a zoom.
        """
        commands = self.numpy(self._check_currents(currents_A))
        starts, ends = self.line_start_s, self.line_end_s
        enabled = self.current_response["enabled"] if response_enabled is None else bool(response_enabled)
        tau = self.current_response["tau_s"]
        boundaries = np.unique(np.r_[starts, ends[np.isfinite(ends)]])
        if times_s is None:
            final = ends[-1] if np.isfinite(ends[-1]) else starts[-1] + (np.median(np.diff(starts)) if starts.size > 1 else 8 * tau.max())
            offsets = np.unique((np.linspace(0, 8, 81)[:, None] * np.unique(tau)).reshape(-1)) if enabled else np.array([0.0])
            limit = final + (8 * tau.max() if enabled and np.isfinite(ends[-1]) else 0)
            times_s = np.unique(np.r_[0, boundaries, (boundaries[:, None] + offsets).reshape(-1),
                                     np.maximum(0, boundaries - 1e-9), final, limit])
            times_s = times_s[times_s <= limit]
        times = _real_array(times_s, "times_s").reshape(-1)
        if not times.size or np.any(times < 0):
            raise ValueError("Waveform times must be nonempty and nonnegative.")
        commanded = commands @ ((times[:, None] >= starts) & (times[:, None] < ends)).T
        if not enabled:
            return {"time_s": times, "actual_A": commanded.copy(), "commanded_A": commanded}
        event_commands = {float(boundary): np.zeros(self.nc) for boundary in boundaries}
        for state, start in enumerate(starts):
            event_commands[float(start)] = commands[:, state]
        timeline = np.unique(np.r_[0, times, boundaries[boundaries <= times.max()]])
        actual_all = np.empty((self.nc, timeline.size))
        actual, command, previous = np.zeros(self.nc), np.zeros(self.nc), 0.0
        for column, now in enumerate(timeline):
            actual = command + (actual - command) * np.exp(-(now - previous) / tau)
            command = event_commands.get(float(now), command)
            actual_all[:, column] = actual
            previous = now
        actual = actual_all[:, np.searchsorted(timeline, times)]
        return {"time_s": times, "actual_A": actual, "commanded_A": commanded}

    def response_diagnostics(self, currents_A, field_limit_Hz=np.inf, total_current_limit_A=np.inf):
        commands = self.numpy(self._check_currents(currents_A))
        times = self.time_s.reshape(-1)
        adc = self.sample_waveform(commands, times)
        command_integrals = commands @ self.numpy(self.H_instant).T
        if self.current_response["enabled"]:
            if self.common_tau:
                integrals = commands @ self.numpy(self.H_response).T
            else:
                integrals = np.einsum("cs,cts->ct", commands, self.numpy(self.H_response))
        else:
            integrals = command_integrals
        line_times = []
        for state in range(1, self.nstates + 1):
            local = times[self.line_index.reshape(-1) == state]
            line_times.append(local[np.argmin(np.abs(local - np.median(local)))])
        line_times = np.asarray(line_times)
        line_currents = self.sample_waveform(commands, line_times)["actual_A"]
        waveform = self.sample_waveform(commands)
        jumps = np.abs(np.column_stack((commands[:, 0], np.diff(commands, axis=1))))
        zoom_channel, zoom_state = np.unravel_index(np.argmax(jumps), jumps.shape)
        tau = self.current_response["tau_s"]
        transition = self.line_start_s[zoom_state]
        zoom_times = np.linspace(max(0, transition - 5 * tau.max()), transition + 5 * tau.max(), 512)
        zoom = self.sample_waveform(commands, zoom_times)
        actual = adc["actual_A"]
        fields = self.maps_numpy[self.quality_mask] @ actual
        max_field = float(np.max(np.abs(fields)))
        current_sums = np.abs(actual).sum(axis=0)
        instantaneous_image, instantaneous_kdata = self.forward(commands, response_enabled=False)
        response_image, response_kdata = self.forward(commands)
        energy = float(np.sum(np.abs(self.reference)**2))
        tau = self.current_response["tau_s"]
        step_time = np.linspace(0, 1.3 * tau.max() * np.log(100), 256)
        return {"adc_time_s": times, "commanded_currents_at_adc_A": adc["commanded_A"],
                "actual_currents_at_adc_A": actual,
                "commanded_integrated_currents_to_adc_As": command_integrals,
                "integrated_currents_to_adc_As": integrals,
                "line_center_time_s": line_times,
                "generated_field_at_line_center_Hz": self.maps_numpy @ line_currents,
                "waveform_time_s": waveform["time_s"],
                "waveform_commanded_currents_A": waveform["commanded_A"],
                "waveform_actual_currents_A": waveform["actual_A"],
                "dense_time_s": zoom_times,
                "dense_commanded_currents_A": zoom["commanded_A"],
                "dense_actual_currents_A": zoom["actual_A"],
                "dense_transition_index": int(zoom_state + 1),
                "dense_transition_channel": int(zoom_channel + 1),
                "step_time_s": step_time,
                "unit_step_response_A": -np.expm1(-step_time[None, :] / tau[:, None]),
                "step_reach99_time_s": tau * np.log(100),
                "maximum_actual_quality_field_Hz": max_field,
                "actual_adc_field_limits_met": max_field <= field_limit_Hz + 1e-8,
                "absolute_actual_current_sum_at_adc_A": current_sums,
                "maximum_actual_current_sum_at_adc_A": float(current_sums.max()),
                "actual_adc_channel_limits_met": bool(np.all(np.abs(actual) <= self.numpy(self.max_current_A)[:, None] + 1e-8)),
                "actual_adc_current_sum_limits_met": bool(np.all(current_sums < total_current_limit_A)),
                "instantaneous_image": instantaneous_image, "instantaneous_kdata": instantaneous_kdata,
                "instantaneous_relative_mse": float(np.sum((np.abs(instantaneous_image) - np.abs(self.reference))**2) / energy),
                "response_image": response_image, "response_kdata": response_kdata,
                "response_relative_mse": float(np.sum((np.abs(response_image) - np.abs(self.reference))**2) / energy),
                "limit_check_domain": "ADC samples only; command-state bounds are optimization constraints"}


@torch.no_grad()
def optimize_image(model, initial_currents_A=None, *, total_current_limit_A=30.0,
                   field_limit_Hz=500.0, max_iterations=150, learning_rate_A=0.01,
                   target_mse=None, patience=25, min_relative_improvement=1e-6,
                   callback=None):
    """Projected Adam with a saved best feasible point and simple plateau stops.

    The exact magnitude-image objective and analytic adjoint run on the GPU.
    Adam is a local optimizer: the returned feasible point can be useful after
    an iteration limit without implying mathematical convergence. History
    reports evaluated values and best values separately.
    """
    if (isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1
            or isinstance(patience, bool) or int(patience) != patience or patience < 1):
        raise ValueError("max_iterations and patience must be positive integers.")
    if not np.isfinite(learning_rate_A) or learning_rate_A <= 0:
        raise ValueError("learning_rate_A must be positive and finite.")
    if target_mse is not None and (not np.isfinite(target_mse) or target_mse <= 0):
        raise ValueError("target_mse must be positive or None.")
    if not np.isfinite(min_relative_improvement) or min_relative_improvement < 0:
        raise ValueError("min_relative_improvement must be finite and nonnegative.")
    if initial_currents_A is None:
        initial_currents_A = np.zeros((model.nc, model.nstates))
    requested = model._check_currents(initial_currents_A)
    requested_value = float(model._objective_tensor(requested, gradient=False)[0])
    currents = model.project_tensor(requested, total_current_limit_A, field_limit_Hz)
    initial = currents.clone()
    best = currents.clone()
    best_value = np.inf
    moment, variance = torch.zeros_like(currents), torch.zeros_like(currents)
    learning_rate = float(learning_rate_A)
    history = {name: [] for name in ("iteration", "fval", "best_fval", "firstorderopt", "constrviolation", "learning_rate_A", "elapsed_s")}
    stale, reductions, adam_step = 0, 0, 0
    status, converged = "iteration_limit", False
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for iteration in range(int(max_iterations) + 1):
        value_tensor, gradient, _, _ = model._objective_tensor(currents)
        value = float(value_tensor)
        if not np.isfinite(value) or not bool(torch.isfinite(gradient).all()):
            status = "nonfinite_evaluation"
            break
        previous_best = best_value
        if value < best_value:
            best_value, best = value, currents.clone()
        significant = not np.isfinite(previous_best) or value < previous_best - min_relative_improvement * max(previous_best, 1e-12)
        stale = 0 if significant else stale + 1
        projected_step = currents - model.project_tensor(currents - gradient, total_current_limit_A, field_limit_Hz)
        first_order = float(projected_step.abs().max())
        elapsed = time.perf_counter() - started
        row = {"iteration": iteration, "fval": value, "best_fval": best_value,
               "firstorderopt": first_order, "constrviolation": 0.0,
               "learning_rate_A": learning_rate, "elapsed_s": elapsed}
        for key, item in row.items():
            history[key].append(item)
        if callback is not None:
            callback(row)
        if target_mse is not None and best_value <= target_mse:
            status, converged = "target_mse", True
            break
        if first_order < (1e-7 if model.dtype == torch.float64 else 1e-6):
            # Radial field contraction is not an orthogonal projection, so
            # this stopping test does not certify constrained stationarity.
            status = "projected_step_small"
            break
        if iteration == max_iterations:
            break
        if stale >= patience:
            if reductions >= 3:
                status = "plateau"
                break
            learning_rate *= 0.5
            currents = best.clone()
            moment.zero_()
            variance.zero_()
            stale, reductions, adam_step = 0, reductions + 1, 0
            # Re-evaluate at the restored best point before its next update.
            continue
        moment.mul_(0.9).add_(gradient, alpha=0.1)
        variance.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
        adam_step += 1
        direction = (moment / (1 - 0.9**adam_step)) / ((variance / (1 - 0.999**adam_step)).sqrt() + 1e-8)
        currents = model.project_tensor(currents - learning_rate * direction,
                                        total_current_limit_A, field_limit_Hz)
    if not np.isfinite(best_value):
        raise RuntimeError("No finite feasible objective evaluation was obtained.")
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    final_value, image, kdata = model._objective_tensor(best, gradient=False)
    commands = model.numpy(best)
    fields = model.maps_numpy @ commands
    limits = model.numpy(model.max_current_A)
    current_ok = bool(np.all(np.abs(commands) <= limits[:, None])
                      and np.all(np.abs(commands).sum(axis=0) < total_current_limit_A))
    field_ok = bool(np.max(np.abs(fields[model.quality_mask])) <= field_limit_Hz)
    initial_used = model.numpy(initial)
    diagnostics = model.response_diagnostics(commands, field_limit_Hz, total_current_limit_A)
    info = {"status": status, "exitflag": 1 if converged else 0,
            "fval": float(final_value), "converged": converged,
            "target_mse": target_mse,
            "target_reached": target_mse is not None and float(final_value) <= target_mse,
            "accepted": bool(current_ok and field_ok and np.isfinite(float(final_value))),
            "current_limits_met": current_ok, "field_limits_met": field_ok,
            "actual_adc_current_limits_met": diagnostics["actual_adc_channel_limits_met"] and diagnostics["actual_adc_current_sum_limits_met"],
            "actual_adc_field_limits_met": diagnostics["actual_adc_field_limits_met"],
            "iterations": len(history["iteration"]) - 1, "seconds": seconds,
            "initial_was_projected": not np.array_equal(initial_used, model.numpy(requested)),
            "algorithm": "projected Adam; box/L1 projection and radial field contraction"}
    return {"currents_A": commands, "initial_currents_A": initial_used,
            "requested_initial_currents_A": model.numpy(requested),
            "requested_initial_relative_mse": requested_value,
            "initial_relative_mse": history["fval"][0],
            "reference": model.reference, "image": model.numpy(image), "kdata": model.numpy(kdata),
            "generated_field_Hz": fields, "field_limit_Hz": field_limit_Hz,
            "current_response": model.current_response,
            "line_start_s": model.line_start_s.copy(), "line_end_s": model.line_end_s.copy(),
            "calibration_mode": "registered", "registration_confirmed": model.registration_confirmed,
            "history": {key: np.asarray(value) for key, value in history.items()},
            "optimization_info": info, "response_diagnostics": diagnostics,
            "runtime_info": model.runtime_info, "elapsed_seconds": seconds}
