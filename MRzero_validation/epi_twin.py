"""Bounded native MRzero baseline twins independent of ACDC calibration.

Both twins share the full object, RF/ADC history, relaxation, receiver weights
and continuous reconstruction. Only the baseline off-resonance is switched.
The input amplitude is an effective transverse amplitude at an explicit time
after RF; it is not implicitly interpreted as quantitative proton density.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from continuous_reconstruction import ContinuousReconstruction
from native_resummation import RESUMMATION_RTOL, RESUMMATION_ROUNDOFF, resum_native_points


class NativeEPITwin:
    """Simulate stationary point spins under actual nominal ADC k coordinates.

    Spatial arrays share the object grid, independently of the acquired image
    matrix. ``rate_sInv`` is a nonnegative decay rate (scalar or object map).
    ``amplitude_reference_time_s`` anchors that amplitude after each RF; MRzero
    receives ``PD=amplitude*exp(anchor*rate)`` and applies its native decay.
    The caller supplies any pixel-area/volume-fraction weights in amplitude.

    Optional complex ``receive_weights[receiver, object_row, object_column]``
    must have unit RSS norm at active points. This distributes an effective
    RSS amplitude among receivers without applying receive magnitude twice.
    No ACDC basis, coil-coverage mask or optimization result is required.
    """

    def __init__(self, acquisition, X_m, Y_m, amplitude, B0_Hz, support_mask,
                 rate_sInv=0, amplitude_reference_time_s=0, device="cpu",
                 sample_chunk=128, receive_weights=None):
        import MRzeroCore as mr0

        self._mr0 = mr0
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select device='cpu'.")
        if (isinstance(sample_chunk, bool) or int(sample_chunk) != sample_chunk
                or sample_chunk < 1):
            raise ValueError("sample_chunk must be a positive integer.")
        self.sample_chunk = int(sample_chunk)
        amplitude = np.asarray(amplitude)
        if (amplitude.ndim != 2 or np.iscomplexobj(amplitude)
                or not np.isfinite(amplitude).all() or np.any(amplitude < 0)):
            raise ValueError("amplitude must be a finite nonnegative real 2D object.")
        self.object_shape = amplitude.shape
        mask = np.asarray(support_mask)
        if mask.shape != self.object_shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("support_mask must be binary and match the object grid.")
        self.support = mask.astype(bool) & (amplitude > 0)
        if not self.support.any():
            raise ValueError("Object support must include positive amplitude.")
        arrays = {}
        for name, value in (("X_m", X_m), ("Y_m", Y_m), ("B0_Hz", B0_Hz)):
            value = np.asarray(value)
            if (value.shape != self.object_shape or np.iscomplexobj(value)
                    or not np.isfinite(value[self.support]).all()):
                raise ValueError(f"{name} must be real, match the object and be finite on its support.")
            arrays[name] = value.astype(np.float64)
        rate = np.asarray(rate_sInv)
        if rate.ndim == 0:
            rate = np.full(self.object_shape, rate.item())
        if (rate.shape != self.object_shape or np.iscomplexobj(rate)
                or not np.isfinite(rate[self.support]).all() or np.any(rate[self.support] < 0)):
            raise ValueError("rate_sInv must be a nonnegative finite scalar or object map.")
        anchor = float(amplitude_reference_time_s)
        if not np.isfinite(anchor) or anchor < 0:
            raise ValueError("amplitude_reference_time_s must be nonnegative finite seconds.")
        self.anchor_s = anchor
        self._rate = np.asarray(rate[self.support], dtype=np.float64)
        with np.errstate(over="ignore"):
            self._pd = np.asarray(amplitude[self.support], dtype=np.float64) * np.exp(anchor * self._rate)
        if not np.isfinite(self._pd).all() or np.any(self._pd > np.finfo(np.float32).max):
            raise ValueError("Amplitude extrapolation to RF overflows native float32 MRzero.")
        self._positions = np.column_stack((arrays["X_m"][self.support], arrays["Y_m"][self.support],
                                          np.zeros(self.support.sum()))).astype(np.float32)
        self._b0 = arrays["B0_Hz"][self.support].astype(np.float32)
        self._t2 = np.divide(1., self._rate, out=np.full_like(self._rate, np.inf), where=self._rate > 0)
        if np.any(self._t2 < 1e-6):
            raise ValueError("Native MRzero requires relaxation times >= 1 microsecond.")
        self.image_shape = np.asarray(acquisition["kx_mInv"]).shape
        if len(self.image_shape) != 2 or min(self.image_shape) < 1:
            raise ValueError("Acquisition needs nonempty two-dimensional ADC arrays.")
        self.nsamples = int(np.prod(self.image_shape))
        self._acquisition = {}
        for name in ("kx_mInv", "ky_mInv", "time_s"):
            value = np.asarray(acquisition[name])
            if value.shape != self.image_shape or np.iscomplexobj(value) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite real acquisition-grid array.")
            self._acquisition[name] = np.asarray(value, dtype=np.float64).reshape(-1)
        for name in ("rf_time_s", "phase_offset_rad"):
            value = np.asarray(acquisition.get(name, np.zeros(self.image_shape)))
            if value.shape != self.image_shape or np.iscomplexobj(value) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite real acquisition-grid array.")
            self._acquisition[name] = np.asarray(value, dtype=np.float64).reshape(-1)
        self._elapsed = self._acquisition["time_s"] - self._acquisition["rf_time_s"]
        if np.any(self._acquisition["rf_time_s"] < 0) or np.any(self._elapsed <= 0):
            raise ValueError("Each ADC must occur strictly after its nonnegative RF time.")
        if receive_weights is None:
            receive = np.ones((1, int(self.support.sum())), dtype=np.complex64)
        else:
            receive = np.asarray(receive_weights)
            if (receive.ndim != 3 or receive.shape[1:] != self.object_shape or receive.shape[0] < 1
                    or not np.isfinite(receive[:, self.support]).all()):
                raise ValueError("receive_weights need [receiver, object row, object column] finite support.")
            receive = receive[:, self.support]
            norm = np.sqrt(np.sum(np.abs(receive)**2, axis=0))
            if not np.allclose(norm, 1., rtol=1e-5, atol=1e-6):
                raise ValueError("receive_weights must have unit RSS norm at every active object point.")
        self._receive = torch.as_tensor(receive, device=self.device, dtype=torch.complex64)
        self.nreceivers = self._receive.shape[0]
        reconstruction_dtype = torch.float32 if self.device.type == "cuda" else torch.float64
        self.reconstruction = ContinuousReconstruction(acquisition, self.device, reconstruction_dtype)

    def _phantom(self, b0_enabled):
        phantom = self._mr0.CustomVoxelPhantom(
            self._positions, PD=self._pd.astype(np.float32),
            B0=self._b0 if b0_enabled else np.zeros_like(self._b0), B1=1.0,
            T1=np.inf, T2=self._t2.astype(np.float32), T2dash=np.inf,
            D=0., voxel_size=0., voxel_shape="box",
        ).build()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                phantom = phantom.cuda()
        phantom.D = torch.zeros_like(phantom.D)
        phantom.dephasing_func = lambda k, _: torch.ones(k.shape[0], device=k.device, dtype=k.dtype)
        # Native one-coil resummation is the oracle; effective receiver weights
        # are applied exactly once to the returned point magnetization below.
        phantom.coil_sens = torch.ones_like(phantom.coil_sens[:1])
        return phantom

    @torch.no_grad()
    def _simulate_history(self, indices, phantom):
        mr0 = self._mr0
        order = np.argsort(self._elapsed[indices], kind="stable")
        chronological = indices[order]
        elapsed = self._elapsed[chronological]
        durations = np.diff(np.r_[0., elapsed])
        if np.any(durations <= 0):
            raise ValueError("ADC times must be distinct within each ideal-RF history.")
        k = np.column_stack((self._acquisition["kx_mInv"][chronological],
                             self._acquisition["ky_mInv"][chronological], np.zeros(len(indices))))
        moments = np.diff(np.vstack((np.zeros(3), k)), axis=0)
        repetition = mr0.Repetition.zero(len(indices))
        repetition.pulse.angle = torch.tensor(np.pi / 2, dtype=torch.float32)
        repetition.pulse.phase = torch.tensor(np.pi / 2, dtype=torch.float32)
        repetition.pulse.usage = mr0.PulseUsage.EXCIT
        repetition.pulse.selective = False
        repetition.event_time[:] = torch.from_numpy(durations.astype(np.float32))
        repetition.gradm[:] = torch.from_numpy(moments.astype(np.float32))
        repetition.adc_usage[:] = 1
        repetition.adc_phase[:] = torch.from_numpy(
            self._acquisition["phase_offset_rad"][chronological].astype(np.float32))
        sequence = mr0.Sequence([repetition], normalized_grads=False)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                sequence = sequence.cuda()
        graph = mr0.compute_graph(sequence, phantom, max_state_count=16, min_state_mag=0.)
        signal, pathways = mr0.execute_graph(
            graph, sequence, phantom, min_emitted_signal=0., min_latent_signal=0.,
            print_progress=False, return_mag_adc=True)
        if len(pathways) != 1 or len(pathways[0]) != 1:
            raise RuntimeError("Expected one transverse pathway after an ideal excitation.")
        point_mag = pathways[0][0].detach()
        native = signal[:, 0].detach()
        resummed, diagnostics = resum_native_points(point_mag, native)
        # Sum in double precision without retaining a double point block.
        # The one uniform receiver uses the already computed validation sum.
        if self.nreceivers == 1 and bool(torch.all(self._receive == 1)):
            receiver_signal = resummed[None].to(torch.complex64)
        else:
            receiver_signal = torch.stack([
                (point_mag * weights[None]).sum(dim=1, dtype=torch.complex128)
                for weights in self._receive
            ]).to(torch.complex64)
        inverse = torch.as_tensor(np.argsort(order), device=self.device)
        return receiver_signal.index_select(1, inverse), diagnostics

    @torch.no_grad()
    def simulate(self, b0_enabled=True):
        """Return RSS image, complex coil images/k-space and native diagnostics."""
        if not isinstance(b0_enabled, (bool, np.bool_)):
            raise ValueError("b0_enabled must be boolean.")
        started = perf_counter()
        phantom = self._phantom(b0_enabled)
        signal = torch.empty((self.nreceivers, self.nsamples), device=self.device, dtype=torch.complex64)
        maxima = {"relative_error": 0., "mass_relative_error": 0., "max_tolerance_ratio": 0.}
        blocks = 0
        for first in range(0, self.nsamples, self.sample_chunk):
            indices = np.arange(first, min(first + self.sample_chunk, self.nsamples))
            rf = self._acquisition["rf_time_s"][indices]
            for center in np.unique(rf):
                selected = indices[rf == center]
                values, diagnostics = self._simulate_history(selected, phantom)
                destinations = torch.as_tensor(selected, device=self.device)
                signal.index_copy_(1, destinations, values)
                for name, error in diagnostics.items():
                    maxima[name] = max(maxima[name], error)
                blocks += 1
        kdata = signal.reshape((self.nreceivers,) + self.image_shape)
        coil_images = self.reconstruction.image(kdata)
        image = torch.sqrt(torch.sum(torch.abs(coil_images)**2, dim=0))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        report = {
            "simulator": "MRzeroCore", "mrzero_version": self._mr0.__version__,
            "native_precision": "float32", "device": str(self.device), "B0_enabled": bool(b0_enabled),
            "object_shape": list(self.object_shape), "active_points": int(self.support.sum()),
            "image_shape": list(self.image_shape), "receivers": int(self.nreceivers),
            "sample_chunk": self.sample_chunk, "native_blocks_simulated": blocks,
            "native_resummation_relative_error": maxima["relative_error"],
            "native_resummation_mass_relative_error": maxima["mass_relative_error"],
            "native_resummation_max_tolerance_ratio": maxima["max_tolerance_ratio"],
            "native_resummation_relative_tolerance": RESUMMATION_RTOL,
            "native_resummation_roundoff_tolerance": RESUMMATION_ROUNDOFF,
            "receiver_accumulation": "complex128, returned as complex64",
            "amplitude_reference_time_s": self.anchor_s,
            "point_magnetization_cache": False,
            "native_point_block_MiB": int(self.support.sum()) * min(self.sample_chunk, self.nsamples) * 8 / 1024**2,
            "amplitude_weighting": "caller-supplied pixel/volume weights, applied once",
            "receiver_normalization": "unit RSS norm at each active point",
            "RF_model": "independent ideal 90-degree excitation per RF history",
            "native_phase_convention": "exp(-2*pi*i*(k*r+B0*t))*exp(+i ADCphase), real initial Mxy",
            "reconstruction": self.reconstruction.report, "elapsed_s": perf_counter() - started,
        }
        return {"image": image.cpu().numpy(), "coil_images": coil_images.cpu().numpy(),
                "kdata": kdata.cpu().numpy(), "report": report}


__all__ = ["NativeEPITwin"]
