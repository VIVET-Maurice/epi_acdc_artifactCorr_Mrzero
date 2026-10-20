"""Native MRzero baseline for the measured ACDC experiment.

MRzero simulates the transverse magnetization of every selected object point
at the measured ADC sample centers. The existing ACDC current integrals then
apply the additional spatial shim phase before the receiver sums the points.
This factorization is exact for stationary point spins after one ideal RF;
it does not require rerunning the native simulator at every optimizer step.

The measured GRE magnitude is an effective transverse-amplitude proxy at TE,
not a measured proton-density or T1 map. The ideal 90-degree excitation and
echo-relative T2* convention therefore preserve the original experiment model.
GRE lines are independently excited, perfectly spoiled acquisitions; this
adapter does not infer a physical GRE steady state, slice profile or receive
coil sensitivity from magnitude data.
"""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch


# Keep the original experiment readers, current model and numerical limits.
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "Experiment"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from acdc_gpu import ACDCModel


class MRzeroExperimentModel(ACDCModel):
    """ACDC experiment model whose fixed voxel magnetization is native MRzero.

    The public constructor, ``forward``, ``objective_and_grad``, projection,
    waveform and response diagnostics match ``Experiment.acdc_gpu.ACDCModel``.
    Native MRzero uses float32; ``precision='float64'`` increases the precision
    of current phase, reconstruction and the adjoint, without changing that
    native simulation precision.

    Every fixed baseline, including the ideal zero-B0 reference, is obtained
    through ``MRzeroCore.compute_graph`` and ``execute_graph``. The analytical
    Fourier baseline in the parent class is never called. Only the exact
    additional current phase and its derivative are evaluated analytically.
    """

    def __init__(self, experiment, calibration, max_current_A=3.0, *,
                 profile="auto", precision="auto", memory_fraction=0.5,
                 sample_chunk=None):
        import MRzeroCore as mr0

        self._mr0 = mr0
        self._native_phantoms = {}
        self._native_cpu_blocks = {}
        self._native_simulation_seconds = 0.0
        self._native_block_count = 0
        self._native_resummation_error = 0.0
        # The parent validates geometry, masks, acquisition order and current
        # timing before lazily calling our _make_base with those attributes.
        super().__init__(experiment, calibration, max_current_A,
                         profile=profile, precision=precision,
                         memory_fraction=memory_fraction,
                         sample_chunk=sample_chunk)
        # With limited GPU memory the parent does not cache its baseline.
        # Prepare native blocks on the CPU once so iterations remain fast.
        if self._base_cache is None:
            for first in range(0, self.nsamples, self.sample_chunk):
                selected = slice(first, min(first + self.sample_chunk, self.nsamples))
                self._make_base(selected, b0_enabled=True)
        # Native phantom arrays are small, but no longer needed by iterations.
        self._native_phantoms.clear()
        self.runtime_info.update({
            "simulator": "MRzeroCore",
            "mrzero_version": mr0.__version__,
            "native_precision": "float32",
            "native_cache_seconds": self._native_simulation_seconds,
            "native_blocks_simulated": self._native_block_count,
            "native_resummation_relative_error": self._native_resummation_error,
            "native_baseline_cache": "GPU" if self._base_cache is not None else "CPU",
            "native_CPU_cache_MiB": sum(v.numel() * v.element_size()
                                        for v in self._native_cpu_blocks.values()) / 1024**2,
            "native_RF_model": "independent ideal 90-degree excitation per RF history",
            "object_amplitude": "measured GRE transverse amplitude at TE",
        })

    def _phantom(self, b0_enabled):
        """Make stationary point spins using the validated experiment arrays."""
        if b0_enabled in self._native_phantoms:
            return self._native_phantoms[b0_enabled]
        mr0 = self._mr0
        rate = self.numpy(self._rate).reshape(-1)
        amplitude = self.numpy(self._amplitude).reshape(-1)
        # MRzero decays from excitation; our proxy is measured at TE.
        # Undo TE decay in the input amplitude, then let MRzero apply T2*.
        with np.errstate(over="ignore", divide="ignore"):
            pd = amplitude * np.exp(self.TE_s * rate)
            t2star = np.divide(1.0, rate, out=np.full_like(rate, np.inf), where=rate > 0)
        if not np.isfinite(pd).all() or np.any(pd > np.finfo(np.float32).max):
            raise ValueError("The TE-relative amplitude overflows native MRzero; inspect T2star_s and TE_s.")
        if np.any(t2star < 1e-6):
            raise ValueError("Native MRzero requires T2star_s >= 1 microsecond.")
        positions = np.column_stack((self.numpy(self._x).ravel(),
                                     self.numpy(self._y).ravel(),
                                     np.zeros(amplitude.size))).astype(np.float32)
        b0 = (self.numpy(self._b0).ravel() if b0_enabled else np.zeros(amplitude.size))
        phantom = mr0.CustomVoxelPhantom(
            positions, PD=pd.astype(np.float32),
            B0=np.asarray(b0, np.float32), B1=1.0,
            T1=np.inf, T2=t2star.astype(np.float32), T2dash=np.inf,
            D=0.0, voxel_size=0.0, voxel_shape="box",
        ).build()
        if self.device.type == "cuda":
            # SimData.cuda() uses the current device rather than a device arg.
            with torch.cuda.device(self.device):
                phantom = phantom.cuda()
        # SimData applies a minimum positive D during construction/cuda().
        # Disable it afterwards and explicitly retain point-spin encoding.
        phantom.D = torch.zeros_like(phantom.D)
        phantom.dephasing_func = lambda k, _: torch.ones(
            k.shape[0], device=k.device, dtype=k.dtype)
        self._native_phantoms[b0_enabled] = phantom
        return phantom

    def _simulate_rf_history(self, sample_indices, b0_enabled):
        """Simulate one ideal-RF history and restore the requested sample order."""
        mr0 = self._mr0
        indices = np.asarray(sample_indices, dtype=np.int64)
        elapsed = self.numpy(self._evolution)[indices]
        order = np.argsort(elapsed, kind="stable")
        elapsed = elapsed[order].astype(np.float64)
        durations = np.diff(np.r_[0.0, elapsed])
        if np.any(durations <= 0):
            raise ValueError("MRzero needs distinct positive ADC sample centers within each RF history.")
        k = np.column_stack((self.numpy(self._kx)[indices][order],
                             self.numpy(self._ky)[indices][order],
                             np.zeros(indices.size))).astype(np.float64)
        moments = np.diff(np.vstack((np.zeros(3), k)), axis=0)
        phases = self.numpy(self._phase)[indices][order]

        repetition = mr0.Repetition.zero(indices.size)
        repetition.pulse.angle = torch.tensor(np.pi / 2, dtype=torch.float32)
        # Native MRzero starts Mxy = -i * exp(i RFphase), so pi/2 starts real.
        repetition.pulse.phase = torch.tensor(np.pi / 2, dtype=torch.float32)
        repetition.pulse.usage = mr0.PulseUsage.EXCIT
        repetition.pulse.selective = False
        repetition.event_time[:] = torch.from_numpy(durations.astype(np.float32))
        # Gradients are integrated moments in cycles/m, not gradient amplitudes.
        repetition.gradm[:] = torch.from_numpy(moments.astype(np.float32))
        repetition.adc_usage[:] = 1
        # RF minus receiver phase is already retained in the measured adapter.
        repetition.adc_phase[:] = torch.from_numpy(phases.astype(np.float32))
        sequence = mr0.Sequence([repetition], normalized_grads=False)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                sequence = sequence.cuda()
        phantom = self._phantom(b0_enabled)

        started = perf_counter()
        graph = mr0.compute_graph(sequence, phantom, max_state_count=16, min_state_mag=0.0)
        with torch.no_grad():
            signal, pathways = mr0.execute_graph(
                graph, sequence, phantom, min_emitted_signal=0.0,
                min_latent_signal=0.0, print_progress=False, return_mag_adc=True,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._native_simulation_seconds += perf_counter() - started
        self._native_block_count += 1
        if len(pathways) != 1 or len(pathways[0]) != 1:
            raise RuntimeError("Expected one MRzero transverse coherence after an ideal excitation.")
        # Native ADC magnetization includes proton density and receiver phase.
        # Apply receive sensitivity once, before summing individual object points.
        magnetization = pathways[0][0].detach() * phantom.coil_sens[0][None, :]
        resummed = magnetization.sum(dim=1)
        native = signal[:, 0].detach()
        scale = torch.linalg.vector_norm(native).clamp_min(torch.finfo(torch.float32).tiny)
        error = float(torch.linalg.vector_norm(resummed - native) / scale)
        self._native_resummation_error = max(self._native_resummation_error, error)
        if not np.isfinite(error) or error > 5e-5:
            raise RuntimeError(f"Native MRzero receive resummation differs from its ADC signal ({error:.3g}).")
        if not bool(torch.isfinite(magnetization).all()):
            raise ValueError("MRzero returned nonfinite voxel magnetization.")
        inverse = torch.as_tensor(np.argsort(order), device=self.device)
        return magnetization.index_select(0, inverse).T.to(self.complex_dtype)

    def _make_base(self, selected, b0_enabled=True):
        """Obtain the fixed signal contribution from native MRzero, never NumPy."""
        first, stop, step = selected.indices(self.nsamples)
        if step != 1:
            raise ValueError("Native baseline blocks require consecutive Cartesian sample indices.")
        key = (first, stop)
        if b0_enabled and key in self._native_cpu_blocks:
            return self._native_cpu_blocks[key].to(self.device)
        indices = np.arange(first, stop)
        result = torch.empty((int(self.support.sum()), indices.size),
                             device=self.device, dtype=self.complex_dtype)
        rf = self.rf_time_s.reshape(-1)[indices]
        # Sorting ADCs preserves bipolar EPI chronology. A spoiled GRE line has
        # its own RF; splitting prevents previous lines' phase or saturation.
        for rf_time in np.unique(rf):
            local = np.flatnonzero(rf == rf_time)
            values = self._simulate_rf_history(indices[local], b0_enabled)
            destination = torch.as_tensor(local, device=self.device)
            result.index_copy_(1, destination, values)
        if b0_enabled and not self._cache_base:
            self._native_cpu_blocks[key] = result.detach().cpu()
        return result


__all__ = ["MRzeroExperimentModel"]
