"""Generation 09: rerun native PDG with ACDC in the B0 field at each evaluation.

The measured slice is one finite voxel thick. Native MRzero box kernels handle
the three local spatial axes and native kernels retain measured unresolved
dispersion. No phase or coherence factor is multiplied onto a cached signal.
The small local executor in native_pdg.py extends native MRzero emission for
event B0 maps and per-voxel encoding trajectories; the package is not patched.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from measured_image_target import magnitude_objective_terms
from native_resummation import resum_native_points
from native_pdg import execute_graph
from voxel_3d import FiniteVoxel3DACDCModel


class NativeB0ACDCModel(FiniteVoxel3DACDCModel):
    """Simulate independent ideal 90-degree excitation histories with native PDG.

    For each ADC, the effective field is B0 + integral(ACDC field)/elapsed.
    This is the exact accumulated field for stationary spins after one ideal
    RF. Its local derivatives are encoded in the trajectories supplied to the
    native voxel kernel. Hardware integrals include the complete preceding
    command history, even when ADC samples are simulated in smaller blocks.

    This adapter deliberately rejects refocusing, diffusion and multiple RF
    histories. It retains the generation 06 effective-amplitude/relaxation
    convention; it does not infer PD, T1 or a selective slice profile.
    """

    def __init__(self, experiment, calibration, max_current_A=3., **options):
        self._dynamic_phantoms = {}
        self.first_native_phantom = None
        self.first_native_simulation = None
        self._dynamic_calls = 0
        self._dynamic_seconds = 0.
        virtual_shape = np.asarray(experiment["m0_proxy"]).shape
        if len(virtual_shape) != 2 or virtual_shape[1] != 1:
            raise ValueError("Generation 09 uses one full-thickness voxel per parent, without z sub-slices.")
        metadata = experiment.get("inplane_voxel", {})
        edges = np.asarray(metadata.get("voxel_edge_vectors_m"), dtype=float)
        if edges.shape != (3, 3) or not np.isfinite(edges).all():
            raise ValueError("Native generation 09 requires finite voxel_edge_vectors_m[3,3].")
        self._native_sizes_numpy = np.linalg.norm(edges, axis=1)
        if np.any(self._native_sizes_numpy <= 0):
            raise ValueError("All three physical native voxel dimensions must be positive.")
        rf = np.asarray(experiment["acquisition"].get("rf_time_s", 0.))
        if np.unique(rf).size != 1:
            raise ValueError("Generation 09 supports one ideal RF history; multiple RF histories need a different adapter.")
        super().__init__(experiment, calibration, max_current_A, **options)
        # The common model initializer reserves a baseline cache. This model
        # emits fresh native signals and never uses that cache.
        self._base_cache = None
        self._native_cpu_blocks.clear()
        self._cache_base = False
        self._retain_contributions = False
        self.runtime_info.update({
            "simulator": "MRzeroCore PDG with local event-B0 executor",
            "native_executor_source": "MRzeroCore 1.2.0 simulation/main_pass.py; local documented extensions",
            "native_baseline_cache": "disabled; compute_graph and execute_graph run at every evaluation",
            "base_cache": False, "retain_contributions": False, "cache_MiB": 0.,
            "native_CPU_cache_MiB": 0., "native_cache_seconds": 0.,
            "native_voxel_size_m": self._native_sizes_numpy.tolist(),
            "native_voxel_dephasing": "native CustomVoxelPhantom voxel_shape=box on three local trajectories",
            "external_voxel_dephasing": False,
            "external_ACDC_signal_phase": False,
            "native_RF_model": "one independent nonselective ideal 90-degree excitation; stationary object; D=0",
            "native_graph_options": {"max_state_count": 16, "min_state_mag": 0.,
                "min_emitted_signal": 0., "min_latent_signal": 0.},
            "B0_gradient_policy": "measured in-plane slopes; unknown through-plane B0 slope is zero; calibrated ACDC slopes in all three axes",
            "ACDC_integration": "event-averaged total B0 given to native emission, including exact hardware-response integrals",
            "native_B0_adapter": "local native_pdg.py; event-averaged B0 plus per-voxel native kernel trajectories",
            "current_derivative": "PyTorch autograd through native PDG emission, evaluated in bounded ADC blocks",
        })
        self.runtime_info["inplane_voxel"].update({
            "native_point_dephasing": "replaced by native three-axis finite box kernel",
            "model": "native MRzero box voxels with locally linear B0/ACDC fields",
            "current_derivative": "native executor PyTorch autograd through effective B0 and local trajectories",
            "through_plane_sampling": "one finite native voxel across the full slice thickness",
        })
        self.runtime_info["subspins"].update({
            "native_voxel_size_m": self._native_sizes_numpy.tolist(),
            "children_per_parent": 1,
            "native_positions": "one center per full-thickness voxel; native local box kernels in all three axes",
        })
        self.runtime_info["multi_echo_coherence"].update({
            "implementation": "MRzero native box/Gaussian kernels inside local PDG executor",
            "device_cache_policy": "no signal cache; native kernel evaluated at each execution",
        })

    def _make_base(self, selected, b0_enabled=True):
        """Satisfy common initialization without running a cached-point model."""
        first, stop, step = selected.indices(self.nsamples)
        if step != 1:
            raise ValueError("Native blocks require consecutive stored sample indices.")
        return torch.zeros((int(self.support.sum()), stop - first),
                           device=self.device, dtype=self.complex_dtype)

    def _dynamic_phantom(self, b0_enabled):
        if b0_enabled in self._dynamic_phantoms:
            return self._dynamic_phantoms[b0_enabled]
        rate = self.numpy(self._rate).ravel()
        amplitude = self.numpy(self._amplitude).ravel()
        with np.errstate(over="ignore"):
            pd = amplitude * np.exp(self.amplitude_reference_time_s * rate)
        if not np.isfinite(pd).all() or np.any(pd > np.finfo(np.float32).max):
            raise ValueError("Echo-envelope extrapolation overflows native float32 MRzero.")
        t2 = np.divide(1., rate, out=np.full_like(rate, np.inf), where=rate > 0)
        if np.any(t2 < 1e-6):
            raise ValueError("Native MRzero relaxation times must be at least one microsecond.")
        positions = np.column_stack((self.numpy(self._x).ravel(), self.numpy(self._y).ravel(),
                                     self._subspin_z_numpy[self.support])).astype(np.float32)
        b0 = self.numpy(self._b0).ravel() if b0_enabled else np.zeros_like(rate)
        phantom = self._mr0.CustomVoxelPhantom(
            positions, PD=pd.astype(np.float32), T1=np.inf, T2=t2.astype(np.float32),
            T2dash=np.inf, B0=b0.astype(np.float32), B1=1., D=0.,
            voxel_size=self._native_sizes_numpy.astype(np.float32), voxel_shape="box").build()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                phantom = phantom.cuda()
        # SimData clamps diffusion during build/cuda. Set the intended D=0
        # afterwards and bind the original native kernel on the chosen device.
        from MRzeroCore.phantom.custom_voxel_phantom import build_dephasing_func
        phantom.D = torch.zeros_like(phantom.D)
        phantom.dephasing_func = build_dephasing_func(
            "box", torch.as_tensor(self._native_sizes_numpy, device=self.device, dtype=torch.float32))
        self._dynamic_phantoms[b0_enabled] = phantom
        return phantom

    def _contribution(self, currents, selected, response_enabled=None, b0_enabled=True):
        self._ensure_voxel_tensors()
        indices = np.arange(self.nsamples)[selected]
        elapsed = self.numpy(self._evolution)[indices]
        order = np.argsort(elapsed, kind="stable")
        indices, elapsed = indices[order], elapsed[order]
        durations = np.diff(np.r_[0., elapsed])
        if np.any(durations <= 0):
            raise ValueError("Native ADC centers must be distinct and positive after RF.")
        integrated = self._voxel_integrated_currents(currents, indices, response_enabled)
        t = self._evolution[indices]
        center_integral = self.B @ integrated
        effective_b0 = center_integral.T / t[:, None]
        if b0_enabled:
            effective_b0 = effective_b0 + self._b0.T
        cycles = torch.stack([
            (self._voxel_edges[axis, 0] * self._kx[indices]
             + self._voxel_edges[axis, 1] * self._ky[indices])[None, :]
            + self._voxel_field_gradient[:, axis] @ integrated
            + (self._voxel_b0_gradient[:, axis, None] * t[None, :]
               if b0_enabled else 0.)
            for axis in range(3)], dim=-1)
        sizes = self.tensor(self._native_sizes_numpy)
        local_k = cycles.permute(1, 0, 2) / sizes

        k = np.column_stack((self.numpy(self._kx)[indices], self.numpy(self._ky)[indices],
                             np.zeros(len(indices))))
        moments = np.diff(np.vstack((np.zeros(3), k)), axis=0)
        repetition = self._mr0.Repetition.zero(len(indices))
        repetition.pulse.angle = torch.tensor(np.pi / 2, dtype=torch.float32)
        repetition.pulse.phase = torch.tensor(np.pi / 2, dtype=torch.float32)
        repetition.pulse.usage = self._mr0.PulseUsage.EXCIT
        repetition.pulse.selective = False
        repetition.event_time[:] = torch.from_numpy(durations.astype(np.float32))
        repetition.gradm[:] = torch.from_numpy(moments.astype(np.float32))
        repetition.adc_usage[:] = 1
        repetition.adc_phase[:] = torch.from_numpy(self.numpy(self._phase)[indices].astype(np.float32))
        sequence = self._mr0.Sequence([repetition], normalized_grads=False)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                sequence = sequence.cuda()
        phantom = self._dynamic_phantom(b0_enabled)
        if self._coherence_numpy is None:
            width = np.zeros(int(self.support.sum()))
            kind = np.zeros(int(self.support.sum()), dtype=np.int8)
        else:
            widths, kinds = self._coherence_numpy
            width, kind = widths[self.support], kinds[self.support]
        started = perf_counter()
        graph = self._mr0.compute_graph(sequence, phantom, max_state_count=16, min_state_mag=0.)
        signal, pathways = execute_graph(graph, sequence, phantom,
            min_emitted_signal=0., min_latent_signal=0., print_progress=False, return_mag_adc=True,
            b0_at_adc_Hz=effective_b0.to(torch.float32), local_k_at_adc_mInv=local_k.to(torch.float32),
            unresolved_width_Hz=torch.as_tensor(width, device=self.device, dtype=torch.float32),
            dispersion_kind=torch.as_tensor(kind, device=self.device))
        if len(pathways) != 1 or len(pathways[0]) != 1:
            raise RuntimeError("Expected one native transverse coherence after one ideal excitation.")
        magnetization = pathways[0][0] * phantom.coil_sens[0][None, :]
        self._dynamic_calls += 1
        self._dynamic_seconds += perf_counter() - started
        self._native_block_count = self._dynamic_calls
        self._native_simulation_seconds = self._dynamic_seconds
        self.runtime_info.update(native_evaluations=self._dynamic_calls,
            native_evaluation_seconds=self._dynamic_seconds,
            native_blocks_simulated=self._dynamic_calls,
            native_cache_seconds=0.)
        if b0_enabled and self.first_native_phantom is None and bool(torch.all(currents.detach() == 0)):
            _, diagnostic = resum_native_points(magnetization.detach(), signal[:, 0].detach())
            self.runtime_info.update(
                native_resummation_relative_error=diagnostic["relative_error"],
                native_resummation_mass_relative_error=diagnostic["mass_relative_error"],
                native_resummation_max_tolerance_ratio=diagnostic["max_tolerance_ratio"])
            self.first_native_phantom = phantom
            self.first_native_simulation = {
                "B0_enabled": bool(b0_enabled), "ADC_samples": len(indices),
                "stage": "first B0-enabled uncalibrated baseline block",
                "zero_currents": bool(torch.all(currents.detach() == 0)),
                "voxel_size_m": self._native_sizes_numpy.tolist(), "voxel_shape": "box",
                "b0_event_min_Hz": float(effective_b0.detach().min()),
                "b0_event_max_Hz": float(effective_b0.detach().max()),
                "b0_center_min_Hz": float(phantom.B0.min()),
                "b0_center_max_Hz": float(phantom.B0.max()),
                "resummation": diagnostic,
                "model": "local native PDG executor; event total B0 and native three-axis box kernel",
            }
        inverse = torch.as_tensor(np.argsort(order), device=self.device)
        return magnetization.index_select(0, inverse).T.to(self.complex_dtype)

    @torch.no_grad()
    def _objective_tensor(self, currents, gradient=True):
        image, kdata = self._forward_tensor(currents)
        value, derivative = magnitude_objective_terms(self, image, gradient=gradient)
        if not gradient:
            return value, image, kdata
        adjoint = self.reconstruction.adjoint(derivative).reshape(-1)
        result = torch.zeros_like(currents)
        # Rebuild only one native ADC block with autograd enabled. The image
        # adjoint holds the exact sensitivity from the complete reconstruction.
        for first in range(0, self.nsamples, self.sample_chunk):
            selected = slice(first, min(first + self.sample_chunk, self.nsamples))
            with torch.enable_grad():
                variable = currents.detach().requires_grad_(True)
                contribution = self._contribution(variable, selected)
                linear_objective = (contribution.sum(0) * adjoint[selected].conj()).real.sum()
                result += torch.autograd.grad(linear_objective, variable)[0].detach()
        return value, result, image, kdata


__all__ = ["NativeB0ACDCModel"]
