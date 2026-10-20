"""Native MRzero subspins with shared ACDC commands and child-wise fields.

The expanded object's two axes are storage axes, ``[parent, child]``. They are
independent of the acquired image matrix. Each child's input amplitude already
contains its volume-fraction weight; the simulator must not weight it again.
Only display fields are averaged back to the original parent grid. Signal
phases, current gradients and hardware field limits use every actual child.
"""

from __future__ import annotations

import numpy as np
import torch

from experiment_realistic import RealisticACDCModel


class SubspinACDCModel(RealisticACDCModel):
    """Extend notebook 04 with native MRzero at explicit XYZ child positions.

    ``experiment['subspins']`` supplies ``parent_shape``, retained
    ``parent_flat_index`` and ``weights[parent, child]`` summing to one per
    parent. All spatial input arrays, including ``geometry.Z_m``, use the
    virtual parent/child shape. Acquisition arrays retain their original image
    shape. The constructor and hardware/reconstruction options match 04.
    """

    def __init__(self, experiment, calibration, max_current_A=3.0, *,
                 hardware=None, reconstruction_chunk=256, **options):
        metadata = experiment.get("subspins")
        if not isinstance(metadata, dict):
            raise ValueError("Expanded experiment requires subspins metadata.")
        weights = np.asarray(metadata["weights"], dtype=np.float64)
        virtual_shape = np.asarray(experiment["m0_proxy"]).shape
        if (weights.ndim != 2 or weights.shape != virtual_shape
                or not np.isfinite(weights).all() or np.any(weights <= 0)
                or not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-12)):
            raise ValueError("Subspin weights must be positive [parent, child] fractions summing to one.")
        parent_shape = tuple(int(value) for value in metadata["parent_shape"])
        if len(parent_shape) != 2 or min(parent_shape) < 1:
            raise ValueError("parent_shape must be a nonempty two-dimensional display grid.")
        parent_indices = np.asarray(metadata["parent_flat_index"])
        if (parent_indices.shape != (weights.shape[0],)
                or not np.isfinite(parent_indices).all()
                or np.any(parent_indices != np.floor(parent_indices))
                or np.any(parent_indices < 0) or np.any(parent_indices >= np.prod(parent_shape))
                or np.unique(parent_indices).size != parent_indices.size):
            raise ValueError("parent_flat_index must select one distinct valid display pixel per parent.")
        z = np.asarray(experiment["geometry"]["Z_m"])
        if z.shape != virtual_shape or np.iscomplexobj(z) or not np.isfinite(z).all():
            raise ValueError("geometry.Z_m must be finite real XYZ child coordinates on the virtual grid.")
        self.parent_shape = parent_shape
        self.parent_flat_index = parent_indices.astype(np.int64)
        self.subspin_weights = weights.copy()
        self._subspin_z_numpy = z.astype(np.float64)
        self.subspin_metadata = metadata
        super().__init__(experiment, calibration, max_current_A,
                         hardware=hardware, reconstruction_chunk=reconstruction_chunk, **options)
        self.runtime_info["subspins"] = {
            "parent_shape": list(self.parent_shape),
            "retained_parents": int(weights.shape[0]),
            "children_per_parent": int(weights.shape[1]),
            "active_children": int(self.support.sum()),
            "signal_weighting": "child m0_proxy already includes its volume fraction",
            "field_constraints": "all reliable child fields; no parent averaging",
            "display_fields": "volume-fraction average only",
            "native_positions": "sequence X, sequence Y, slice-normal offset; ADC kz remains zero",
            "native_voxel_size_m": 0.0,
            "native_diffusion_enabled": False,
            "through_plane_B0": metadata.get("through_plane_B0", "see sampling report"),
        }

    def _phantom(self, b0_enabled):
        """Native weighted point spins, with each child's true XYZ and B0."""
        if b0_enabled in self._native_phantoms:
            return self._native_phantoms[b0_enabled]
        rate = self.numpy(self._rate).reshape(-1)
        amplitude = self.numpy(self._amplitude).reshape(-1)
        with np.errstate(over="ignore", divide="ignore"):
            pd = amplitude * np.exp(self.TE_s * rate)
            t2star = np.divide(1.0, rate, out=np.full_like(rate, np.inf), where=rate > 0)
        if not np.isfinite(pd).all() or np.any(pd > np.finfo(np.float32).max):
            raise ValueError("The weighted TE-relative amplitude overflows native MRzero.")
        if np.any(t2star < 1e-6):
            raise ValueError("Native MRzero requires T2star_s >= 1 microsecond.")
        positions = np.column_stack((self.numpy(self._x).ravel(),
                                     self.numpy(self._y).ravel(),
                                     self._subspin_z_numpy[self.support])).astype(np.float32)
        b0 = self.numpy(self._b0).ravel() if b0_enabled else np.zeros(amplitude.size)
        phantom = self._mr0.CustomVoxelPhantom(
            positions, PD=pd.astype(np.float32),
            B0=np.asarray(b0, np.float32), B1=1.0,
            T1=np.inf, T2=t2star.astype(np.float32), T2dash=np.inf,
            D=0.0, voxel_size=0.0, voxel_shape="box",
        ).build()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                phantom = phantom.cuda()
        # Each child is already an explicit spin. Additional finite-voxel or
        # diffusion attenuation would count an unmodeled second dispersion.
        phantom.D = torch.zeros_like(phantom.D)
        phantom.dephasing_func = lambda k, _: torch.ones(
            k.shape[0], device=k.device, dtype=k.dtype)
        self._native_phantoms[b0_enabled] = phantom
        return phantom

    def _parent_image(self, values, fill_value=np.nan):
        values = np.asarray(values)
        output = np.full((int(np.prod(self.parent_shape)),) + values.shape[1:],
                         fill_value, dtype=np.result_type(values.dtype, np.float64))
        output[self.parent_flat_index] = values
        return output.reshape(self.parent_shape + values.shape[1:])

    def aggregate_to_parent(self, values, fill_value=np.nan):
        """Return volume-fraction averages for display, never signal/limits.

        ``values`` must have leading virtual ``[parent, child]`` axes and may
        have trailing state dimensions. Unretained parent pixels receive NaN
        by default. The returned NumPy array has the original parent grid.
        """
        if isinstance(values, torch.Tensor):
            values = self.numpy(values)
        values = np.asarray(values)
        if values.shape[:2] != self.subspin_weights.shape:
            raise ValueError("Display aggregation needs leading [parent, child] virtual axes.")
        weights = self.subspin_weights.reshape(self.subspin_weights.shape + (1,) * (values.ndim - 2))
        return self._parent_image(np.sum(values * weights, axis=1), fill_value)

    @staticmethod
    def _maximum_child_field_hz(quality_maps, actual_currents, adc_chunk=64):
        maximum = 0.0
        for first in range(0, actual_currents.shape[1], adc_chunk):
            fields = quality_maps @ actual_currents[:, first:first + adc_chunk]
            maximum = max(maximum, float(np.max(np.abs(fields))))
        return maximum

    def response_diagnostics(self, currents_A, field_limit_Hz=np.inf, total_current_limit_A=np.inf):
        """Keep all physical limit checks on children; aggregate display only."""
        # Mirror the parent's waveform diagnostics while reducing ADC field
        # maxima in blocks. A full child-by-ADC float64 array can exceed 2 GiB
        # for the in-plane/thickness variant and is unnecessary for a maximum.
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
        quality_maps = self.maps_numpy[self.quality_mask]
        max_field = self._maximum_child_field_hz(quality_maps, actual)
        current_sums = np.abs(actual).sum(axis=0)
        instantaneous_image, instantaneous_kdata = self.forward(commands, response_enabled=False)
        response_image, response_kdata = self.forward(commands)
        energy = float(np.sum(np.abs(self.reference)**2))
        step_time = np.linspace(0, 1.3 * tau.max() * np.log(100), 256)
        result = {
            "adc_time_s": times, "commanded_currents_at_adc_A": adc["commanded_A"],
            "actual_currents_at_adc_A": actual,
            "commanded_integrated_currents_to_adc_As": command_integrals,
            "integrated_currents_to_adc_As": integrals,
            "line_center_time_s": line_times,
            "generated_field_at_line_center_Hz": self.maps_numpy @ line_currents,
            "waveform_time_s": waveform["time_s"],
            "waveform_commanded_currents_A": waveform["commanded_A"],
            "waveform_actual_currents_A": waveform["actual_A"],
            "dense_time_s": zoom_times, "dense_commanded_currents_A": zoom["commanded_A"],
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
            "actual_adc_channel_limits_met": bool(np.all(
                np.abs(actual) <= self.numpy(self.max_current_A)[:, None] + 1e-8)),
            "actual_adc_current_sum_limits_met": bool(np.all(current_sums < total_current_limit_A)),
            "instantaneous_image": instantaneous_image, "instantaneous_kdata": instantaneous_kdata,
            "instantaneous_relative_mse": float(np.sum(
                (np.abs(instantaneous_image) - np.abs(self.reference))**2) / energy),
            "response_image": response_image, "response_kdata": response_kdata,
            "response_relative_mse": float(np.sum(
                (np.abs(response_image) - np.abs(self.reference))**2) / energy),
            "limit_check_domain": "all reliable subspins at ADC samples; command-state bounds remain constraints",
            "field_maximum_ADC_chunk": 64,
        }
        fields = result["generated_field_at_line_center_Hz"]
        result["generated_field_at_line_center_subspin_Hz"] = fields
        result["generated_field_at_line_center_Hz"] = self.aggregate_to_parent(fields)
        result["generated_field_at_line_center_min_Hz"] = self._parent_image(fields.min(axis=1))
        result["generated_field_at_line_center_max_Hz"] = self._parent_image(fields.max(axis=1))
        result["maximum_child_field_range_at_line_center_Hz"] = float(np.max(np.ptp(fields, axis=1)))
        result["field_display_domain"] = "parent volume-fraction average; constraints retain every reliable child"
        return result


__all__ = ["SubspinACDCModel"]
