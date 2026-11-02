"""Nonperiodic residual-field guidance for single-excitation EPI and spiral.

Fourier image reconstruction already models signal crossing a FOV boundary.
This additional objective distinguishes physical phase slopes that differ by
whole turns. It uses the measured B0 branch, never a wrapped image distance.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "Experiment"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from acdc_gpu import response_matrices


class FieldGuidance:
    """Quadratic residual phase-slope penalty, with exact command response.

    Between successive chronological command-state readout centers, the effective residual
    frequency is ``B0 + B @ ((integral(t2)-integral(t1))/(t2-t1))``. The penalty
    is its mean square over intervals and reliable object centers, divided by
    ``scale_Hz**2``. All intervals and reliable centers have equal weight.

    Readout centers are the temporal midpoint of their first/last ADC centers;
    no ADC interpolation or averaging of instantaneous current is used. The
    exact integral uses the model's command windows and first-order response.
    A state is a Cartesian line or one spiral turn. Unequal ADC counts and
    durations are retained; samples need not be stored in chronological order.
    This is a field guide, not an additional signal model or hardware limit.
    It does not identify the absolute frequency branch of the measured B0.

    A thin QR factorization of ``[B, B0]`` reduces the spatial objective to at
    most ``channels+1`` rows. Squaring that residual avoids cancellation in an
    expanded quadratic near perfect shimming and avoids voxel-by-ADC arrays
    during optimization. Finite-voxel hardware corners remain independent.
    """

    def __init__(self, model, *, scale_Hz=100.):
        scale = np.asarray(scale_Hz)
        if (isinstance(scale_Hz, (bool, np.bool_)) or scale.ndim != 0
                or np.iscomplexobj(scale) or not np.issubdtype(scale.dtype, np.number)
                or not np.isfinite(scale) or scale <= 0):
            raise ValueError("scale_Hz must be a finite positive real scalar.")
        if model.sequence_kind not in {"epi", "spiral"}:
            raise ValueError("Residual phase-slope field guidance requires single-excitation EPI or spiral.")
        if model.nstates < 2:
            raise ValueError("Residual phase-slope field guidance requires at least two command states.")
        self.model = model
        self.scale_Hz = float(scale)
        self.nc, self.nstates = model.nc, model.nstates

        times = np.asarray(model.time_s, dtype=np.float64)
        labels = np.asarray(model.line_index)
        rf = np.asarray(model.rf_time_s, dtype=np.float64)
        if (times.shape != labels.shape or rf.shape != times.shape
                or not np.isfinite(times).all() or not np.isfinite(rf).all()
                or np.any(np.abs(rf) > 1e-12)):
            raise ValueError("Field guidance requires finite ADC times relative to one RF center at zero.")
        centers = []
        for state in range(1, self.nstates + 1):
            local = times[labels == state]
            if local.size == 0:
                raise ValueError("Field guidance requires every chronological line_index.")
            centers.append(.5 * (float(local.min()) + float(local.max())))
        self.state_center_times_s = np.asarray(centers)
        # Retain the public notebook-06 attribute for existing diagnostic code.
        self.line_center_times_s = self.state_center_times_s
        durations = np.diff(self.state_center_times_s)
        if np.any(durations <= 0):
            raise ValueError("Readout center times must increase in chronological command-state order.")
        self.interval_durations_s = durations
        self.nintervals = len(durations)

        starts = np.asarray(model.line_start_s, dtype=np.float64)
        ends = np.asarray(model.line_end_s, dtype=np.float64)
        response_enabled = bool(model.current_response["enabled"])
        if response_enabled:
            tau = np.asarray(model.current_response["tau_s"], dtype=np.float64).reshape(-1)
            common_tau = bool(np.all(tau == tau[0]))
            integrals, _ = response_matrices(
                starts, ends, self.line_center_times_s,
                tau[:1] if common_tau else tau, np.zeros(self.nstates))
            slopes = np.diff(integrals, axis=1) / durations[None, :, None]
            slopes = slopes[0] if common_tau else slopes
        else:
            integrals = np.maximum(0., np.minimum(self.line_center_times_s[:, None], ends)
                                   - np.maximum(0., starts))
            slopes = np.diff(integrals, axis=0) / durations[:, None]
        self._slopes = model.tensor(slopes)
        self._channel_response = self._slopes.ndim == 3

        mask = np.asarray(model.quality_mask, dtype=bool)
        maps = np.asarray(model.maps_numpy, dtype=np.float64)[mask]
        b0 = np.asarray(model.b0_numpy, dtype=np.float64)[mask]
        if (maps.shape != (b0.size, self.nc) or b0.size == 0
                or not np.isfinite(maps).all() or not np.isfinite(b0).all()):
            raise ValueError("Field guidance requires finite B0 and coil maps on reliable object centers.")
        augmented = np.column_stack((maps, b0)) / np.sqrt(b0.size)
        compact = np.linalg.qr(augmented, mode="r")
        self._basis = model.tensor(compact[:, :self.nc])
        self._offset = model.tensor(compact[:, self.nc:])
        self.report = {
            "objective": "mean((B0 + B @ mean_actual_current_between_state_centers)**2) / scale_Hz**2",
            "sequence_kind": model.sequence_kind,
            "command_state_unit": "spiral turn" if model.sequence_kind == "spiral" else "Cartesian line",
            "scale_Hz": self.scale_Hz,
            "frequency_branch": "measured unwrapped B0 as supplied; no phase wrapping or branch fitting",
            "domain": "model.quality_mask object centers; finite-voxel hardware corners remain separate",
            "quality_centers": int(b0.size),
            "spatial_weighting": "equal weight per reliable center",
            "temporal_weighting": "equal weight per consecutive chronological readout-center interval",
            "state_centers": "midpoint of first and last ADC centers within each chronological command state",
            "state_center_times_s": self.state_center_times_s.tolist(),
            "line_centers": "legacy key: midpoint of first and last ADC centers within each chronological command state",
            "line_center_times_s": self.line_center_times_s.tolist(),
            "interval_durations_s": durations.tolist(),
            "interval_count": self.nintervals,
            "response_enabled": response_enabled,
            "channel_specific_response": self._channel_response,
            "response_integral": "same exact rectangular-command integrals and configured response as the forward model",
            "compression": "thin QR of [B, B0] / sqrt(number of quality centers)",
            "compressed_spatial_rows": int(compact.shape[0]),
            "current_limits_modified": False,
            "field_limits_modified": False,
        }

    @torch.no_grad()
    def value_gradient(self, currents, *, gradient=True):
        """Return ``(penalty, gradient_or_None, residual_rms_Hz)`` tensors.

        Currents have the same channel/state order and units as the model.
        The optimizer remains responsible for projecting all hardware limits,
        including exact zero on disabled channels. This method has no state.
        """
        if torch.is_tensor(currents):
            if currents.is_complex():
                raise ValueError("Field guidance currents must be real.")
        elif np.iscomplexobj(currents):
            raise ValueError("Field guidance currents must be real.")
        commands = self.model.tensor(currents)
        if tuple(commands.shape) != (self.nc, self.nstates):
            raise ValueError("Field guidance currents must have shape [channels, imaging states].")
        if self._channel_response:
            mean_currents = torch.bmm(self._slopes, commands[:, :, None]).squeeze(-1)
        else:
            mean_currents = commands @ self._slopes.T
        residual = self._basis @ mean_currents + self._offset
        scaled = residual / self.scale_Hz
        penalty = scaled.square().sum() / self.nintervals
        rms = residual.square().sum().div(self.nintervals).sqrt()
        if not gradient:
            return penalty, None, rms
        mean_gradient = (2. / (self.nintervals * self.scale_Hz)) * (self._basis.T @ scaled)
        if self._channel_response:
            derivative = torch.bmm(mean_gradient[:, None, :], self._slopes).squeeze(1)
        else:
            derivative = mean_gradient @ self._slopes
        return penalty, derivative, rms


__all__ = ["FieldGuidance"]
