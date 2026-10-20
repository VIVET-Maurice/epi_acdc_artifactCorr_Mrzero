"""Physical-current constraints for notebook 04, after the DAC-command bounds.

The published 133 us settling time is represented by the first-order response
already used by ``acdc_gpu``. This is an approximation to the filtered system,
not a measured transfer function for each installed coil. Currents start at zero;
the residual at the end of the supplied cycle is reported explicitly.
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


class HardwareConstraints:
    """Check the continuous response and contract all commands if necessary.

    ``config`` is a plain dictionary. Optional per-channel values accept either
    a scalar or one value per channel. Missing limits remain disabled:

    * ``max_slew_A_per_s``: maximum absolute derivative of physical current.
    * ``avg_abs_current_A``: mean absolute physical current over the cycle.
    * ``coil_R_ohm``, ``coil_L_H``, ``max_voltage_V``: optional RL compliance
      model, V = R I + L dI/dt, without amplifier-internal voltage drops.
    * ``max_coil_power_W``: maximum mean resistive coil power, R mean(I**2).

    ``cycle_duration_s`` should be the measured complete EPI TR. Without it,
    the diagnostic window ends 12 tau after switch-off; heating limits require
    an explicit cycle duration. The 8 us minimum update interval is checked
    without changing the acquisition. The effective 0.6 mA DAC resolution is
    informational: it is neither an exact DAC code step nor a quantization rule.
    """

    def __init__(self, model, config=None):
        self.model = model
        self.config = dict(config or {})
        if not model.current_response["enabled"]:
            raise ValueError("Hardware constraints require current_response.enabled=True.")
        self.tau_s = self._channel_value(model.current_response["tau_s"], "tau_s")
        starts = np.asarray(model.line_start_s, dtype=float).reshape(-1)
        ends = np.asarray(model.line_end_s, dtype=float).reshape(-1)
        if (starts.shape != (model.nstates,) or ends.shape != starts.shape
                or not np.isfinite(starts).all() or not np.isfinite(ends).all()
                or np.any(starts < 0) or np.any(ends <= starts)
                or np.any(np.diff(starts) <= 0) or np.any(ends[:-1] > starts[1:])):
            raise ValueError("Hardware constraints need finite ordered, nonoverlapping command windows.")
        self.starts_s, self.ends_s = starts, ends
        self.explicit_cycle = self.config.get("cycle_duration_s") is not None
        self.cycle_duration_s = float(self.config.get("cycle_duration_s")
                                      if self.explicit_cycle else ends[-1] + 12 * self.tau_s.max())
        if not np.isfinite(self.cycle_duration_s) or self.cycle_duration_s < ends[-1]:
            raise ValueError("cycle_duration_s must be finite and include the final command end.")

        self.limits = {}
        for name in ("max_slew_A_per_s", "avg_abs_current_A", "coil_R_ohm",
                     "coil_L_H", "max_voltage_V", "max_coil_power_W"):
            value = self.config.get(name)
            self.limits[name] = None if value is None else model.tensor(
                self._channel_value(value, name, allow_zero=name == "coil_L_H"))
        if self.limits["max_voltage_V"] is not None and (
                self.limits["coil_R_ohm"] is None or self.limits["coil_L_H"] is None):
            raise ValueError("max_voltage_V requires measured coil_R_ohm and coil_L_H.")
        if self.limits["max_coil_power_W"] is not None and self.limits["coil_R_ohm"] is None:
            raise ValueError("max_coil_power_W requires measured coil_R_ohm.")
        if not self.explicit_cycle and any(self.limits[key] is not None for key in (
                "avg_abs_current_A", "max_coil_power_W")):
            raise ValueError("Average-current and power limits require cycle_duration_s (complete TR).")

        self.min_update_interval_s = self._positive_scalar(
            self.config.get("min_update_interval_s", 8e-6), "min_update_interval_s")
        self.dac_resolution_A = self._positive_scalar(
            self.config.get("dac_resolution_A", 0.0006), "dac_resolution_A")
        edges = np.unique(np.r_[starts, ends])
        spacing = np.diff(edges)
        minimum = float(spacing.min()) if spacing.size else np.inf
        self.timing_report = {
            "minimum_command_interval_s": minimum,
            "minimum_allowed_interval_s": self.min_update_interval_s,
            "update_intervals_met": bool(minimum + 1e-12 >= self.min_update_interval_s),
            "effective_DAC_resolution_A": self.dac_resolution_A,
            "cycle_duration_s": self.cycle_duration_s,
            "cycle_duration_supplied": self.explicit_cycle,
            "initial_current_assumption": "zero; no periodic steady-state assumption",
        }
        if not self.timing_report["update_intervals_met"]:
            raise ValueError("Command start/end transitions are closer than the configured DAC update interval.")

        # One segment per constant-command interval, including zero-current gaps
        # and the final switch-off tail. At a boundary the physical current is
        # continuous; only its derivative changes.
        boundaries = np.unique(np.r_[0.0, edges, self.cycle_duration_s])
        # Keep a zero-duration final segment to check dI/dt and voltage just
        # after switch-off even when the final command ends exactly at TR.
        self.segment_start_s = boundaries
        self.segment_end_s = np.r_[boundaries[1:], boundaries[-1]]
        _, weights = response_matrices(starts, ends, self.segment_start_s, self.tau_s)
        self.response_at_start = model.tensor(weights)
        self.command_selector = model.tensor(
            (self.segment_start_s[:, None] >= starts)
            & (self.segment_start_s[:, None] < ends))
        self.dt = model.tensor(self.segment_end_s - self.segment_start_s)[None, :]
        self.tau = model.tensor(self.tau_s)[:, None]
        self.decay = torch.exp(-self.dt / self.tau)
        self.common_tau = bool(np.all(self.tau_s == self.tau_s[0]))
        self.positive_field_basis = model.B_quality.clamp_min(0)
        self.negative_field_basis = model.B_quality.clamp_max(0)

    @staticmethod
    def _positive_scalar(value, name):
        if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
            raise ValueError(f"{name} must be a positive finite scalar.")
        result = float(value)
        if not np.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be a positive finite scalar.")
        return result

    def _channel_value(self, value, name, *, allow_zero=False):
        if np.iscomplexobj(value) or np.asarray(value).dtype.kind == "b":
            raise ValueError(f"{name} must contain real numeric values.")
        result = np.asarray(value, dtype=float)
        result = np.full(self.model.nc, result.item()) if result.size == 1 else result.reshape(-1)
        if (result.shape != (self.model.nc,) or not np.isfinite(result).all()
                or np.any(result < 0 if allow_zero else result <= 0)):
            raise ValueError(f"{name} needs one finite {'nonnegative' if allow_zero else 'positive'} value per channel.")
        return result

    def _segments(self, currents):
        actual_start = torch.einsum("ces,cs->ce", self.response_at_start, currents)
        command = currents @ self.command_selector.T
        actual_end = command + (actual_start - command) * self.decay
        return command, actual_start, actual_end

    def _metrics(self, currents):
        command, first, last = self._segments(currents)
        difference = first - command
        derivative_first = -difference / self.tau
        derivative_last = derivative_first * self.decay
        peak_channel = torch.maximum(first.abs(), last.abs()).amax(dim=1)
        peak_slew = derivative_first.abs().amax(dim=1)
        if self.common_tau:
            # A common exponential traces a straight line in channel space.
            # Convex absolute-current and field maxima occur at its endpoints.
            total_bound = torch.maximum(first.abs().sum(0), last.abs().sum(0)).max()
            field_bound = torch.maximum((self.model.B_quality @ first).abs(),
                                        (self.model.B_quality @ last).abs()).max()
        else:
            # Each channel is monotone on a segment. Its endpoint interval gives
            # an all-time bound even when unequal taus break field cancellation.
            low, high = torch.minimum(first, last), torch.maximum(first, last)
            total_bound = torch.maximum(first.abs(), last.abs()).sum(0).max()
            lower_field = self.positive_field_basis @ low + self.negative_field_basis @ high
            upper_field = self.positive_field_basis @ high + self.negative_field_basis @ low
            field_bound = torch.maximum(lower_field.abs(), upper_field.abs()).max()

        # Analytic time integrals for I = command + difference * exp(-t/tau).
        decay_integral = -self.tau * torch.expm1(-self.dt / self.tau)
        signed = command * self.dt + difference * decay_integral
        square = (command.square() * self.dt + 2 * command * difference * decay_integral
                  - difference.square() * self.tau / 2 * torch.expm1(-2 * self.dt / self.tau))
        # A monotone exponential can cross zero once. Split its signed integral
        # there before taking absolute values; abs(integral(I)) is insufficient.
        safe_command = torch.where(command == 0, torch.ones_like(command), command)
        ratio = (-difference / safe_command).clamp_min(torch.finfo(currents.dtype).tiny)
        crossing = (first * last < 0) & (command != 0)
        crossing_time = torch.where(crossing, self.tau * ratio.log(), torch.zeros_like(first))
        crossing_time = torch.minimum(crossing_time.clamp_min(0), self.dt)
        signed_before = (command * crossing_time
                         - difference * self.tau * torch.expm1(-crossing_time / self.tau))
        absolute = torch.where(crossing, signed_before.abs() + (signed - signed_before).abs(), signed.abs())
        result = {
            "peak_actual_channel_current_A": peak_channel,
            "peak_slew_A_per_s": peak_slew,
            "peak_total_current_bound_A": total_bound,
            "peak_field_bound_Hz": field_bound,
            "avg_abs_current_A": absolute.sum(1) / self.cycle_duration_s,
            "mean_square_current_A2": square.clamp_min(0).sum(1) / self.cycle_duration_s,
            "end_current_A": last[:, -1],
        }
        resistance, inductance = self.limits["coil_R_ohm"], self.limits["coil_L_H"]
        if resistance is not None:
            result["average_coil_power_W"] = resistance * result["mean_square_current_A2"]
        if resistance is not None and inductance is not None:
            voltage_first = resistance[:, None] * first + inductance[:, None] * derivative_first
            voltage_last = resistance[:, None] * last + inductance[:, None] * derivative_last
            result["peak_coil_voltage_V"] = torch.maximum(voltage_first.abs(), voltage_last.abs()).amax(1)
        return result

    def _checks(self, currents, metrics, total_current_limit_A, field_limit_Hz):
        return {
            "command_channel": (currents.abs() <= self.model.max_current_A[:, None]).all(),
            "command_total": (currents.abs().sum(0) <= total_current_limit_A).all(),
            "command_field": ((self.model.B_quality @ currents).abs() <= field_limit_Hz).all(),
            "actual_channel": (metrics["peak_actual_channel_current_A"] <= self.model.max_current_A).all(),
            "actual_total_bound": metrics["peak_total_current_bound_A"] <= total_current_limit_A,
            "actual_field_bound": metrics["peak_field_bound_Hz"] <= field_limit_Hz,
        }

    @staticmethod
    def _check_global_limits(total_current_limit_A, field_limit_Hz):
        if not np.isfinite(total_current_limit_A) or total_current_limit_A <= 0:
            raise ValueError("total_current_limit_A must be positive and finite.")
        if np.isnan(field_limit_Hz) or field_limit_Hz <= 0:
            raise ValueError("field_limit_Hz must be positive; infinity disables it.")

    @torch.no_grad()
    def additional_projection(self, currents, total_current_limit_A, field_limit_Hz=np.inf):
        """Uniform contraction preserves the preceding box/L1/field projection.

        This is a feasible contraction, not a Euclidean projection onto the full
        feasible set. Every response metric is homogeneous of degree one except
        resistive power, which is degree two. No clipping of physical waveforms
        or alteration of acquisition times is performed.
        """
        self._check_global_limits(total_current_limit_A, field_limit_Hz)
        metrics = self._metrics(currents)
        tiny = torch.finfo(currents.dtype).tiny
        scale = currents.new_tensor(1.0)
        total_budget = total_current_limit_A - min(1e-4, total_current_limit_A * 1e-3)
        scale = torch.minimum(scale, total_budget / metrics["peak_total_current_bound_A"].clamp_min(tiny))
        if np.isfinite(field_limit_Hz):
            field_budget = field_limit_Hz - min(.05, field_limit_Hz * 1e-3)
            scale = torch.minimum(scale, field_budget / metrics["peak_field_bound_Hz"].clamp_min(tiny))
        for config_name, metric_name in (
                ("max_slew_A_per_s", "peak_slew_A_per_s"),
                ("avg_abs_current_A", "avg_abs_current_A"),
                ("max_voltage_V", "peak_coil_voltage_V"),
                ("max_coil_power_W", "average_coil_power_W")):
            limit = self.limits[config_name]
            if limit is not None:
                ratio = (limit * (1 - 1e-5) / metrics[metric_name].clamp_min(tiny)).amin()
                scale = torch.minimum(scale, ratio.sqrt() if config_name == "max_coil_power_W" else ratio)
        return currents * scale.clamp(0, 1)

    @torch.no_grad()
    def diagnostics(self, currents, total_current_limit_A, field_limit_Hz=np.inf):
        """JSON-ready checks, including conservative bounds between ADC samples."""
        self._check_global_limits(total_current_limit_A, field_limit_Hz)
        currents = self.model._check_currents(currents)
        metrics = self._metrics(currents)
        checks = self._checks(currents, metrics, total_current_limit_A, field_limit_Hz)
        for config_name, metric_name in (
                ("max_slew_A_per_s", "peak_slew_A_per_s"),
                ("avg_abs_current_A", "avg_abs_current_A"),
                ("max_voltage_V", "peak_coil_voltage_V"),
                ("max_coil_power_W", "average_coil_power_W")):
            if self.limits[config_name] is not None:
                checks[config_name] = (metrics[metric_name] <= self.limits[config_name]).all()
        checks = {key: bool(value) for key, value in checks.items()}
        result = {key: self.model.numpy(value).tolist() for key, value in metrics.items()}
        result.update({
            "checks": checks, "all_limits_met": all(checks.values()),
            "timing": dict(self.timing_report),
            "bound_method": "exact segment endpoints (common tau)" if self.common_tau else
                            "conservative channel intervals (unequal taus)",
            "disabled_limits": [key for key in ("max_slew_A_per_s", "avg_abs_current_A",
                                "max_voltage_V", "max_coil_power_W") if self.limits[key] is None],
            "assumptions": ["First-order filtered response; zero initial current.",
                            "Mean currents/power use this cycle only; repeated-shot carryover is not modeled.",
                            "RL coil voltage excludes amplifier drops, mutual inductance and eddy currents."],
        })
        return result


__all__ = ["HardwareConstraints"]
