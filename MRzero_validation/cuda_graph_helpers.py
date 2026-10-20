"""Replay the fixed CUDA work of the MRzero objective and current projection.

CUDA graphs avoid submitting the same small tensor operations from Python on
every iteration. They do not alter the model, objective or projection. The
MRzero baseline must already reside on the GPU; otherwise evaluation keeps
the ordinary direct path so there is no hidden CPU transfer inside a graph.
"""

from __future__ import annotations

from inspect import signature
from math import isfinite
from time import perf_counter

import torch


class GpuEvaluation:
    """Evaluate and project currents, optionally with two independent graphs.

    ``evaluate`` returns graph-owned ``(loss, gradient, image, kdata)`` buffers.
    The next evaluation overwrites those buffers: clone a result that must be
    retained, or copy it to NumPy for a live figure. ``project`` always returns
    an owned tensor, so a subsequent projection cannot overwrite the optimizer's
    current candidate. The wrapper is intended for one sequential optimizer;
    it is not safe for concurrent callers or a model modified after capture.

    ``enabled``, ``status`` and ``report`` state whether capture was possible.
    CPU, an uncached baseline, an explicitly disabled graph, or an unsupported
    CUDA operation uses the same direct model calls instead.
    """

    @torch.no_grad()
    def __init__(self, model, total_current_limit_A, field_limit_Hz, *, enabled=True):
        self.model = model
        self.total_current_limit_A = total_current_limit_A
        self.field_limit_Hz = field_limit_Hz
        self.enabled = False
        self.status = "disabled by configuration"
        self._evaluation_graph = self._projection_graph = None
        started = perf_counter()
        if enabled:
            if model.device.type != "cuda":
                self.status = "direct evaluation: model uses CPU"
            elif model._base_cache is None:
                self.status = "direct evaluation: MRzero baseline is not cached on the GPU"
            elif model._base_cache.device.type != "cuda":
                self.status = "direct evaluation: MRzero baseline is on CPU"
            else:
                self._prepare_graphs()
        self.report = {
            "enabled": self.enabled,
            "status": self.status,
            "setup_seconds": perf_counter() - started,
            "evaluation_outputs_reused": self.enabled,
            "projection_output_owned": True,
        }

    def _prepare_graphs(self):
        # Allocate persistent inputs once. All later input copies and graph
        # replays occur on the caller's current stream in their original order.
        self._evaluation_input = torch.zeros(
            (self.model.nc, self.model.nstates),
            device=self.model.device, dtype=self.model.dtype,
        )
        self._projection_input = torch.zeros_like(self._evaluation_input)
        # Validate limits before capture. Invalid parameters should raise their
        # original ValueError, rather than silently disabling acceleration.
        self.model.project_tensor(
            self._projection_input, self.total_current_limit_A, self.field_limit_Hz,
        )
        # The original projection constructs a scalar CUDA margin with
        # limits.new_tensor(...). Creating it from a CPU value during capture
        # is unsupported, so prepare its unchanged limits outside the graph.
        # Read the original defaults rather than inventing new safety margins.
        from acdc_gpu import project_currents_tensor
        defaults = signature(project_currents_tensor).parameters
        current_margin = defaults["current_margin_A"].default
        field_margin = defaults["field_margin_Hz"].default
        limits = torch.as_tensor(
            self.model.max_current_A, device=self.model.device, dtype=self.model.dtype,
        ).reshape(-1, 1)
        self._safe_channel_limits = torch.clamp(
            limits - torch.minimum(limits * 1e-3, limits.new_tensor(current_margin)), min=0,
        )
        total = float(self.total_current_limit_A)
        self._safe_total_limit = max(0.0, total - min(total * 1e-3, current_margin))
        field = float(self.field_limit_Hz)
        self._safe_field_limit = max(0.0, field - min(field * 1e-3, field_margin)) if isfinite(field) else None
        try:
            self._evaluation_graph, self._evaluation_outputs = self._capture(
                lambda: self.model._objective_tensor(self._evaluation_input)
            )
            self._projection_graph, self._projection_output = self._capture(
                lambda: self._project_capture_safe(self._projection_input)
            )
        except RuntimeError as error:
            # Do not expose half of a captured evaluator. Both methods keep the
            # direct path if either graph is unsupported on this installation.
            self._evaluation_graph = self._projection_graph = None
            self.status = f"direct evaluation: CUDA graph capture failed ({error})"
            # A failed capture can leave a CUDA error pending. Only announce
            # direct fallback if the model can still evaluate successfully.
            try:
                self.model._objective_tensor(self._evaluation_input)
                torch.cuda.synchronize(self.model.device)
            except RuntimeError as fallback_error:
                raise RuntimeError(
                    "CUDA graph capture failed and the CUDA context cannot continue. "
                    "Restart the notebook kernel and set USE_CUDA_GRAPH=False. "
                    f"Capture error: {error}"
                ) from fallback_error
            return
        self.enabled = True
        self.status = "CUDA graphs enabled for objective/adjoint and current projection"

    def _capture(self, operation):
        device = self.model.device
        caller_stream = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(caller_stream)
        # CUDA libraries create plans and internal workspaces on first use.
        # Warm up on a side stream, then finish all work before capture starts.
        with torch.cuda.stream(stream):
            for _ in range(3):
                operation()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = operation()
        caller_stream.wait_stream(stream)
        return graph, output

    def _project_capture_safe(self, currents):
        """Original projection arithmetic with its margins already on CUDA.

        Keep this bisection and radial contraction in the same order as
        Experiment/acdc_gpu.py::project_currents_tensor. Tests compare the
        returned values bit for bit for both supported floating precisions.
        """
        magnitude = currents.abs()
        low = torch.zeros(
            (1, currents.shape[1]), device=currents.device, dtype=currents.dtype,
        )
        high = magnitude.amax(dim=0, keepdim=True)
        clipped = torch.minimum(magnitude, self._safe_channel_limits)
        needs_l1 = clipped.sum(dim=0, keepdim=True) > self._safe_total_limit
        for _ in range(54 if currents.dtype == torch.float64 else 28):
            middle = (low + high) / 2
            candidate = torch.minimum(
                torch.clamp(magnitude - middle, min=0), self._safe_channel_limits,
            )
            over_budget = candidate.sum(dim=0, keepdim=True) > self._safe_total_limit
            low = torch.where(over_budget, middle, low)
            high = torch.where(over_budget, high, middle)
        threshold = torch.where(needs_l1, high, torch.zeros_like(high))
        projected = currents.sign() * torch.minimum(
            torch.clamp(magnitude - threshold, min=0), self._safe_channel_limits,
        )
        if self._safe_field_limit is not None:
            peak = (self.model.B_quality @ projected).abs().amax(dim=0, keepdim=True)
            scale = torch.clamp(
                self._safe_field_limit / peak.clamp_min(torch.finfo(currents.dtype).tiny), max=1,
            )
            projected *= scale
        return projected

    def _check_input(self, currents):
        # Device/dtype/shape are metadata checks and never synchronize CUDA.
        # copy_ must not quietly insert a CPU transfer or a precision change.
        expected = self._evaluation_input
        if (not isinstance(currents, torch.Tensor)
                or currents.shape != expected.shape
                or currents.dtype != expected.dtype
                or currents.device != expected.device):
            raise ValueError("Graph currents must match the model's CUDA device, dtype and [channel, state] shape.")

    @torch.no_grad()
    def evaluate(self, currents):
        """Return loss, analytic gradient, image and k-space for this candidate."""
        if not self.enabled:
            return self.model._objective_tensor(currents)
        self._check_input(currents)
        self._evaluation_input.copy_(currents)
        self._evaluation_graph.replay()
        return self._evaluation_outputs

    @torch.no_grad()
    def project(self, currents):
        """Apply the unchanged box/L1 and radial field-limit projection."""
        if not self.enabled:
            return self.model.project_tensor(
                currents, self.total_current_limit_A, self.field_limit_Hz,
            )
        self._check_input(currents)
        self._projection_input.copy_(currents)
        self._projection_graph.replay()
        # The optimizer retains projected currents while making another
        # projection for its stopping diagnostic. Own the result to avoid an
        # alias that would overwrite those retained currents on the next call.
        return self._projection_output.clone()


__all__ = ["GpuEvaluation"]
