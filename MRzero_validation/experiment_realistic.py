"""Notebook 04: native MRzero, continuous k-space, and ACDC hardware limits.

The signal and exact first-order current integrals come from notebook 03's
model. Only reconstruction, its adjoint, and the current projection change.
No existing notebook or simulator is patched.
"""

from __future__ import annotations

import numpy as np
import torch

from acdc_hardware import HardwareConstraints
from continuous_reconstruction import ContinuousReconstruction
from experiment_mrzero import MRzeroExperimentModel


class RealisticACDCModel(MRzeroExperimentModel):
    """Use the actual nominal Pulseq k coordinates throughout optimization.

    ``hardware`` contains explicitly configured limits; missing electrical
    measurements stay unset. ``reconstruction_chunk`` bounds the direct
    Fourier calculation for trajectories that cannot use a separable form.
    Use the ordinary solver with ``use_cuda_graph=False``: the older CUDA
    graph helper has a separate projection that does not know these limits.
    """

    def __init__(self, experiment, calibration, max_current_A=3.0, *,
                 hardware=None, reconstruction_chunk=256, **options):
        # The parent constructs its ideal image during initialization. Set the
        # reconstruction inputs first; _image creates the operator when needed.
        self._reconstruction_acquisition = experiment["acquisition"]
        self._reconstruction_chunk = reconstruction_chunk
        self.reconstruction = None
        super().__init__(experiment, calibration, max_current_A, **options)
        config = dict(hardware or {})
        config.setdefault("cycle_duration_s", experiment["acquisition"]["TR_s"])
        self.hardware = HardwareConstraints(self, config)
        self.runtime_info["reconstruction"] = self.reconstruction.report
        self.runtime_info["hardware_timing"] = self.hardware.timing_report

    def _image(self, kdata):
        if self.reconstruction is None:
            self.reconstruction = ContinuousReconstruction(
                self._reconstruction_acquisition, self.device, self.dtype,
                sample_chunk=self._reconstruction_chunk,
            )
        return self.reconstruction.image(kdata)

    @torch.no_grad()
    def _objective_tensor(self, currents, gradient=True):
        """Magnitude MSE and the same signal derivative as notebook 03.

        The only change in the derivative is R^H instead of the FFT adjoint.
        The scalar objective remains image MSE, without a hidden penalty.
        """
        retain = gradient and self._retain_contributions
        output = self._forward_tensor(currents, retain=retain)
        image, kdata = output[:2]
        amplitude = image.abs()
        residual = amplitude - self._reference_amplitude
        value = residual.square().sum() / self._reference_energy
        if not gradient:
            return value, image, kdata

        derivative = 2 * residual * image / amplitude.clamp_min(
            torch.finfo(self.dtype).tiny)
        derivative /= self._reference_energy
        adjoint = self.reconstruction.adjoint(derivative).reshape(-1)

        result = torch.zeros_like(currents)
        for block, first in enumerate(range(0, self.nsamples, self.sample_chunk)):
            selected = slice(first, min(first + self.sample_chunk, self.nsamples))
            contribution = (output[2][block] if retain
                            else self._contribution(currents, selected))
            sensitivity = (2 * np.pi) * (
                contribution * adjoint[selected].conj()).imag
            channel_sensitivity = self.B.T @ sensitivity
            if self._channel_response:
                result += torch.bmm(
                    channel_sensitivity[:, None, :], self._H[:, selected, :]
                ).squeeze(1)
            else:
                result += channel_sensitivity @ self._H[selected]
        return value, result, image, kdata

    @torch.no_grad()
    def project_tensor(self, currents, total_current_limit_A,
                       field_limit_Hz=np.inf):
        """First constrain commands, then the complete physical waveform."""
        commands = super().project_tensor(
            currents, total_current_limit_A, field_limit_Hz)
        return self.hardware.additional_projection(
            commands, total_current_limit_A, field_limit_Hz)

