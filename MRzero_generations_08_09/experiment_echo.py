"""Echo-anchored native MRzero subspins for the complete ACDC pipeline.

This variant retains child-wise current phase, exact analytic current gradients
and hardware checks from notebook 04. Its effective transverse amplitude is
anchored at an explicitly measured echo time, independently of the EPI TE.
An optional reconstruction factory supplies a jointly transformed output grid.
"""

from __future__ import annotations

import numpy as np
import torch

from experiment_subspins import SubspinACDCModel


class EchoSubspinACDCModel(SubspinACDCModel):
    """Use explicit echo amplitude, actual child fields and shared commands.

    ``amplitude_reference_time_s`` may be supplied directly, in the experiment
    dictionary, or under ``experiment['relaxation']``. It is never silently
    replaced with EPI TE. Child ``m0_proxy`` already includes volume weights.

    A custom ``reconstruction`` must expose ``image``, ``adjoint``, ``report``,
    ``device`` and ``dtype``. Alternatively ``reconstruction_factory(acq,
    device, dtype)`` is called lazily, after the model selects its device and
    before its ideal reference is formed. Original reconstruction is retained
    if neither option is supplied. The two options are mutually exclusive.
    """

    def __init__(self, experiment, calibration, max_current_A=3.0, *,
                 amplitude_reference_time_s=None, reconstruction=None,
                 reconstruction_factory=None, hardware=None,
                 reconstruction_chunk=256, **options):
        if reconstruction is not None and reconstruction_factory is not None:
            raise ValueError("Supply reconstruction or reconstruction_factory, not both.")
        if reconstruction_factory is not None and not callable(reconstruction_factory):
            raise ValueError("reconstruction_factory must be callable.")
        anchor = amplitude_reference_time_s
        if anchor is None:
            anchor = experiment.get("amplitude_reference_time_s")
            relaxation_anchor = experiment.get("relaxation", {}).get("amplitude_reference_time_s")
            if anchor is None:
                anchor = relaxation_anchor
            elif relaxation_anchor is not None and not np.isclose(anchor, relaxation_anchor, rtol=0, atol=1e-12):
                raise ValueError("Experiment amplitude reference times disagree.")
        if anchor is None or not np.isscalar(anchor) or not np.isreal(anchor):
            raise ValueError("An explicit amplitude_reference_time_s is required.")
        anchor = float(np.real(anchor))
        if not np.isfinite(anchor) or anchor < 0:
            raise ValueError("amplitude_reference_time_s must be nonnegative finite seconds.")
        self.amplitude_reference_time_s = anchor
        self._echo_reconstruction = reconstruction
        self._echo_reconstruction_factory = reconstruction_factory
        self._echo_reconstruction_validated = False
        super().__init__(experiment, calibration, max_current_A,
                         hardware=hardware, reconstruction_chunk=reconstruction_chunk, **options)
        self.runtime_info.update({
            "object_amplitude": "effective transverse amplitude at explicit measured echo time",
            "amplitude_reference_time_s": self.amplitude_reference_time_s,
            "EPI_TE_s": self.TE_s,
            "amplitude_at_RF": "weighted child amplitude multiplied by exp(reference_time * decay_rate)",
            "simulation_domain": "all positive supported child amplitudes",
            "quality_domain": "independent reliable child mask used for field constraints",
            "simulation_children": int(self.support.sum()),
            "quality_children": int(self.quality_mask.sum()),
            "custom_reconstruction": self._echo_reconstruction is not None,
        })

    def _image(self, kdata):
        if self._echo_reconstruction is None and self._echo_reconstruction_factory is not None:
            self._echo_reconstruction = self._echo_reconstruction_factory(
                self._reconstruction_acquisition, self.device, self.dtype)
        if self._echo_reconstruction is None:
            return super()._image(kdata)
        if not self._echo_reconstruction_validated:
            operator = self._echo_reconstruction
            if (not callable(getattr(operator, "image", None))
                    or not callable(getattr(operator, "adjoint", None))
                    or not isinstance(getattr(operator, "report", None), dict)):
                raise ValueError("Custom reconstruction requires image, adjoint and a report dictionary.")
            if not hasattr(operator, "device") or not hasattr(operator, "dtype"):
                raise ValueError("Custom reconstruction must declare device and dtype.")
            device = torch.device(operator.device)
            if device.type != self.device.type or (device.index is not None and device.index != self.device.index):
                raise ValueError("Custom reconstruction device differs from the model device.")
            if operator.dtype not in (self.dtype, self.complex_dtype):
                raise ValueError("Custom reconstruction dtype differs from the model precision.")
            self._echo_reconstruction_validated = True
        self.reconstruction = self._echo_reconstruction
        return self.reconstruction.image(kdata)

    def _phantom(self, b0_enabled):
        """Native point-spin phantom anchored at the measured echo, once."""
        if b0_enabled in self._native_phantoms:
            return self._native_phantoms[b0_enabled]
        rate = self.numpy(self._rate).reshape(-1)
        amplitude = self.numpy(self._amplitude).reshape(-1)
        with np.errstate(over="ignore", divide="ignore"):
            pd = amplitude * np.exp(self.amplitude_reference_time_s * rate)
            t2star = np.divide(1., rate, out=np.full_like(rate, np.inf), where=rate > 0)
        if not np.isfinite(pd).all() or np.any(pd > np.finfo(np.float32).max):
            raise ValueError("Echo-anchored amplitude extrapolation overflows native MRzero float32.")
        if np.any(t2star < 1e-6):
            raise ValueError("Native MRzero requires relaxation times >= 1 microsecond.")
        positions = np.column_stack((self.numpy(self._x).ravel(), self.numpy(self._y).ravel(),
                                     self._subspin_z_numpy[self.support])).astype(np.float32)
        b0 = self.numpy(self._b0).ravel() if b0_enabled else np.zeros(amplitude.size)
        phantom = self._mr0.CustomVoxelPhantom(
            positions, PD=pd.astype(np.float32), B0=np.asarray(b0, np.float32), B1=1.,
            T1=np.inf, T2=t2star.astype(np.float32), T2dash=np.inf,
            D=0., voxel_size=0., voxel_shape="box",
        ).build()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                phantom = phantom.cuda()
        phantom.D = torch.zeros_like(phantom.D)
        phantom.dephasing_func = lambda k, _: torch.ones(k.shape[0], device=k.device, dtype=k.dtype)
        self._native_phantoms[b0_enabled] = phantom
        return phantom


__all__ = ["EchoSubspinACDCModel"]
