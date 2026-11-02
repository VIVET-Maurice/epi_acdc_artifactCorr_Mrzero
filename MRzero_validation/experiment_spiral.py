"""Native MRzero spiral signals with complex receiver encoding before RSS.

An RSS object is an amplitude estimate, not a physical receiver. In a long
spiral readout, off-resonance redistributes complex signal before receiver
combination. Reconstructing a single RSS-weighted signal therefore need not
give the RSS of the measured receiver images. This model retains the native
MRzero histories, finite-voxel integrals and multi-echo coherence, and adds
fixed spatial complex receiver weights before summing object points.
"""

from __future__ import annotations

import numpy as np
import torch

from experiment_echo import EchoSubspinACDCModel
from experiment_multiecho import MultiEchoVoxelACDCModel
from acdc_gpu import ACDCModel
from measured_image_target import magnitude_objective_terms


class SpiralReceiverACDCModel(MultiEchoVoxelACDCModel):
    """Reconstruct each complex receiver, then combine its image magnitude.

    ``experiment['receive_weights']`` optionally has shape
    ``[receivers] + m0_proxy.shape``. Each supported child's receiver vector
    is normalized to unit RSS norm, preserving its complex phases. Thus the
    existing RSS amplitude proxy and through-plane volume weights are applied
    once; receiver magnitude is not counted twice. Weights are fixed during
    ACDC optimization, assumed constant within each finite in-plane box, and
    must come from independently acquired complex receiver data.

    With explicit weights, ``forward`` returns an RSS image (stored as a
    complex array with zero imaginary part) and complex data with a leading
    receiver axis. ``receiver_forward`` also exposes the separate complex
    receiver images. Omitting weights reproduces the original single-receiver
    model exactly, including its complex image and data-array shapes.

    ``receive_chunk`` bounds receiver matrix products and reconstruction
    batches. Native point histories are shared across receivers; no extra
    MRzero spin simulation or receiver-by-point-by-time cache is introduced.
    All current constraints, channel disabling and response timing are
    inherited without modification.
    """

    def __init__(self, experiment, calibration, max_current_A=3., *,
                 receive_chunk=4, **options):
        if (isinstance(receive_chunk, (bool, np.bool_))
                or int(receive_chunk) != receive_chunk or receive_chunk < 1):
            raise ValueError('receive_chunk must be a positive integer.')
        shape = np.asarray(experiment['m0_proxy']).shape
        self._receive_explicit = 'receive_weights' in experiment
        weights = np.asarray(experiment.get('receive_weights', np.ones((1,) + shape)))
        if (weights.ndim != len(shape) + 1 or weights.shape[1:] != shape
                or weights.shape[0] < 1
                or not np.issubdtype(weights.dtype, np.number)
                or not np.isfinite(weights).all()):
            raise ValueError(f'receive_weights must be finite numeric with shape [receivers] + {shape}.')
        self._receive_numpy = weights.astype(np.complex128, copy=True)
        self.nreceivers = weights.shape[0]
        self.receive_chunk = min(int(receive_chunk), self.nreceivers)
        self._receive_weights = None
        self._receive_input_norm_range = None
        # The parent forms its zero-B0 reference by calling our forward during
        # initialization. Receiver tensors must therefore initialize lazily.
        super().__init__(experiment, calibration, max_current_A, **options)
        self.runtime_info['receive_model'] = {
            'receivers': self.nreceivers,
            'explicit_complex_weights': self._receive_explicit,
            'receiver_chunk': self.receive_chunk,
            'encoding': 'fixed complex receive weights multiply each native child before the receiver sum',
            'combination': 'root sum of squared complex receiver-image magnitudes after reconstruction'
                if self._receive_explicit else 'original single complex receiver',
            'normalization': 'unit RSS receiver norm at every supported child; object RSS amplitude applied once',
            'input_RSS_norm_range': list(self._receive_input_norm_range),
            'within_voxel_assumption': 'receiver weights constant inside each finite in-plane box',
            'current_dependence': 'none; only physical field phase and finite-voxel coherence depend on currents',
            'native_point_cache_expansion': 1,
            'receiver_point_time_cache': False,
        }

    def _ensure_receive_weights(self):
        if self._receive_weights is not None:
            return
        selected = self._receive_numpy[:, self.support]
        norm = np.sqrt(np.sum(np.abs(selected)**2, axis=0))
        if np.any(norm <= 0) or not np.isfinite(norm).all():
            raise ValueError('receive_weights need a finite nonzero RSS norm at every supported child.')
        self._receive_input_norm_range = float(norm.min()), float(norm.max())
        self._receive_weights = torch.as_tensor(selected / norm[None],
            device=self.device, dtype=self.complex_dtype)

    def _receiver_forward_tensor(self, currents, *, response_enabled=None,
                                 b0_enabled=True, retain=False):
        self._ensure_receive_weights()
        signal = torch.empty((self.nreceivers, self.nsamples),
                             device=self.device, dtype=self.complex_dtype)
        saved = []
        for first in range(0, self.nsamples, self.sample_chunk):
            selected = slice(first, min(first + self.sample_chunk, self.nsamples))
            contribution = self._contribution(currents, selected, response_enabled, b0_enabled)
            for receiver in range(0, self.nreceivers, self.receive_chunk):
                rx = slice(receiver, min(receiver + self.receive_chunk, self.nreceivers))
                signal[rx, selected] = self._receive_weights[rx] @ contribution
            if retain:
                saved.append(contribution)
        kdata = signal.reshape((self.nreceivers,) + self.sampling_shape)
        images = torch.empty((self.nreceivers,) + self.image_size,
                             device=self.device, dtype=self.complex_dtype)
        for first in range(0, self.nreceivers, self.receive_chunk):
            rx = slice(first, min(first + self.receive_chunk, self.nreceivers))
            images[rx] = self._image(kdata[rx])
        return (images, kdata, saved) if retain else (images, kdata)

    def _combined_image(self, images):
        if not self._receive_explicit:
            return images[0]
        return torch.sqrt(images.abs().square().sum(dim=0)).to(self.complex_dtype)

    def _forward_tensor(self, currents, *, response_enabled=None,
                        b0_enabled=True, retain=False):
        output = self._receiver_forward_tensor(currents, response_enabled=response_enabled,
                                               b0_enabled=b0_enabled, retain=retain)
        images, kdata = output[:2]
        image = self._combined_image(images)
        if not self._receive_explicit:
            kdata = kdata[0]
        return (image, kdata, output[2]) if retain else (image, kdata)

    @torch.no_grad()
    def receiver_forward(self, currents_A, response_enabled=None, *, b0_enabled=True):
        """Return complex ``[receivers, rows, columns]`` images and ADC data."""
        images, kdata = self._receiver_forward_tensor(self._check_currents(currents_A),
            response_enabled=response_enabled, b0_enabled=b0_enabled)
        return self.numpy(images), self.numpy(kdata)

    @torch.no_grad()
    def historical_rss_forward(self, currents_A, response_enabled=None, *, b0_enabled=True):
        """Reproduce the old single-RSS-object approximation using this cache.

        This is a diagnostic comparison, not receiver combination. Sharing the
        fixed native histories avoids allocating a second large MRzero model.
        """
        image, kdata = ACDCModel._forward_tensor(self, self._check_currents(currents_A),
            response_enabled=response_enabled, b0_enabled=b0_enabled)
        return self.numpy(image), self.numpy(kdata)

    @torch.no_grad()
    def _objective_tensor(self, currents, gradient=True):
        if not self._receive_explicit:
            # Preserve the exact existing one-receiver objective and adjoint.
            return super()._objective_tensor(currents, gradient=gradient)
        receiver_images, kdata = self._receiver_forward_tensor(currents)
        image = self._combined_image(receiver_images)
        value, derivative = magnitude_objective_terms(self, image, gradient=gradient)
        if not gradient:
            return value, image, kdata

        # d RSS = Re(sum(conj(I_rx) * dI_rx)) / RSS. The generalized
        # derivative at exactly zero RSS is zero. No receiver phase is lost
        # before this chain rule or before the matched reconstruction adjoint.
        scale = derivative.real / image.real.clamp_min(torch.finfo(self.dtype).tiny)
        adjoint = torch.empty((self.nreceivers, self.nsamples),
                              device=self.device, dtype=self.complex_dtype)
        for first in range(0, self.nreceivers, self.receive_chunk):
            rx = slice(first, min(first + self.receive_chunk, self.nreceivers))
            receiver_derivative = scale[None] * receiver_images[rx]
            adjoint[rx] = self.reconstruction.adjoint(receiver_derivative).reshape(-1, self.nsamples)

        result = torch.zeros_like(currents)
        for first in range(0, self.nsamples, self.sample_chunk):
            selected = slice(first, min(first + self.sample_chunk, self.nsamples))
            factor, drow, dcolumn = self._voxel_factors(currents, selected, derivatives=True)
            # Obtain native centers with unresolved coherence but without the
            # in-plane sinc factor: its derivative is nonzero at sinc roots.
            point = EchoSubspinACDCModel._contribution(self, currents, selected)
            weighted_adjoint = torch.zeros_like(point)
            for receiver in range(0, self.nreceivers, self.receive_chunk):
                rx = slice(receiver, min(receiver + self.receive_chunk, self.nreceivers))
                weighted_adjoint += self._receive_weights[rx].T @ adjoint[rx, selected].conj()
            product = point * weighted_adjoint
            sensitivity = (2 * np.pi) * (product * factor).imag
            channel_sensitivity = self.B.T @ sensitivity
            channel_sensitivity += self._voxel_field_gradient[:, 0].T @ (product.real * drow)
            channel_sensitivity += self._voxel_field_gradient[:, 1].T @ (product.real * dcolumn)
            if self._channel_response:
                result += torch.bmm(channel_sensitivity[:, None], self._H[:, selected]).squeeze(1)
            else:
                result += channel_sensitivity @ self._H[selected]
        return value, result, image, kdata


__all__ = ['SpiralReceiverACDCModel']
