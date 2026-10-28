"""Native MRzero finite pixels with multi-echo residual coherence.

The measured echo train can identify a residual frequency width, but a single
2D slice does not locate that width along a spatial axis. Its sinc or Gaussian
coherence is therefore current independent and stays in the zero-B0 target.
Measured in-plane gradients and ACDC fields retain their separate exact box
integrals in :class:`FiniteVoxelEchoACDCModel`.
"""

from __future__ import annotations

import numpy as np
import torch

from experiment_voxel import FiniteVoxelEchoACDCModel


class MultiEchoVoxelACDCModel(FiniteVoxelEchoACDCModel):
    """Apply a fitted unresolved frequency distribution to native MRzero spins.

    ``experiment['multi_echo_coherence']['width_Hz']`` must match ``m0_proxy``.
    ``dispersion_kind`` selects 0=none, 1=uniform, 2=Gaussian per child.
    Width means full frequency width for uniform dispersion and frequency
    standard deviation for Gaussian dispersion. Their coherence factors are
    normalized ``sinc(width_Hz * time_since_RF)`` and
    ``exp(-2*pi**2*(width_Hz*time_since_RF)**2)``. Sinc can change sign after
    a null. Omitting ``dispersion_kind`` selects uniform dispersion.

    Amplitudes are the exponential envelope at the explicit reference time,
    before both in-plane and unresolved sinc factors. Residual R2 belongs in
    the existing ``relaxation`` field. A fitted width is not a through-plane
    gradient: neither ACDC compensation nor recovery under ``b0_enabled=False``
    can be inferred from these data, so this factor remains in both cases.

    Only one extra real width per native child is stored. The native CPU
    cache holds unmodified MRzero points; the normal device baseline cache
    stores the factor once. Current gradients inherited from the finite-pixel
    model therefore include this signed factor without differentiating it or
    dividing by it at its nulls.
    """

    def __init__(self, experiment, calibration, max_current_A=3., **options):
        metadata = experiment.get('multi_echo_coherence')
        if not isinstance(metadata, dict):
            raise ValueError("MultiEchoVoxelACDCModel requires experiment['multi_echo_coherence'] metadata.")
        shape = np.asarray(experiment['m0_proxy']).shape
        width = np.asarray(metadata.get('width_Hz'))
        if (width.shape != shape or np.iscomplexobj(width)
                or not np.issubdtype(width.dtype, np.number)
                or not np.isfinite(width).all() or np.any(width < 0)):
            raise ValueError(f'multi_echo_coherence.width_Hz must be finite nonnegative real with shape {shape}.')
        self._coherence_width_numpy = width.astype(np.float64, copy=True)
        kind = np.asarray(metadata.get('dispersion_kind', np.ones(shape, dtype=np.int8)))
        if (kind.shape != shape or np.iscomplexobj(kind)
                or not np.issubdtype(kind.dtype, np.number)
                or not np.isin(kind, (0, 1, 2)).all()):
            raise ValueError(f'multi_echo_coherence.dispersion_kind must contain 0, 1 or 2 with shape {shape}.')
        self._coherence_kind_numpy = kind.astype(np.int8, copy=True)
        self._coherence_width = None
        self._coherence_kind = None
        super().__init__(experiment, calibration, max_current_A, **options)
        selected_width = self._coherence_width_numpy[self.support]
        self.runtime_info['object_amplitude'] = (
            'multi-echo exponential envelope at explicit reference time, before finite-voxel coherence')
        self.runtime_info['multi_echo_coherence'] = {
            'model': 'unresolved frequency distribution: none, symmetric uniform signed sinc, or Gaussian coherence',
            'dispersion_kind': {'0': 'none', '1': 'uniform', '2': 'Gaussian'},
            'width_definition': 'uniform: full frequency width in Hz; Gaussian: frequency standard deviation in Hz',
            'width_min_Hz': float(selected_width.min()),
            'width_max_Hz': float(selected_width.max()),
            'nonzero_width_children': int(np.count_nonzero(selected_width)),
            'uniform_children': int(np.count_nonzero(self._coherence_kind_numpy[self.support] == 1)),
            'Gaussian_children': int(np.count_nonzero(self._coherence_kind_numpy[self.support] == 2)),
            'spatial_assignment': 'unresolved; no through-plane or other spatial direction inferred',
            'current_dependence': 'none; measured ACDC center and in-plane gradients remain separate',
            'zero_B0_target_policy': 'retain unresolved coherence; remove only measured center B0 and known in-plane B0 gradients',
            'amplitude_policy': 'exponential envelope at reference time; coherence factors applied once at time since RF',
            'native_point_cache_expansion': 1,
            'CPU_cache_policy': 'unmodified native MRzero point magnetization',
            'device_cache_policy': 'native magnetization multiplied by unresolved coherence exactly once',
        }
        self.runtime_info['inplane_voxel']['effective_R2star_overlap_possible'] = False
        self.runtime_info['inplane_voxel']['effective_R2star_assumption'] = (
            'Residual R2 and unresolved width were fitted jointly after accounting for the same fixed known in-plane gradients.')

    def _coherence_factor(self, selected):
        if self._coherence_width is None:
            self._coherence_width = self.tensor(self._coherence_width_numpy[self.support])[:, None]
            self._coherence_kind = torch.as_tensor(self._coherence_kind_numpy[self.support],
                                                   device=self.device)[:, None]
        q = self._coherence_width * self._evolution[selected][None]
        uniform = torch.sinc(q)
        gaussian = torch.exp(-2 * torch.pi**2 * q.square())
        return torch.where(self._coherence_kind == 1, uniform,
                           torch.where(self._coherence_kind == 2, gaussian, torch.ones_like(q)))

    def _make_base(self, selected, b0_enabled=True):
        # The superclass caches its own native result before returning. Never
        # modify it in place: otherwise CPU-cache reuse would apply sinc twice.
        native = super()._make_base(selected, b0_enabled)
        return native * self._coherence_factor(selected)


__all__ = ['MultiEchoVoxelACDCModel']
