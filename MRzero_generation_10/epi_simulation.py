"""EPI simulation with ACDC currents: the signal model of generation 08.

The k-space signal is the sum of the object voxels,

    s(t) = sum_p  M_p(t) * exp(-2 pi i * B_p . J(t)) * D_p(t)

1. ``M_p(t)`` is computed once by MRzero: magnetisation at the centre of
   voxel ``p`` after one ideal 90 degree pulse, with T2* decay, the imaging
   gradients and the static off-resonance B0.
2. ``J(t)`` [A s] is the time integral of the real ACDC currents since the RF
   pulse. ``B_p`` [Hz/A] is the coil field at the voxel centre, so the middle
   factor is the phase added by ACDC. MRzero does not need to be run again
   when the currents change: for static spins and a single pulse this
   factorisation is exact.
3. ``D_p(t) = sinc(q_x) sinc(q_y) sinc(q_z)`` is the signal loss inside one
   voxel. ``q`` is the number of phase turns across the voxel along each
   axis, caused by the imaging gradients, by the variation of B0 and by the
   variation of the ACDC field. One voxel spans the whole slice thickness.

Everything after MRzero is written with differentiable PyTorch operations, so
an image-domain objective can also be optimised (see image_optimization.py).
"""

import numpy as np
import torch

from acdc_hardware import integrated_current_s


def choose_device(requested="auto", minimum_free_GiB=8.0):
    """Use the GPU only if it has enough free memory; otherwise the CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes / 1024**3 >= minimum_free_GiB:
            return torch.device("cuda")
    return torch.device("cpu")


def mrzero_voxel_magnetization(x_m, y_m, amplitude, T2star_s, B0_Hz,
                               kx_mInv, ky_mInv, time_s, device):
    """Run MRzero once: magnetisation of every voxel at every ADC sample.

    The voxels are point spins at the voxel centres. The sequence is one ideal
    90 degree pulse followed by the EPI trajectory. Returns a complex tensor
    [voxels, samples], in the order of the given samples.
    """
    import MRzeroCore as mr0

    def float32(values):
        return np.asarray(values, dtype=np.float32)

    def tensor(values):
        return torch.from_numpy(float32(values))

    # --- Object: point spins without T1 recovery or diffusion ---------------
    positions = np.column_stack((x_m, y_m, np.zeros(len(x_m))))
    phantom = mr0.CustomVoxelPhantom(
        float32(positions), PD=float32(amplitude), B0=float32(B0_Hz), B1=1.0,
        T1=float("inf"), T2=float32(T2star_s), T2dash=float("inf"),
        D=0.0, voxel_size=0.0, voxel_shape="box").build()
    if device.type == "cuda":
        phantom = phantom.cuda()
    # MRzero imposes a small diffusion and a voxel shape; switch both off so
    # that every spin is exactly a point (the voxel integral is applied later).
    phantom.D = torch.zeros_like(phantom.D)
    phantom.dephasing_func = lambda k, _: torch.ones(k.shape[0], device=k.device, dtype=k.dtype)

    # --- Sequence: MRzero needs the ADC samples in chronological order -------
    order = np.argsort(time_s)
    durations = np.diff(time_s[order], prepend=0.0)
    k = np.column_stack((kx_mInv[order], ky_mInv[order], np.zeros(len(order))))
    gradient_moments = np.diff(k, axis=0, prepend=np.zeros((1, 3)))

    repetition = mr0.Repetition.zero(len(order))
    repetition.pulse.angle = torch.tensor(np.pi / 2)
    repetition.pulse.phase = torch.tensor(np.pi / 2)   # magnetisation starts real
    repetition.pulse.usage = mr0.PulseUsage.EXCIT
    repetition.pulse.selective = False
    repetition.event_time[:] = tensor(durations)
    repetition.gradm[:] = tensor(gradient_moments)     # in cycles per metre
    repetition.adc_usage[:] = 1
    sequence = mr0.Sequence([repetition], normalized_grads=False)
    if device.type == "cuda":
        sequence = sequence.cuda()

    # --- Simulation --------------------------------------------------------
    graph = mr0.compute_graph(sequence, phantom, max_state_count=16, min_state_mag=0.0)
    with torch.no_grad():
        _, magnetization = mr0.execute_graph(
            graph, sequence, phantom, min_emitted_signal=0.0, min_latent_signal=0.0,
            print_progress=False, return_mag_adc=True)
    # One repetition, one coherence pathway: tensor [samples, voxels].
    magnetization = magnetization[0][0] * phantom.coil_sens[0][None, :]

    # Back to the order of the input samples, as [voxels, samples].
    inverse = torch.as_tensor(np.argsort(order), device=magnetization.device)
    return magnetization.index_select(0, inverse).T.contiguous()


class EpiSimulator:
    """Simulate the EPI image of a phantom for any ACDC current commands.

    ``magnetization`` [voxels, samples] is the MRzero result (with B0, without
    ACDC); ``ideal_image`` is the image of the same object in a perfect magnet.
    """

    def __init__(self, phantom, sequence, coil_fields, hardware,
                 device="auto", sample_chunk=1024):
        self.device = choose_device(device)
        self.image_shape = sequence.kx_mInv.shape
        self.n_samples = sequence.kx_mInv.size
        self.sample_chunk = int(sample_chunk)
        support = phantom.support

        # Per ADC sample (flattened [lines, readout] order).
        self._kx = self._tensor(sequence.kx_mInv.ravel())
        self._ky = self._tensor(sequence.ky_mInv.ravel())
        self._time = self._tensor(sequence.time_s.ravel())
        # Integrated current of every state at every ADC sample [samples, states].
        self._H = self._tensor(integrated_current_s(
            sequence.time_s.ravel(), sequence.state_start_s, sequence.state_end_s,
            hardware.response_tau_s))

        # Per voxel.
        self._coil_maps = self._tensor(coil_fields.maps_Hz_per_A)            # [voxels, channels]
        self._coil_gradients = self._tensor(coil_fields.gradients_Hz_per_A)  # [voxels, 3, channels]
        self._voxel_size = phantom.voxel_size_m

        # MRzero inputs shared by the two simulations below.
        arguments = dict(x_m=phantom.X_m[support], y_m=phantom.Y_m[support],
                         amplitude=phantom.amplitude[support], T2star_s=phantom.T2star_s[support],
                         kx_mInv=sequence.kx_mInv.ravel(), ky_mInv=sequence.ky_mInv.ravel(),
                         time_s=sequence.time_s.ravel(), device=self.device)
        zero_currents = np.zeros((hardware.n_channels, sequence.n_lines))

        # Ideal image: the same object in a perfect magnet (no B0, no current).
        self._B0_gradient = self._tensor(np.zeros((phantom.n_voxels, 3)))
        self.magnetization = mrzero_voxel_magnetization(B0_Hz=np.zeros(phantom.n_voxels), **arguments)
        self.ideal_image = self.simulate(zero_currents)

        # Actual object: off-resonance at the voxel centre (inside MRzero) and
        # its change across the voxel along x, y, z (used for the signal loss).
        self._B0_gradient = self._tensor(phantom.B0_gradient_Hz[support])    # [voxels, 3]
        self.magnetization = mrzero_voxel_magnetization(B0_Hz=phantom.B0_Hz[support], **arguments)

    # -- helpers ------------------------------------------------------------

    def _tensor(self, values):
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=self.device)

    def sample_chunks(self):
        """Slices that cover all ADC samples, a few at a time (limits memory)."""
        for first in range(0, self.n_samples, self.sample_chunk):
            yield slice(first, min(first + self.sample_chunk, self.n_samples))

    # -- signal model ---------------------------------------------------------

    def kspace_chunk(self, currents, samples):
        """Signal of the selected ADC samples; ``currents`` is a tensor [channels, states]."""
        integrated = currents @ self._H[samples].T            # [channels, samples] A s

        # Phase added by ACDC at the voxel centre.
        acdc_turns = self._coil_maps @ integrated             # [voxels, samples]
        signal = self.magnetization[:, samples] * torch.polar(
            torch.ones_like(acdc_turns), -2 * torch.pi * acdc_turns)

        # Signal loss inside the voxel: phase turns across it along x, y, z.
        k = (self._kx[samples], self._ky[samples], None)
        for axis in range(3):
            turns = self._coil_gradients[:, axis] @ integrated
            if k[axis] is not None:
                turns = turns + self._voxel_size[axis] * k[axis][None, :]
            turns = turns + self._B0_gradient[:, axis, None] * self._time[samples][None, :]
            signal = signal * torch.sinc(turns)
        return signal.sum(dim=0)

    def kspace(self, currents):
        """Complete k-space [lines, readout] as a complex tensor."""
        chunks = [self.kspace_chunk(currents, samples) for samples in self.sample_chunks()]
        return torch.cat(chunks).reshape(self.image_shape)

    @staticmethod
    def reconstruct(kspace):
        """Centred inverse Fourier transform (no correction of any kind)."""
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace)))

    def simulate(self, currents_A):
        """Return the complex EPI image (NumPy) for commands [channels, states]."""
        with torch.no_grad():
            image = self.reconstruct(self.kspace(self._tensor(currents_A)))
        return image.cpu().numpy()
