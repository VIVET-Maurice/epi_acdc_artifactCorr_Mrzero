"""Explicit Fourier sums with the same inverse convention as generation 06."""

from inplane_geometry import FrameReconstruction


class FrameDFTReconstruction(FrameReconstruction):
    """Use the actual ADC coordinates and a positive-sign direct Fourier sum.

    Reconstruction is sum(signal * exp(+2*pi*i*k.x)) / sample_count.
    The positive sign matches IFFT; DFT here names its explicit evaluation,
    rather than a change to the physical encoding convention. The inherited
    adjoint evaluates the conjugate negative-sign sum with the same scale.
    No ADC is rounded onto a Cartesian grid.
    """

    def __init__(self, acquisition, device, dtype, frame_angle_deg=0.,
                 sample_chunk=256, frame_matrix=None):
        acquisition = dict(acquisition, reconstruction_method="direct")
        super().__init__(acquisition, device, dtype, frame_angle_deg,
                         sample_chunk, frame_matrix)
        # ContinuousReconstruction normally detects an exact Cartesian grid
        # and uses IFFT, even when reconstruction_method='direct'. Override
        # that acceleration while keeping its tested direct/adjoint pair.
        self.base.method = "direct_chunked"
        self.base._offset_phase = self.base._row_x = self.base._row_y = None
        self.base.report.update(method="direct_chunked", cached_kernel_MiB=0.,
                                evaluation="explicit positive-sign Fourier sum; no FFT")
        self.report.update(method="direct_chunked", cached_kernel_MiB=0.,
            evaluation="explicit positive-sign Fourier sum; no FFT",
            operator="explicit normalized Fourier adjoint on the common output frame")


__all__ = ["FrameDFTReconstruction"]
