"""Timing of the single-shot gradient-echo EPI simulated in generation 10.

Only what the simulation and the optimisation need is described here:

* where every ADC sample lies in k-space and when it is acquired;
* the time window during which each ACDC current state is played
  (one state per EPI line, as in generation 08);
* the echo time ``T_n`` of every line, i.e. the instant at which line ``n``
  crosses the centre of the readout (kx = 0).

The default values reproduce the sequence used on the scanner,
``gre_epi_128x128_fov200mm_nav3_cimax_snr_t2star40ms_line_triggers_v6.seq``:
128 x 128 matrix, 200 mm field of view, 5 us dwell time, 0.88 ms echo spacing,
first imaging line 4.57 ms after the RF centre, TE = 61.33 ms.

All times are in seconds after the centre of the excitation pulse.
k-space coordinates are in cycles per metre.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpiSequence:
    """Sampling pattern and current windows of one EPI readout train.

    The lines are acquired from the bottom of k-space to the top, so row ``n``
    of every array is also the ``n``-th line in time. Inside a row the samples
    are stored with kx increasing; ``time_s`` remembers that every other line
    is actually traversed from right to left.
    """

    kx_mInv: np.ndarray        # [lines, readout] readout k-space coordinate
    ky_mInv: np.ndarray        # [lines, readout] phase-encoding k-space coordinate
    time_s: np.ndarray         # [lines, readout] centre of every ADC sample
    state_start_s: np.ndarray  # [lines] start of the current state of each line
    state_end_s: np.ndarray    # [lines] end of that state (start of the next line)
    echo_time_s: np.ndarray    # [lines] T_n: time at which line n crosses kx = 0
    fov_m: float
    dwell_s: float
    echo_spacing_s: float

    @property
    def n_lines(self):
        return self.kx_mInv.shape[0]

    @property
    def n_readout(self):
        return self.kx_mInv.shape[1]

    @property
    def TE_s(self):
        """Echo time of the image: the line that crosses the k-space centre."""
        return float(self.echo_time_s[self.n_lines // 2])


def make_epi_sequence(matrix=128, fov_m=0.2, echo_spacing_s=0.88e-3, dwell_s=5e-6,
                      first_line_start_s=4.57e-3, adc_delay_s=0.12e-3):
    """Build the EPI timing from a few scanner parameters.

    ``first_line_start_s`` is the start of the first imaging line (the three
    navigator echoes are played before it, without ACDC current).
    ``adc_delay_s`` is the delay between the start of a line and its first
    ADC sample (gradient ramp).
    """
    readout_duration_s = matrix * dwell_s
    if adc_delay_s + readout_duration_s > echo_spacing_s:
        raise ValueError("The readout does not fit inside one echo spacing.")

    # Centred Cartesian grid: index matrix // 2 is k = 0.
    k_axis = (np.arange(matrix) - matrix // 2) / fov_m
    kx, ky = np.meshgrid(k_axis, k_axis)

    line = np.arange(matrix)
    state_start = first_line_start_s + line * echo_spacing_s
    state_end = state_start + echo_spacing_s

    # Position of every stored sample in the acquisition order of its line.
    # Even lines are read from right to left, like on the scanner.
    acquisition_index = np.tile(np.arange(matrix), (matrix, 1))
    right_to_left = line % 2 == 0
    acquisition_index[right_to_left] = matrix - 1 - acquisition_index[right_to_left]
    time = state_start[:, None] + adc_delay_s + (acquisition_index + 0.5) * dwell_s

    return EpiSequence(kx_mInv=kx, ky_mInv=ky, time_s=time,
                       state_start_s=state_start, state_end_s=state_end,
                       echo_time_s=time[:, matrix // 2].copy(),
                       fov_m=float(fov_m), dwell_s=float(dwell_s),
                       echo_spacing_s=float(echo_spacing_s))
