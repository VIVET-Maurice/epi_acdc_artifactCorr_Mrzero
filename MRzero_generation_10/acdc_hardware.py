"""ACDC amplifier model and hardware limits (same assumptions as generation 08).

The optimisation variables are the *commands* sent to the amplifiers: one
current per channel and per EPI line, ``currents_A[channel, line]``.

Three things are modelled:

1. the amplifier response: the real current follows a command with a
   first-order lag of time constant ``response_tau_s``;
2. the limits: maximum current per channel, maximum summed absolute current,
   and maximum ACDC field anywhere in the object;
3. the DAC: commands are stored on a signed 16-bit scale of +/- 5 A.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AcdcHardware:
    max_current_A: np.ndarray   # [channels] limit of each channel; 0 disables it
    total_current_A: float      # limit of sum(|current|) over the channels
    field_limit_Hz: float       # limit of |ACDC field| anywhere in the object
    response_tau_s: float       # amplifier time constant; 0 = instantaneous
    dac_full_scale_A: float = 5.0
    dac_levels: int = 32767

    @property
    def n_channels(self):
        return len(self.max_current_A)


def make_hardware(n_channels=32, max_current_A=3.0, disabled_channels=(),
                  total_current_A=30.0, field_limit_Hz=1000.0,
                  response_tau_s=133e-6 / np.log(100.0)):
    """Create the limits. ``disabled_channels`` are physical outputs, 1-based."""
    limits = np.full(n_channels, float(max_current_A))
    for output in disabled_channels:
        limits[output - 1] = 0.0
    return AcdcHardware(max_current_A=limits, total_current_A=float(total_current_A),
                        field_limit_Hz=float(field_limit_Hz),
                        response_tau_s=float(response_tau_s))


# ---------------------------------------------------------------------------
# 1. Amplifier response
# ---------------------------------------------------------------------------

def integrated_current_s(times_s, state_start_s, state_end_s, tau_s):
    """Time integral of the real current of every state, up to given times.

    Returns ``H[time, state]`` in seconds. If state ``i`` is commanded to 1 A
    during its window, the real current integrated from the RF pulse to
    ``times_s[t]`` is ``H[t, i]`` ampere-seconds. For any commands,

        integral of the current of one channel  =  H @ commands_of_that_channel

    With an instantaneous amplifier, ``H[t, i]`` is simply the time that state
    ``i`` has already been on. The first-order lag only rounds the edges.
    """
    times = np.asarray(times_s, dtype=float).reshape(-1, 1)

    def step_integral(elapsed):
        """Integral of the response to a unit step, ``elapsed`` after the step."""
        elapsed = np.maximum(elapsed, 0.0)
        if tau_s <= 0:
            return elapsed
        return elapsed + tau_s * np.expm1(-elapsed / tau_s)

    # A rectangular command is a step up at the start minus a step at the end.
    return step_integral(times - state_start_s) - step_integral(times - state_end_s)


# ---------------------------------------------------------------------------
# 2. Limits
# ---------------------------------------------------------------------------

def peak_field_Hz(currents_A, coil_maps, coil_gradients=None):
    """Largest |ACDC field| in the object for every state, shape [states].

    ``coil_maps`` [voxels, channels] gives the field at the voxel centres.
    With ``coil_gradients`` [voxels, 3, channels] (field change across the full
    voxel along x, y, z) the maximum is taken over the eight voxel corners.
    """
    field = np.abs(coil_maps @ currents_A)
    if coil_gradients is not None:
        for axis in range(3):
            field += 0.5 * np.abs(coil_gradients[:, axis] @ currents_A)
    return field.max(axis=0)


def project_onto_limits(currents_A, hardware, coil_maps=None, coil_gradients=None):
    """Return the closest commands that respect the hardware limits.

    Every state (column) is treated separately, as in generation 08:

    1. exact projection onto the channel limits and the total-current limit;
    2. if the field limit is still exceeded, the state is scaled down.
    """
    limits = hardware.max_current_A[:, None]
    magnitude = np.abs(currents_A)

    # Channel limits: clip every current.
    projected = np.sign(currents_A) * np.minimum(magnitude, limits)

    # Total-current limit, only for the states that exceed it. The exact
    # projection reduces every magnitude by the same threshold before clipping;
    # the smallest sufficient threshold is found by bisection.
    exceeding = np.abs(projected).sum(axis=0) > hardware.total_current_A
    if exceeding.any():
        excess = magnitude[:, exceeding]

        def reduced(threshold):
            return np.minimum(np.maximum(excess - threshold, 0.0), limits)

        low, high = np.zeros(excess.shape[1]), excess.max(axis=0)
        for _ in range(50):
            middle = 0.5 * (low + high)
            too_much = reduced(middle).sum(axis=0) > hardware.total_current_A
            low, high = np.where(too_much, middle, low), np.where(too_much, high, middle)
        projected[:, exceeding] = np.sign(currents_A[:, exceeding]) * reduced(high)

    if coil_maps is not None and np.isfinite(hardware.field_limit_Hz):
        peak = peak_field_Hz(projected, coil_maps, coil_gradients)
        projected = projected * np.minimum(1.0, hardware.field_limit_Hz / np.maximum(peak, 1e-30))
    return projected


def check_limits(currents_A, hardware, coil_maps, coil_gradients=None):
    """Summarise how close the commands are to every hardware limit."""
    largest = np.abs(currents_A).max(axis=1)
    total = np.abs(currents_A).sum(axis=0).max()
    field = peak_field_Hz(currents_A, coil_maps, coil_gradients).max()
    tolerance = 1e-9
    return {
        "max |current| [A]": float(largest.max()),
        "channel limits met": bool(np.all(largest <= hardware.max_current_A + tolerance)),
        "max summed |current| [A]": float(total),
        "total-current limit met": bool(total <= hardware.total_current_A + tolerance),
        "max |ACDC field| [Hz]": float(field),
        "field limit met": bool(field <= hardware.field_limit_Hz * (1 + 1e-9)),
    }


# ---------------------------------------------------------------------------
# 3. DAC
# ---------------------------------------------------------------------------

def quantize_to_dac(currents_A, hardware):
    """Round the commands toward zero on the signed 16-bit DAC scale."""
    codes = np.trunc(currents_A / hardware.dac_full_scale_A * hardware.dac_levels)
    return codes / hardware.dac_levels * hardware.dac_full_scale_A
