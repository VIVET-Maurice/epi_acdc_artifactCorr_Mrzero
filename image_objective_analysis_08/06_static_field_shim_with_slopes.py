"""Does a field-only static shim work once it also penalises the coil slopes?

Section 3.4 of README.md says the image sees the residual field AND its slopes
inside the voxel. This script keeps the classical recipe (constant currents,
quadratic field criterion, no image simulation in the solver) and adds the
slope terms:

    minimise  mean_p (dB0_p + B_p.I)^2
              + beta_z  * mean_p (Gz_p.I)^2                      slice slope of the coils
              + beta_xy * mean_p |g0_p + Gxy_p.I|^2              in-plane slope of B0 + coils
    subject to the same current and field limits as the notebook.

Slopes are in Hz per voxel width, so beta is dimensionless. Each solution is
then scored with the full image model.

Run:  python 06_static_field_shim_with_slopes.py
Writes outputs/06_static_slopes.json and outputs/06_static_slopes.npz.
"""

import json
from pathlib import Path

import numpy as np
import torch

from acdc08_twin import Twin

OUT = Path(__file__).resolve().parent / "outputs"


def solve(twin, beta_z, beta_xy, iterations=4000):
    """Projected accelerated gradient on the 32 constant currents."""
    P = twin.n_voxels
    rows = [twin.B / np.sqrt(P), np.sqrt(beta_z / P) * twin.G[:, 2]]
    targets = [-twin.b0 / np.sqrt(P), np.zeros(P)]
    for axis in (0, 1):
        rows.append(np.sqrt(beta_xy / P) * twin.G[:, axis])
        targets.append(-np.sqrt(beta_xy / P) * twin.g0[:, axis])
    A, b = np.concatenate(rows), np.concatenate(targets)
    AtA, Atb = A.T @ A, A.T @ b
    step = 1.0 / np.linalg.eigvalsh(AtA)[-1]
    project = lambda v: twin.project(torch.as_tensor(v[:, None])).numpy()[:, 0]
    x = project(np.zeros(twin.n_channels))
    y, momentum = x.copy(), 1.0
    for _ in range(iterations):
        new = project(y - step * (AtA @ y - Atb))
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        y = new + (momentum - 1) / next_momentum * (new - x)
        x, momentum = new, next_momentum
    return x


def main():
    twin = Twin()
    zero = np.zeros((twin.n_channels, twin.n_lines))
    with torch.no_grad():
        perfect = twin.simulate(zero, b0=False).abs()
    settings = [(0.0, 0.0), (0.1, 0.0), (0.3, 0.0), (1.0, 0.0), (3.0, 0.0), (10.0, 0.0), (30.0, 0.0),
                (1.0, 1.0), (3.0, 3.0), (10.0, 10.0)]
    report, currents = {}, {}
    print(f"{'beta_z':>7s} {'beta_xy':>7s} {'vs GRE':>7s} {'vs perfect':>10s} {'signal':>7s} "
          f"{'field Hz':>8s} {'shift px':>8s} {'slice cyc':>9s} {'sum|I| A':>8s}")
    for beta_z, beta_xy in settings:
        solution = solve(twin, beta_z, beta_xy)
        full = np.repeat(solution[:, None], twin.n_lines, axis=1)
        row = twin.summary(full, perfect)
        key = f"beta_z={beta_z:g}, beta_xy={beta_xy:g}"
        report[key], currents[key] = row, solution
        print(f"{beta_z:7g} {beta_xy:7g} {row['J_vs_GRE']:7.3f} {row['J_vs_perfect']:10.3f} "
              f"{row['signal_vs_perfect']:7.2f} {row['field_rms_Hz']:8.1f} {row['rms_shift_pixels']:8.2f} "
              f"{row['rms_slice_dephasing_at_TE_cycles']:9.2f} {row['mean_total_current_A']:8.1f}")
    reference = twin.saved["initial_currents_A"][:, 0].astype(float)
    difference = float(np.abs(currents["beta_z=0, beta_xy=0"] - reference).max())
    report["check"] = {"max_abs_difference_to_notebook_static_shim_A": difference}
    print(f"\nbeta = 0 against the notebook's static shim: largest difference {difference:.3f} A")
    (OUT / "06_static_slopes.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / "06_static_slopes.npz", **{k.replace("=", "_").replace(", ", "__"): v
                                                         for k, v in currents.items()})


if __name__ == "__main__":
    main()
