"""One table for every waveform: the notebook's and the re-optimised ones.

Run after 03_optimise_variants.py:
    python 05_compare_variants.py
Writes outputs/05_comparison.json and prints a Markdown table.
"""

import json
from pathlib import Path

import numpy as np
import torch

from acdc08_twin import Twin

OUT = Path(__file__).resolve().parent / "outputs"

COLUMNS = [("J_vs_GRE", "Loss vs GRE target", "{:.3f}"),
           ("J_vs_perfect", "Loss vs perfect-B0 EPI", "{:.3f}"),
           ("signal_vs_perfect", "Signal / perfect", "{:.2f}"),
           ("field_rms_Hz", "Residual field RMS [Hz]", "{:.0f}"),
           ("rms_shift_pixels", "Shift RMS [pixels]", "{:.1f}"),
           ("rms_slice_dephasing_at_TE_cycles", "Slice dephasing at TE [cycles]", "{:.2f}"),
           ("median_local_echo_ms", "Median local echo [ms]", "{:.0f}"),
           ("time_varying_fraction", "Time-varying part", "{:.2f}"),
           ("mean_total_current_A", "Mean total current [A]", "{:.0f}")]


def main():
    twin = Twin()
    zero = np.zeros((twin.n_channels, twin.n_lines))
    waveforms = {
        "No ACDC": zero,
        "Static field shim (least squares)": twin.saved["initial_currents_A"].astype(float),
        "Notebook: GRE target, dynamic": twin.saved["currents_A"].astype(float)}
    slopes = OUT / "06_static_slopes.npz"
    if slopes.exists():
        constant = np.load(slopes)["beta_z_10__beta_xy_0"]
        waveforms["Static field shim + slice-slope penalty"] = np.repeat(constant[:, None], twin.n_lines, axis=1)
    for label, name in (("GRE target, static", "03_gre_static"),
                        ("Self-consistent target, static", "03_perfect_static"),
                        ("Self-consistent target, dynamic", "03_perfect_dynamic"),
                        ("Self-consistent, dynamic from best static", "03_perfect_dynamic_from_static")):
        path = OUT / f"{name}.npz"
        if path.exists():
            waveforms[label] = np.load(path)["currents_A"]
        else:
            print(f"missing {path.name}: run 03_optimise_variants.py")
    with torch.no_grad():
        perfect = twin.simulate(zero, b0=False).abs()
        images = {label: twin.simulate(currents).abs().numpy() for label, currents in waveforms.items()}
    report = {label: twin.summary(currents, perfect) for label, currents in waveforms.items()}
    (OUT / "05_comparison.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / "05_images.npz", **images)
    np.savez_compressed(OUT / "05_currents.npz", **waveforms)
    print("| Waveform | " + " | ".join(title for _, title, _ in COLUMNS) + " |")
    print("|---|" + "---:|" * len(COLUMNS))
    for label, row in report.items():
        print(f"| {label} | " + " | ".join(fmt.format(row[key]) for key, _, fmt in COLUMNS) + " |")


if __name__ == "__main__":
    main()
