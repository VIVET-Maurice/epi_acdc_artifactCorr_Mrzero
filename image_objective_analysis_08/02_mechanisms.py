"""What does the image objective reward?  Term-by-term dissection of run 08.

For four current waveforms (none, static field shim, optimised, time-average
of the optimised) the script evaluates the notebook loss and switches single
physical terms of the model off, one at a time. It also computes, per voxel,
the time at which its neighbourhood is in phase along the phase-encode
direction (the local echo), which sets the T2*-weighting of that voxel.

Run:  python 02_mechanisms.py        (about one minute on CPU)
Writes outputs/02_mechanisms.json and outputs/02_arrays.npz.
"""

import json
from pathlib import Path

import numpy as np
import torch

from acdc08_twin import Twin

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

SWITCHES = {
    "full model": {},
    "ACDC slice gradient removed": dict(acdc_gradient_axes=(0, 1)),
    "all ACDC gradients removed": dict(acdc_gradient_axes=()),
    "point spins (no voxel integral)": dict(voxel=False),
    "amplitude frozen at TE": dict(decay="frozen"),
}


def main():
    twin = Twin()
    optimised = twin.saved["currents_A"].astype(float)
    static = twin.saved["initial_currents_A"].astype(float)
    waveforms = {
        "no ACDC": np.zeros_like(optimised),
        "static field shim": static,
        "image-optimised (notebook)": optimised,
        "time-average of optimised": np.repeat(optimised.mean(axis=1, keepdims=True), twin.n_lines, axis=1),
    }
    report, arrays = {}, {}
    with torch.no_grad():
        perfect = twin.simulate(waveforms["no ACDC"], b0=False).abs()
        perfect_frozen = twin.simulate(waveforms["no ACDC"], b0=False, decay="frozen").abs()
        perfect_kspace = twin.signal(waveforms["no ACDC"], b0=False).abs().numpy()
        arrays["perfect"] = perfect.numpy()
        arrays["line energy perfect"] = (perfect_kspace**2).sum(axis=1)
        report["perfect B0, no ACDC"] = {
            "J_vs_GRE": float(twin.image_loss(perfect)),
            "J_vs_GRE_amplitude_frozen_at_TE": float(twin.image_loss(perfect_frozen)),
            "signal_vs_GRE": float(perfect[twin.mask].sum() / twin.reference[twin.mask].sum()),
            "kspace_energy_share_centre_line_percent": 100 * float(
                arrays["line energy perfect"][twin.n_lines // 2] / arrays["line energy perfect"].sum()),
            "kspace_energy_share_centre_pm8_lines_percent": 100 * float(
                arrays["line energy perfect"][twin.n_lines // 2 - 8:twin.n_lines // 2 + 9].sum()
                / arrays["line energy perfect"].sum()),
            "amplitude_at_TE_vs_first_echo": float(twin.amplitude_at(twin.TE).sum() / twin.amplitude.sum())}
        for name, currents in waveforms.items():
            entry = twin.summary(currents, perfect)
            for label, options in SWITCHES.items():
                image = twin.simulate(currents, **options).abs()
                reference = perfect_frozen if options.get("decay") == "frozen" else perfect
                entry[label] = {
                    "J_vs_GRE": float(twin.image_loss(image)),
                    "J_vs_perfect_simulation": float(twin.image_loss(image, reference)),
                    "signal_vs_perfect": float(image.sum() / reference.sum())}
                if label == "full model":
                    arrays["image " + name] = image.numpy()
            kspace = twin.signal(currents).abs().numpy()
            line_energy = (kspace**2).sum(axis=1)
            entry["kspace_energy_centroid_ms"] = 1e3 * float((line_energy * twin.line_centres).sum() / line_energy.sum())
            entry["kspace_energy_before_TE_percent"] = 100 * float(line_energy[:twin.n_lines // 2].sum() / line_energy.sum())
            echo = twin.local_echo(currents)
            arrays["echo " + name] = twin.to_pixels(echo["time"])
            arrays["qz " + name] = echo["q_slice"]
            arrays["line energy " + name] = line_energy
            residual = twin.b0[:, None] + twin.B @ twin.mean_line_current(currents).numpy()
            arrays["field rms per line " + name] = np.sqrt((residual**2).mean(axis=0))
            report[name] = entry
    arrays.update(reference=twin.reference, mask=twin.mask, line_centres=twin.line_centres,
                  currents_optimised=optimised, currents_static=static)
    np.savez_compressed(OUT / "02_arrays.npz", **arrays)
    (OUT / "02_mechanisms.json").write_text(json.dumps(report, indent=2))

    print(f"{'waveform':30s} {'model variant':34s} {'J vs GRE':>9s} {'J vs perfect':>13s} {'signal':>7s}")
    print(f"{'perfect B0, no ACDC':30s} {'full model':34s} {report['perfect B0, no ACDC']['J_vs_GRE']:9.3f} {0:13.3f} {1:7.2f}")
    for name in waveforms:
        for label in SWITCHES:
            r = report[name][label]
            print(f"{name:30s} {label:34s} {r['J_vs_GRE']:9.3f} {r['J_vs_perfect_simulation']:13.3f} {r['signal_vs_perfect']:7.2f}")
        scalars = {k: v for k, v in report[name].items() if not isinstance(v, dict)}
        print(" " * 30 + " " + "; ".join(f"{k}={v:.3g}" for k, v in scalars.items()))


if __name__ == "__main__":
    main()
