"""Check that the closed-form twin reproduces what notebook 08 saved.

The pipeline obtained its k-space from MRzero (float32) plus analytic factors.
Here the same k-space is recomputed without MRzero and without any pipeline
module. Agreement to ~1e-6 shows that the equations in README.md are the ones
the optimiser actually used.

Run:  python 01_validate_twin.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from acdc08_twin import Twin

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


def relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    twin = Twin()
    zero = np.zeros((twin.n_channels, twin.n_lines))
    report = {"voxels": twin.n_voxels, "TE_ms": 1e3 * twin.TE, "echo_spacing_ms": 1e3 * twin.esp}
    with torch.no_grad():
        cases = {
            "no ACDC": (zero, "uncalibrated_baseline_kdata", "baseline_image"),
            "optimised currents": (twin.saved["currents_A"], "optimized_kdata", "optimized_image"),
        }
        for name, (currents, kname, iname) in cases.items():
            k = twin.signal(currents)
            image = twin.image(k)
            report[name] = {
                "kspace_relative_error": relative(k.numpy(), twin.saved[kname]),
                "image_relative_error": relative(image.numpy(), twin.saved[iname]),
                "J_image_twin": float(twin.image_loss(image)),
                "J_image_from_saved_image": float(twin.image_loss(twin.saved[iname])),
            }
        ideal = twin.simulate(zero, b0=False)
        report["zero B0, no ACDC"] = {
            "image_relative_error": relative(ideal.numpy(), twin.saved["ideal_image"]),
            "J_image_twin": float(twin.image_loss(ideal)),
        }
        static = twin.simulate(twin.saved["initial_currents_A"])
        report["static field shim (notebook start)"] = {
            "J_image_twin": float(twin.image_loss(static)),
            "J_image_notebook_history": float(twin.saved["relative_mse"][0]),
        }
        report["field penalty"] = {
            "static_twin": float(twin.field_penalty(twin.saved["initial_currents_A"])),
            "static_notebook": float(twin.saved["optimization_field_penalty"][0]),
            "optimised_twin": float(twin.field_penalty(twin.saved["currents_A"])),
            "optimised_notebook": float(twin.saved["optimization_field_penalty"][-1]),
        }
    print(json.dumps(report, indent=2))
    (OUT / "01_validation.json").write_text(json.dumps(report, indent=2))
    worst = max(report[c][key] for c in ("no ACDC", "optimised currents")
                for key in ("kspace_relative_error", "image_relative_error"))
    assert worst < 1e-4, f"twin does not reproduce the saved run (relative error {worst:.2e})"
    print(f"\nOK: largest relative error against the saved run = {worst:.2e}")


if __name__ == "__main__":
    main()
