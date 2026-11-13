"""Re-run the image optimisation with one ingredient changed at a time.

Same optimiser as notebook 08 (projected Adam, 0.01 A step, same limits and
field guidance), applied to the closed-form twin with automatic
differentiation. Two things can be changed:

  --target gre       measured GRE / fixed gain (the notebook target)
  --target perfect   the simulated EPI with zero B0 and no ACDC: a target the
                     model can reach exactly, with the same T2* weighting
  --mode dynamic     one current vector per EPI line (32 x 128 unknowns)
  --mode static      one current vector for the whole train (32 unknowns)

Examples:
  python 03_optimise_variants.py --target gre --mode dynamic --iterations 20 --tag parity
  python 03_optimise_variants.py --target perfect --mode dynamic --iterations 600
  python 03_optimise_variants.py --target perfect --mode dynamic --tag from_static \
         --init-file outputs/03_perfect_static.npz      # what does time variation add?

Writes outputs/03_<target>_<mode>[_tag].npz with currents, images and history.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from acdc08_twin import Twin

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("gre", "perfect"), default="gre")
    parser.add_argument("--mode", choices=("dynamic", "static"), default="dynamic")
    parser.add_argument("--init", choices=("static", "zero"), default="static")
    parser.add_argument("--init-file", default=None,
                        help="start from the currents_A of a previous outputs/03_*.npz instead")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--guidance", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    twin = Twin(threads=args.threads)
    zero = np.zeros((twin.n_channels, twin.n_lines))
    with torch.no_grad():
        perfect = twin.simulate(zero, b0=False).abs()
    reference = None if args.target == "gre" else perfect
    start = twin.saved["initial_currents_A"].astype(float) if args.init == "static" else zero
    if args.init_file:
        start = np.load(args.init_file)["currents_A"].astype(float)
    lines = twin.n_lines if args.mode == "dynamic" else 1
    expand = (lambda c: c) if args.mode == "dynamic" else (lambda c: c.expand(-1, twin.n_lines))
    currents = twin.project(torch.as_tensor(start[:, :lines]))

    moment, variance = torch.zeros_like(currents), torch.zeros_like(currents)
    history = {key: [] for key in ("J_image", "J_total", "field_rms_Hz")}
    best_value, best = np.inf, currents.clone()
    started = time.perf_counter()
    for iteration in range(args.iterations + 1):
        variable = currents.clone().requires_grad_(True)
        full = expand(variable)
        image_loss = twin.image_loss(twin.simulate(full), reference)
        penalty = twin.field_penalty(full)
        total = image_loss + args.guidance * penalty
        total.backward()
        gradient = variable.grad
        history["J_image"].append(float(image_loss))
        history["J_total"].append(float(total))
        history["field_rms_Hz"].append(100 * float(penalty)**0.5)
        if float(total) < best_value:
            best_value, best = float(total), currents.clone()
        if iteration % 10 == 0 or iteration == args.iterations:
            print(f"{iteration:5d}  J_image {float(image_loss):.4f}  J_total {float(total):.4f}  "
                  f"field rms {history['field_rms_Hz'][-1]:6.1f} Hz  "
                  f"[{time.perf_counter() - started:6.0f} s]", flush=True)
        if iteration == args.iterations:
            break
        step = iteration + 1
        moment = 0.9 * moment + 0.1 * gradient
        variance = 0.999 * variance + 0.001 * gradient**2
        direction = (moment / (1 - 0.9**step)) / ((variance / (1 - 0.999**step)).sqrt() + 1e-8)
        currents = twin.project(currents - args.learning_rate * direction)

    final = expand(best).detach().numpy().copy()
    with torch.no_grad():
        image = twin.simulate(final).abs()
        result = dict(
            currents_A=final, image=image.numpy(), perfect=perfect.numpy(),
            J_vs_GRE=float(twin.image_loss(image)),
            J_vs_perfect=float(twin.image_loss(image, perfect)),
            signal_vs_perfect=float(image.sum() / perfect.sum()),
            field_rms_Hz=100 * float(twin.field_penalty(final))**0.5,
            **{"history_" + key: np.asarray(value) for key, value in history.items()})
    name = f"03_{args.target}_{args.mode}" + (f"_{args.tag}" if args.tag else "")
    np.savez_compressed(OUT / f"{name}.npz", **result)
    print(f"\n{name}: J vs GRE {result['J_vs_GRE']:.4f} | J vs perfect {result['J_vs_perfect']:.4f} | "
          f"signal/perfect {result['signal_vs_perfect']:.2f} | field rms {result['field_rms_Hz']:.1f} Hz")
    if args.tag == "parity":
        saved = twin.saved["relative_mse"][:len(history["J_image"])]
        worst = float(np.max(np.abs(np.asarray(history["J_image"]) - saved)))
        print("notebook history :", np.round(saved[:8], 4))
        print("twin history     :", np.round(history["J_image"][:8], 4))
        print(f"largest difference over {len(saved)} iterations: {worst:.2e}")


if __name__ == "__main__":
    main()
