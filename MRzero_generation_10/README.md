# Generation 10 — optimising the phase at every EPI echo

Generation 08 chooses the ACDC currents by comparing simulated images.
Generation 10 tests a different objective, on a numerical phantom: the phase
accumulated by the object at the echo of every EPI line,

```text
phi_n(p) = 2*pi*B0(p)*T_n + sum_{i<=n} A(p) x_i        (T_n: echo time of line n, x_i: currents of line i)
Psi      = sum_n ||phi_n||^2 ,   ||phi_n||^2 = mean over the object of phi_n(p)^2
```

`Psi` is quadratic in the currents and the hardware limits are convex, so the
minimum is unique and is found in about ten seconds on a CPU, without running
any simulation during the optimisation.

This folder is self-contained. It only reads `../data/ACDC_3T.npz`.

## Files

| File | Content |
| --- | --- |
| `10_acdc_phase_optimization_MRzero.ipynb` | The experiment, from the phantom to the comparison table |
| `phase_optimization.py` | **New in generation 10**: the phase at every echo, `Psi`, its minimisation (FISTA) and a static shim |
| `phantom.py` | Numerical head phantom: contrast, grid, air cavities with their exact dipole field in 3D |
| `epi_sequence.py` | EPI timing: k-space, ADC times, echo times `T_n`, one current window per line |
| `acdc_calibration.py` | Reads the 3D calibration and samples the coil fields on the voxels |
| `acdc_hardware.py` | Amplifier lag, limits (per channel, total, field), DAC rounding |
| `epi_simulation.py` | Signal model of generation 08: MRzero at the voxel centres, ACDC phase, signal loss inside the voxel along x, y, z |
| `image_optimization.py` | Image objective of generation 08 (with field guidance), used as the reference |
| `display_helpers.py` | Tables and figures of the notebook |

## Running

Use the environment of generations 08/09 (Python 3.11, PyTorch 2.8.0,
MRzeroCore 1.2.0; see `requirements.txt`). Open the notebook from this folder
and run the cells in order, or from the repository root:

```bash
python MRzero_validation/execute_notebook.py \
  MRzero_generation_10/10_acdc_phase_optimization_MRzero.ipynb --timeout 3600
```

The whole notebook takes about two minutes. Each run writes its figures,
currents and summary to `results/mrzero10_<date>_<time>/`.
The parameters are in the first code cells. `DEVICE="auto"` uses the GPU only
if at least 8 GiB are free; the CPU is sufficient.

## What is kept from generation 08

- One voxel per pixel spanning the whole 5 mm slice; MRzero gives the
  magnetisation at the voxel centre; the loss of signal inside the voxel is the
  product of three sinc factors, including the gradients of the ACDC field.
- One current state per EPI line, zero current before the first imaging line,
  first-order amplifier response (99 % in 133 µs).
- Limits: 3 A per channel, 30 A in total, 1000 Hz, output 9 disabled.
- EPI protocol of acquisition 029: 128 × 128, FOV 200 mm, echo spacing
  0.88 ms, TE 61.3 ms.

Left out because the object is a numerical phantom: Twix readers, the 32-echo
fit, baseline calibration, and CSV export of the waveforms.

## The phantom

A head-sized ellipse with white matter, cortex, ventricles and a grid of darker
lines, in the coil frame of the calibration. Three spherical air cavities lie
below the slice (nasal cavity with sinuses, two ear canals). Each produces the
exact dipole field of a sphere of air in tissue at 3 T. The off-resonance is
between −36 and +210 Hz (standard deviation 36 Hz), close to the map measured
in acquisition 029 (−87 to +251 Hz, 38 Hz). Because the field is known in 3D,
its variation through the slice is known too; above the cavities the signal is
lost at TE, as in a real EPI.

## Options of the phase objective

`PhaseProblem(..., intravoxel_weight=0, voxel_weights=None)` is the formula
above taken literally: each voxel is a point and all voxels count equally.

- `intravoxel_weight` adds the phase change across each voxel along x, y, z.
  With `1/12` the objective is the exact mean of `phi^2` over the voxel volume,
  so dephasing through the slice is also penalised.
- `voxel_weights` weights the voxels in the mean. The notebook uses a mask
  that leaves out the voxels keeping less than half of their signal at TE
  without correction.

## Result of the first run (12 November 2026)

Image error is the NRMSE of the magnitude image against the ideal image of the
phantom (no off-resonance). Signal is the summed magnitude relative to it.

| Currents | Image error [%] | Signal [%] | Psi [rad²] | Mean total current [A] |
| --- | ---: | ---: | ---: | ---: |
| No ACDC | 37.8 | 90.4 | 30 598 | 0.0 |
| Static shim, voxel centres | 71.9 | 36.2 | 2 718 | 30.0 |
| **Phase, voxel centres** (the formula above) | 72.3 | 36.6 | 2 602 | 29.2 |
| Static shim, voxel volume + signal mask | 27.7 | 89.0 | 9 889 | 18.8 |
| **Phase, voxel volume + signal mask** | 28.6 | 87.7 | 9 170 | 18.1 |
| Image objective of generation 08 (300 iterations) | 17.9 | 98.8 | 19 110 | 8.4 |

On this phantom, optimising the phase is not better than optimising the image.

- **The formula taken literally makes the image worse than no ACDC.** It
  divides `Psi` by 12, but it uses the full 30 A to do so. At that level the
  coils create a phase variation of more than one turn through the 5 mm slice
  over large parts of the object, which cancels the signal (36 % left). A
  conventional static shim on the same voxel centres fails in the same way.
  Generation 08 shows the same effect on the measured data of acquisition 029:
  its static-shim starting point has a relative image error of 0.62, against
  0.45 without ACDC.
- **With the voxel volume and the signal mask, the phase objective improves the
  image** (28.6 % against 37.8 %). This holds for other settings: thresholds
  0.2 to 0.8 and intravoxel weights 1/12 to 10 gave 26 to 30 %.
- **The phase objective does not benefit from changing the currents during the
  readout.** Its solution is almost constant in time; replacing it by its time
  average, or by a static shim with the same criterion, gives the same image
  (27.7 to 28.6 %).
- **The image objective is better (17.9 %) and uses less current (8.4 A).**
  Its currents do vary during the readout: their time average gives 38.1 %.
  It reaches a lower image error with a `Psi` seven times larger than the
  phase optimum, so `Psi` is not a reliable predictor of image quality here.
- Cost: about 12 s on a CPU for the phase objective, 71 s on a shared GPU for
  300 iterations of the image objective.

Why `Psi` and the image disagree: `Psi` counts phase in radians everywhere,
including where it is many turns and cannot be corrected by the coils, and it
penalises a phase that is constant in time, which does not change a magnitude
image. The image depends on phase differences inside a voxel and between
echoes.

Limits of this result: one phantom, one slice, noise-free, with the object
defined on the image grid and a B0 map known exactly. The comparison uses the
same simulator for optimisation and evaluation. The image objective was run
for 300 iterations from zero currents and had not fully converged. On an
earlier version of the phantom, 1000 iterations lowered its error from 19.0 to
16.9 %, and starting it from the phase solution did not help (18.6 % after 300
iterations).
