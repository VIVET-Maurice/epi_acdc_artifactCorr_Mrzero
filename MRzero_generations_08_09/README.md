# Generations 08 and 09

These three notebooks use the **active** configuration of
`MRzero_validation/06_acdc_echo32_b0_s0_t2star_MRzeroV3.ipynb`: acquisition
`acquisition_029_(06)`, 32 MEGRE echoes, 90° alignment, a 90° common frame,
disabled output 9, the hardware response, and the measured GRE target.

| Notebook | Intravoxel dephasing | ACDC | Reconstruction |
| --- | --- | --- | --- |
| `08_acdc_echo32_b0_s0_t2star_MRzero.ipynb` | Analytic x/y/z integral, one MRzero center per voxel | Integrated phase applied after MRzero, as in 06 | Existing operator: IFFT or Fourier sums, depending on the trajectory |
| `08_V2_acdc_echo32_b0_s0_t2star_MRzero.ipynb` | Same as 08 | Same as 08 | DFT encoding matrix, SVD least-squares inverse, optional Tikhonov; no FFT/IFFT |
| `09_acdc_echo32_b0_s0_t2star_MRzero.ipynb` | Physical voxel dimensions and native MRzero voxel kernels inside the simulator | Total B0 history inside the simulator; PDG recomputed at each evaluation | Same operator as 08 |

## Installation and required data

Use a separate Python 3.11 environment for these generations. Their native
executor uses **MRzeroCore 1.2.0 and PyTorch 2.8.0**; the older root requirements
pin a different MRzeroCore version. From the repository root:

```bash
python3.11 -m venv .venv-generations
source .venv-generations/bin/activate
# Install the PyTorch 2.8.0 CPU/CUDA build appropriate for this device first.
python -m pip install -r MRzero_generations_08_09/requirements.txt
python -m pip check
python -m ipykernel install --user --name acdc-generations --display-name "Python (ACDC generations)"
```

Select this kernel when opening a notebook. An installed CUDA build such as
`torch==2.8.0+cu128` satisfies the requirement and is preserved. Install the
generation requirements independently of `requirements-server.txt` and the
older `MRzero_validation/requirements*.txt` files.

Keep the acquisition folder, its GRE/EPI/B0 Twix `.dat` files and their matching
Pulseq `.seq` files on the destination device. The defaults select
`acquisition_029_(06)`. The notebooks also require **`data/ACDC_3T.npz` at the
repository root**, separately from those acquisitions. Use the calibrated 3D
field maps matching the configured source hash, channel order, units and
registration; an arbitrary calibration file will fail validation. Retain the
repository layout, including `Experiment/`, which the notebooks use to locate
the project root. Adapt the input paths in the configuration cell if necessary.

Generated results and caches are excluded from Git. The three simulation
notebooks regenerate their results from the inputs above and need no saved
results, compiled Python files or prepared presentation cache.

The optional `presentation/` source has additional dependencies in
`presentation/requirements_presentation.txt` and system tools listed there.
Its historical comparison also needs the external legacy
`mrzero06_optimization.json` and `.npz` files from the run documented in
[the presentation README](presentation/README.md); acquisition data alone do
not reproduce that comparison. On a fresh device, set `REFRESH_INPUTS=True`
and `RUN_EXPENSIVE=True` in the presentation notebook to regenerate its other
numerical inputs. Its local pickle cache must be rebuilt on the destination.

## Usage

Open a notebook in this folder and restart the kernel before switching
generations. Edit the paths and parameters at the beginning, then run the cells
in order. Acquisitions and `data/ACDC_3T.npz` remain in the parent folder.
The required functions are copied here to isolate these generations from the
earlier notebooks. Only imported runtime modules are included, without older
notebooks, tests, or generation scripts.

The GPU settings and local exports from 06 are retained. For CPU execution, set
`GPU_REQUIRED=False` and `GPU_PROFILE="cpu"`. To inspect diagnostics without
optimization, set `RUN_OPTIMIZATION=False`. Field exports remain independent;
the three `EXPORT_*WAVEFORM*` switches are explained in the parameters.
Each run writes its results to `results/mrzero08_*`, `results/mrzero08_V2_*`,
or `results/mrzero09_*`. CSV files remain in `INPUT_DIR/waveform`, as in 06.

## Changes

- **One voxel across the full slice thickness**, with weight 1. The eight
  sub-slices are removed.
- ACDC variations across the full width of all three axes and at the eight
  corners come from the 3D calibration in the measured slice geometry. Voxels
  without complete coverage are excluded; the remaining signal is not renormalized.
- The measured B0 map is 2D, so its normal derivative is unknown and assumed
  zero. The residual dispersion fitted from 32 echoes has no identified spatial direction.
- 08 applies the three sinc integrals once and differentiates all three factors
  during optimization. The residual coherence from 06 is retained once.
- V2 now solves the matrix inverse from the meeting diagram, `s=A m`, rather
  than only applying the normalized adjoint. All notebook text, comments,
  figure labels, console messages and generated HTML are in English.

## Matrix reconstruction in 08 V2

`reconstruction_08.py` forms the actual-trajectory encoding
`A[i,j]=exp(-2πi k[i]·r[j])`. The factor `2π` is required because the sequence
reader gives k-space in cycles/metre. One row corresponds to an ADC sample;
one column corresponds to an output voxel. Receiver data are reconstructed
separately before RSS combination. Rotation, mirrors and the Fourier origin
are preserved.

`DFT_REGULARIZATION=0.0` uses the Moore-Penrose inverse, evaluated by SVD.
Positive values solve `min ||A m-s||²/N_ADC + lambda*||m||²`, equivalent to
`m=(A.H A + lambda*N_ADC*I)^-1 A.H s`. This is the whiteboard's Tikhonov
formula with a normalized data term. Small general problems use full SVD,
including rectangular or rank-deficient encoding matrices. Singular values
below `rcond=1e-10` times the largest are discarded at zero regularization.
The code never explicitly forms or inverts a large normal matrix.

For the 128×128 EPI, a full complex128 matrix would occupy 4 GiB. Instead,
the code factors the same encoding into one phase matrix `Y` and readout
matrices `X_r`, with `s_r=X_r (Y m)_r`. At full rank, it solves `X_r` then
`Y`, using only small matrices. Positive regularization requires orthogonal
phase encoding; that condition is checked before the exact factorized ridge
solve. Unsupported large geometries fail explicitly. No acquired sample is
rounded to an ideal trajectory.

The local `MATRIX_DFT_ENABLED` switch uses matrix least squares in all V2
image reconstructions. `FORCE_DIRECT_DFT` also makes raw navigator transforms
use matrix products. Input GRE/B0 maps use zero regularization; the configured
EPI regularization is shared by measured baseline, simulation, nuisance
calibration and final EPI comparisons. B0 and ACDC remain in the forward model;
the reconstruction matrix stays fixed during current optimization. The optimizer
uses `R.H`, the exact adjoint of the inverse `R`, so its gradient remains valid.

The first EPI simulation displays matrix dimensions, numerical rank, condition
number, regularization and memory use. It saves `dft_reconstruction.json` and,
when the singular spectrum is available, `dft_singular_values.png`. The notebook
introduction derives the equations and explains their connection to the meeting
photo. The lambda curve in that photo is schematic; no optimal lambda is inferred
from an unknown ground truth. Rerun the cells to generate the English figures.
The actual 128×128 EPI encoding has a condition number close to one. On that
nearly Cartesian grid, its matrix inverse closely matches the Fourier adjoint,
as expected; the reconstruction change alone does not remove B0 distortion.

## MRzero in 09

The standard MRzero API uses a static B0 value per voxel and a common voxel
dephasing model. `native_pdg.py` adapts its executor with two local extensions,
marked `GEN09`: effective B0 at each ADC and a local trajectory per voxel for
the native kernel. The installed package is unchanged.

After the single ideal excitation, the field supplied to the native rotation is
`B0_effective(t)=B0 + integral_RF→t(ACDC_field)/(t−RF)`. The integral retains the
complete command history and hardware response. Local trajectories also include
the integrated ACDC and B0 gradients. MRzero applies phase, relaxation, reception,
and its own `box`/`gauss` kernels. No ACDC phase or dephasing factor is applied to
the output signal. Unresolved residual coherence uses these native kernels inside
the executor and remains independent of currents, as in 08.

This representation is exact within native numerical precision for the inherited
model: stationary spins, one ideal nonselective 90° excitation, no diffusion,
and no refocusing. Other RF histories are rejected. T1, the slice profile, and
the steady state are not measured or simulated; effective amplitude comes from
the data. Gradients are locally linear within each voxel. 09 recomputes and
executes the PDG at each evaluation and during gradient calculation in blocks,
so it takes substantially longer than 08.

The local executor is adapted from **MRzeroCore 1.2.0**, under AGPL v3; its
license is retained in `LICENSE_MRzeroCore.txt`. The code was checked with
MRzeroCore 1.2.0 and PyTorch 2.8.0, pinned in this folder's `requirements.txt`.

## First simulation report

The "First simulation parameters" cell appears before EPI calibration and
optimization. It reports RF, TE/TR, ADC, effective PD, B0, T1/T2/T2dash, B1,
diffusion, voxel dimensions and count, precision, version, PDG settings,
and the gain/NRMSE for comparison with the EPI scan. Differences between
simulation assumptions and acquired values are shown explicitly.

`first_simulation.json` and `first_simulation.npz` preserve this first result.
In 09, statistics from the first native block with B0 enabled are identified
as such; image comparison covers all baseline ADC samples.

## Validation

The associated tests are in `../MRzero_validation/test_generations08_09.py`:
independent 3D quadrature, displaced-spin sums, current gradients, corner
constraints, coverage/calibration, DFT and adjoint, 08/09 parity with the
hardware response and uniform/Gaussian dispersion, native gradients, and the
JSON report. Run them from the project root:

```bash
python -m unittest discover -s MRzero_validation -p 'test_generations08_09.py'
python -m unittest discover -s MRzero_validation -p 'test_matrix_dft_08v2.py'
```

Validation here used the CPU. A complete CUDA optimization and a new scanner
measurement have not been performed.

The first 08 run on acquisition 029, with the configured 60 fitting iterations
and before EPI calibration, simulated 7,785 voxels, each spanning the full 5 mm
slice thickness. B0 ranged from −87.11 to 250.76 Hz; simulated TE was 61.3275 ms,
compared with 61.328 ms in the scan. Magnitude NRMSE after a global comparison
gain was 45.98%. This confirms execution without establishing quantitative
agreement with the scan. The report also shows the simulated ideal 90° excitation
and the acquired 15° flip angle. The effective-amplitude convention from 06
is retained without introducing an inferred T1 map.

The 12 original CPU tests and 14 matrix-reconstruction tests pass. The matrix
tests cover dense least squares, rank-deficient and rectangular matrices,
Tikhonov normal equations, factorized solves, phase sign, frames, exact adjoints,
and FFT-free execution. Actual EPI/GRE matrix roundtrips have relative errors
below `7e-15` in float64; the factorized 128×128 reconstruction caches 64.5 MiB
in complex128 instead of storing a 4 GiB encoding matrix.

All updated 08 V2 cells ran successfully on acquisition 029 with a shortened
fit, optimization disabled, and waveform exports disabled. All 08 cells also
ran successfully with a shortened fit,
optimization disabled, and waveform exports disabled. The first 09 simulation
on the same prepared measured object agrees with 08 to `1.25e-7` relative error
in the complex signal.
