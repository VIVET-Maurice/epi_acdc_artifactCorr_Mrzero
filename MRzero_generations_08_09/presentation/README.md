# ACDC / MRzero weekly presentation: reproduction and changes

This folder supports the selected **15-slide presentation: 11 main slides and
4 backups**, in the user's retained order. The working source is
`../2026_11_11_ACDC_MRzero_Generations08_09_WeeklyResearchUpdate.pptx`, SHA-256
`46bcea5282a73fcf3d1158b49ddb4ef8062db5619225ba4fbe5bd425f2d1c3af`.
The earlier 19-slide selection is not used. The source deck and the original
08/08V2/09 physics notebooks are not modified by the presentation helpers.

The output is `../2026_11_11_ACDC_MRzero_WeeklyUpdate_REVISED.pptx`.
`presentation_build_manifest.json` records its source/output hashes, slide count,
used equations/figures, and whether a build is a layout draft. A draft build is
not a completed scientific deliverable.

## What the figures establish

- Measured EPI OFF fidelity, simulated current optimization against measured
  GRE, numerical agreement between 08 and 09, and measured ACDC ON/OFF validation
  are separate questions. Their targets, gains, masks and units are recorded
  separately; an implementation-parity result does not validate the scanner model.
- The reconstruction control uses **one identical complex Gen08 k-space array**
  for automatic actual-coordinate reconstruction, an explicit Fourier adjoint,
  the current 08 V2 matrix inverse, and a forced nominal-grid IFFT diagnostic.
  Ordinary IFFT and direct inverse Fourier summation must agree on an exactly
  Cartesian grid; a separate synthetic control checks this.
- **Current 08 V2 is an SVD-based matrix least-squares inverse**, with a matched
  Hermitian adjoint for optimization. It is not the same operator as the
  separately evaluated normalized direct Fourier-adjoint sum. Both are labeled
  explicitly. The current EPI source is read as it exists today, rather than
  assuming that the 9 November source archive is unchanged.
- The coordinates and ADC center timestamps are **calculated from the supplied
  Pulseq sequence and validated against Twix acquisition metadata**. They are
  not a measured field-probe trajectory. The tests address numerical/grid
  handling for that supplied trajectory; they do not exclude physical scanner
  trajectory errors, gradient delays, or an unknown receive filter.
- The uniform-phantom control deliberately changes object amplitude to a constant,
  removes fitted decay/unresolved dispersion, and uses zero ACDC current. These
  are labeled diagnostic interventions, not changes to the production object.
  Its finite support and voxel/k-space windows can generate spatial variation
  even with B0=0. CV and neighboring-pixel differences are descriptive metrics,
  not a noise model or proof of a particular mechanism.
- The finite-dwell calculation is a separate **hypothetical rectangular ADC
  aperture** experiment, with numerical quadrature/convergence controls.
  Production still samples at ADC centers; the scanner's actual receive-filter
  impulse response has not been identified.

## Receiver audit: implemented versus missing

The current EPI 08/08V2/09 forward models use **one uniform complex receiver**.
They do not emit separate receive-coil ADC signals or reconstruct simulated
receiver RSS. Measured EPI, GRE and MEGRE data do contain 32 receive channels.
Their physical IDs are identical and ordered as
`[0–13, 15–31, 33]`, verified across 131 EPI blocks, 128 GRE blocks and
4,096 MEGRE blocks. These IDs are distinct from the 32 ACDC shim outputs
`1–32`; disabled ACDC output 9 does not mean receive channel 9 is disabled.

The complex multi-echo fit estimates a complex amplitude per receiver, then
passes its scalar RSS envelope to the EPI object. On 8,068 finite positive
fitted voxels, RSS of those complex amplitudes agrees with the scalar envelope
to relative error `1.2707e-16`. Unit-RSS normalization of their directions has
maximum norm error `1.1102e-15`; these directions are **not used by the EPI
forward model**. Multiplying the existing RSS-weighted object by unnormalized
receiver amplitudes would double-count the receive envelope. A future transfer
must justify its normalization, phase conventions and cross-acquisition validity.
Measured receiver images contain tissue contrast and initial phase as well as
reception; they are not isolated quantitative sensitivity maps.

The separate spiral pipeline does implement receiver weights and per-channel
forward/RSS reconstruction in `../../MRzero_validation/spiral_receive.py` and
`experiment_spiral.py`. That preparation rejects EPI inputs and is not imported
by the EPI notebooks. No new receive-sensitive EPI model was added for this
presentation, and no before/after EPI receiver-model improvement is claimed.
The native transmit setting `B1=1` is not a measured receive sensitivity.

Evidence, source line numbers and hashes are in `receiver_audit.json` and
`legacy_and_receiver_audit.json`; the figure shows measured first-echo MEGRE
channels on one common magnitude scale.

## Legacy observation and current results

The exact saved run referenced by the textual outputs of
`../../MRzero_validation/06_acdc_echo32_b0_s0_t2star_MRzeroV3.ipynb` is
`../../MRzero_validation/results_experiment_echo32/mrzero06_20261103_121718_079562/`.
Slide 2 is regenerated from its numerical `mrzero06_optimization.npz` arrays,
not notebook/deck screenshots. All six recorded scan/sequence input hashes and
the ACDC calibration hash match the files currently present. The historical
simulation helper hashes were not saved, so this is an archived simulation with
fresh figure/metric computation, not a claimed fresh execution of its old code.

Its recomputed full-FOV OFF NRMSE is **45.975603%**, after one least-squares
baseline-to-measured gain of **5.16156907**. That same gain is applied to the
B0=0 diagnostic. Pattern metrics use a fixed retained-parent support after two
3×3 binary erosions: 4,941 pixels, 9,252 neighboring pairs.

| Archived comparison | CV | Adjacent RMS difference / mean |
| --- | ---: | ---: |
| Legacy B0=0 diagnostic | 0.761842 | 0.345718 |
| Legacy measured-B0 baseline | 0.680401 | 0.370522 |
| Measured EPI OFF RSS | 0.589513 | 0.395407 |

These statistics include object contrast. The B0=0 diagnostic also retains fitted
residual relaxation and unresolved coherence. Neither these values nor the panels
establish that all structured variation disappears, or identify the cause of the
PI's apparent-speckle observation. Scanner trajectory/grid handling, finite ADC
aperture, receive/RSS contrast and B0 modeling remain distinct hypotheses.

**Fresh controlled results**, from the current 7,785-voxel / 16,384-ADC object,
are recorded in `presentation_summary.json`, `results_metrics.csv` and
`data/*.json`. The full fit uses all configured 60 iterations. The current
uncalibrated Gen08 full-FOV OFF NRMSE is **45.975605%**; its rounded `45.98%`
therefore remains close to the archived result, but is now reproduced. Its
fixed reporting gain is **5.161569272**, fitted on all 16,384 finite output
pixels. The acquisition is 128×128 over 200×200 mm, with 5 mm slice thickness,
5 μs dwell and TR 119.35 ms. Calculated TE is 61.3275 ms versus 61.328 ms in
the scan; ADC centers span 4.6925–117.0875 ms after RF. The ideal simulated
90° RF history and acquired 15° flip angle remain distinct effective-amplitude
assumptions, not an inferred T1/steady-state model. At zero
ACDC, the assumed uniform B0/amplitude/relaxation through thickness and kz=0
make eight weighted legacy z children equivalent to one 08 center with qz=0.
This OFF comparison alone does not test through-slice ACDC compensation.

| Controlled experiment | Fresh result | Scope |
| --- | ---: | --- |
| 08 versus 09 complex signal, zero currents | `1.270856e-7` relative L2 | All ADC samples, identical object and reconstruction |
| 08 versus 09 complex signal, deterministic nonzero currents | `6.358020e-6` relative L2 | Diagnostic currents, not an optimization |
| Automatic versus actual-coordinate direct adjoint | `4.101085e-15` complex relative L2 | Exactly the same Gen08 complex k-space |
| Current 08 V2 matrix inverse versus automatic reconstruction | `1.837110e-12` complex relative L2 | Same complex k-space, grid and scale; inverse is a distinct operator |
| Exactly Cartesian FFT versus direct adjoint sanity check | `1.360613e-15` complex relative L2 | Asymmetric 16×16 synthetic complex image; fixed seed 20261111 |
| Forced nominal-grid IFFT versus actual-coordinate direct adjoint | `2.477960e-4` complex relative L2; `1.009062e-9%` magnitude NRMSE | Phase-sensitive difference; negligible magnitude change on this supplied trajectory |
| Uniform phantom, B0=0 interior CV | `0.112859%` | 4,783 pixels after three default-connectivity erosions |
| Same uniform phantom, measured-B0 interior CV | `36.430482%` | Same fixed mask and B0=0-derived normalization |
| Hypothetical 5 μs rectangular aperture | `9.327796%` magnitude-image change from center sampling | Zero ACDC; five-node quadrature; not an identified scanner filter |
| Three- versus five-node aperture convergence | `1.705295e-5` complex relative L2 | Center analytic oracle versus native MRzero error: `7.713830e-6` |

The computational automatic/direct explanation is not supported for these
arrays: their reconstructed images agree numerically. Physical scanner
trajectory errors remain untested. The controlled B0 and aperture results
show sensitivity within their stated models, not the cause of the measured
intensity pattern. Receiver-sensitive EPI remains unimplemented and untested.
These masks differ from the archived 4,941-pixel pattern mask; the CV values
are not interchangeable with anatomical-image CV or OFF-model NRMSE.

An archived command replay, when present, evaluates saved commands on the fresh
object. It is **not a new optimization**, and no measured ACDC-ON scan has been
linked to that waveform. Predicted image changes are not physical ACDC success.

## Reproduce the figures

Open `00_generate_weekly_figures.ipynb` with this Python environment:

```text
/space/guerin/STUDENTS/MAURICE/main/experimental_pipeline/.venv/bin/python
```

The notebook exposes `RUN_EXPENSIVE` and its derived `RUN_FAST` mode indicator.
The fast path regenerates
figures/metrics from validated local prepared/numerical caches; it does not
pretend to rerun expensive simulation or timing. The expensive path regenerates
the numerical controls, setup/forward/reconstruction/gradient benchmarks, and
ADC-window diagnostics. Rebuilding the input cache performs the original
60-iteration complex echo fit unless an explicit override is recorded. Keep the
production input paths, fit settings, geometry and object masks unchanged for
matched comparisons. Modified inputs/code invalidate their dependent caches.
Set `REFRESH_INPUTS=True` with `RUN_EXPENSIVE=True` for a complete fresh run;
otherwise a newly prepared cache intentionally invalidates older controls.

The following commands run from the project directory:

```bash
cd /autofs/space/guerin/STUDENTS/MAURICE/to_be_sorted/epi_acdc_artifactCorr_MrZero_pc/epi_acdc_artifactCorr_Mrzero
TASK_PY=/space/guerin/STUDENTS/MAURICE/main/experimental_pipeline/.venv/bin/python
MPLCONFIGDIR=/tmp/acdc-presentation-mpl "$TASK_PY" MRzero_generations_08_09/presentation/prepare_data.py
MPLCONFIGDIR=/tmp/acdc-presentation-mpl "$TASK_PY" MRzero_generations_08_09/presentation/controlled_experiments.py MRzero_generations_08_09/presentation/cache/prepared_context.pkl MRzero_generations_08_09/presentation/data --threads 2 --warmup 1 --repeats 3 --setup-repeats 3
MPLCONFIGDIR=/tmp/acdc-presentation-mpl "$TASK_PY" MRzero_generations_08_09/presentation/generate_figures.py
```

`prepare_data.py` explicitly refreshes input preparation when run as a script.
Omit that first command when reusing the validated existing prepared context.
`generate_figures.py --run-expensive` also requests numerical regeneration;
the explicit commands above expose timing repetitions. The current combined runner
also exports local q maps and times the actual-coordinate adjoint / forced-grid
IFFT. Its optional `--supplement-only` command regenerates just those additions
for an already completed numerical run; it is not needed after a full run.
The full numerical run can take substantially longer for 09 gradients than
for cached 08 evaluation. Do not run competing heavy jobs during timing.

Recorded benchmark setup: Intel Xeon 6760P CPU, Python 3.11.13, PyTorch
2.8.0+cu128, two PyTorch CPU threads; model/reconstruction float64 and native
MRzero float32. OMP/MKL/OpenBLAS thread environment variables were unset.
Setup uses three fresh constructions; repeated operations use one warm-up and
three recorded evaluations, reporting raw times, median and IQR. Raw Twix I/O,
echo fitting, imports, artifact writing, projection and field guidance are
excluded. Construction, first B0-enabled forward, complete baseline availability,
warm forward+reconstruction, reconstruction alone and loss+gradient are separated.
No CUDA timing or GPU peak-memory measurement is claimed by this CPU run.

| Generation | Fresh constructor [s] | Complete cold baseline [s] | Warm forward + reconstruction [s] | Warm loss + gradient [s] | Production reconstruction only [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 08 | 16.2882 | 19.6507 | 3.30509 | 11.8280 | 0.00106706 |
| 08 V2 matrix inverse | 15.4888 | 18.5466 | 3.31807 | 11.9588 | 0.00152142 |
| 09 | 9.43175 | 18.8221 | 9.30804 | 25.1511 | 0.00109416 |

These are measured medians, not exclusive-host guarantees. IQR and all raw
repetitions are in `data/runtime_benchmarks.json` and `.csv`. A separate
same-k-space reconstruction timing gives **3.14661 s** for actual-coordinate
direct adjoint and **0.000126653 s** for forced nominal-grid IFFT. Those are
reconstruction-only comparators, not production 08 V2 runtime or a claim that
an IFFT evaluates nonuniform samples exactly.

## Build and visually verify the PowerPoint

The scientific notebook calls `export_slide_data.py` to write `presentation_slide_data.json`,
which supplies the generated values and figure aliases to the deck builder.
After numerical figure generation and slide-data export:

```bash
"$TASK_PY" MRzero_generations_08_09/presentation/export_slide_data.py
PYTHONPATH=/tmp/presentation_deps /usr/bin/python MRzero_generations_08_09/presentation/build_presentation.py
PYTHONPATH=/tmp/presentation_deps /usr/bin/python MRzero_generations_08_09/presentation/render_presentation.py
```

The builder requires `python-pptx`, Pillow and lxml; this machine's local
support installation is `/tmp/presentation_deps`. Rendering/equations require
`pdflatex`, `pdftocairo`, `pdftoppm`, `pdfinfo`, LibreOffice and `xvfb-run`.
That temporary dependency directory is machine-specific and is not a portable
project dependency. Install the same packages in a retained environment on a
different machine. Headless slide rendering uses the local Xvfb virtual display.

Text and conceptual diagrams use editable native PowerPoint objects. Equations
are genuine typeset LaTeX embedded as vector SVG, with compatibility previews;
they are not native OMML equation objects. Literal LaTeX, `.tex`, SVG/PDF files,
hashes and slide assignments are retained in `equations/`,
`equations_manifest.json` and speaker notes. Numerical figures are replaceable
embedded images, generated at 300 dpi with accompanying SVG/PDF exports.

Inspect every `rendered/slide-*.png`, the rendered PDF and `contact_sheet.png`.
`rendered/render_audit.json` checks slide count, missing notes/placeholders and
out-of-bounds objects, but does not replace visual/scientific inspection.
The final visual inspection status is recorded in `rendered/visual_audit.md`.
The figure-generation notebook has been executed top to bottom in the scientific
environment: all 12 code cells completed in 55.2 s using the validated numerical
caches. That cache-based execution time is not a simulation benchmark.

## Changes to the retained slides

The machine-readable record is `slide_changes.json`. All slides preserve the
selected scientific topics and general order; no removed conceptual section is
restored. The main discussion remains approximately 30 minutes.

| Slide | Change and scientific qualification |
| --- | --- |
| 1 | Preserve the central question and pipeline; separate experimental OFF-model fidelity from simulated current optimization. |
| 2 | Replace slide screenshots with archived 06 V3 numerical-array panels; label the PI observation, competing hypotheses and editable diagnostic decision diagram without assigning a cause. |
| 3 | Add fresh measured-OFF versus simulated-OFF panels, signed residual, fixed-mask gain/NRMSE definitions and proper LaTeX. |
| 4 | Preserve GRE/EPI/MEGRE/ACDC input roles; improve editable physical alignment/geometry; distinguish nonconsecutive MRI receiver IDs from ACDC shim outputs and show measured receiver images. |
| 5 | Provide the large editable voxel → one ADC → all samples → EPI flow; link RF-relative time, integrated actual currents, local q, three-sinc factor, voxel sum and reconstruction equations. Every voxel contributes at every ADC. |
| 6 | Correct what 08 caches; show effective PD and once-only unresolved coherence. Fitted receiver amplitudes are not silently described as EPI receive-sensitive simulation. |
| 7 | Derive the 3D normalized-coordinate box integral with genuine LaTeX, an editable voxel, q units and the measured/unmeasured B0-versus-calibrated-ACDC slope distinction. |
| 8 | Insert matched same-k-space reconstructions and a Cartesian sanity control; distinguish the direct Fourier adjoint, existing automatic method and current V2 matrix inverse. Conclusions apply to the supplied Pulseq trajectory. |
| 9 | Preserve the 08/09 implementation comparison; show modified local `native_pdg.execute_graph`, effective event B0/local voxel trajectories and autograd. Equivalent outputs do not establish new physical accuracy. |
| 10 | Replace figure slots/stale numbers with matched GRE, measured EPI OFF, fresh simulated OFF and residual images, with common axes/scales and stated mask/gain/stage/timing. |
| 11 | Show freshly calculated complex per-ADC/image differences and measured setup-versus-repeated-operation timings for 08/08V2/09; keep computational parity separate from scanner agreement. |
| 12, backup | Retain ADC dwell integration; typeset center/window formulas and show the explicit rectangular-aperture diagnostic and limitations. |
| 13, backup | Retain coordinate/convention checks; add Pulseq-calculated k/timestamp plots and genuine Fourier-sign equations, preserving data/time reversal and origin/frame conventions. |
| 14, backup | Update assumptions after the receiver audit; distinguish transmit B1+ from reception, normal-B0 assumptions and effective relaxation. Receiver-sensitive EPI/RSS equations are clearly planned, not implemented here. |
| 15, backup | Replace README-only provenance with exact data/code hashes, executed preparation cells, notebook/figure paths and limitations; retain the 15-slide selection and consistent counters. |

## Data and code provenance

The numerical input root is `../../acquisition_029_(06)/`:

- GRE: `meas_MID00162_FID50187_reference.dat`.
- EPI OFF: `meas_MID00163_FID50188_scan.dat`.
- 32-echo MEGRE: `meas_MID00164_FID50189_b0_map.dat`.
- EPI Pulseq: `sequences/gre_epi_128x128_fov200mm_nav3_cimax_snr_t2star40ms_line_triggers_v6.seq`.
- GRE Pulseq: `sequences/gre_128x128_fov200mm_flip15deg_tr15ms.seq`.
- MEGRE Pulseq: `sequences/megre_b0_s0_t2star_128x128_fov200mm_32echo.seq`.
- 3D ACDC calibration: `../../data/ACDC_3T.npz`.

Git HEAD is `3b41c8ef8edb9c2ced983b6b673ba91130de7fb0`; the generation/spiral
sources include untracked working-tree files. **HEAD alone does not identify
the executed code.** Use SHA-256 hashes in `figures_manifest.json`,
`cache/prepared_context_manifest.json`, the per-experiment JSON and source audit.
The source deck hash, executed preparation-cell text, selected configuration,
input hashes, masks, gains, devices/dtypes, timing scope and code hashes are
recorded independently. Original scan data and historical result archives remain
unchanged. Only load the trusted local `cache/prepared_context.pkl` pickle.

`figures/` holds all data-derived figures; `data/` holds their numerical arrays
and per-experiment metadata. `results_metrics.csv` is the numerical table;
`presentation_summary.json` and `figures_manifest.json` connect values, figures,
slide assignments, sources and hashes. Current-run conclusions should be taken
from those artifacts, not copied from an older presentation or README.

Independent recomputation from the exported NPZ arrays and raw timing samples
passed all 56 checks in `scientific_audit.json`. `artifact_verification.json`
records final notebook, deck, notes, embedded-figure hashes and render consistency.
