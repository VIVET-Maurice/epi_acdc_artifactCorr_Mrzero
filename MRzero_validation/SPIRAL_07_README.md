# Notebook 07: spiral ACDC optimization

The updated forward model now reconstructs the independently measured complex
receivers before RSS combination and can calibrate two bounded effective
gradient delays. Both changes affect the simulated signal. Measured images
keep the original nominal NUFFT trajectory and density weights.

`SPIRAL_RECEIVE_MODEL="auto"` selects complex receive-weighted initial
magnetization from the independent multi-echo fit. `RECEIVER_CHUNK=4` bounds
receiver batches. These receiver channels are separate from the 32 ACDC output
channels. `SPIRAL_GRADIENT_DELAY_CALIBRATION=True` enables the optional delay
fit, bounded by `SPIRAL_MAX_GRADIENT_DELAY_US=20.0` in each native gradient axis.
Positive delay means the actual gradient lags the nominal waveform. Exact
Pulseq integration includes the gradient history before the imaging ADC.

Up to three bounded delay proposals use training pixels and a training-only scalar gain.
Only an exact training improvement is retained; reserved pixels never choose proposal steps.
The full nonlinear signal is then recomputed; application requires improved
reserved validation and test errors, an identifiable Jacobian, no bound hit,
and agreement with rebuilt native MRzero. ADC times, turn triggers, ACDC
response integrals and hardware constraints stay fixed. The fitted delays are
effective parameters inferred from this scan, not an independent measurement
of the scanner gradient response. A rejected proposal keeps nominal encoding.

The later global-frequency/rigid-translation calibration checks physical
coverage and fixed receive encoding before selecting among its nested
candidates. A translation that changes object membership is rejected while
a valid simpler frequency model can still be retained. Unchanged parameters
reuse the existing native cache.

The acquisition-032 diagnostic gives the following NRMSE values. Every model
has one intensity gain fitted on training pixels and frozen for the other
subsets; the complex receive maps come from the independent MEGRE measurement.

| Forward model | Full field of view | Reserved test pixels |
| --- | ---: | ---: |
| Previous RSS object | 29.20% | 34.47% |
| Complex receivers | 19.51% | 25.64% |
| Complex receivers and effective gradient delays | 15.17% | 17.74% |

The fitted native-axis delays are approximately +9.267 microseconds in X and
-0.247 microseconds in Y. These are descriptive internal checks of this scan,
not validation on a separate acquisition. The full-object benchmark uses a
stationary-spin signal oracle; separate native-MRzero tests verify receiver
sums, current gradients, delayed-trajectory rebuilds and phase parity. The
remaining error includes model uncertainty and measurement noise; exact scan
reproduction has not been established. Reproducible evidence and English
figures are in [receiver_model_update](results_experiment_spiral07/validation/receiver_model_update/README.md).

Run `07_acdc_spiral_echo32_b0_s0_t2star_MRzero.ipynb` from the project root or
`MRzero_validation` directory. Restart the kernel before the first run so the
updated shared model and reconstruction modules are loaded. The notebook code,
figures and optimization-image viewer use English.

If the initial CUDA test fails with an out-of-memory error while allocating a
2 × 2 tensor, the failure occurs before this run loads data or builds its model.
Restart your own notebook kernel to release tensors from previous runs, then
check `nvidia-smi` on the same host and GPU allocation used by that kernel if the
error persists. Another kernel or process may still occupy the GPU. The
`GPU_MEMORY_FRACTION` and `SAMPLE_CHUNK` settings control the later model cache
and work blocks; lowering them does not address this initial CUDA-test failure.

Notebook 06 keeps its Cartesian workflow. Notebook 07 uses the same physical
object and hardware model, with spiral acquisition and reconstruction adapters.
Each optimization state is one complete vector of 32 commanded ACDC currents.
The state changes at an actual Pulseq output pulse marking a spiral turn.

## Supplied acquisition 032

| Item | Verified value |
| --- | --- |
| Imaging sequence | `gre_spiral_128x128_fov200mm_nav3_cimax_turn_triggers_v1.seq` |
| Imaging measurement | `meas_MID00193_FID50216_spiral_scan.dat` |
| Reference GRE | `meas_MID00196_FID50217_reference.dat` |
| B0 / temporal-model measurement | `meas_MID00197_FID50218_b0_map.dat` |
| Reconstructed matrix / FOV | 128 × 128 / 200 × 200 mm |
| Spiral imaging ADC | 29,084 samples at 5 µs dwell |
| Imaging transport packets | 11 packets of 2,644 samples, `cSet=0..10` |
| Leading navigators | Three Cartesian records, 128 samples each at 6.5 µs dwell |
| ACDC imaging states | 65 turns, each delimited by an actual `ext1` pulse |
| Excitation / shots / interleaves | One / one / one |
| First imaging ADC after RF centre | 8.4425 ms |
| Last imaging ADC after RF centre | 153.8575 ms |
| TR | 155.44 ms |
| Mapping echoes | 32, from approximately 2.1231 to 58.5431 ms |

The 11 Twix packets are parts of one continuous imaging ADC. They are not
spiral turns, Cartesian lines or independent repetitions. The loader
concatenates them in file order, excluding the navigators and `SYNCDATA`
packets. It verifies sample counts, packet counters, FOV, matrix, TE/TR and
slice geometry. It does not crop, pad, sort or reflect the imaging samples.

The scanner's general dwell header describes the first navigator ADC here.
The spiral imaging dwell comes from the selected Pulseq ADC event. Likewise,
the scanner protocol's advisory flip-angle header differs from the Pulseq
excitation definition; both values are retained in the input metadata.

The first turn begins 8.43 ms after the RF centre; the next begins at
11.16 ms. Later intervals are not all equal. State assignment uses the actual
pulse times and ADC sample centres, rather than dividing the samples into 65
equal groups. The trajectory figure shows these variable sample counts and
command-window durations explicitly.

## Reconstruction and signal model

The acquisition stores chronological sample arrays with shape `(1, 29084)`;
the separate `image_shape` is `(128, 128)`. The historical model key
`line_index` now identifies a one-based spiral-turn state. It does not imply a
Cartesian line. Complex spin contributions use the actual nominal Pulseq k
coordinates and sample times.

The defaults are:

```python
SPIRAL_RECONSTRUCTION_METHOD = "nufft"
SPIRAL_DENSITY_COMPENSATION = "pipe_menon"
```

`torchkbnufft` provides the NUFFT reconstruction and its matched Hermitian
adjoint. Pipe-Menon density weights are computed once from the trajectory and
output grid, normalized to mean one, and held fixed. Measured data, simulated
data, baseline calibration and optimization use those same weights. The
operator is a normalized, density-weighted Fourier adjoint; it is not an
iterative inverse reconstruction. `"direct"` selects exact Fourier sums for
small validation cases, with a substantial runtime cost on this acquisition.
Density compensation can also be explicitly set to `"radial"` or `"none"`.

The three navigators are retained for inspection. Their Cartesian odd/even
phase correction is not applied to the spiral imaging samples.

### Complex receive encoding

```python
SPIRAL_RECEIVE_MODEL = "auto"
RECEIVER_CHUNK = 4
```

The default uses complex receiver amplitudes from the independently acquired
32-echo MEGRE fit. Each supported voxel's receiver vector is normalized to
unit RSS norm, preserving its relative amplitudes and spatial phases. Its
reference-echo phase is restored to excitation time so MRzero applies the
measured B0 phase evolution exactly once. Object amplitude, support,
relaxation and quadrature weights stay fixed.

`SpiralReceiverACDCModel` sums spins separately for every receiver, reconstructs
the complex receiver images, and then forms RSS. A single RSS object followed
by one reconstruction cannot in general reproduce that operation, especially
when off-resonance redistributes signal. `"complex_echo"` requires complex
MEGRE data; `"rss_proxy"` selects the earlier single-receiver approximation.
The comparison figure reuses the same native spin histories for both models.
`RECEIVER_CHUNK` limits receiver batches without multiplying the native cache.

The 32 receive channels measure signal; the 32 ACDC outputs drive shim currents.
These are distinct devices and channel lists. Disabling ACDC output 9 leaves
all receive channels available. Twix `ChannelId` headers verify receiver order
throughout B0 echoes, spiral packets and navigators. Acquisition 032 uses
nonconsecutive receiver IDs `[0..13, 15..31, 33]`; receiver array indices are
not assumed to equal physical IDs. Weights and their assumptions are saved in
the session NPZ/JSON archives and remain fixed during nuisance calibration.

The coefficients contain effective receive sensitivity and initial transverse
phase. They are not separately measured coil-sensitivity maps. Their transfer
assumes stable complex spatial encoding between the MEGRE and spiral scans,
and constant weights within each finite voxel and its thickness children.
It does not establish exact scanner agreement.

The archived acquisition-032 comparison reduces full-FOV NRMSE from 29.20%
to 19.51%, with one scalar gain fitted on training pixels only. Validation
NRMSE changes from 27.38% to 16.60%; test NRMSE from 34.47% to 25.64%.
These guarded subsets belong to one spiral acquisition. The comparison uses
an analytical stationary-spin history equivalent to the zero-current native
model, rather than a full native execution. Reproduction code, images and
limitations are in `results_experiment_spiral07/validation/receiver_model_update`.

Useful features retained from version 06 include:

- Independently validated reference GRE and 32-echo B0 measurements;
  independent complex-noise estimation and object-support diagnostics.
- Complex multi-echo fitting, held-out echo diagnostics and the guarded
  reference-amplitude prior.
- Analytic finite-voxel integration, thickness subspins, and 3D calibration-map
  sampling at the measured slice translation and orientation.
- Separate physical image-to-coil alignment and computational-frame settings.
- Static-shim initialization, measured-GRE image optimization and optional
  physical residual-B0 guidance.
- First-order amplifier response, per-channel and total-current constraints,
  field checks at voxel corners, DAC quantization checks and current previews.
- Disabled physical outputs. The inherited default is `DISABLED_CHANNELS=[9]`;
  `[]` enables all outputs. A disabled output remains exactly zero during the
  solve, and the other outputs are reoptimized.
- Session archives, source hashes, optimization-image history and comparison
  of explicitly selected corrected scans using one baseline-to-GRE gain.
- Independent static and dynamic field-shim exports.

The spiral baseline calibration uses a bounded global frequency offset and
two in-plane object translations. There is no EPI odd/even-phase parameter.
Training, validation and test image blocks remain separate, with guard bands,
identifiability and boundary checks. Accepted translations rebuild the object
and resample physical coil maps; native MRzero sample-phase parity is checked.

## Timing and exports

The image-optimized waveform contains all 65 turn commands, including any
initial all-zero states, followed by a terminal zero row. Its timestamps come
from the verified turn windows and, by default, start at the first imaging
trigger. The terminal zero timestamp is the end of the imaging window; it is
not an additional measured turn trigger.

`waveform_adc.csv` describes current-monitor sampling times relative to the
latest verified trigger. Its sidecar records how local times map to the
sequence clock. Static and dynamic field shims retain their own explicit time
grids; their rows should not be interpreted as one command per spiral turn.

A `_zero_time.csv` companion retains current rows but removes timing
information. Its interpretation depends on the external controller's playback
mode. Verifying the Pulseq trigger train does not establish how the controller
advances rows, handles repeated triggers or applies the terminal zero. The
notebook writes local files and does not configure or program that controller.

## Supported scope and validation limits

The current adapter supports a two-dimensional, single-shot, single-interleaf
spiral-out readout following one excitation, with zero or three leading
navigators and one actual output pulse per turn. Other trajectories require
their own verified sample/state and raw-packet contracts. Both the GRE
reference and the B0 mapping sequence remain Cartesian.

The trajectory is calculated from nominal Pulseq gradients. No measured
gradient-delay, eddy-current or gradient-nonlinearity correction is inferred.
Good numerical agreement cannot independently establish coil registration or
the physical cause of signal loss. The inherited 2D B0 map does not resolve
the true B0 variation through the slice thickness.

This spiral readout extends to approximately 154 ms, beyond the final mapping
echo at approximately 59 ms. The notebook reports that extrapolation; late
spiral signal predictions remain dependent on the fitted temporal model.

Validation includes the actual acquisition-032 GRE/spiral/B0 read and geometry
checks, exact Fourier/NUFFT comparisons, matched-adjoint and current-gradient
tests, packet-ordering and navigator-separation tests, and the actual 65-trigger
CSV timing. Reloading the baseline as its own comparison scan gives identical
reconstructed arrays. The actual notebook input and object-preparation cells
also pass on the full acquisition, including the 60-iteration complex fit,
reference-prior gate and 3D coil-map coverage checks. Evidence and English
diagnostic figures are in
`results_experiment_spiral07/validation/full_input_preparation`.

The reproducible `results_experiment_spiral07/validation/reduced_native_smoke.py`
check uses measured object pixels and actual spiral samples, retains all 65
original turn windows and the 128 × 128 output, and tests native MRzero,
optimization, gradients, disabled output 9, DAC checks and the 66-row CSV.
Its reduced object and sample domains validate integration, rather than
full-phantom image quality. CPU checks do not substitute for a full
optimization on the intended GPU or a new physical ACDC acquisition.
