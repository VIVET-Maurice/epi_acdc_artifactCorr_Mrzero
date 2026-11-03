# V4: choose a plane, optimize the acquired 3D EPI

Open `06_acdc_echo32_b0_s0_t2star_MRzeroV4 (3D).ipynb`. V3 is unchanged.

1. Run through **Select the plane before any simulation**.
2. Choose patient X, Y or Z and move the slice slider. All three panels show
   the same physical plane: GRE reference, acquired EPI and measured B0.
3. Click **Use this slice**, then run the remaining cells. Changing the plane
   invalidates the old selection; rebuild the scene before optimizing again.
4. Inspect the baseline simulation and coverage before interpreting the
   optimized image. Results go into a new `results_volume_v4/v4_*` folder.

Section 6 retains V3's live optimization figure: reference, baseline, current
image, signed magnitude residual, image-MSE curve and channel-current heatmap.
The curve includes every evaluated iteration. Images use the confirmed patient
plane, a fixed intensity scale and the same orientation as the volume viewer.
`LIVE_UPDATE_EVERY=20` controls the display and periodic image archive interval;
the last evaluated state also updates the live figure when optimization stops.
The retained best combined-objective candidate is shown separately in section 7.

The same output folder contains `optimization_progress.png`,
`optimization_images.npz` and `optimization_images.html`. The HTML provides
V3's reference/baseline/iteration slider and works offline. Archives contain
only 2D magnitude planes at iteration zero and exact multiples of the configured
interval. Full 3D images and currents are not stored for every iteration.
Display images reuse the forward calculation already needed by the objective.

`ipywidgets` is included in `requirements-experiment.txt` and has been installed
in the experimental pipeline environment used for validation. Restart an
already running notebook kernel after installing dependencies or updating modules.
Volume reconstruction and browsing work without CUDA. The full 128³ optimization
defaults to requiring CUDA; a complete full-resolution run was not performed in
the CPU-only validation environment.

## What is optimized

The full 3D object, measured 3D B0 and 32 calibration responses remain on the
native EPI readout/phase/partition grid. The selected patient plane supplies a
fixed image-loss mask. It does not create a fictitious slice-selective 2D scan.

The same 128-command chronological line waveform is repeated for each of the
128 acquired kz partitions. The exact matching Pulseq sequence is included in
`b0map/seq/`; its signature is checked against the Twix. Measured ADC
timing, readout polarity, phase encoding and all partition encodings are retained.

The dedicated differentiable PyTorch forward model sums the full complex 3D
point-spin signal. It uses explicit phase encoding, FFT readout/partition
encoding and temporal interpolation within each actual readout. Interpolation
is checked against increasing node counts, including the final candidate and
its nominal DAC version. Exact ADC-time evaluation is available for small tests.
This is a new 3D Bloch-phase/Fourier implementation, not the 2D MRzero graph.

V3's first-order current response, disabled-channel controls, static
initialization, image/field objective and hardware checks are reused. Channel 9
is disabled by default. Current limits, timing, field bounds and objective
improvement are checked before export. The fixed calibration anchor is copied
from V3; selecting another plane does not move the coil frame.

## Input interpretation and current data limitations

- No B0 Pulseq file is required. This is a native Siemens B0 acquisition;
  the reader uses its Twix echo times, encoding sizes, oversampling and geometry.
  The supplied EPI and GRE files in `b0map/seq/` both match the sequence
  signatures recorded in their respective Twix scans.
- GRE and EPI reconstruct as 128³ volumes. The Siemens B0 scan reconstructs as
  112³ after removing oversampling in **all three axes**, not only readout.
- Every selected receiver is reconstructed independently before RSS/echo-pair
  combination, limiting memory use. Navigator correction is separate for each
  EPI partition. Cache keys include source hashes and reconstruction options.
- B0 was confirmed by the user to have all ACDC currents at zero. Its filename
  `CAL_p1_ch0` is not used as evidence of an applied current.
- The B0 and EPI reference frequencies differ by 3 Hz; the map is converted
  to the EPI reference. A two-echo field map still has an unknown absolute
  integer frequency branch. `B0_GLOBAL_OFFSET_HZ` is an explicit known offset,
  not an automatically fitted answer.
- The two-echo B0 data cannot support V3's 32-echo relaxation/dispersion fit.
  V4 uses GRE receive-weighted amplitude and no inferred decay by default;
  a constant `T2STAR_MS` can be supplied. Receive-coil phase, RF/T1 steady state
  and unresolved intravoxel dispersion are not inferred.
- Missing B0 or calibration is never extrapolated. Unsupported object signal
  is excluded without redistribution, with separate B0/calibration counts.
  For the supplied data and default alignment, approximately **17.8%** of
  reference object amplitude lacks measured B0, and calibration coverage removes
  **28.0% of the remaining B0-covered amplitude**: **40.8% excluded overall**.
  This is a substantial limitation for a quantitative full-phantom prediction.
  A passing numerical or hardware check does not resolve missing physical data.

## Coordinate conventions

Raw MDH quaternion columns describe Siemens phase/readout/slice directions.
Image arrays are reconstructed in readout/phase/partition order. Siemens native
Cartesian voxel-index directions also require the pixel signs implemented in
`twixtools.geometry`: `prs_to_pcs() @ rps_to_prs() = -Q[:, [1,0,2]]`.
V4 explicitly applies this conversion to the native Siemens B0 reconstruction;
it does not flip the measured values or fit anatomy.

The Pulseq scans retain their declared native gradient mapping. The actual
console **OrientationMapping** may differ from MDH orientation, as described in
the [Pulseq interpreter discussion](https://github.com/pulseq/pulseq/discussions/285).
An explicit `PATIENT_FROM_NATIVE_M` override is available for a separately
verified mapping; a supplied full affine is not modified by the default B0
pixel-sign conversion. The configuration and all effective affines are saved.

For these data, correcting the native B0 voxel convention improves reference/B0
magnitude correlation from about 0.054 to 0.896 in the diagnostic sample. This
supports the relative registration, but does not independently establish the
physical calibration axes or the Pulseq console setting. No free rotation,
translation or sign-search result is silently fitted into the production pipeline.

## Full-acquisition export

V4 exports a **timed** CSV in `b0map/waveform_v4/`, with a JSON sidecar linking
the full-volume waveform to its scene, selection, sources, geometry and checks.
It repeats the line commands across all partitions and inserts an explicit
zero-current event after every EPI train, before the next RF/navigators.

A zero-time copy is not equivalent: those inter-partition switch-off events
have no imaging trigger, so a one-row-per-trigger player would shift later
commands. The export therefore refuses zero-time playback. The controller must
support the timed schedule from the first imaging trigger. No hardware is
configured, messaged or started by the notebook.

## Code layout and validation

| Module | Responsibility |
| --- | --- |
| `volume_twix.py` | Memory-bounded 3D GRE/B0 reconstruction and metadata/cache |
| `volume_epi.py` | Per-partition navigators and exact 3D sequence timing |
| `volume_slice_viewer.py` | Linked X/Y/Z viewer and explicit selection |
| `volume_inputs.py` | Physical registration, coil sampling and full-volume ROI |
| `volume_epi_model.py` | Differentiable full 3D signal and accuracy checks |
| `volume_optimization.py` | Constrained optimizer and timed full-volume export |

The `test_volume_*` suite covers synthetic raw 3D reconstruction, oversampling,
physical coordinates, slider selection, direct complex 3D Fourier agreement,
current gradients, temporal convergence, disabled outputs, hardware constraints,
inter-partition off events, and an executable notebook simulation-to-export test.
The actual raw data, widget controls, full-volume scene and matching sequence
were also checked. A measured preview and validation reports are in
`results_volume_v4/validation/`. Full 128³ optimization and scanner validation
remain to be performed on the intended GPU/hardware setup.
