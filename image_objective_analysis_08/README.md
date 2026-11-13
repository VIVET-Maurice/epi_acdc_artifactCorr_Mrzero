# What the image-based optimisation of generation 08 really minimises

Analysis of `MRzero_generations_08_09/08_acdc_echo32_b0_s0_t2star_MRzero.ipynb`,
using the run saved on 12 November 2026 for acquisition 029
(`MRzero_generations_08_09/results/mrzero08_20261112_135622_906979`).

**Everything here is simulation.** The numbers describe the digital twin of
acquisition 029, not a measured ACDC-on scan. No pipeline file was modified.

## 1. Short answer

1. **Your intuition is a theorem, for the field criterion.** The constant
   least-squares shim minimises the mean squared residual field, and no
   time-varying waveform can beat it on that criterion (section 3.1). The run
   confirms it: 13 Hz RMS for the static shim, 70 Hz for the image-optimised
   waveform.
2. **The image loss is a different functional.** The magnitude image does not
   depend on the residual field at each instant. It depends on the *phase
   history* of every voxel: where the voxel lands depends on the field around
   its local echo, and how bright it is depends on when that local echo occurs
   and on the phase spread accumulated inside the voxel since the RF pulse
   (section 3).
3. **In this run the least-squares static shim is a bad image shim** (loss 0.62,
   worse than no ACDC at 0.45). It is computed on voxel centres of a 2D slice,
   and the coil combination it picks adds 24 Hz RMS across the 5 mm slice. At
   TE = 61 ms that is 1.4 cycles: the slice signal cancels and half the total
   signal is lost. With that single term switched off, the same currents score
   0.31, as good as a perfect shim.
4. **The optimiser then goes further than shimming, because of the target.**
   A perfectly shimmed simulated EPI scores 0.307 against the measured-GRE
   target; the optimiser reaches 0.187. It does so by making the image 36%
   brighter than a perfectly shimmed EPI: it uses the coils as extra
   phase-encoding gradients to move the local echo earlier (median 65 ms to
   53 ms), where T2* decay is smaller. Geometry is not improved: the residual
   shift is 4.3 pixels RMS, against 3.2 without ACDC and 1.1 for the static shim.
5. **With a target the model can reach, the optimiser behaves like a shim
   again** (section 5). Against the simulated zero-B0 EPI, the same optimiser
   leaves the local echo at TE. Constant currents chosen for the image reach
   0.075 (least-squares static shim 0.595, no ACDC 0.329). Letting the currents
   vary in time lowers this to about 0.05. Most of the gain comes from the
   criterion, not from changing the currents during the train: a static field
   shim with one extra quadratic term on the coils' slice slope, solved without
   any image simulation, already reaches 0.125 with 8 A.

## 2. What the code computes

### 2.1 Signal model

One voxel $p$ per pixel (7785 voxels), 128 EPI lines, one 32-channel command
per line, commands `currents[channel, line]`.

$$
s(t)=\sum_p A_p(t)\,D_p(t)\,e^{-2\pi i\,[\,\mathbf k(t)\cdot\mathbf r_p+\Phi_p(t)\,]}
$$

| Symbol | Meaning | Units | Code |
|---|---|---|---|
| $\Phi_p(t)=\Delta B_{0,p}\,t+\sum_c B_{pc}\,J_c(t)$ | phase accumulated at the voxel centre since the RF | cycles | [acdc_gpu.py:493-512](../MRzero_generations_08_09/acdc_gpu.py#L493-L512) |
| $J_c(t)=\int_0^t I_c(\tau)\,d\tau$ | time integral of the real current (first-order amplifier lag, $\tau$ = 29 µs) | A·s | [acdc_gpu.py:167-194](../MRzero_generations_08_09/acdc_gpu.py#L167-L194) |
| $A_p(t)=a_p\,e^{-R_p(t-t_{\rm ref})}\,C_p(t)$ | amplitude fitted on the 32 echoes: relaxation and unresolved coherence loss | a.u. | [experiment_echo.py:93-120](../MRzero_generations_08_09/experiment_echo.py#L93-L120), [voxel_3d.py:223-235](../MRzero_generations_08_09/voxel_3d.py#L223-L235) |
| $D_p(t)=\prod_{a\in\{\rm PE,RO,z\}}\operatorname{sinc}\,q_{pa}(t)$ | signal of a uniform box voxel with a linear phase across it | – | [voxel_3d.py:246-264](../MRzero_generations_08_09/voxel_3d.py#L246-L264) |
| $q_{pa}(t)=\mathbf e_a\cdot\mathbf k(t)+g^0_{pa}\,t+\sum_c G_{pac}\,J_c(t)$ | phase difference between the two faces of the voxel along axis $a$ | cycles | same |

$\Delta B_0$ is in Hz, $B$ in Hz/A, $g^0$ in Hz per voxel width, $G$ in Hz/A
per voxel width, $\operatorname{sinc}(q)=\sin(\pi q)/(\pi q)$. The measured B0
map is 2D, so $g^0_{pz}=0$: the model has no through-slice B0 slope, only the
through-slice slope of the coils.

The image is the normalised Fourier adjoint,
$\hat m(\mathbf r_j)=\frac1{N_s}\sum_i s(t_i)\,e^{+2\pi i\,\mathbf k_i\cdot\mathbf r_j}$.

### 2.2 Loss and optimiser

$$
J_{\rm total}(I)=\underbrace{\frac{\sum_{j\in M}\big(|\hat m_j(I)|-m^{\rm GRE}_j/g\big)^2}{\sum_{j\in M}\big(m^{\rm GRE}_j/g\big)^2}}_{J_{\rm image}}
+\lambda\;\underbrace{\frac1{(N-1)P}\sum_{n,p}\Big(\frac{\Delta B_{0,p}+\sum_c B_{pc}\bar I_{cn}}{f_s}\Big)^2}_{\text{field guidance}}
$$

- $m^{\rm GRE}$ is the **measured GRE** (TE = 4.3 ms), $M$ the whole field of
  view, and $g$ = 2.135 a gain fixed once from the unshimmed images
  ([measured_image_target.py:40-58](../MRzero_generations_08_09/measured_image_target.py#L40-L58),
  [157-184](../MRzero_generations_08_09/measured_image_target.py#L157-L184)).
- $\lambda$ = 0.05, $f_s$ = 100 Hz, $\bar I_{cn}$ the mean real current between
  the centres of lines $n$ and $n+1$
  ([field_guidance.py:133-164](../MRzero_generations_08_09/field_guidance.py#L133-L164)).
- Projected Adam, step 0.01 A, starting from the static field shim; after each
  step the commands are projected onto $|I_{cn}|\le 3$ A (0 A on output 9),
  $\sum_c|I_{cn}|\le 30$ A, and contracted so that the coil field stays below
  1000 Hz at the eight voxel corners
  ([experiment_optimization.py:172-176](../MRzero_generations_08_09/experiment_optimization.py#L172-L176),
  [acdc_gpu.py:197-245](../MRzero_generations_08_09/acdc_gpu.py#L197-L245)).
- The saved run stopped at the iteration limit (1000), not at convergence.

The **static field shim** used as the starting point is
$\arg\min_I\sum_p(\Delta B_{0,p}+\sum_c B_{pc}I_c)^2$ over voxel centres with
the same limits (notebook cell 26).

## 3. Why the least-squares static shim does not minimise the image loss

### 3.1 On the field criterion, static is optimal

Let $F(I)=\frac1P\sum_p(\Delta B_{0,p}+\sum_cB_{pc}I_c)^2$. $F$ is convex, and
the set of allowed commands (box, total-current and field limits) is convex, so
for any waveform $I_1,\dots,I_N$ with time average $\bar I$:

$$
\frac1N\sum_nF(I_n)\;\ge\;F(\bar I)\;\ge\;F(I_{\rm static}).
$$

Changing the currents during the train can only increase the mean squared
residual field. This is exactly your statement, and it is what the field
guidance term measures.

### 3.2 The image sees the phase history, through one PSF per voxel

Write the history of voxel $p$ as $z_p(t)=A_p(t)D_p(t)e^{-2\pi i\Phi_p(t)}$,
sample it at the centre $t_n$ of each line ($t_n=t_0+n\,T_{\rm esp}$, phase
encode $k_{y,n}$), and neglect its change during the 0.64 ms of one readout.
Then

$$
\hat m(x,y)\;\approx\;\sum_p\delta_{x,x_p}\;h_p(y-y_p),\qquad
h_p(y)=\frac1N\sum_{n}z_p(t_n)\,e^{2\pi i\,k_{y,n}\,y}.
$$

Every voxel is imaged through its own point-spread function along the
phase-encode direction, and that PSF is the Fourier transform, over the line
index, of the voxel's history. A perfect image needs $z_p(t_n)$ constant in $n$.

- A constant residual field $f_p$ gives $\Phi_p=f_pt$, a linear phase over
  lines, hence a pure shift of $f_p\,N\,T_{\rm esp}$ pixels. Here
  $N\,T_{\rm esp}$ = 112.6 ms, so 0.113 pixel per Hz. This is the distortion
  formula $D=\Delta B_0\cdot{\rm ESP}\cdot{\rm FOV}$ of the Arango/Stockmann
  chapter, and it is the only effect the field criterion controls.
- The same field also changes $D_p$ and the time at which the voxel's
  neighbourhood is in phase. Those two effects set the brightness.

### 3.3 Brightness is set at the local echo

Neighbouring voxels along the phase-encode direction have nearly the same
history, apart from the phase difference $q_{p,\rm PE}(t)$ defined in 2.1
(encoding plus field slope). Their sum is coherent only when
$q_{p,\rm PE}(t^*_p)=0$: that instant is the **local echo** of voxel $p$.
Without any field it is the $k_y=0$ line, $t^*=T_E$. Around voxel $p$,

$$
|\hat m|\;\approx\;\frac{A_p(t^*_p)\;\operatorname{sinc}q_{p,\rm RO}(t^*_p)\;\operatorname{sinc}q_{p,z}(t^*_p)}{\big|N\,\partial q_{p,\rm PE}/\partial n\big|_{t^*_p}} .
$$

For a constant field with phase-encode slope $g_p$ (Hz per pixel, sign set by
the blip polarity):

$$
t^*_p=\frac{T_E}{1+g_p/g_c},\qquad g_c=\frac1{N\,T_{\rm esp}}=8.9\ \text{Hz/pixel}=5.7\ \text{Hz/mm}.
$$

So the brightness of a voxel depends on three things that the field criterion
does not contain:

| Quantity | What it does | Depends on |
|---|---|---|
| local echo time $t^*_p$ | T2* weighting $A_p(t^*_p)$; no echo at all if $g_p\le -g_c$ | in-plane slope of the field along phase encode |
| $q_{p,z}(t^*_p)=\sum_cG_{pzc}\,J_c(t^*_p)$ | through-slice cancellation | slice slope of the coils **times the time since the RF** |
| $\partial q_{p,\rm PE}/\partial n$ | local stretching or pile-up | same in-plane slope |

In this dataset the amplitude decays fast: summed over the object,
$A(T_E)$ is 14% of the amplitude at the first echo, and
$\sum_pA_p(50\,\text{ms})=1.45\sum_pA_p(T_E)$. Moving the local echo is a
strong lever on brightness.

### 3.4 Small-field limit: which quadratic form replaces $\sum f^2$

For a small, smooth, constant residual $f$ and a target equal to the perfectly
shimmed image $m$, combining 3.2 and 3.3 to first order (my derivation;
$T=N\,T_{\rm esp}$, derivatives per pixel along phase encode and per slice
thickness, $R$ the local decay rate of $A$):

$$
J_{\rm image}\;\approx\;\frac{\Big\|\;T\big[\,f\,\partial_ym+(1-R\,T_E)\,m\,\partial_yf\,\big]+\tfrac{\pi^2}{6}\,m\,(T_E\,\partial_zf)^2\Big\|^2}{\|m\|^2}.
$$

The field criterion is $\|f\|^2$ with the same weight on every voxel. The image
loss instead sees $f$ only where the image has edges along the phase-encode
direction, sees the in-plane slope of $f$ wherever there is signal, and sees
the slice slope of $f$ multiplied by $T_E$. The two criteria have the same
minimiser only if the coils can cancel $\Delta B_0$ exactly. With 32 channels
and current limits they cannot, and the two criteria then choose different
compromises, even among constant currents.

### 3.5 What time-varying currents add

With $\Phi_p(t)=\Delta B_{0,p}t+\sum_cB_{pc}J_c(t)$:

- the **position** of voxel $p$ depends on the field around its local echo,
  $\Delta B_{0,p}+\sum_cB_{pc}I_c(t^*)$;
- its **brightness** depends on the average current from the RF to the local
  echo, through $q(t^*)=t^*\big[g^0+\sum_cG_{pc}\,\bar I_c(0\!\to\!t^*)\big]$;
- the local echo time itself depends on the same average.

A constant waveform forces $I(t^*)=\bar I(0\!\to\!t^*)$. A time-varying one can
choose them separately. Lines far from the local echo carry little signal (for
the perfectly shimmed image, 39% of the k-space energy is in the centre line
and 83% within ±8 lines), so the currents played there act almost only through
their integral $J$. That is the freedom the optimiser uses, for better (keep
the slice in phase at the echo while cancelling the field there) or for worse
(move the echo).

## 4. What the optimiser did in the saved run

### 4.1 Scoreboard

Both losses use the same mask $M$ (whole field of view) for every row:

- "Loss vs GRE" is $J_{\rm image}$ of section 2.2.
- "Loss vs perfect" is
  $\sum_M(|\hat m|-|\hat m_{\rm perfect}|)^2/\sum_M|\hat m_{\rm perfect}|^2$,
  where $\hat m_{\rm perfect}$ is the simulated EPI at zero B0 and no ACDC
  (`model.ideal_image` in the notebook). It has the same T2* weighting, blur
  and voxel model as every candidate, so it is zero for a perfect shim.
- "Shift" is the phase-encode displacement implied by the phase slope over
  ±4 lines around each voxel's local echo, amplitude-weighted.

| Waveform | Loss vs GRE (notebook) | Loss vs perfect-B0 EPI | Signal / perfect | Residual field RMS [Hz] | Shift RMS [pixels] | Slice dephasing at TE [cycles RMS] | Median local echo [ms] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Perfect B0, no ACDC | 0.307 | 0 | 1.00 | 0 | 0 | 0 | 61 |
| No ACDC | 0.445 | 0.329 | 0.91 | 38 | 3.2 | 0 | 65 |
| Static field shim | 0.625 | 0.595 | 0.47 | 13 | 1.1 | 1.39 | 60 |
| Image-optimised (notebook) | **0.187** | 0.310 | **1.36** | 70 | 4.3 | 0.40 | 53 |
| Time average of the optimised waveform | 0.462 | 0.479 | 0.90 | 31 | 3.5 | 0.63 | 57 |

![images](figures/fig1_images.png)

### 4.2 The static field shim fails through the slice, not in the plane

| Static field shim, model variant | Loss vs GRE | Loss vs perfect | Signal / perfect |
|---|---:|---:|---:|
| Full model | 0.625 | 0.595 | 0.47 |
| ACDC slice slope removed | 0.310 | 0.127 | 1.08 |
| All ACDC slopes removed | 0.314 | 0.116 | 1.03 |

The static currents create 24 Hz RMS across the slice (the B0 map has none by
assumption), 79% of the voxels exceed half a cycle at TE, and the signal falls
linearly in time (figure 2, bottom). In the plane the static shim barely
changes the slopes (3.6 Hz/pixel against 3.2 for B0 alone). The Rev. D paper
handles the same problem by including two neighbouring slices above and below
each target slice in the field fit.

### 4.3 The optimised waveform brightens the image by moving the local echo

| Image-optimised waveform, model variant | Loss vs GRE | Signal / perfect |
|---|---:|---:|
| Full model | 0.187 | 1.36 |
| Amplitude frozen at its TE value (no gain from an earlier echo) | 0.323 | 1.04 |
| ACDC slice slope removed | 0.226 | 1.51 |

- With the amplitude frozen, the extra signal disappears (1.36 to 1.04) and the
  advantage over the perfect shim shrinks from 0.120 to 0.040 (the perfect
  shim scores 0.363 in that frozen model).
- The local echo moves earlier (figure 3): median 65 ms without ACDC, 53 ms
  optimised; 89% of the k-space energy is acquired before TE, against 43%
  without ACDC. The sum of amplitudes at the local echoes predicts a signal
  ratio of 1.30, close to the 1.36 observed.
- Removing the slice slope makes the loss worse (0.226): the remaining slice
  dephasing is part of the intensity match, not a defect the optimiser failed
  to remove.
- The time average of the waveform scores 0.46: the effect lives in the time
  variation (70% of the waveform norm), including line-to-line sign
  alternations on several outputs around the centre of k-space (figure 4).

![time structure](figures/fig2_time_structure.png)
![local echo](figures/fig3_local_echo.png)
![currents](figures/fig4_currents.png)

### 4.4 Why the loss rewards this

The target is a GRE at TE = 4.3 ms; the model is an EPI at TE = 61 ms with
T2* decay. With a perfect shim their normalised squared difference is 0.307,
and refitting the gain only lowers it to 0.304: the mismatch is contrast and
blur, not scale. The gradient of the loss is therefore not zero at the perfect
shim, and the only way the model can raise intensity towards the GRE is to
sample each region earlier. The field guidance term is too weak to stop it:
at the end it contributes 0.05 × 0.49 = 0.025 to a total of 0.211.

## 5. Re-optimisation with one ingredient changed at a time

Same optimiser, limits and field guidance as the notebook, applied to the
closed-form twin. Only the target (measured GRE, or the perfect-B0 simulated
EPI) and the freedom (one command per line, or one constant command) change.
Static runs: 500 iterations; dynamic: 1000; "dynamic from best static": 300
more iterations starting from the static solution of the row above it.

| Waveform | Loss vs GRE | Loss vs perfect-B0 EPI | Signal / perfect | Residual field RMS [Hz] | Shift RMS [pixels] | Slice dephasing at TE [cycles] | Median local echo [ms] | Time-varying part of the waveform | Mean total current [A] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No ACDC | 0.445 | 0.329 | 0.91 | 38 | 3.2 | 0.00 | 65 | 0 | 0 |
| Static field shim (least squares, notebook start) | 0.625 | 0.595 | 0.47 | 13 | 1.1 | 1.39 | 60 | 0 | 20 |
| Notebook: GRE target, dynamic | 0.187 | 0.310 | 1.36 | 70 | 4.3 | 0.40 | 53 | 0.70 | 24 |
| GRE target, static | 0.275 | 0.148 | 1.04 | 26 | 1.9 | 0.27 | 59 | 0 | 12 |
| Static field shim + slice-slope penalty (closed form) | 0.328 | 0.125 | 1.04 | 16 | 1.3 | 0.15 | 61 | 0 | 8 |
| Self-consistent target, static | 0.313 | 0.075 | 1.03 | 22 | 1.5 | 0.19 | 61 | 0 | 9 |
| Self-consistent target, dynamic | 0.309 | 0.052 | 1.05 | 43 | 1.6 | 0.31 | 58 | 0.77 | 16 |
| Self-consistent target, dynamic from best static | 0.313 | **0.049** | 1.03 | 30 | 1.6 | 0.17 | 61 | 0.59 | 10 |

![variants](figures/fig5_variants.png)

What the table shows:

1. **The echo shift is a response to the GRE target.** With the
   self-consistent target the local echo stays at TE (58 to 61 ms) and the
   signal stays within 5% of the perfect image, for constant and for
   time-varying currents.
2. **The criterion matters most.** Constant currents chosen for the image reach
   0.075; constant currents chosen for the field reach 0.595. The image choice
   accepts a larger residual field (22 Hz against 13 Hz) in exchange for seven
   times less slice dephasing (0.19 against 1.39 cycles), with less than half
   the current.
3. **Time variation adds a smaller gain**, 0.075 to about 0.05. It does not
   come from one model term: the dynamic-from-static solution still scores
   0.050, 0.051 and 0.059 when the slice slope, all coil slopes, or the
   amplitude evolution are switched off. I did not identify its mechanism.
4. **A field-only recipe gets most of the way.** Adding
   $\beta_z\,\mathrm{mean}_p(\sum_cG_{pzc}I_c)^2$ to the least-squares field
   fit (`06_static_field_shim_with_slopes.py`, no image simulation in the
   solver):

   | $\beta_z$ | 0 | 0.1 | 0.3 | 1 | 3 | 10 | 30 |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | Loss vs perfect-B0 EPI | 0.656 | 0.273 | 0.192 | 0.148 | 0.132 | 0.125 | 0.128 |
   | Residual field RMS [Hz] | 13.0 | 13.5 | 13.8 | 14.1 | 14.7 | 16.0 | 17.9 |
   | Mean total current [A] | 28 | 20 | 18 | 15 | 11 | 8 | 6 |

   Penalising the in-plane slopes as well changes little (0.120 at
   $\beta_z=\beta_{xy}=10$).
5. **The least-squares field shim is not well defined as an image shim.** My
   $\beta_z=0$ solution and the notebook's static shim have the same residual
   field within 0.1 Hz but differ by up to 2.8 A per channel, and score 0.656
   and 0.595. The field criterion is almost flat along current combinations
   that change the slice slope a lot.
6. **Even constant currents exploit the GRE target a little**: 0.275, below
   the perfect shim's 0.307.

None of these runs has converged. The loss was still falling by about 0.001
per 50 iterations (0.007 for "GRE target, static"), so the ranking is reliable
but the last digit is not.

## 6. What this suggests for the pipeline

These are suggestions drawn from the simulations above. Nothing was changed in
the pipeline.

1. **Choose what the target means.** With the measured GRE and a fixed gain, a
   lower loss is not a better shim: the perfect shim scores 0.307 and the
   optimiser can go below it. A target the model can reach removes this:
   the zero-B0 simulated EPI (`model.ideal_image`, already computed), or the
   GRE rescaled voxel by voxel with the fitted decay $A_p(T_E)/A_p(T_{E,\rm GRE})$.
2. **Start from a static shim that knows about the slice.** The current
   starting point is worse than zero currents. One extra quadratic block on
   the coil slice slope fixes it (section 5, point 4). The Rev. D paper does
   the equivalent by fitting neighbouring slices, and the chapter cites a
   joint $\nabla B_0$ and $\Delta B_0$ objective (Willey et al., 2021).
3. **Report three numbers next to the loss**: the loss against the perfect-B0
   EPI, the total signal relative to it, and the median local echo time. A
   signal above 1 or an echo far from TE means the optimiser is matching
   contrast, not shimming.
4. **Measure the through-slice B0 slope.** The model sets it to zero. The real
   slope could add to or cancel the coil slope, so every slice-dephasing number
   here depends on that assumption. The 3D B0 acquisitions in `b0map/` and the
   06 V4 notebook could supply it.
5. **Look for the signature in the measured scans** of acquisitions 024 to 027
   before interpreting them: an image-optimised scan brighter than the
   unshimmed one, with fine texture and no reduction of the shift.

## 7. Validation, limits, reproduction

**Validation of this analysis**

- `acdc08_twin.py` rebuilds the generation-08 model in closed form from the
  saved arrays, without MRzero and without any pipeline module. It reproduces
  the saved k-space and images to 7e-6 relative error (float32 precision of the
  saved data) for zero currents and for the optimised currents, and the four
  loss values 0.4455, 0.6245, 0.1865 and 0.3069 (`01_validate_twin.py`). The
  comparison is on complex k-space, so a wrong phase sign, a swapped axis or a
  missing $2\pi$ would fail it.
- The re-implemented optimiser reproduces the first 21 iterations of the
  notebook history to 1.5e-7 (`03_optimise_variants.py --tag parity`).

**Limits**

- One acquisition (029), one slice, one saved run that had not converged.
- Simulated only. The measured ACDC-on scans of acquisitions 024 to 027 were
  not analysed; whether they show the brightening and the texture of figure 1
  is the natural next check.
- The model assumes no through-slice B0 slope and trusts the 3D coil
  calibration for the coil slopes. Both drive the slice-dephasing result.
- The local-echo and shift numbers are first-crossing, locally linear
  summaries. They are diagnostics; the losses come from the full model.
- Section 3.4 is a first-order sketch for small smooth fields, not a fit.

**Reproduce** (Python 3.11 with numpy, torch, matplotlib; CPU is enough)

```bash
PY=../../../epi_acdc_artifactCorr_analysis/MRzero_simulation/.venv/bin/python
$PY 01_validate_twin.py                                   # 10 s
$PY 02_mechanisms.py                                      # 1 min
$PY 03_optimise_variants.py --target perfect --mode dynamic --iterations 1000   # 45 min
$PY 03_optimise_variants.py --target perfect --mode static  --iterations 500    # 20 min
$PY 03_optimise_variants.py --target gre     --mode static  --iterations 500    # 20 min
$PY 03_optimise_variants.py --target perfect --mode dynamic --iterations 300 \
    --tag from_static --init-file outputs/03_perfect_static.npz                 # 13 min
$PY 06_static_field_shim_with_slopes.py                   # 1 min
$PY 05_compare_variants.py
$PY 04_figures.py
```

| File | Content |
|---|---|
| `acdc08_twin.py` | Closed-form model, loss, projection, per-voxel diagnostics |
| `01_validate_twin.py` | Agreement with the saved run |
| `02_mechanisms.py` | Sections 4.1 to 4.3 |
| `03_optimise_variants.py` | Re-optimisations of section 5 |
| `06_static_field_shim_with_slopes.py` | Closed-form static shim with slope penalties (section 5, point 4) |
| `05_compare_variants.py` | Table of section 5 |
| `04_figures.py` | Figures |
| `outputs/` | JSON reports, arrays, logs |
| `figures/` | Figures 1 to 5 |
