# Dynamic ACDC shimming with MRzero

### 06 — cartographie 32 échos et correction bruit/support

[06_acdc_echo32_b0_s0_t2star_MRzero.ipynb](06_acdc_echo32_b0_s0_t2star_MRzero.ipynb)
reprend le 05 avec `megre_b0_s0_t2star_128x128_fov200mm_32echo.seq` pour la
cartographie B0/relaxation. La séquence EPI à corriger reste sélectionnée séparément.
Les événements de la nouvelle cartographie donnent 32 échos monopolaire 128²,
de 2,123 à 58,543 ms, TR 61,97 ms et durée 9,9152 s. La définition accessoire
`EchoSamples=64` est périmée ; les ADC réels contiennent 128 points. Le lecteur
conserve les petits décalages de kx par écho et les TE réels lors de la
reconstruction complexe.

Renseigner `B0_SCAN_PATH` avec le Twix acquis avec cette séquence (chemin relatif
à `INPUT_DIR`, ou absolu). `None` sélectionne un unique `*b0_map.dat`, puis vérifie
sa compatibilité ; le fichier historique trois échos est explicitement refusé
avec la séquence 32. Les six premiers échos servent par défaut au B0
(`B0_FIELD_ECHO_COUNT`), tous les 32 restent disponibles pour R2*.

`prepare_echo_object` estime le bruit dans des coins indépendants du seuil objet,
ou dans un `noise_mask` explicitement fourni. La covariance complexe débiaise
le RSS avant le support SNR ; les petites composantes et les trous sans signal
ne sont plus absorbés par un remplissage global. Un contrôle de moyenne complexe
signale certains fonds contaminés ; les régions de bruit doivent aussi être
inspectées sur les figures. Le notebook demande une calibration du bruit par
défaut et utilise les taux locaux acceptés (`RATE_MODE="guarded"`). Les API du 05
restent compatibles et bénéficient aussi du correctif de bruit/support.

Le précontrôle affiche l'extrapolation vers le TE EPI (environ 19,5 ms au-delà
du dernier écho pour l'EPI actuelle), et vers le dernier ADC, plus tardif.
Les critères de fiabilité de relaxation restent actifs, notamment le SNR à
tous les échos ; un signal tardif insuffisant peut encore donner un repli
explicite sans décroissance. L'amplitude est une amplitude GRE effective,
pas une mesure indépendante de PD/T1 ou du B0 à travers la coupe.
Les masques, covariance, diagnostics de fit et paramètres sont sauvegardés sous
`results_experiment_echo32/mrzero06_*`, avec le reste du pipeline du 05.

Two sub-spin extensions of notebook 04 are available:

- [04_acdc_hardware_continuous_MRzero_V2.ipynb](04_acdc_hardware_continuous_MRzero_V2.ipynb)
  uses eight midpoint spins through the EPI slice thickness. Each spin samples
  the original 3D ACDC calibration. The measured 2D B0 is explicitly constant
  through thickness; no measured through-plane susceptibility gradient is inferred.
- [04_acdc_hardware_continuous_MRzero_V3.ipynb](04_acdc_hardware_continuous_MRzero_V3.ipynb)
  retains those eight thickness positions and adds 2 by 2 in-plane positions.
  Every child samples B0 directly from the native measured 128 by 128 map,
  before the phase calculation, rather than the parent 64 by 64 interpolation.

`subspin_sampling.py` prepares weighted children and rejects a whole parent if
any child lacks a reliable interpolation stencil. `experiment_subspins.py`
simulates their native MRzero magnetization and applies each child's dynamic
ACDC phase in every objective/gradient evaluation. Hardware field limits cover
all reliable children; only display fields are averaged. The 32 current commands
per imaging state and the reconstructed scan matrix remain unchanged.
V2 checks zero-current signal agreement with the original 04 on the same retained
support. Both versions display child B0 spread and a field-only coherence
diagnostic. The GRE amplitude remains an effective, already weighted signal
proxy; subdividing does not recover unmeasured tissue detail or lost GRE signal.
Results use separate `results_experiment_realistic_v2/` and
`results_experiment_realistic_v3/` folders, including child positions, weights,
B0 values, calibration, and numerical histories. Large sub-spin caches may use
CPU memory automatically when the GPU cache budget would be exceeded.

For the hardware-aware extension of notebook 03, open
[04_acdc_hardware_continuous_MRzero.ipynb](04_acdc_hardware_continuous_MRzero.ipynb).
It retains the measured-data preparation and live image optimizer, with the
filtered current response described in the supplied controller manuscript
(99% rise in 133 us, equivalent first-order tau, recommended +/-3 A per channel,
minimum 8 us command spacing). The chapter's per-channel and total absolute
current constraints are applied to commands and the predicted physical waveform.
The user confirmed 3 A/channel, 30 A total and the 133 us filter on 2026-10-16.
The total limit is an installation choice, not a specification established by
the papers; `CURRENT_PRESET="article_demo"` selects the manuscript's
2 A/channel and 10 A experimental protocol.

`acdc_hardware.py` checks complete constant-command intervals, including gaps,
startup and switch-off. Optional measured slew, average-current, RL voltage and
mean resistive-power limits remain explicitly unset until supplied. Current and
power integrals use the full sequence TR, with final residual current reported.
The 0.6 mA effective resolution in the manuscript is informational; the existing
controller's nominal DAC conversion is checked separately.

`continuous_reconstruction.py` uses actual nominal Pulseq k-space coordinates,
without rounding or dropping ADC samples, and its exact Hermitian adjoint.
It is the normalized Fourier adjoint, not a general nonuniform inverse.
`measured_continuous.py` applies the same reconstruction per measured receiver
before RSS and to the complex B0 echoes. Native MRzero magnetization and current
phase derivatives are reused by `experiment_realistic.py`. CUDA graphs are
disabled in 04 because the older graph helper contains its own projection.
Results are saved under `results_experiment_realistic/<session>/`; CSV export
remains optional. The original 03 model and notebook are unchanged.

The validation notebooks use English. The experimental notebook includes a
French walkthrough, with code and plot labels in English.

For the measured Siemens experiment, open
[03_acdc_experiment_MRzero_live.ipynb](03_acdc_experiment_MRzero_live.ipynb).
It follows `Experiment/02_acdc_experiment_step_by_step_v3.2_fast.ipynb`, using
native MRzero voxel magnetization for the ideal and measured-B0 simulations.
The image-optimization figure updates every `LIVE_UPDATE_EVERY` iterations
(20 by default; set 10 or 1 for more frequent updates), plus the initial and final
states. It shows the current image, matching relative MSE, residual and
channel/line commands. The loss curve retains every iteration between updates.
Select `Python (MRzero GPU Windows)` and set `INPUT_DIR` to the acquisition
containing the Twix files, measured calibration and the three Pulseq sequences.
CSV export is optional and disabled by default. Install the extra raw-data
readers with `python -m pip install -r requirements-experiment.txt`.
The final currents, images and per-iteration numerical history are saved as
NPZ and JSON in `results_experiment_mrzero/<session>/`.

`experiment_mrzero.py` contains the native simulation adapter,
`experiment_optimization.py` the projected Adam loop with image snapshots,
and `live_optimization.py` the single updating Matplotlib display.
`USE_CUDA_GRAPH = True` uses `cuda_graph_helpers.py` to replay the unchanged
objective, adjoint and constraint projections with less Python launch overhead.
The solver reports whether graphs are active and separates plotting time from
optimization time. `GPU_PROFILE = "gtx1660ti"` requires CUDA; select `"auto"`
or `"cpu"` explicitly for a machine without CUDA. Data reading and Matplotlib
still use the CPU. Current
response, constraints, reconstruction and raw-data readers reuse the original
experiment modules. Native MRzero magnetization is cached once; the exact
ACDC phase is then applied for the source notebook's ideal-RF, stationary
point-spin model. The measured GRE magnitude supplies effective transverse
amplitude, rather than an unmeasured proton-density or T1 map.

On acquisition 014 with the GTX 1660 Ti, float32, 256 ADC samples per block,
150 iterations and display every 20 iterations, the solver loop took 4.91 s
(3.83 s for snapshots/display and 1.08 s otherwise). Its best relative MSE was
0.0226029139, identical to the previous every-iteration display run. This timing
excludes input preparation, graph setup and the final summary figures. GPU
utilization varies while reading data and drawing figures; sustained 100% is
not required for CUDA execution.

Start with [02_dynamic_shimming_MRzero.ipynb](02_dynamic_shimming_MRzero.ipynb)
to run the complete Python pipeline: load the original input maps, generate the
EPI sequence, simulate the baseline with MRzero, optimize ACDC currents using
PyTorch, reconstruct images, and compare with the saved reference results.
The optimizer uses CUDA when available and supports a CPU fallback. MATLAB is
not required to run either notebook.

The Python files mirror the processing stages: [data.py](data.py) loads the maps,
[sequence.py](sequence.py) generates the EPI waveform, [model.py](model.py)
caches MRzero magnetization and computes dynamic ACDC images, and
[optimization.py](optimization.py) initializes and optimizes the currents.

[01_audit_comparison_MRzero.ipynb](01_audit_comparison_MRzero.ipynb) is the initial
audit and numerical comparison. It reads the supplied MATLAB results, compares
them with NumPy and MRzeroCore 1.1.1, counts ADC grid collisions and discarded
samples, checks field limits on both masks, illustrates current-response
sensitivity, and examines transfer to subjects 007–009. Its synthetic MATLAB
references are already stored locally. This notebook does not optimize currents.

The new optimization follows the supplied method: the same maps and coordinates,
one ideal RF pulse, original EPI timing and reconstruction, image-magnitude
objective, and original current and sampled-field constraints. The audit's
trajectory, sparse-mask and hardware-response findings therefore still apply.
Changing those assumptions would define a separate physical validation study.

The Python kernel needs the packages in [requirements.txt](requirements.txt).
The isolated CUDA environment for this project is:

```text
MRzero_validation/.venv-gpu/bin/python
```

To create the same environment on another Linux machine, run these commands
from `MRzero_validation`. Install CUDA PyTorch before the remaining requirements
so that the CPU package does not replace it:

```bash
python3.11 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install --upgrade pip
.venv-gpu/bin/python -m pip install torch==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
.venv-gpu/bin/python -m pip install -r requirements.txt
```

For Windows PowerShell, install Python 3.11 first, then run the following from
`MRzero_validation`. The Windows environment is created in the repository root
as `.venv-gpu-win`, alongside this directory:

```powershell
$python311 = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'
& $python311 -m venv ..\.venv-gpu-win
..\.venv-gpu-win\Scripts\python.exe -m pip install --upgrade pip
..\.venv-gpu-win\Scripts\python.exe -m pip install torch==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
..\.venv-gpu-win\Scripts\python.exe -m pip install -r requirements.txt
..\.venv-gpu-win\Scripts\python.exe -m ipykernel install --user --name mrzero-gpu-win --display-name "Python (MRzero GPU Windows)"
```

In the notebook kernel picker, select `Python (MRzero GPU Windows)` or the
interpreter `..\.venv-gpu-win\Scripts\python.exe`. The existing `.venv-gpu`
was created on Linux and cannot be used as a Windows interpreter.

The CUDA package selection follows the official
[PyTorch 2.6 installation instructions](https://pytorch.org/get-started/previous-versions/).
Select this interpreter as the notebook kernel. The CUDA 11.8 wheel bundles its
CUDA runtime libraries; it requires a compatible NVIDIA driver. CUDA operations
on this workstation must run outside the execution sandbox so that the GPU
device is visible. The original shared CPU environment remains unchanged.
GPU complex-phase autograd and fused Adam with float64 current parameters were
verified on the Tesla V100S-PCIE-32GB.

In the notebook settings, leave `GPU_PROFILE = 'auto'` for automatic selection,
or choose an explicit preset:

| GPU_PROFILE | ADC samples per block | Intended device |
|---|---:|---|
| `v100` | 1024 | Tesla V100 with 32 GB |
| `gtx1660ti` | 256 | GTX 1660 Ti with 6 GB |
| `cpu` | 1024 | CPU |

`DEVICE = 'auto'` selects CUDA when visible. Set `DEVICE = 'cuda:0'` to require
GPU execution. These presets change block size while preserving the model and
objective. The notebook benchmarks phase integration and optimizer choices on
the selected device before running all field limits.

Input map selection, units and coordinate-frame assumptions are documented in
[subject006_slice120_provenance.json](subject006_slice120_provenance.json).
The original large archives are unnecessary for rerunning the notebooks:
`subject006_slice120_inputs.npz` through `subject009_slice120_inputs.npz` contain
the exact selected arrays for all four subjects. The
repository's `sol_*.mat` and `eval_other_subj` files provide saved comparison
results and are read with SciPy.

[mrzero_dynamic.py](mrzero_dynamic.py) applies dynamic ACDC phase to MRzero's
per-voxel magnetization before receive summation. This adaptation is valid for
one ideal RF pulse, stationary point spins, and uniform receive sensitivity. It
preserves the exact X/Y positions and phase history during transitions.
The first-order current response and clipping in the audit are illustrative
sensitivity studies without reoptimization or hardware identification.

After the audit executes, [comparison_report.json](comparison_report.json)
stores numerical errors and [saved_results_comparison.png](saved_results_comparison.png)
shows the saved images on a common scale. The dynamic notebook saves its Python
optimization outputs and timing measurements for comparison.

For future physical validation, use a verified trajectory, measured hardware
response, and a 3D simulation with subvoxels and realistic slice excitation.
The current point-spin comparison establishes numerical agreement with the
original method and does not establish intravoxel signal recovery.

The optional legacy synthetic MATLAB reference can be regenerated from the
repository root. This command is unnecessary for the Python pipeline:

```bash
matlab -batch "run('MRzero_validation/validate_synthetic_matlab.m')"
```

### V4 — validation du jumeau EPI mesuré (acquisition 015)

`04_acdc_hardware_continuous_MRzero_V4.ipynb` vérifie le modèle sans ACDC contre
`meas_MID00080_FID46887_scan.dat`. Les versions précédentes sont conservées.

Le retournement de repère à 180° transforme ensemble X/Y et k, au lieu de tourner
les seuls tableaux sur des grilles fixes. L'objet utilise les échos complexes
natifs 128². Le domaine simulé dépend de l'objet visible et du B0 fiable, et non
de la ROI d'optimisation ou de la couverture des bobines ACDC. Les poids d'aire
sont appliqués une seule fois et la fraction d'amplitude non couverte est rapportée.

Les trois échos ajustent une amplitude effective ancrée au premier TE et un R2*
effectif. Le bruit RSS, l'incertitude à TE EPI et les replis sont explicites.
Ils ne déterminent ni PD ni T1 ; extrapoler de 5,8 ms vers 98,4 ms reste une hypothèse
physique. Le B0 brut est comparé à une régularisation par plans locaux d'échelle
fixée qui préserve les champs linéaires, le masque et le B0 moyen. Aucun champ
manquant n'est inventé et aucune déformation n'est ajustée sur l'image EPI.

Les deux jumeaux sont calculés par MRzero avec le même objet ; seul B0 change.
Huit sous-spins identiques suivant z seraient équivalents pour cette acquisition
sans ACDC et à kz=0 ; la distribution B0 suivant l'épaisseur n'est pas mesurée.

Modules : `twin_inputs.py`, `twin_geometry.py`, `echo_object.py`,
`b0_regularization.py`, `epi_twin.py`. Les résultats JSON/NPZ se trouvent sous
`results_experiment_realistic_v4/`. La NRMSE utilise tout le FOV et un gain global
par variante ; les gradients, le Dice des contours et les centroïdes sont aussi
rapportés. Le choix d'une variante sur ce scan n'est pas une validation indépendante.
Cette V4 valide le signal de base ; elle n'optimise ni n'exporte les courants ACDC.

L'audit `acquisition_015_geometry_audit.json` vérifie les réglages scanner communs
et les impulsions fournies : fréquence, shim et géométrie enregistrés identiques ;
angles RF intégrés de 90° pour EPI/GRE et de 15° pour B0, malgré des valeurs déclarées
incohérentes. Les fichiers d'acquisition et les séquences ne sont pas modifiés.

### 05 — optimisation ACDC avec objet natif, rotations et miroirs

`05_acdc_echo128_relaxation_orientation_MRzero.ipynb` intègre l'objet V4 dans le
pipeline complet du 04 : lecture, sous-spins, shim statique initial, optimisation
d'image avec affichage dynamique, réponse matérielle, contrôles DAC, sauvegarde
et export CSV facultatif. Il utilise par défaut acquisition_015, 150 itérations,
±3 A/canal, 30 A au total et le filtre équivalent 133 µs à 99 %.

Les paramètres d'orientation sont regroupés au début :

```python
ALIGNMENT_ANGLE_DEG = 180.0
ALIGNMENT_MIRROR_X = False
ALIGNMENT_MIRROR_Y = False
```

Les miroirs changent le signe de la coordonnée nommée avant la rotation positive
de X vers Y. Cette orientation Q place tout le repère d'imagerie dans celui des
bobines : positions, ADC et grille de reconstruction sont transformés ensemble,
les réponses ACDC sont interrogées aux nouvelles positions patient. L'optimisation
s'adapte donc à cet alignement. Il n'y a ni interpolation des images ni décalage
lié au centre des matrices. Les paramètres `FRAME_*` définissent un second changement
de repère commun P, sans changer le placement relatif aux bobines. Modifier P seul
doit préserver les courants physiques ; la reconstruction utilise O=P@Q.

Les cartes natives 128² restent associées aux mêmes indices. Huit positions suivant
l'épaisseur utilisent les réponses ACDC 3D. Les limites portent sur chaque enfant.
Les parents sans B0 ou sans couverture ACDC complète sont explicitement exclus et
leur fraction d'amplitude est rapportée, sans renormalisation du signal restant.
Le temps d'ancrage de l'amplitude est le premier TE mesuré, pas le TE EPI.

Les modules nouveaux sont `echo_shim_inputs.py`, `experiment_echo.py`,
`inplane_geometry.py` et `shim_export_timing.py`. Ils réutilisent le solveur et les
contraintes du 04. Les résultats sont enregistrés sous `results_experiment_echo/`,
avec les angles, miroirs, matrices Q/P/O, positions patient, champs, courants,
historique, comparaisons et contrôles avant/après conversion nominale DAC.

L'export est désactivé par défaut. L'origine `first_imaging_trigger` est vérifiée
dans le `.seq` : pour la séquence 015, t=0 est le trigger ext1 du premier bloc
image, 10,370 ms après la RF ; le retour à zéro est à 173,440 ms après ce trigger.
Les temps CSV sont conservés et les canaux/polarités restent configurables. Aucun
matériel n'est piloté. Les scans corrigés historiques sont affichés comme tels :
ils n'utilisent pas la nouvelle forme d'onde calculée par ce notebook.
