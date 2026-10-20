"""Build the MRzero experiment notebook from the original step-by-step workflow."""

from pathlib import Path
import json


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "Experiment" / "02_acdc_experiment_step_by_step_v3.2_fast.ipynb"
TARGET = HERE / "03_acdc_experiment_MRzero_live.ipynb"


def set_source(cell, text):
    cell["source"] = text.strip("\n").splitlines(keepends=True)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]
for cell in cells:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    cell["metadata"] = {}

set_source(cells[0], """
# ACDC expérimental avec MRzero — optimisation d'image en direct

Ce troisième notebook reprend les étapes de
`Experiment/02_acdc_experiment_step_by_step_v3.2_fast.ipynb` : lecture des données
Siemens, reconstruction GRE/EPI, estimation du B0, cartes ACDC, shim statique,
shim dynamique de champ, puis optimisation de l'image.

**Toutes les images simulées proviennent de MRzeroCore.** Le simulateur construit
une séquence à partir des temps ADC et des moments de gradients réellement lus
dans les `.seq`, puis appelle `compute_graph` et `execute_graph`. La cible idéale
est le même objet simulé avec B0 nul. Aucun moteur MATLAB n'est lancé.

Pour l'optimisation, on conserve l'aimantation MRzero de chaque voxel à chaque ADC.
La phase des courants ACDC est ensuite intégrée exactement avec la réponse
temporelle choisie. Cette réutilisation est exacte dans le modèle du notebook
source : spins ponctuels immobiles, une excitation idéale par train EPI, réception
uniforme et spoiling idéal pour le GRE. La magnitude GRE mesurée reste une
**amplitude transverse effective**, sans ajout d'une carte PD/T1 non mesurée.
La simulation ne modélise pas le profil de coupe RF ni le déphasage intravoxel.

Dans la section **11**, une seule figure se met à jour toutes les
`LIVE_UPDATE_EVERY` itérations (**20 par défaut**, réglable à 10 ou 1) :
référence, baseline sans ACDC, image du candidat courant, différence de magnitude,
MSE relative et commandes par canal/ligne. L'image et la MSE correspondent au
même candidat évalué. Le résultat final conserve le meilleur candidat admissible.

1. Choisir le noyau **Python (MRzero)**.
2. Vérifier `INPUT_DIR` et les paramètres dans les deux premières cellules.
3. Exécuter de haut en bas. Le CSV est facultatif (`EXPORT_WAVEFORM = False`).
""")

imports = '''
# 1. Imports and input/output paths
%matplotlib inline
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import display, Markdown

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "Experiment").is_dir() else Path.cwd().parent
EXPERIMENT_DIR = PROJECT_ROOT / "Experiment"
VALIDATION_DIR = PROJECT_ROOT / "MRzero_validation"
for folder in (EXPERIMENT_DIR, VALIDATION_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from prepare_acdc_inputs import read_gre, read_scan, check_reference_sequence, check_physical_geometry
from read_acdc_sequence import read_sequence
from read_acdc_b0 import read_b0_coil_images, b0_from_coil_images
from read_acdc_calibration import read_calibration
from resample_acdc_maps import resample_to_grid
from prepare_acdc_simulation_grid import prepare_simulation_reference
from run_acdc_experiment import check_calibration, current_windows
from export_acdc_waveform import export_waveform, quantize_currents
from acdc_gpu import select_device
from acdc_gpu_field import solve_field_gpu
from experiment_mrzero import MRzeroExperimentModel
from experiment_optimization import optimize_mrzero_image
from live_optimization import LiveOptimizationPlot

INPUT_DIR = PROJECT_ROOT / "acquisition_014"

def unique_input(pattern):
    """Find exactly one input; explain missing or ambiguous acquisitions."""
    # Baseline acquisitions are at the input root; results/ contains later scans.
    matches = sorted(INPUT_DIR.glob(pattern))
    if not matches:
        matches = sorted(path for path in INPUT_DIR.rglob(pattern)
                         if "results" not in path.relative_to(INPUT_DIR).parts)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {pattern} under {INPUT_DIR}; found {len(matches)}. "
            "Set INPUT_DIR to the acquisition folder."
        )
    return matches[0]

FILES = {
    "reference": unique_input("*reference.dat"),
    "b0": unique_input("*b0_map.dat"),
    "scan": unique_input("*scan.dat"),
    "calibration": INPUT_DIR / "ACDC_3T.npz",
    "scan_sequence": INPUT_DIR / "sequences/gre_epi_64x64_fov200mm_nav3_rise1ms_ext1_single_trigger_first_readout_v4.seq",
    "reference_sequence": INPUT_DIR / "sequences/gre_32x32_fov200mm_tr1s.seq",
    "b0_sequence": INPUT_DIR / "sequences/b0_map.seq",
}
SESSION = datetime.now().strftime("mrzero_%Y%m%d_%H%M%S_%f")
OUTPUT_DIR = VALIDATION_DIR / "results_experiment_mrzero" / SESSION
print("Input folder:", INPUT_DIR)
print("Results folder:", OUTPUT_DIR)
print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda)
'''
set_source(cells[1], imports)

parameters = "".join(cells[3]["source"])
parameters = parameters.replace("EXPORT_WAVEFORM = True", "EXPORT_WAVEFORM = False")
# Keep the portable auto GPU profile from the template.
parameters += "\n# Plot every 20 iterations; set 10 for more frequent updates, or 1 for every step.\n"
parameters += "LIVE_UPDATE_EVERY = 20\n"
parameters += "# Replay identical CUDA operations without resubmitting each small kernel.\n"
parameters += "USE_CUDA_GRAPH = True\n"
set_source(cells[3], parameters)

set_source(cells[24], """
## 9. Cartes ACDC et préparation du simulateur MRzero

On conserve les cartes mesurées, les coordonnées, la ROI et les unités du
notebook source. L'aimantation de base est maintenant simulée par MRzero,
avec les mêmes temps ADC, phases RF/ADC et trajectoires du fichier Pulseq.
Les deux simulations natives (B0 mesuré et B0 nul) sont préparées une fois.
Les blocs de calcul règlent l'utilisation mémoire sans retirer de voxels ni d'ADC.
""")
preparation = "".join(cells[27]["source"])
preparation = preparation.replace("# Préparer les données physiques, puis le modèle GPU une seule fois.",
                                  "# Prepare the physical inputs, then native MRzero magnetization once.")
preparation = preparation.replace("model = ACDCModel(", "model = MRzeroExperimentModel(")
preparation = preparation.replace("Uniform receive sensitivity; ideal GRE spoiling.",
                                  "Native MRzero point spins; uniform receive sensitivity; ideal GRE spoiling.")
preparation = preparation.replace("Données et modèle conservés en mémoire Python.",
                                  "Native MRzero magnetization and current-response matrices are ready.")
set_source(cells[27], preparation)

for index in (29, 33):
    text = "".join(cells[index]["source"])
    text = text.replace("EXPORT_STATIC_CURRENTS = True", "EXPORT_STATIC_CURRENTS = EXPORT_WAVEFORM")
    text = text.replace("EXPORT_DYNAMIC_B0_WAVEFORM = True", "EXPORT_DYNAMIC_B0_WAVEFORM = EXPORT_WAVEFORM")
    set_source(cells[index], text)

set_source(cells[40], """
## 10. Jumeaux numériques MRzero et temps de calcul

La référence idéale et le jumeau avec B0 mesuré utilisent l'aimantation produite
par MRzero. Les ADC sont remis dans l'ordre cartésien de la reconstruction,
indépendamment de l'ordre chronologique des commandes.

La magnitude GRE mesurée sert à définir l'objet. L'objectif reste la MSE relative
de magnitude par rapport au **jumeau idéal MRzero**, sur toute l'image.
Le scan RSS mesuré est affiché séparément avec son échelle propre.
""")
twins = "".join(cells[41]["source"])
twins = twins.replace("# Le modèle et les tableaux fixes sont déjà prêts depuis l'étape 9.",
                      "# Native MRzero magnetization was prepared in section 9.")
twins = twins.replace("Jumeau idéal, B0 nul", "MRzero: ideal zero-B0 twin")
twins = twins.replace("Jumeau avec B0 mesuré", "MRzero: measured-B0 twin")
set_source(cells[41], twins)

set_source(cells[42], """
## 11. Optimisation MRzero sur GPU avec suivi visuel réglable

Adam met à jour une commande DAC par canal et par ligne. Les limites par canal,
la somme des valeurs absolues des courants et le champ ACDC sont imposés après
chaque mise à jour. Le modèle garde tout l'historique de la réponse temporelle.

**Pendant l'exécution de la cellule suivante**, la même figure est remplacée
toutes les `LIVE_UPDATE_EVERY` itérations (20 par défaut), ainsi qu'au début
et à la dernière itération. La courbe conserve les valeurs de **toutes** les
itérations, même entre deux affichages. Les échelles des images
restent fixes. La MSE affichée est celle de l'image courante, pas celle d'un
candidat différent. Une MSE relative de 0,01 correspond à une NRMSE de 10 %.

`GPU_PROFILE = "gtx1660ti"` exige CUDA. `USE_CUDA_GRAPH = True` regroupe les
opérations répétées d'objectif, de gradient et de projection pour réduire
les lancements depuis Python. Les équations et les contraintes restent identiques.
Le tableau final indique si cette accélération est active.

L'affichage utilise Matplotlib et `display_id`, sans dépendance à `ipympl`.
La lecture des données et la préparation des figures utilisent le CPU ; le calcul
d'image, du gradient et des projections utilise CUDA. Le temps total inclut
la génération des figures. Le meilleur candidat admissible
est conservé pour les figures finales et les vérifications de la section 12.
""")

set_source(cells[44], '''
# Refresh the display and solver when rerunning this cell in an existing kernel.
from importlib import reload
import live_optimization
import experiment_optimization
LiveOptimizationPlot = reload(live_optimization).LiveOptimizationPlot
optimize_mrzero_image = reload(experiment_optimization).optimize_mrzero_image

solution = None
if RUN_OPTIMIZATION:
    print("MRI optimization device:", model.device, "| GPU:", model.runtime_info["gpu_name"])
    live_plot = LiveOptimizationPlot(
        reference=ideal_image,
        baseline=baseline_image,
        current_limit_a=MAX_CURRENT_A,
        title="MRzero image optimization: current evaluated candidate",
    )

    def report_progress(snapshot):
        """Display the image, currents and loss from the same evaluated candidate."""
        live_plot.set_history(snapshot["history"]["iteration"], snapshot["history"]["fval"])
        live_plot.update(
            iteration=snapshot["iteration"],
            image=snapshot["image"],
            currents_a=snapshot["currents_A"],
            loss=snapshot["fval"],
            nrmse_percent=100 * np.sqrt(snapshot["fval"]),
            max_field_hz=snapshot["max_field_Hz"],
            field_limit_hz=FIELD_LIMIT_HZ,
            phase="Projected Adam",
        )

    try:
        solution = optimize_mrzero_image(
            model, image_initial_currents_A,
            total_current_limit_A=TOTAL_CURRENT_LIMIT_A,
            field_limit_Hz=FIELD_LIMIT_HZ,
            max_iterations=MAX_ITERATIONS,
            learning_rate_A=IMAGE_LEARNING_RATE_A,
            target_mse=IMAGE_TARGET_MSE,
            patience=IMAGE_PATIENCE,
            callback=report_progress,
            callback_every=LIVE_UPDATE_EVERY,
            use_cuda_graph=USE_CUDA_GRAPH,
        )
    except BaseException:
        live_plot.finish("Stopped: last evaluated candidate remains visible")
        raise
    else:
        info = solution["optimization_info"]
        live_plot.finish(f"{info['status']}; best feasible MSE = {info['fval']:.6g}")

    show_table(list(info.items()), ("Solver result", "Value"))
    show_table([
        ("Initialization", IMAGE_INITIALIZATION),
        ("Initial relative MSE", solution["initial_relative_mse"]),
        ("Best feasible relative MSE", info["fval"]),
        ("Best feasible NRMSE [%]", 100 * np.sqrt(info["fval"])),
        ("Live display and snapshot time [s]", info["callback_seconds"]),
        ("Optimization excluding live display [s]", info["seconds_without_callback"]),
    ])
    history = solution["history"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, key, title in zip(axes, ("fval", "firstorderopt", "constrviolation"),
                             ("Relative image MSE", "Projected step [A]", "Constraint violation")):
        ax.plot(history["iteration"], history[key], ".-")
        ax.set(xlabel="Iteration", ylabel=title, title=title)
    axes[0].plot(history["iteration"], history["best_fval"], "--", label="Best feasible MSE")
    axes[0].legend()
    show_figure(fig)
else:
    print("Image optimization disabled in the parameter cell.")
''')

# Update simulator-specific wording, keeping all original diagnostics and exports.
for cell in cells[45:]:
    text = "".join(cell["source"])
    text = text.replace("Python/PyTorch", "MRzero/PyTorch")
    set_source(cell, text)

results = "".join(cells[46]["source"])
results += '''

# Save the best feasible result and the numerical convergence history.
if solution is not None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "mrzero_optimization.npz",
        reference=solution["reference"], baseline=baseline_image,
        optimized=solution["image"], currents_A=solution["currents_A"],
        initial_currents_A=solution["initial_currents_A"],
        iteration=history["iteration"], relative_mse=history["fval"],
        best_relative_mse=history["best_fval"],
        X_m=X_m, Y_m=Y_m, simulation_mask=support,
    )
    result_summary = {
        "optimization": solution["optimization_info"],
        "runtime": model.runtime_info,
        "source_sha256": source_hashes,
        "image_target": "native MRzero zero-B0 reference",
    }
    (OUTPUT_DIR / "mrzero_optimization.json").write_text(
        json.dumps(result_summary, indent=2), encoding="utf-8",
    )
    print("Numerical results saved in:", OUTPUT_DIR)
'''
set_source(cells[46], results)

set_source(cells[55], '''
# Optional measured comparison: explicitly choose one corrected acquisition.
RESULTS_DIR = INPUT_DIR / "results"
CORRECTED_SCAN_NAME = "meas_MID00351_FID46584_scan.dat"
CORRECTED_SCAN_FILE = RESULTS_DIR / (CORRECTED_SCAN_NAME or "scan_corr.dat")
CORRECTED_SEQUENCE_FILE = FILES["scan_sequence"]
RMSE_USE_ROI = True
NORMALIZE_COMPARISON = True
print("Available result scans:", [path.name for path in sorted(RESULTS_DIR.glob("*.dat"))])
print("Selected corrected scan:", CORRECTED_SCAN_NAME)
''')
comparison = "".join(cells[57]["source"])
comparison = comparison.replace(
    'print(f"Comparison skipped: acquire the corrected scan and place it at {CORRECTED_SCAN_FILE}.")',
    'print("Comparison skipped: choose CORRECTED_SCAN_NAME in the preceding cell.")',
)
set_source(cells[57], comparison)

notebook["metadata"]["kernelspec"] = {
    "display_name": "Python (MRzero)",
    "language": "python",
    "name": "python3",
}
notebook["metadata"].pop("widgets", None)
notebook["metadata"]["language_info"]["version"] = "3.11.9"
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Created {TARGET.name}: {len(cells)} cells")
