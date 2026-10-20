"""Build the two teaching notebooks, using V4 as the validation source.

Only paths, synthetic acquisition metadata and plot labels are adapted in the
V4 copy; its numerical cells are retained. Run with the project .venv-gpu-win interpreter.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent / "MRzero_validation"


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    result = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True),
                  id=hashlib.sha1(source.encode()).hexdigest()[:12])
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def write_notebook(path, cells, metadata=None):
    metadata = deepcopy(metadata or {})
    metadata.update(kernelspec=dict(display_name="Python (MRzero)",
        language="python", name="python3"), language_info=dict(name="python", version="3.11"))
    path.write_text(json.dumps(dict(cells=cells, metadata=metadata, nbformat=4, nbformat_minor=5),
        indent=1, ensure_ascii=False), encoding="utf-8")


SETUP = '''
from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import display, Markdown

# Find the project from either this folder or its parents, on Linux or Windows.
PROJECT_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
                    if (p / "Experiment").is_dir() and (p / "MRzero_validation").is_dir())
VALIDATION_DIR = PROJECT_ROOT / "MRzero_validation"
WORK_DIR = PROJECT_ROOT / "Mrzero_simulation"
for folder in (PROJECT_ROOT / "Experiment", VALIDATION_DIR, WORK_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from synthetic_acquisition import SimulationConfig, generate_acquisition, read_adc_samples, verify_acquisition_files
from synthetic_phantom import make_phantom
from twin_inputs import read_twin_inputs
from synthetic_validation import evaluate_maps, compare_truth_epi, serialisable

print("Interpréteur :", sys.executable)
print("MRzero GPU disponible :", torch.cuda.is_available())
print("Environnement prévu :", PROJECT_ROOT / ".venv-gpu-win/Scripts/python.exe")
'''


def build_generator():
    cells = [cell("markdown", '''
    # 1 — Créer une acquisition synthétique avec MRzero

    Ce notebook génère une GRE **32²**, une cartographie B0 à **trois échos 128²**
    et une EPI **64² avec trois navigateurs**, à partir des fichiers Pulseq de
    `acquisition_015`. Il ne lit ni ne copie ses signaux Twix ou son dossier `results`.

    Les données sont des **fichiers Siemens Twix `.dat`**, identifiés comme synthétiques.
    Les lecteurs Twix habituels de V4 reconstruisent leurs images et réestiment B0/R2*.
    Les cartes de vérité terrain servent uniquement à évaluer les erreurs.

    Utiliser le noyau `.venv-gpu-win/Scripts/python.exe` du projet sous Windows.
    La création refuse d'écraser une acquisition existante.
    '''), cell("code", SETUP), cell("markdown", '''
    ## Paramètres à modifier

    `b0_scale=0` donne un contrôle à champ homogène ; `1` produit une distorsion EPI.
    Le bruit est faible et reproductible. `noise_std_image` désigne l'écart-type de
    chaque composante réelle/imaginaire après la reconstruction normalisée.
    Un bruit strictement nul empêcherait l'estimation de covariance utilisée par V4.

    Un récepteur uniforme est le cas conseillé pour tester V4 : son modèle de
    simulation par défaut ne contient pas de cartes de sensibilité multi-récepteurs.
    Le générateur accepte davantage de récepteurs pour des essais dédiés.
    '''), cell("code", '''
    SOURCE_DIR = PROJECT_ROOT / "acquisition_015"
    DATASET_NAME = "acquisition_synthetic_twix_001"
    DATASET_DIR = WORK_DIR / DATASET_NAME
    METADATA_DIR = WORK_DIR / "simulation_metadata" / DATASET_NAME
    CONFIG = SimulationConfig(
        object_matrix=128, receiver_count=1, b0_scale=1.0,
        noise_std_image=0.0005, seed=15,
        device="auto", adc_chunk=256,
        copy_calibration=True,  # Copie ACDC_3T.npz pour conserver l'organisation d'acquisition.
    )
    # Réexécuter le notebook réutilise le même jeu si sa configuration est identique.
    # Pour changer la configuration, choisir un nouveau DATASET_NAME.
    print("Données à créer :", DATASET_DIR)
    '''), cell("markdown", '''
    ## Un objet et des cartes connus

    Le fantôme est une ellipse asymétrique à amplitude lisse, sans données patient.
    Ses coins ne contiennent aucun objet : ils permettent de mesurer le bruit.
    B0 est en Hz, les positions en mètres et R2* en s⁻¹. L'origine est le pixel
    `n//2`, comme dans le pipeline. Le modèle représente une coupe intégrée sur 5 mm.
    '''), cell("code", '''
    phantom = make_phantom(CONFIG.object_matrix, 0.2,
                          b0_scale=CONFIG.b0_scale, ncoils=CONFIG.receiver_count)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    panels = [(phantom.proton_density, "Amplitude à RF", "gray"),
              (np.where(phantom.support_mask, phantom.B0_Hz, np.nan), "B0 connu [Hz]", "RdBu_r"),
              (np.where(phantom.support_mask, phantom.R2star_sInv, np.nan), "R2* connu [s⁻¹]", "magma")]
    for ax, (image, title, cmap) in zip(axes, panels):
        im = ax.imshow(image, origin="lower", cmap=cmap, extent=[-100, 100, -100, 100])
        ax.set(title=title, xlabel="X [mm]", ylabel="Y [mm]")
        fig.colorbar(im, ax=ax, shrink=.8)
    plt.show()
    plt.close(fig)
    '''), cell("markdown", '''
    ## Lire les ADC réels et simuler avec MRzeroCore

    Les gradients deviennent des moments intégrés en cycles/m et chaque événement
    atteint exactement un centre ADC. Le temps de relaxation et la phase B0 partent
    du centre RF associé à cet échantillon, y compris les lignes EPI négatives.

    MRzero reçoit des points stationnaires, une excitation idéale de 90° et
    `T2 = 1/R2*`. Les excitations GRE sont indépendantes et parfaitement spoilées.
    Le contraste de saturation T1, le profil RF et les imperfections de gradients
    sont laissés hors de ce test. Les `.seq` fournissent ici le k-space et le timing,
    pas un modèle complet des impulsions sélectives.

    Le simulateur est indépendant de `NativeEPITwin`, qui sera testé dans V4.
    Treize ADC par acquisition sont comparés à une équation analytique indépendante
    en double précision pour vérifier phases, relaxation, unités et réception.
    '''), cell("code", '''
    source_sequences = {
        "GRE": next((SOURCE_DIR / "sequences").glob("gre_32*.seq")),
        "B0": SOURCE_DIR / "sequences/b0_map.seq",
        "EPI": next((SOURCE_DIR / "sequences").glob("*epi*.seq")),
    }
    for name, path in source_sequences.items():
        samples = read_adc_samples(path)
        print(name, "matrice", samples["matrix_size"], "TE [ms]", samples["TE_s"] * 1000,
              "TR [ms]", samples["TR_s"] * 1000,
              "ADC", samples["elapsed_s"].shape, "navigateurs", samples["navigator_count"])
    '''), cell("code", '''
    from dataclasses import asdict
    if DATASET_DIR.exists():
        manifest = json.loads((DATASET_DIR / "acquisition.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "mrzero_synthetic_twix" or manifest["config"] != asdict(CONFIG):
            raise ValueError("Configuration différente : choisir un nouveau DATASET_NAME.")
        verify_acquisition_files(DATASET_DIR, manifest)
        print("Acquisition Twix existante réutilisée ; hashes vérifiés.")
    else:
        DATASET_DIR, manifest = generate_acquisition(SOURCE_DIR, DATASET_DIR, CONFIG)
    for kind, report in manifest["reports"].items():
        assert (DATASET_DIR / manifest["files"][kind]).suffix == ".dat"
        print(manifest["files"][kind], "|", report["stored_shape"], "| MRzero :", report["device"],
              "| erreur analytique :", f"{report['analytic_relative_error']:.3g}")
    '''), cell("markdown", '''
    ## Repasser par les reconstructions du pipeline

    `read_twin_inputs` lit directement les trois `.dat` avec les lecteurs Twix
    habituels : reconstruction continue par récepteur, correction des navigateurs
    et estimation B0. Les blocs binaires conservent les compteurs de lignes et
    d’échos, les indicateurs REFLECT et les trois navigateurs PHASCOR.
    Leurs en-têtes sont créés pour ce fantôme et marqués MRZERO_SYNTHETIC.
    '''), cell("code", '''
    inputs = read_twin_inputs(DATASET_DIR)
    print("GRE", inputs["gre"]["magnitude"].shape, "EPI", inputs["scan"].shape,
          "B0", inputs["b0_raw"]["coil_images"].shape)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, image, title, cmap in zip(axes,
        [inputs["gre"]["magnitude"], inputs["field"]["b0Hz"], inputs["scan"]],
        ["GRE synthétique 32²", "B0 réestimé [Hz]", "EPI synthétique 64²"],
        ["gray", "RdBu_r", "gray"]):
        im = ax.imshow(image, origin="lower", cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.8)
    plt.show()
    plt.close(fig)
    '''), cell("markdown", '''
    ## Comparer aux cartes connues et au moteur de V4

    Les erreurs B0/R2* sont calculées dans un intérieur érodé, car un pixel de bord
    mélange des contributions de différentes amplitudes et fréquences.
    Les estimations utilisent uniquement les échos synthétiques reconstruits.

    Le contrôle suivant fournit ensuite les **cartes exactes** au moteur de V4,
    pour vérifier séparément son signal EPI et celui du générateur. Le signal lu
    contient le bruit ajouté ; leur différence doit être de l'ordre de ce bruit.
    Ce contrôle utilise les ADC bruts avant correction des navigateurs. Le
    changement introduit par cette correction est mesuré séparément.
    Ce contrôle technique n'est pas une validation de l'estimation des cartes.
    Le notebook 2 utilise uniquement les cartes estimées, comme V4.
    '''), cell("code", '''
    map_validation, fit = evaluate_maps(inputs, phantom)
    truth_epi, signal_validation = compare_truth_epi(
        inputs, phantom, device=CONFIG.device, adc_chunk=CONFIG.adc_chunk)
    validation = dict(maps=map_validation, epi_signal=signal_validation,
                      generation=manifest["reports"], independent_scanner_validation=False)
    print(json.dumps(serialisable(validation), indent=2, ensure_ascii=False, allow_nan=False))
    (METADATA_DIR / "generation_validation.json").write_text(
        json.dumps(serialisable(validation), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    '''), cell("markdown", '''
    ## Tester maintenant V4

    Ouvrir `02_tester_pipeline_V4.ipynb` et exécuter toutes les cellules avec le même noyau.
    Cette copie reprend les calculs du notebook V4 ; seuls les chemins, les
    métadonnées d'audit et les libellés de données sont adaptés au jeu synthétique.
    Son dossier de sortie est distinct du dossier d'acquisition généré.

    L'acquisition créée contient `reference.dat`, `b0_map.dat`, `scan.dat`,
    les trois séquences, le manifeste et éventuellement `ACDC_3T.npz`. Les cartes
    de vérité terrain et diagnostics sont dans `simulation_metadata/<DATASET_NAME>/`.
    Elle ne contient aucun sous-dossier `results` de l'acquisition réelle.
    ''')]
    write_notebook(HERE / "01_generer_acquisition_MRzero.ipynb", cells)


def build_pipeline_copy():
    source_path = VALIDATION / "04_acdc_hardware_continuous_MRzero_V4.ipynb"
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    cells = deepcopy(source["cells"])
    cells[0] = cell("markdown", '''
    # 2 — Tester le pipeline V4 sur l'acquisition synthétique

    Ce notebook reprend les cellules numériques de
    `04_acdc_hardware_continuous_MRzero_V4.ipynb`. Les chemins et l'audit des
    métadonnées sont adaptés : les headers de l'acquisition réelle 015 ne sont
    pas utilisés pour décrire le fantôme. La vérité terrain n'entre pas dans
    les reconstructions, l'estimation des cartes ou le choix des variantes.

    Exécuter d'abord `01_generer_acquisition_MRzero.ipynb`.
    ''')
    setup = "".join(cells[1]["source"])
    setup = setup.replace('PROJECT_ROOT = Path.cwd() if (Path.cwd()/"Experiment").is_dir() else Path.cwd().parent',
        'PROJECT_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)\n'
        '                    if (p/"Experiment").is_dir() and (p/"MRzero_validation").is_dir())')
    setup = setup.replace('INPUT_DIR = PROJECT_ROOT/"acquisition_015"',
        'WORK_DIR = PROJECT_ROOT/"Mrzero_simulation"\n'
        'DATASET_NAME = "acquisition_synthetic_twix_001"\nINPUT_DIR = WORK_DIR/DATASET_NAME')
    setup = setup.replace('PROJECT_ROOT/"MRzero_validation/results_experiment_realistic_v4"',
        'WORK_DIR/"pipeline_validation"')
    setup = setup.replace('"twin015_%Y%m%d_%H%M%S_%f"', '"synthetic_%Y%m%d_%H%M%S_%f"')
    cells[1] = cell("code", setup)
    reader = "".join(cells[3]["source"])
    reader = reader[:reader.index('audit = json.loads')]
    reader += '''manifest = inputs["manifest"]
print("Données synthétiques :", manifest["format"])
print("Modèle RF :", manifest["model"]["RF"])
print("Vérité terrain utilisée par ce pipeline :", manifest["ground_truth_used_by_pipeline"])
assert not manifest["ground_truth_used_by_pipeline"]
'''
    cells[3] = cell("code", reader)
    for item in cells:
        text = "".join(item["source"])
        text = text.replace("EPI réelle sans ACDC", "EPI synthétique sans ACDC")
        text = text.replace("EPI mesurée, repère commun", "EPI synthétique, repère commun")
        text = text.replace('"Réel x=', '"Synthétique x=')
        text = text.replace("maximum réel", "maximum synthétique")
        text = text.replace("Trois échos mesurés", "Trois échos synthétiques")
        text = text.replace('"Comparaison à la mesure"', '"Comparaison aux ADC synthétiques"')
        item["source"] = text.splitlines(keepends=True)
        if item["cell_type"] == "code":
            item.update(outputs=[], execution_count=None)
    cells.append(cell("markdown", '''
    ## Interpréter ce test

    Les différences entre le meilleur jumeau et l'EPI synthétique mesurent ici
    les erreurs de reconstruction, de couverture, de fit et d'extrapolation du
    pipeline. Même avec un simulateur exact, des échos courts n'identifient pas
    parfaitement la décroissance jusqu'à 98 ms. La régularisation B0 change aussi
    légèrement le champ connu.

    Le notebook de génération a vérifié séparément le moteur EPI avec les cartes
    exactes. Ce test de cohérence numérique ne démontre pas la fidélité du modèle
    aux acquisitions réelles, au profil RF ou aux imperfections des gradients.
    '''))
    metadata = source.get("metadata", {})
    metadata["synthetic_source"] = dict(file=source_path.name,
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        adaptations=["project/input/output paths", "replace measured header audit by synthetic manifest", "plot labels"])
    write_notebook(HERE / "02_tester_pipeline_V4.ipynb", cells, metadata)


if __name__ == "__main__":
    build_generator()
    build_pipeline_copy()
    print("Created 01_generer_acquisition_MRzero.ipynb and 02_tester_pipeline_V4.ipynb")
