# Acquisition synthétique MRzero pour tester V4

Exécuter **01_generer_acquisition_MRzero.ipynb**, puis
**02_tester_pipeline_V4.ipynb**. Le premier crée une acquisition contenant de
vrais fichiers binaires Siemens Twix `.dat`. Le second les lit avec les lecteurs
Twix habituels du pipeline V4.

Tous les fichiers décrits ici se trouvent dans
`epi_acdc_artifactCorr_MrZero/Mrzero_simulation`.

## Fichiers et organisation

| Fichier | Rôle |
|---|---|
| `01_generer_acquisition_MRzero.ipynb` | Paramètres, fantôme, simulation des trois acquisitions, export Twix et contrôles |
| `02_tester_pipeline_V4.ipynb` | Calculs de V4 appliqués aux fichiers `.dat` générés |
| `synthetic_phantom.py` | Fantôme analytique et cartes de vérité terrain |
| `synthetic_acquisition.py` | Lecture des ADC Pulseq, simulation MRzero, bruit et génération du dossier |
| `synthetic_twix.py` | Construction des en-têtes synthétiques et écriture Twix avec `twixtools.write_twix` |
| `synthetic_inputs.py` | Utilitaires des jeux synthétiques ; lecture des anciens jeux NPZ conservée |
| `synthetic_validation.py` | Comparaison des cartes et du signal EPI aux valeurs connues |
| `build_synthetic_notebooks.py` | Construction reproductible des notebooks à partir du notebook V4 source |
| `verify_synthetic_outputs.py` | Vérification des sorties exécutées et des calculs repris de V4 |

Le jeu créé par défaut est organisé comme une acquisition du projet :

```text
acquisition_synthetic_twix_001/
    reference.dat           # GRE : [phase=32, receiver=1, readout=32]
    b0_map.dat              # B0 : [echo=3, receiver=1, phase=128, readout=128]
    scan.dat                # EPI : 64 × 64 et trois navigateurs
    ACDC_3T.npz             # Calibration de l'exemple, facultative
    sequences/              # Les trois fichiers .seq de l'exemple
    acquisition.json        # Paramètres, provenance et SHA-256 ; schéma 2

simulation_metadata/acquisition_synthetic_twix_001/
    ground_truth.npz        # Amplitude, B0, R2*, coordonnées et réception connus
    generation_validation.json

pipeline_validation/<session>/
    ...                     # Sorties du notebook 2
```

Les données acquises sont donc dans `reference.dat`, `b0_map.dat` et `scan.dat`.
Les NPZ restants contiennent la calibration ou la vérité terrain, pas les signaux
acquis. La vérité terrain et les diagnostics sont placés à côté du dossier
d'acquisition ; aucun dossier `results` de l'exemple n'est copié.

`acquisition.json` porte le format `mrzero_synthetic_twix`. Les anciens jeux
`acquisition_synthetic_001` en NPZ restent des sorties historiques ; ils ne sont
plus le jeu par défaut des notebooks.

## Comment les fichiers `.dat` sont créés

Les séquences de `acquisition_015` fournissent les coordonnées ADC, leurs
instants au centre des fenêtres et les phases RF/ADC. MRzero simule tous les
échantillons complexes, y compris les navigateurs EPI. Un bruit complexe
gaussien reproductible est ensuite ajouté.

`synthetic_twix.py` construit les en-têtes texte Siemens, les en-têtes de mesure
(MDH), les compteurs de lignes/échos et les indicateurs `REFLECT` et `PHASCOR`.
`twixtools.write_twix` écrit ces structures et les signaux dans le format binaire
Twix. Aucun en-tête ni signal d'un `.dat` réel n'est lu ou copié. Les fichiers
obtenus sont synthétiques et destinés au test logiciel du pipeline ; ils ne
proviennent pas d'un scanner.

Le notebook V4 original reste intact. Pour ce nouveau jeu, les fonctions
habituelles `read_scan` et `read_b0_coil_images` ouvrent les `.dat` ; elles
n'utilisent pas l'ancien adaptateur des données NPZ. Cela teste aussi le
décodage Twix, le classement des lignes et échos, le traitement des lignes
inversées et la correction par les navigateurs.

## Modèle et conventions

Le fantôme possède une forme asymétrique, une amplitude lisse, un champ B0
variable et un R2* connu. Son support est éloigné des coins où V4 estime le
bruit. Un récepteur uniforme est utilisé par défaut.

| Acquisition | Matrice | TE [ms] | TR [ms] |
|---|---:|---|---:|
| GRE de référence | 32 × 32 | 4,1173 | 1000 |
| Cartographie B0 | 128 × 128 | 2,1231 / 3,9431 / 5,7631 | 9,19 |
| EPI | 64 × 64 | 98,4395 | 185,94 |

Le FOV est de 200 × 200 mm et la coupe représente une amplitude intégrée sur
5 mm. Le générateur appelle directement `MRzeroCore.compute_graph` et
`execute_graph`, indépendamment de `NativeEPITwin`, le moteur testé dans V4.

Le modèle emploie des spins ponctuels stationnaires et une excitation idéale
indépendante de 90° par histoire RF. MRzero applique `T2 = 1/R2*` et une phase :

```text
PD × poids_surface × exp(-R2* × t)
   × exp[-2πi × (kx × X + ky × Y + B0 × t)] × exp[i × (phase_RF - phase_ADC)]
```

Les coordonnées sont en mètres, k en cycles/m, B0 en Hz et t en secondes depuis
le centre RF correspondant. L'origine de grille est `n//2`. Un découpage en
blocs de calcul ne redémarre ni la phase ni la relaxation à chaque écho.

Les poids de surface `aire_pixel_objet / aire_pixel_acquis` valent ici 1/16
pour la GRE 32², 1 pour B0 128² et 1/4 pour l'EPI 64². Ils maintiennent la
cohérence d'amplitude entre résolutions. Le bruit par composante réelle ou
imaginaire est adapté à chaque matrice pour obtenir un niveau voisin dans les
images reconstruites.

Les trois navigateurs sont simulés et leur correction est effectivement
appliquée par le lecteur EPI natif. Aucune erreur matérielle pair/impair n'est
injectée. Le contrôle du signal MRzero utilise le k-space brut avant cette
correction ; l'effet de la correction est mesuré séparément.

V4 réestime B0 par les différences de phase des échos, puis l'amplitude et R2*
par leurs magnitudes. Il ne lit pas la vérité terrain. Celle-ci sert seulement
aux diagnostics du notebook 1.

Le modèle ne décrit pas le profil RF, le contraste lié à l'angle nominal, la
saturation T1, l'état stationnaire GRE, les imperfections des gradients, le
mouvement ou la dispersion B0 suivant z. Les résultats évaluent le pipeline
sous ces hypothèses. Les API sont décrites dans la
[documentation MRzeroCore](https://mrsources.github.io/MRzero-Core/).

## Exécution avec le venv demandé

Depuis `F:\Documents_bis\Martinos_center\epi_acdc_artifactCorr_MrZero`, dans
PowerShell :

```powershell
.\.venv-gpu-win\Scripts\python.exe .\MRzero_validation\execute_notebook.py .\Mrzero_simulation\01_generer_acquisition_MRzero.ipynb --timeout 1200
.\.venv-gpu-win\Scripts\python.exe .\MRzero_validation\execute_notebook.py .\Mrzero_simulation\02_tester_pipeline_V4.ipynb --timeout 1200
```

Le lanceur emploie son propre interpréteur pour le noyau, sans installer de
kernelspec global. En mode interactif, sélectionner
`F:\Documents_bis\Martinos_center\epi_acdc_artifactCorr_MrZero\.venv-gpu-win\Scripts\python.exe`.
Le notebook choisit CUDA si disponible ; `device="cuda"` permet de l'exiger.

Le venv Windows existant contient Python 3.11.9, MRzeroCore 1.1.1, PyPulseq
1.5.0.post1 et PyTorch 2.6.0+cu118 ; la machine dispose d'une GTX 1660 Ti.

Pour refaire un jeu, changer `DATASET_NAME` et les paramètres dans le notebook 1,
puis reporter ce nom dans le notebook 2. Une acquisition existante n'est jamais
écrasée ; à configuration identique elle est réutilisée et ses hashes contrôlés.
Conserver `object_matrix=128` pour comparer directement aux cartes B0 natives.
Mettre `b0_scale=0` pour un contrôle sans distorsion, ou modifier le bruit et la
relaxation dans les paramètres et `synthetic_phantom.py`.

`copy_calibration=False` évite la copie d'environ 396 Mo, inutile pour V4.
Garder `receiver_count=1` pour le premier test : les variantes V4 ne reçoivent
pas les sensibilités des récepteurs supplémentaires.

Les contrôles de la génération sont enregistrés dans
`simulation_metadata/<DATASET_NAME>/generation_validation.json`. Les contrôles
des notebooks se trouvent dans `synthetic_notebooks_validation.json` et
`synthetic_unit_validation.json`. Rebâtir les notebooks avec
`build_synthetic_notebooks.py` efface leurs sorties exécutées ; le fichier
source V4 et son hash sont enregistrés dans la copie.
