# Shimming dynamique ACDC : reconstruire et vérifier dans le notebook

Le point d'entrée est
[02_acdc_experiment_step_by_step.ipynb](02_acdc_experiment_step_by_step.ipynb).
Les chemins et paramètres sont dans ses premières cellules. Exécuter les
cellules dans l'ordre permet de voir les données, les figures et les
contrôles avant chaque calcul suivant. **Il n'est pas nécessaire de créer
ou modifier un fichier JSON.** Les notebooks, les commentaires du code et
les figures sont en anglais ; ce guide est en français.

Le notebook réutilise les lecteurs Python existants, prépare des fichiers
MAT, puis appelle **le forward model et l'optimiseur existants dans
MATLAB**. Les résultats MATLAB sont relus et affichés dans le notebook.
L'export écrit uniquement **`waveform.csv`** après examen des résultats.
Les figures et paramètres restent dans le notebook ; les MAT et journaux
MATLAB utilisent un dossier de travail temporaire. Le pipeline ne se
connecte pas au FPGA et ne produit aucun script de commande.

La version [02_acdc_experiment_step_by_step_v2.ipynb](02_acdc_experiment_step_by_step_v2.ipynb)
prend aussi en compte la réponse temporelle des canaux ACDC pour l'EPI.
Les réglages et les figures de ce modèle sont décrits dans la section
[Réponse temporelle des canaux ACDC](#réponse-temporelle-des-canaux-acdc-dans-le-notebook-v2).

## 1. Rassembler les fichiers

Le dossier d'acquisition contient les données réellement acquises et
les séquences correspondantes :

```text
acquisition_001/
├── gre_reference.dat
├── b0_map.dat
├── scan.dat
├── ACDC_3T.npz
└── sequences/
    ├── scan.seq
    ├── reference_gre.seq
    └── b0_map.seq
```

Dans `acquisition_001`, la référence brute `gre_reference.dat` reste à
déposer. L'ancien `gre_reference.dcm` est conservé comme donnée historique,
mais ne remplace pas ce Twix. Fournir aussi le `reference_gre.seq` qui a
réellement servi à acquérir la nouvelle référence, même si ses paramètres
diffèrent de ceux de l'ancienne référence.

La ROI est définie dans une étape du notebook ; aucun masque préexistant
n'est nécessaire dans ce dossier. Les noms des données sont configurables
dans les cellules de paramètres. Le notebook n'infère
ni la géométrie ni le type de séquence à partir de leurs noms.

| Entrée actuelle | Rôle |
| --- | --- |
| `gre_reference.dat` | Twix brut de référence GRE/FLASH 2D mono-écho. Sa magnitude RSS reconstruite fournit l'amplitude du modèle. |
| `b0_map.dat` | Twix GRE 128 × 128 à trois échos, reconstruit en B0 en Hz. |
| `scan.dat` | EPI 32 × 32 avec trois navigateurs, conservée pour cet essai selon le choix de l'utilisateur. |
| `ACDC_3T.npz` | Volume des 32 réponses ACDC Bay5, importé directement en Hz/A. |
| `scan.seq` | Séquence de l'EPI à corriger. |
| `reference_gre.seq` | Séquence GRE/FLASH propre à la référence. |
| `b0_map.seq` | Séquence multi-écho propre à la mesure B0. |

Le recalage des cartes Bay5 et leur correspondance avec les bobines du
montage ont déjà été confirmés. Conserver cette confirmation pour ce
montage. Une autre bobine ou un autre placement demande son propre
recalage ; une simple modification de résolution n'en demande pas un nouveau.

## 2. Installer et ouvrir le notebook

Toutes les commandes partent de la racine du dépôt :

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r Experiment/requirements.txt
```

Python 3.11 ou plus récent est nécessaire. Ouvrir le notebook dans Jupyter
ou VS Code et sélectionner le Python de `.venv`. Les sorties des cellules,
les figures et les tableaux sont conservés dans le fichier `.ipynb`.

L'exécution complète depuis la ligne de commande est possible :

```bash
.venv/bin/python Experiment/execute_notebook.py \
  Experiment/02_acdc_experiment_step_by_step.ipynb
```

Le script utilise le Python qui le lance et conserve les sorties. Pour
un premier essai ou une modification d'acquisition, l'exécution cellule
par cellule permet de lire les contrôles au moment où ils sont produits.

Les étapes MATLAB exigent un exécutable accessible, avec Optimization
Toolbox et Parallel Computing Toolbox. Le CPU suffit ; le GPU est
facultatif. Indiquer dans `MATLAB_EXECUTABLE` un chemin valide vers le
fichier exécutable MATLAB, par exemple le chemin absolu de l'installation.
Le nom `matlab` seul n'est pas recherché dans le `PATH` par ce notebook.
Une erreur de lancement laisse les figures Python
consultables et le journal de l'étape permet d'identifier l'installation
ou la licence manquante.

## 3. Choisir les paramètres dans les cellules

Choisir d'abord le dossier d'entrée et un **nouveau dossier de sortie**.
Le résultat doit rester associé aux paramètres et aux sources qui l'ont
produit. Après un changement de fichier, de masque ou de modèle, choisir
une nouvelle sortie et repartir des cellules concernées.

Les réglages se répartissent en quatre groupes :

| Groupe | Ce qui se renseigne |
| --- | --- |
| Données | Chemins de la référence, du Twix B0, du scan, des cartes ACDC et des trois `.seq`. |
| Reconstruction et géométrie | Seuils B0, multiplicateur de signe B0, seuil de référence GRE, unités, repère et transformation de calibration connue. |
| Calcul MATLAB | Exécutable, cible de MSE relative, limite de champ ACDC, nombre maximal d'itérations, CPU/GPU, nombre de workers et de threads, bornes de courant. |
| CSV | Correspondance et polarité des 32 canaux, bornes de courant, décalage temporel facultatif et intervalle minimal facultatif entre changements. |

Dans la cellule **1. Imports and paths**, modifier `INPUT_DIR` et `FILES`.
Le notebook ne charge plus l'ancien masque `roi_b0_reliable.npy`.
La nouvelle étape **4b. Define and inspect the ROI**, placée après la reconstruction de la référence,
permet de construire et examiner la région avant les contrôles B0.
La sortie `OUTPUT_DIR` utilise un nom `notebook_<date_et_heure>` sous
`Experiment/acdc_output`. Le dossier est créé seulement lorsque le CSV
peut être exporté. `SCRATCH_DIR = Path("/scratch/mv1289")` définit le
répertoire des fichiers temporaires. Le notebook le crée si nécessaire ;
`WORK_DIR` est un sous-dossier temporaire de ce répertoire et contient les
échanges avec MATLAB. Le stockage temporaire des workers MATLAB utilise
également `SCRATCH_DIR`. Après un changement de ce chemin, réexécuter le
notebook depuis la cellule **1. Imports and paths** afin que le dossier de
travail et les requêtes MATLAB utilisent le nouveau répertoire.

La cellule **2. Experiment parameters** regroupe `B0_PARAMETERS`,
`CALIBRATION_PARAMETERS`, les paramètres MATLAB et `WAVEFORM_PARAMETERS`.
L'exécutable initial `MATLAB_EXECUTABLE` est celui de MATLAB R2026a
sur ce poste. Sur un autre ordinateur, le remplacer par son chemin réel.
Relire les valeurs de `TARGET_MSE`, `MAX_ITERATIONS`, `DEVICE`, `WORKERS` et
`THREADS_PER_WORKER` dans cette cellule : elles décrivent le calcul choisi
pour la session. `DEVICE` sélectionne le CPU ou le GPU. Si le pilote GPU
déclenche `parallel:gpu:device:OldDriver`, choisir `DEVICE = "cpu"`.
`DEVICE = "gpu"` demande une installation compatible avec MATLAB.
Le modèle et ses contraintes restent les mêmes avec le CPU.

Après un changement, réexécuter la cellule de paramètres, puis la cellule
qui commence par `# The bridge only orchestrates the existing MATLAB functions.`
Elle écrit la requête MAT depuis les valeurs actuellement exécutées.
Relancer ensuite l'étape MATLAB voulue. Réexécuter le notebook depuis
les paramètres réalise aussi ces étapes ; relancer uniquement
l'optimisation peut conserver une ancienne requête MAT.

Les trois interrupteurs de calcul valent initialement `True` :

| Paramètre | Pour examiner les reconstructions | Pour examiner le forward model | Parcours complet |
| --- | --- | --- | --- |
| `RUN_FORWARD_MODEL` | `False` | `True` | `True` |
| `RUN_OPTIMIZATION` | `False` | `False` | `True` |
| `EXPORT_WAVEFORM` | `False` | `False` | `True`, seulement si les contrôles du calcul et du CSV réussissent. |

Ces réglages n'effacent pas les figures des étapes précédentes. L'export
écrit un seul CSV ; aucun programme de contrôle matériel n'est appelé.

La matrice, le FOV, TE, TR et dwell sont lus dans les fichiers. Ils ne
sont pas saisis une seconde fois à la main. La référence et le scan ont
leur propre grille ; les 32 canaux ACDC ne sont pas une dimension d'image.

### Seuils de reconstruction B0

Les paramètres initiaux sont :

| Paramètre | Valeur | Effet |
| --- | --- | --- |
| `magnitude_fraction` | `0.05` | Signal minimal entre échos, relatif à son maximum. |
| `max_phase_residual_rad` | `0.5` | Cohérence maximale tolérée entre incréments de phase. |
| `min_component_pixels` | `10` | Taille minimale d'une région connexe retenue. |
| `frequency_offset_hz` | `0` | Décalage connu indépendamment, s'il existe. |
| `b0_sign` | `1` | Multiplicateur appliqué une seule fois à la convention du lecteur. |

Ces seuils définissent la fiabilité de la mesure. Les modifier change la
carte utilisée ; examiner les magnitudes, les résidus et les masques avant
de retenir un nouveau réglage. Un seuil moins strict ne crée pas une
mesure fiable dans une zone de faible signal.

Le lecteur Twix utilise la convention du modèle
`signal(TE) = amplitude × exp(-2π i B0 TE)`. Le multiplicateur `b0_sign`
permet la convention vérifiée expérimentalement. Le signe ne doit pas
être choisi après coup pour améliorer une simulation.

Les échos actuels ont ΔTE ≈ 1,820 ms. La bande principale au voxel
d'ancrage est donc ±274,7 Hz ; deux branches diffèrent d'environ 549,45 Hz.
Chaque région connexe est déroulée séparément et ancrée sur son pixel de
plus fort signal. Le déroulement peut produire ailleurs des valeurs hors
bande principale. Il ne mesure pas la branche absolue ni le décalage
relatif entre régions disjointes. `frequency_offset_hz` sert uniquement
à consigner une correction connue par une mesure indépendante.

### Région d'intérêt et limites de courant

Exécuter l'étape **4b. Define and inspect the ROI** après avoir examiné
la référence. Elle construit d'abord le support avec
`reference_magnitude > REFERENCE_THRESHOLD * reference_magnitude.max()`.
Le seuil `REFERENCE_THRESHOLD` retient les pixels dont la magnitude
dépasse cette fraction du maximum. Par exemple, `0.1` signifie 10 %.
C'est un seuil de signal GRE, indépendant de la qualité B0.

Dans la cellule ROI, les deux réglages suivants permettent de restreindre
le support dans les coordonnées physiques de la référence :

| Paramètre | Valeur initiale | Utilisation |
| --- | --- | --- |
| `ROI_X_LIMITS_MM` | `None` | Conserver tout l'axe X, ou renseigner `(minimum, maximum)` en mm. |
| `ROI_Y_LIMITS_MM` | `None` | Conserver tout l'axe Y, ou renseigner `(minimum, maximum)` en mm. |

Avec les deux valeurs `None`, la ROI correspond à tout le support GRE
obtenu par seuil. La figure affiche le masque binaire, sa superposition
verte sur la référence et les pixels du support exclus par les limites.
Le nombre de pixels et la part du signal exclue sont également affichés.
Modifier le seuil ou les limites, puis réexécuter la cellule pour vérifier
la sélection. Le masque reste en mémoire et est repris dans l'entrée
temporaire MATLAB. Aucun fichier de masque n'est ajouté à la sortie finale.

La ROI choisie et le masque de fiabilité B0 restent distincts : ils
décrivent respectivement l'objet visé et la qualité de sa mesure de champ.
Le contrôle de couverture ultérieur montre les pixels de ROI sans B0
fiable et arrête la préparation s'il en existe. Il ne modifie pas la ROI
pour contourner ce problème.

Les bornes numériques déjà choisies sont **1 A par canal** et
**somme absolue des courants <30 A par état**. Elles peuvent servir au
calcul numérique sans valider le câblage ni le timing du contrôleur.
La limite `FIELD_LIMIT_HZ` porte sur le champ **produit par l'ACDC**,
pas sur le résidu `B0 + champ ACDC`. Relire sa valeur dans les paramètres
de la session.

## 4. Reconstruire et examiner les données Python

Chaque groupe de cellules explique les tableaux utilisés, affiche une
figure et donne des nombres permettant de repérer une incohérence.
Les premières étapes ne dépendent pas de MATLAB ni du contrôleur.

| Étape | Figures à lire | Contrôle à effectuer |
| --- | --- | --- |
| Fichiers et séquences | Tableau des tailles et temps ; trajectoire ADC et calendrier RF/ADC. | Les `.seq` doivent correspondre aux acquisitions, avec matrice, FOV, TE, TR, dwell et ordre des lignes cohérents. |
| Référence | K-espace brut, magnitude et phase d'un canal, magnitude RSS et coordonnées physiques. | Vérifier matrice/FOV/timing face à `reference_gre.seq`, orientation, coupe et région visée. |
| Définition de la ROI | Masque binaire, superposition verte sur la GRE et pixels exclus. | Choisir le seuil et, si nécessaire, les limites X/Y en mm ; vérifier la région retenue avant les contrôles B0. |
| Scan | K-espace et image reconstruite ; correction des navigateurs pour l'EPI. | Vérifier sens de lecture, fantômes pair/impair et combinaison RSS. |
| Échos B0 | Magnitude de chaque écho et k-espace. | Un écho absent ou de faible signal est visible avant le calcul B0. |
| Phase B0 | Différences de phase entre échos et résidu. | Repérer phases incohérentes, trous de signal et désaccord entre paires. |
| Masques B0 | Seuil de magnitude, seuil de résidu, filtre de composantes et masque final. | Identifier précisément quel contrôle exclut une zone. |
| Carte B0 | B0 masquée, régions connexes et distribution des fréquences. | Vérifier champ, unités, signe et hypothèses de branche. |
| Calibration | Mosaïque des réponses ACDC et couverture de calibration. | Vérifier les 32 cartes, leur signe, leur repère et leur couverture. |
| Grille commune et ROI | B0 et calibration sur la référence, contours du support et exclusions. | Toute la région choisie doit être couverte par des mesures fiables. |

La figure du filtrage des composantes B0 compare le masque avant et après
le filtre de taille des régions, ainsi que les pixels supprimés. Elle
permet de distinguer ce filtre des seuils de magnitude et de résidu.

### Vérifier la correction des navigateurs EPI dans la version v3

L'étape 5 estime séparément les phases des paires de navigateurs 2–1 et
2–3, puis les moyenne. Elle évite la moyenne complexe des navigateurs
positifs : ceux-ci peuvent presque s'annuler lorsque le B0 fait évoluer
leur phase entre les lectures. Le masque et les poids utilisent le signal
des trois échos séparément. Le déroulement de phase et l'ajustement affine
se limitent à la région continue la plus fiable de chaque récepteur.

Inspecter les images avant/après sur leur échelle commune, le signal des
navigateurs, les phases des deux paires, les pixels retenus et le résidu
de l'ajustement. Un récepteur avec moins de quatre points fiables
consécutifs conserve une correction nulle et apparaît dans le tableau.
Les temps issus du `.seq` doivent donner trois échos également espacés.
Le B0 est supposé approximativement constant pendant ces trois échos.

`NAVIGATOR_PHASE_BRANCH = 0` choisit la petite différence de phase
pair/impair au point de signal le plus fort ; `1` ajoute π et sélectionne
l'autre branche. Trois navigateurs ne lèvent pas seuls cette ambiguïté.
Il faut vérifier cette convention expérimentalement. Le même paramètre
s'applique aux scans sans et avec ACDC, chacun avec son propre ajustement.
La reconstruction ne recale pas l'EPI sur la GRE : le déplacement et les
distorsions dus au B0 restent visibles pour évaluer le shimming.

La cellule de l'étape 5 recharge le lecteur Python et remplace la fonction
importée dans le noyau. Cela évite l'erreur « unexpected keyword argument
'navigator_phase_branch' » lorsqu'une ancienne version reste en mémoire.
Réexécuter cette cellule suffit à rafraîchir `read_scan`, sans recréer le
dossier temporaire MATLAB. La première cellule recharge aussi les lecteurs
lors d'une exécution complète. La comparaison des résultats acquis se
trouve à l'étape 14 ; réexécuter ses cellules pour recalculer les RMSE
après une nouvelle reconstruction.

### Distinguer le déplacement B0 du fantôme résiduel dans v3

L'étape **8b**, après la reconstruction et le rééchantillonnage de la carte
B0, propose un diagnostic activé par `SHOW_MEAN_B0_DIAGNOSTIC = True`.
Elle calcule le champ moyen uniquement sur les pixels fiables de la ROI
et affiche leur nombre, la moyenne, l'écart-type et le déplacement attendu
dans la direction d'encodage de phase.

Pour une EPI de 64 lignes avec un espacement de 2,710 ms, une fréquence
moyenne de 94 Hz représente environ 16 pixels de déplacement. Le phantom
peut alors franchir le bord inférieur et réapparaître en haut. Ce repliement
n'est pas à lui seul une copie pair/impair.

La cellule retire la phase correspondant à cette fréquence aux échantillons
bruts d'imagerie **et** aux navigateurs, à leurs temps ADC. Elle réestime
ensuite la correction des navigateurs. Les deux images sont affichées avec
la même échelle. Ce calcul suppose le signe, la branche de fréquence et la
géométrie B0 déjà vérifiés. Il retire un seul offset constant ; les variations
spatiales du champ et un fantôme résiduel à demi-champ peuvent subsister.

L'image obtenue est `mean_b0_diagnostic_image`. Elle sert uniquement à
comprendre les artefacts : `scan_image`, les entrées MATLAB, les exports
CSV et les RMSE de la comparaison finale restent inchangés. La carte B0
sans shim n'est pas appliquée au scan corrigé, dont le champ change avec
les courants ACDC. Passer le paramètre à `False` désactive ce diagnostic ;
les scans GRE sont ignorés automatiquement.

### Reconstruire la référence Twix

Le lecteur de référence réutilise la reconstruction cartésienne du scan :
il remet les lignes acquises dans leur ordre spatial, corrige le sens de
lecture indiqué par les données, puis applique une transformée de Fourier
inverse centrée à chaque canal de réception. La magnitude finale vaut
`RSS = sqrt(sum(abs(coil_images)**2, axis=receiver))`. Les figures permettent
ainsi de distinguer un problème dans le k-espace, la reconstruction d'un
canal ou la combinaison des canaux.

Le Twix de référence doit être une GRE/FLASH classique 2D mono-écho,
avec une RF et une lecture par ligne, une grille cartésienne complète
et une seule image. Son propre `reference_gre.seq` fournit la matrice,
le FOV et le calendrier à vérifier contre les données. La référence peut
avoir une matrice et un FOV différents de ceux du scan à corriger ;
les cartes B0 et ACDC seront rééchantillonnées dans ses coordonnées
physiques, sans extrapolation.

L'adaptateur actuel suit le protocole axial centré déjà utilisé : axe
modèle `X = patient Y`, axe modèle `Y = -patient X`. Il contrôle la
position et l'orientation de coupe dans les en-têtes Twix. Les figures
utilisent les coordonnées physiques associées aux lignes et colonnes ;
une rotation visuelle n'est pas une preuve de recalage. Une orientation
ou un déplacement non pris en charge est refusé avant MATLAB.

Les images du scan combinent les canaux de réception IRM en RSS. Leur
nombre est lu dans le Twix et ne désigne pas le nombre de canaux de shim.
La carte B0 est calculée à partir des produits conjugués d'échos voisins,
combinés entre canaux après annulation de leur phase statique.

### Lire la couverture de la ROI

Les nombres de pixels sont recalculés sur la nouvelle référence Twix
et avec les paramètres de la session. Comme `gre_reference.dat` n'est
pas encore présent dans `acquisition_001`, aucune nouvelle mesure de
couverture sur cette référence n'est disponible.

La figure de couverture B0 superpose en rouge les pixels de la ROI
sans B0 fiable. S'il en existe, la préparation s'arrête avant MATLAB.
Examiner les figures de magnitude, phase et masques pour comprendre
les exclusions B0. Si cette région est celle que l'on veut corriger,
il faut obtenir une mesure B0 fiable sur toute cette région. Si
l'expérience vise une région plus petite, la définir explicitement
dans la cellule ROI et vérifier sa figure. Le notebook ne réduit pas
automatiquement le support et ne complète pas les trous par des
valeurs inconnues.

L'interpolation utilise les coordonnées physiques sans extrapolation.
Un point B0 n'est conservé que si les valeurs contribuant à son
interpolation sont fiables. Le nombre de pixels fiables sur la grille
de référence ne se déduit donc pas d'un simple rapport de matrices.

Le contrôle historique sur l'ancien DICOM, avec un seuil GRE de 3 %,
avait sélectionné 359 pixels, dont 88 couverts par une B0 fiable. Ces
comptes ne décrivent pas la référence Twix ni les paramètres actuellement
choisis dans le notebook.

## 5. Préparer et vérifier l'entrée MATLAB

La préparation rassemble la référence normalisée, le scan reconstruit,
la B0, les coordonnées, la trajectoire ADC et les fenêtres de courant.
L'import de `ACDC_3T.npz` extrait les 32 réponses du volume sur la coupe
choisie. Le MAT nécessaire à MATLAB est produit automatiquement ; le
volume source reste inchangé.

La transformation patient → cartes Bay5 déjà confirmée est l'identité.
Les translations d'une transformation différente seraient exprimées en
mètres. Une confirmation de recalage enregistre une vérification physique ;
elle ne réalise pas ce recalage.

Relire les figures de l'entrée préparée avant d'appeler MATLAB : les
pixels du support doivent être inclus dans la qualité B0 et la couverture
ACDC. La référence et la carte B0 peuvent avoir une résolution différente
du scan, mais les coordonnées doivent décrire la même coupe.

Les modèles MATLAB utilisés restent ceux du dépôt, notamment
`forward_model_experiment.m`, `forward_model_experiment_acdc.m`,
`prepare_experiment_optimizer.m` et `run_experiment_fast_parallel.m`.
Le point de passage
[run_acdc_notebook_stage.m](run_acdc_notebook_stage.m) sépare la simulation
et l'optimisation et conserve un état et un message d'erreur par étape.

## Premier scan avec shim statique dans le notebook v3

Ouvrir [02_acdc_experiment_step_by_step_v3.ipynb](02_acdc_experiment_step_by_step_v3.ipynb)
et exécuter les cellules jusqu'aux sections **9b et 9c**, après la
validation de la ROI, de la B0 et des cartes ACDC. Ce calcul reprend
`solve_shim_static_qp` du dossier `ACDC_shim_static_dyn_haotian` ; il lui
fournit la cible **`-B0`** et les cartes sur la ROI sélectionnée pour
réduire `B0 + champ ACDC`. Il produit un courant constant par canal.
Le solveur reste Python, comme dans l'exemple d'origine ; le forward
model et l'optimisation de l'image restent MATLAB.

`RUN_STATIC_SHIM = True` lance le calcul et
`EXPORT_STATIC_CURRENTS = True` autorise l'export après les contrôles.
`STATIC_MAX_CURRENT_A` et `STATIC_TOTAL_CURRENT_LIMIT_A` reprennent par
défaut les limites choisies pour le dynamique. `quadprog` et `tqdm` sont
nécessaires ; les installer avec les autres dépendances :

```bash
.venv/bin/python -m pip install -r Experiment/requirements.txt
```

Examiner la B0 initiale, le champ ACDC prédit, le résidu et les 32 courants
avant l'export. Les RMS et NRMSE de cette étape décrivent le **champ prévu
une fois les courants établis**, et non une image mesurée ou la NRMSE des
images. Le tableau distingue les courants du modèle et ceux destinés aux
sorties physiques, avec les mêmes correspondances et polarités que le
CSV dynamique.

`static_currents.csv` est écrit dans `STATIC_OUTPUT_DIR`, un dossier frère
de la sortie dynamique terminé par `_static`. Il contient exactement
deux lignes sans en-tête, avec des timestamps en secondes et des courants
en ampères. Les courants des deux lignes sont identiques :

```text
0,I1,I2,...,I32
1,I1,I2,...,I32
```

Le dernier timestamp correspond à `STATIC_CSV_LAST_TIME_S`, initialement
1 seconde. Avec les 32 sorties par défaut, le fichier compte 33 colonnes.
Choisir ce timestamp pour couvrir **tout le scan après le trigger** du
premier readout d'imagerie, après les navigateurs. La première consigne
est appliquée à ce trigger puis atteint progressivement sa valeur ; la
phase accumulée auparavant reste présente. Il n'y a pas de ligne finale
à zéro : garder les courants pendant ce scan, puis les couper avec le
contrôleur externe avant de charger les waveforms dynamiques suivantes,
qui restent indépendantes. Cette étape exporte seulement le CSV et ne
contrôle pas le FPGA ; elle ne modifie ni la B0 d'entrée ni les calculs
ou l'export du parcours dynamique.

## Shimming dynamique du champ B0 dans le notebook v3

Les sections **9d, 9e et 9f**, juste après le shim statique, reprennent
`solve_shim_waveform_fista` du dossier `ACDC_shim_static_dyn_haotian`.
Elles calculent **une consigne par canal et par ligne EPI**, avec la même
ROI, la même calibration et les mêmes limites de courant. Avec l'EPI
32 lignes actuelle, le tableau des consignes a 32 états et 32 canaux.
Le nombre d'états suit le nombre de lignes lu dans la séquence.

Cette étape utilise la **carte B0 mesurée, constante pendant le scan**.
La cible `-B0` est répétée à chaque échantillon ADC d'imagerie ; aucune
variation temporelle de B0 n'est inventée. Les consignes peuvent néanmoins
varier pour compenser l'établissement du courant et les transitions entre états.
Si les courants sont déjà établis avant les ADC, les consignes peuvent
rester presque constantes. Le solveur minimise le champ résiduel
`B0 + champ ACDC` à **chaque échantillon ADC d'imagerie**. Il ne minimise
pas directement une erreur d'image. Le forward model et l'optimisation de l'image des sections MATLAB suivantes
restent indépendants et inchangés.

### Réglages de la section 9d

| Paramètre | Utilisation |
| --- | --- |
| `RUN_DYNAMIC_B0_SHIM` | `True` lance le calcul FISTA. |
| `EXPORT_DYNAMIC_B0_WAVEFORM` | `True` autorise l'écriture du CSV après les contrôles. |
| `B0_DYNAMIC_MAX_ITERATIONS` | `2000` itérations au maximum ; examiner le suivi et les résultats avant de retenir cette valeur. |
| `B0_DYNAMIC_RIDGE_X` | `1e-6`, régularisation des consignes dans le problème mis à l'échelle par le solveur Haotian. |
| `B0_DYNAMIC_TRIGGER_DELAY_S` | `0.0` seconde depuis le **début du premier bloc de readout d'imagerie**, après les navigateurs. Renseigner le délai réellement mesuré. |
| `B0_DYNAMIC_OUTPUT_DIR` | Dossier frère de `OUTPUT_DIR`, terminé par `_dynamic_b0`. |

Les paramètres `MAX_CURRENT_A`, `TOTAL_CURRENT_LIMIT_A` et
`FIELD_LIMIT_HZ` sont repris du notebook. La borne de champ porte sur le
champ **produit par l'ACDC**, pas sur le résidu B0. Les correspondances et
polarités des canaux sont celles de `WAVEFORM_PARAMETERS`.
`CURRENT_RESPONSE_ENABLED` active le premier ordre et
`CURRENT_RESPONSE_TAU_US` accepte un τ commun ou 32 valeurs dans l'ordre
des canaux de la calibration. Avec l'option désactivée, les transitions
sont instantanées.

Le trigger reste celui du **premier readout d'imagerie après les
navigateurs**. `B0_DYNAMIC_TRIGGER_DELAY_S` est mesuré depuis le début du
bloc de ce readout et doit être compris entre zéro et le premier
échantillon ADC. Il est distinct de `RF_DELAY_S`, utilisé par le parcours
MATLAB existant. Le courant initial vaut zéro ; la première consigne est
appliquée au trigger. La phase accumulée avant ce trigger reste présente,
et une réduction du champ aux ADC ne garantit pas son annulation.

### Lire les figures et exporter

La section 9e utilise la réponse exacte du premier ordre à tous les
échantillons ADC d'imagerie, en conservant le courant atteint lors des
transitions précédentes. Les temps réels des ADC dans la séquence
permettent de prendre en compte l'établissement du courant pendant les
premiers échantillons. Le solveur conserve une seule consigne par ligne,
pas une nouvelle consigne à chaque ADC.

La section 9f affiche les consignes, les courants prédits et les résidus
de champ. Les cartes aux centres des lignes servent à la visualisation ;
les RMS et NRMSE résument le **champ simulé dans la ROI à tous les ADC**.
Elles ne sont ni la MSE de magnitude de l'optimiseur MATLAB ni la NRMSE
des scans mesurés comparés à la GRE.

Avant l'export, le notebook vérifie les limites sur les consignes, sur les
courants et champs prédits à **tous les échantillons ADC**, ainsi que sur
les consignes après quantification nominale. Ces contrôles ne constituent
pas une mesure des transitoires analogiques réels. La réponse du premier
ordre doit toujours être validée expérimentalement.

La sortie est indépendante du CSV statique et du CSV de l'optimisation
d'image MATLAB :

```text
Experiment/acdc_output/notebook_<date_et_heure>_dynamic_b0/
└── waveform.csv
```

Ce CSV contient les **consignes DAC**, avec le mapping et les polarités
du système. La première consigne est à `0` seconde ; les consignes
suivantes suivent les débuts des blocs de readout de la séquence, exprimés
depuis le trigger. Une dernière ligne remet toutes les sorties à zéro à
la fin du dernier bloc de readout. Pour 32 lignes EPI, il contient donc
**33 lignes numériques**, plus l'en-tête commenté du format d'export
existant. Les 32 sorties par défaut donnent 33 colonnes : le temps puis
les courants en ampères.

Pour préparer uniquement cette expérience de shimming du champ, exécuter
le notebook jusqu'à la section 9f ; les étapes MATLAB suivantes ne sont
pas nécessaires à cet export. Pour les lancer aussi, poursuivre les
cellules comme auparavant. Le notebook ne charge aucune waveform sur le
contrôleur et ne pilote pas le FPGA.

## Initialisation de l'optimisation d'image dans le notebook v3

La section **9g**, après les deux étapes de shimming du champ et avant
le forward model MATLAB, prépare le point de départ du solveur. Choisir
`IMAGE_INITIALIZATION` dans la cellule de paramètres :

| Valeur | Point de départ du solveur MATLAB |
| --- | --- |
| `"static"` — valeur par défaut | Les 32 courants calculés en 9b sont répétés pour chaque état de la séquence. |
| `"zero"` | Tous les courants de chaque état valent zéro. |

Le nombre d'états suit la séquence préparée. Avec l'EPI 32 lignes
actuelle, les courants initiaux forment donc un tableau **32 canaux ×
32 états**. Le solveur peut ensuite modifier chaque état indépendamment.
L'option `"static"` demande un calcul statique disponible et valide ;
elle n'utilise pas la waveform dynamique de B0 de la section 9f et ne
remplace pas silencieusement un résultat manquant par des zéros.

Examiner la figure des courants initiaux et celle du champ produit dans
la ROI. Le tableau de contrôles vérifie les dimensions et les limites
de courant et de champ avant la préparation de la requête MATLAB.

Pour changer l'option dans une session déjà ouverte, réexécuter la
première cellule, puis poursuivre toutes les cellules dans l'ordre.
Elle crée un nouveau dossier de travail temporaire et une nouvelle
sortie. Ce nouveau dossier est nécessaire après une optimisation déjà
exécutée : MATLAB refuse de réutiliser son dossier de résultats.
Le parcours recalcule les données et le shim statique, puis la section
**9g** prépare les courants initiaux et la cellule qui commence par
`# The bridge only orchestrates the existing MATLAB functions.` réécrit
la requête MAT avant l'optimisation. Appliquer le même parcours si la B0,
la ROI ou la calibration change.

Cette initialisation est le **point de départ numérique du solveur
d'image**. Elle ne soustrait pas le shim statique de la B0 mesurée et
n'ajoute pas une seconde correction aux courants optimisés. Le modèle
de réponse temporelle conserve un courant physique nul avant les
premières consignes, même si le solveur démarre depuis les valeurs
statiques. Aucun CSV supplémentaire n'est créé et aucun courant n'est
appliqué au système à cette étape.

## 6. Afficher le forward model avant l'optimisation

Le notebook demande à MATLAB deux simulations du même objet choisi :

1. B0 nul, scénario idéal de référence.
2. B0 mesurée, avant toute correction ACDC.

Les images sont relues dans Python et affichées avec le scan acquis.
La figure des k-espaces du forward model montre aussi les k-espaces
idéal et à B0 mesurée, ainsi que leur différence complexe. Examiner les mêmes
structures, orientations et échelles.
Les cartes de différences localisent les désaccords visibles ; elles ne
calibrent pas les causes physiques de ces désaccords.

L'objet simulé est limité à la ROI définie dans le notebook, alors que
le scan mesuré contient l'objet acquis complet. Une différence peut donc
provenir du support choisi, du contraste GRE/EPI, de la réception RSS ou
des hypothèses du modèle, en plus de B0. Une image idéalement corrigée
d'un objet limité à une ROI n'est pas une prédiction validée de tout le scan.

Le modèle utilise la GRE comme proxy d'amplitude et une réception
uniforme. Il ne mesure ni densité protonique ni sensibilités de réception.
Il conserve T2* infini si aucune mesure ou hypothèse explicite n'est fournie.
Les différences de TR, TE et angle entre référence et scan peuvent changer
le contraste ; lire ces paramètres ne corrige pas cet effet.

Si MATLAB échoue, la cellule affiche le rapport détaillé, avec la cause
MATLAB et sa pile d'appels, ainsi que le chemin du journal sous `WORK_DIR`.
Lire cette cause avant le message générique éventuel du solveur, tel que
`The function evaluation completed with an error.`. Par exemple,
`parallel:gpu:device:OldDriver` indique un problème de pilote GPU ; choisir
`DEVICE = "cpu"`, réexécuter la cellule de paramètres et relancer l'étape.
Les sorties de reconstruction restent disponibles pour revenir au premier
contrôle qui diverge. Ne poursuivre vers l'optimisation qu'après examen
de l'entrée et de la simulation.

## 7. Optimiser, puis lire les diagnostics

Activer l'étape d'optimisation dans les cellules de paramètres lorsque
la préparation et le forward model ont été examinés. Le notebook
transmet les paramètres à MATLAB sous forme MAT et relit ses résultats.
Le solveur et ses gradients ne sont pas réimplémentés en Python.

La cellule de paramètres définit `TARGET_MSE = 0.005`. Le calcul demande
l'arrêt dès qu'une itération atteint une MSE relative de magnitude
**strictement inférieure à 0,005**, avec les limites de courant et de champ
respectées. Le test porte sur l'objectif calculé par MATLAB :

```text
sum((abs(image_simulated) - abs(image_ideal))**2) / sum(abs(image_ideal)**2)
```

`image_ideal` est la référence simulée à B0 nul sur le support préparé
pour cette session. La cible est donc une erreur relative sur les
magnitudes, et non une MSE absolue ni une limite sur le B0 résiduel.
Modifier `TARGET_MSE` dans la cellule de paramètres permet de choisir
un autre compromis ; `TARGET_MSE = None` désactive ce critère d'arrêt.
Réexécuter cette cellule, puis la cellule qui écrit la requête MAT,
avant de relancer l'optimisation.

Le maximum d'itérations et les critères habituels du solveur restent
actifs. Atteindre la cible n'est pas garanti pour toutes les acquisitions.
Une MSE faible ne suffit pas à autoriser l'arrêt si les contraintes
physiques sont encore violées.

La sortie de la cellule reçoit les messages MATLAB au fil du calcul,
sans attendre la fin de l'étape. Les lignes de progression indiquent :

| Indicateur | Comment le lire |
| --- | --- |
| Itération / maximum demandé | Avancement du solveur ; le calcul peut s'arrêter avant ce maximum. |
| Nombre d'évaluations | Appels de la fonction objectif, qui peuvent être plus nombreux que les itérations. |
| Erreur relative quadratique sur les magnitudes | Valeur de l'objectif `fval` sur l'image reconstruite du modèle préparé à partir de la ROI. |
| Optimalité du premier ordre | Indicateur du solveur sur le gradient ; sa baisse ne suffit pas à valider les contraintes. |
| Violation des contraintes | Écart évalué par le solveur ; examiner aussi les bornes physiques vérifiées à la fin. |
| Temps écoulé | Durée écoulée depuis le début de l'optimisation. |

Ces messages sont aussi enregistrés dans le journal temporaire de
l'étape. La première évaluation ou l'initialisation des workers peut
prendre du temps avant la première ligne d'itération. À la fin, le
notebook affiche `exitflag`, `converged`, `target_reached`, `accepted`,
`current_limits_met` et `field_limits_met`. `stopping_reason` précise la
raison de l'arrêt et `target_mse` conserve le seuil demandé. Les valeurs
d'optimalité et de violation affichées
en cours de calcul sont des diagnostics ; la convergence est déterminée
après l'arrêt du solveur et les contrôles de sortie.

Un arrêt demandé sur la cible produit `exitflag = -1`, car MATLAB le
considère comme un arrêt par la fonction de suivi. Le code conserve ce
diagnostic : `converged` décrit la convergence habituelle du solveur,
distincte de l'arrêt sur la cible. `target_reached` indique l'atteinte
du seuil avec les contraintes respectées ; `accepted` indique que la
solution est admissible par convergence ou atteinte de cette cible.
L'export exige encore les contrôles des courants, du champ et de la
forme d'onde, y compris après quantification nominale des courants.

Une limite de champ et un petit nombre d'itérations peuvent d'abord servir
à vérifier le chemin de calcul. Un essai court peut produire des courants
et des figures sans convergence. Pour un résultat final, ajuster le
maximum d'itérations selon le diagnostic du solveur, en conservant une
nouvelle sortie pour chaque configuration.

Examiner ensemble :

- Image idéale, image avant correction et image simulée après correction.
- Différences d'image et mesures d'erreur sur le même support.
- Courants des 32 canaux, dans l'ordre chronologique des lignes acquises.
- Maximum par canal et somme absolue à chaque état.
- Cartes du champ ACDC et du résidu `B0 + champ ACDC` pour plusieurs états.
- État du solveur, convergence ou arrêt sur la cible, et respect des contraintes.

La figure des k-espaces optimisés montre le k-espace initial, le
k-espace après optimisation et le changement apporté par la correction.
Dans la figure des champs dynamiques, la première rangée représente le champ
ACDC généré, la seconde le champ instantané `B0 + champ ACDC` pour les
mêmes états. La limite `FIELD_LIMIT_HZ` s'applique à la première rangée ; elle
ne borne pas le résidu de la seconde. En EPI, ces champs instantanés
ne suffisent pas à expliquer l'image : la phase accumulée depuis la RF
reste calculée par le modèle MATLAB.

Un état `complete` indique que le calcul a écrit ses sorties ; il ne
suffit pas à conclure à la convergence. Une solution doit avoir convergé
ou s'être arrêtée sur la cible demandée ; les limites de courant et de
champ et tous les contrôles du CSV doivent être respectés pour permettre
l'export. Une amélioration numérique de
l'image ne remplace pas une mesure indépendante après correction.

### Exemple historique exécuté le 6 octobre 2026

Le forward model et une optimisation complète ont été exécutés sur
`acquisition_001`, pour une ancienne ROI de 88 pixels. Cette ROI n'est
plus chargée par le notebook actuel. Ce calcul utilisait la référence
DICOM ; il ne décrit pas une ROI issue de la nouvelle référence Twix.
Les résultats historiques sont conservés
dans
[notebook_20261006_115330](acdc_output/notebook_20261006_115330).

| Mesure du calcul MATLAB | Résultat |
| --- | --- |
| Erreur relative quadratique sur les magnitudes, avant → après | `0,8735772490 → 0,3024949604`. |
| Itérations | 428, avec 429 points dans l'historique, état initial compris. |
| Temps du solveur | Environ 544,3 s sur le poste utilisé. |
| Courant absolu maximal par canal | `0,998482 A`, pour une borne de 1 A. |
| Somme absolue maximale à un état | `27,994167 A`, pour une borne de 30 A. |
| Champ ACDC absolu maximal sur la région | `99,973071 Hz`, pour une borne de 100 Hz. |
| Arrêt du solveur | `exitflag = 2` : variation relative du pas devenue inférieure à la tolérance, contraintes respectées. |
| CSV historique | Aucun courant n'a été appliqué ; cet ancien calcul ne décrit pas la ROI actuelle. |

L'erreur compare les images **simulées** à la référence idéale simulée,
pas au scan acquis. Le critère d'arrêt porte sur le pas relatif
(`StepTolerance = 10⁻⁶`), avec une violation de contrainte nulle ; il
ne correspond pas à un arrêt certifiant l'optimalité du premier ordre.
Ce résultat ne prouve ni un optimum global ni un gain physique sur
l'objet complet, dont environ 65,3 % du signal GRE a été exclu du support.
La remise à jour des figures a relu ces résultats après comparaison
numérique des entrées ; elle n'a pas relancé le solveur une seconde fois.

### Réponse temporelle des canaux ACDC dans le notebook v2

Dans [02_acdc_experiment_step_by_step_v2.ipynb](02_acdc_experiment_step_by_step_v2.ipynb),
la cellule de paramètres contient :

```python
CURRENT_RESPONSE_ENABLED = True
CURRENT_RESPONSE_TAU_US = 133 / np.log(100)
```

La mesure « 99 % en 133 µs » donne une constante de temps de premier
ordre `tau = 133 / ln(100) ≈ 28,9 µs`. Après un changement de consigne,
le courant est calculé par :

```text
I(t) = I_new + (I_old - I_new) * exp(-t / tau)
```

`I_old` est le **courant réellement atteint** au moment du changement.
Il peut être différent de la consigne précédente si cette consigne n'a
duré que quelques τ. Le calcul conserve donc toute l'histoire des
transitions. Il suppose un courant initial nul avant le premier état ;
si une fenêtre de consigne finit sans nouvelle fenêtre, la consigne
revient à zéro et le courant décroît suivant la même réponse.

Pour utiliser une valeur commune, modifier le scalaire
`CURRENT_RESPONSE_TAU_US`. Pour caractériser les canaux séparément,
remplacer ce scalaire par une liste ou un tableau de **32 valeurs
positives en µs**, dans l'ordre des cartes de calibration :

```python
# Replace these example values with the measured values for all 32 channels.
CURRENT_RESPONSE_TAU_US = [28.9] * 32
```

L'ordre est celui des canaux du modèle, avant la permutation et les
polarités de `WAVEFORM_PARAMETERS` utilisées pour l'export. Le notebook
convertit ces valeurs en secondes pour MATLAB. Une valeur nulle,
négative ou non finie est refusée. `CURRENT_RESPONSE_ENABLED = False`
rétablit exactement le modèle historique à transitions instantanées.
Cette première version active la réponse uniquement pour une EPI avec
une RF et un état de courant par ligne ; laisser l'option désactivée
pour une GRE classique.

MATLAB calcule le courant à chaque échantillon ADC puis le champ
`B_acdc(t) = sum(map_channel * I_channel(t))`. Pour simuler le signal,
il utilise **l'intégrale de ce champ depuis le centre RF**, avec un
calcul analytique de premier ordre, dans le forward model et dans
l'optimisation. Remplacer simplement les courants par leur valeur à
l'ADC manquerait la phase accumulée pendant les transitions précédentes,
même lorsque le courant a déjà atteint sa consigne avant l'ADC.

Après le calcul, examiner les figures de réponse : la comparaison des
consignes DAC aux courants prédits, le détail d'une transition et la
différence entre les images simulées avec et sans cette dynamique.
La comparaison utilise les mêmes consignes optimisées et permet
d'isoler l'effet du modèle temporel ; ce n'est pas une nouvelle mesure
sur le scanner. Les champs et courants prédits aux ADC sont également
contrôlés. Les contraintes du solveur continuent de porter sur les
consignes et leurs champs ; les contrôles supplémentaires aux ADC ne
certifient pas les bornes entre deux acquisitions, notamment lorsque
les τ diffèrent d'un canal à l'autre.

Le CSV exporté contient toujours les **consignes DAC optimisées**.
Exporter les courants prédits introduirait une seconde réponse
temporelle lorsque le matériel les applique. Aucune commande du
contrôleur n'est ajoutée au pipeline.

Après une modification de τ ou de l'activation, réexécuter le notebook
depuis la cellule de paramètres : l'entrée MAT, l'optimisation et les
figures doivent utiliser la même configuration. Un résultat enregistré
avec une autre configuration est refusé par les contrôles du notebook.
Le premier ordre décrit une approximation de la mesure d'échelon ; il
reste à valider sur les canaux, les amplitudes et les changements de
signe de l'expérience. Il ne modélise pas les courants de Foucault, un
délai pur supplémentaire ni d'éventuelles non-linéarités du montage.

## 8. Examiner puis exporter `waveform.csv`

Le notebook écrit un seul fichier final. La correspondance entre les
canaux du modèle et les sorties physiques sert uniquement à ordonner les
colonnes du CSV. Aucun réglage du FPGA, lecture de courant, chargement ou
lancement de forme d'onde ne fait partie du pipeline.

### Paramètres du CSV

| Paramètre | Rôle et valeur initiale |
| --- | --- |
| `WAVEFORM_PARAMETERS["channel_map"]` | Liste de 32 sorties physiques distinctes, numérotées de 1 à 64 ; vérifier chaque entrée avec le câblage réel. `None` bloque l'export. |
| `WAVEFORM_PARAMETERS["channel_polarity"]` | 32 signes `1` ou `-1`, initialement `1`, pour convertir les courants du modèle dans la convention des sorties. |
| `WAVEFORM_PARAMETERS["max_current_A"]` | Borne par canal utilisée dans le calcul et le contrôle du CSV. |
| `WAVEFORM_PARAMETERS["total_current_limit_A"]` | Borne de la somme absolue des courants à chaque état. |
| `WAVEFORM_PARAMETERS["min_dac_interval_s"]` | `None` : contrôle facultatif de l'intervalle entre changements de courant ; saisir une mesure positive en secondes pour l'activer. |
| `RF_DELAY_S` | `0.0` : temps exprimés depuis le premier centre RF ; saisir le délai trigger → RF mesuré pour des temps depuis le trigger. |
| `EXPORT_WAVEFORM` | `True` : écrire le CSV si les contrôles réussissent ; `False` permet de lire les résultats sans créer ce fichier. |

L'ordre saisi doit correspondre au câblage réel. Une liste `1, 2, …, 32`
ne convient que si cet ordre est celui de l'installation.
`channel_map = None` bloque l'export jusqu'à ce que les sorties soient renseignées. Une sortie physique `1`
devient la colonne `ch0_A`. Les sorties inutilisées valent zéro ; les
colonnes sont complétées par groupes de huit sorties. Ne réappliquer
aucun signe déjà présent dans les cartes de calibration.

Le paramètre `RF_DELAY_S` est un décalage temporel, pas une commande de
synchronisation. La valeur initiale zéro ne prétend pas mesurer un délai
nul sur le banc : elle choisit le centre RF comme origine. Pour obtenir
des temps utilisables depuis un trigger externe, saisir son délai mesuré
jusqu'au **premier centre RF**, excitations factices comprises.

### Lire la forme d'onde

La figure finale montre les courants en fonction du temps. En GRE, les
courants sont constants entre la fin du bloc RF et la fin du bloc ADC de
chaque ligne, puis reviennent à zéro avant la ligne suivante. En EPI,
les états de courant se suivent après les navigateurs et la phase
continue de s'accumuler depuis l'unique RF, comme dans le modèle MATLAB.

Le calendrier couvre une **image complète**. Le CSV commence et se termine
avec des courants nuls. Une GRE de `Ny` lignes comporte `2Ny + 1` lignes
de CSV ; une EPI comporte `Ny + 2` lignes. Changer la matrice change le
nombre d'états, pas le nombre de bobines.

Le fichier est compatible avec les colonnes du programme `waveform` du
dépôt [`LincolnCB/rev_d_shim`](https://github.com/LincolnCB/rev_d_shim/tree/52a4235144ccddc19d1b7e6a4c50038c341f9016/software/waveform) :

```text
# time_s,ch0_A,ch1_A,...
```

Les temps sont en secondes et les courants en ampères, sans conversion
manuelle en codes DAC. Le contrôle de la quantification nominale des
courants et de la borne de champ est conservé. Il ne mesure pas les
gains, offsets ni transitoires analogiques du montage.

### Sortie finale et échanges MATLAB

Après validation de la ROI, de la calibration, de la convergence ou de
l'arrêt sur la cible, des courants et de la forme d'onde, la cellule
d'export produit :

```text
Experiment/acdc_output/notebook_<date_et_heure>/
└── waveform.csv
```

Le dossier de sortie ne contient aucun MAT, JSON, masque, manifeste,
PNG ni script. Les figures, tableaux et paramètres sont dans le notebook.
Les MAT, requêtes et journaux utilisés par le forward model et le solveur
restent dans `WORK_DIR`, qui est temporaire et distinct de cette sortie.
Le chemin est affiché pour pouvoir examiner une erreur MATLAB pendant
la session.

Si une étape échoue ou si l'export est désactivé, aucun CSV final n'est
créé. Tant que la nouvelle référence Twix manque, le parcours s'arrête
au contrôle des fichiers. Les contrôles de ROI et de couverture seront
exécutés une fois cette référence fournie.

### Comparer les acquisitions après shimming dans le notebook v2

La fin de
[02_acdc_experiment_step_by_step_v2.ipynb](02_acdc_experiment_step_by_step_v2.ipynb)
compare les trois acquisitions mesurées : `gre_reference.dat`, `scan.dat`
sans ACDC et `results/scan_corr.dat` avec ACDC. Déposer le nouveau Twix
dans le sous-dossier `results` du même dossier d'acquisition, puis
renseigner les chemins dans la cellule de comparaison :

```python
RESULTS_DIR = INPUT_DIR / "results"
CORRECTED_SCAN_FILE = RESULTS_DIR / "scan_corr.dat"
CORRECTED_SEQUENCE_FILE = FILES["scan_sequence"]
RMSE_USE_ROI = True
NORMALIZE_COMPARISON = True
```

Le dernier chemin suppose que les deux EPI utilisent la même séquence.
Si le scan corrigé utilise une autre séquence, indiquer son propre fichier
`.seq` dans `CORRECTED_SEQUENCE_FILE`.

Cette comparaison réutilise la GRE et le scan sans ACDC déjà reconstruits
dans le notebook. Le Twix avec ACDC est reconstruit séparément : ses
navigateurs, s'ils sont présents, fournissent sa propre correction de
phase EPI. Cette correction de reconstruction est distincte du shimming
ACDC appliqué sur le scanner. La comparaison ne relance pas l'optimiseur,
ne commande aucun équipement et n'ajoute aucun fichier au dossier
d'export.

Par défaut, les mesures utilisent la ROI finale employée par
l'optimisation, limitée à la couverture commune des trois images. Le
masque et le nombre de pixels retenus sont affichés. Avec
`RMSE_USE_ROI = False`, la comparaison utilise toute la couverture
commune, sans restriction à la ROI.

Avec `NORMALIZE_COMPARISON = True`, un seul gain positif est ajusté entre
la magnitude du scan sans ACDC et celle de la GRE dans cette région. Le
même gain est appliqué au scan avec ACDC, puis les trois images sont
divisées par le maximum GRE dans la région retenue. La RMSE porte sur ces
magnitudes normalisées ; la NRMSE exprime cette erreur en pourcentage de
la moyenne quadratique des magnitudes de référence. Avec `False`, la RMSE
utilise les magnitudes RSS brutes, en unités arbitraires du scanner.

Les RMSE mesurées comparent les deux EPI à la GRE. Elles diffèrent de la
MSE du modèle simulé utilisée par l'optimiseur : les différences de
contraste GRE/EPI, notamment de TE, contribuent aussi à l'écart. Si
`scan_corr.dat` manque, la comparaison est simplement ignorée. Une fois
le fichier acquis, exécuter les cellules finales de comparaison dans la
même session, sans relancer l'optimisation.

## 9. Revenir à l'étape qui explique un arrêt

| Message ou observation | Étape à revoir |
| --- | --- |
| Fichier absent | Chemins et dossier d'entrée dans les cellules de paramètres. |
| Matrice, FOV, TE, TR, dwell ou ordre de lignes incohérents | Tableau d'acquisition et trajectoire de la séquence correspondante. |
| Fantômes EPI ou changement après correction | Navigateurs, polarités et figures de reconstruction avant/après. |
| Peu de pixels B0 fiables | Magnitudes des échos, phases, résidu et masques de filtrage séparés. |
| Plusieurs régions B0 déconnectées | Figure des composantes et hypothèse de branche absolue. |
| ROI hors couverture B0 | Superposition de la carte rééchantillonnée et du masque choisi ; améliorer la mesure ou choisir explicitement une région adaptée. |
| Calibration hors couverture | Géométrie patient, transformation connue, coupe et voisinage valide du volume ACDC. |
| Simulation différente du scan | Orientation, support limité, B0, contrastes et hypothèses de réception avant d'accuser le solveur. |
| MATLAB échoue | Exécutable, licences, état MAT et journal de l'étape. |
| `parallel:gpu:device:OldDriver` | Pilote GPU incompatible avec MATLAB ; choisir `DEVICE = "cpu"`, réexécuter les paramètres, puis l'étape MATLAB. |
| Solveur terminé sans convergence | Diagnostic de convergence, gradients, contraintes et nombre d'itérations. |
| `exitflag = -1` après atteinte de la cible | Vérifier `target_reached`, `accepted` et les contraintes ; ce code signale ici l'arrêt demandé par la fonction de suivi. |
| Correspondance de canaux invalide | Tableau de câblage : exactement 32 sorties distinctes de 1 à 64. |
| Intervalle de changement trop court | Calendrier de la séquence et `min_dac_interval_s`, si ce contrôle facultatif est activé. |
| Origine temporelle inattendue | `RF_DELAY_S = 0.0` signifie centre RF ; un temps depuis le trigger demande son délai mesuré vers la RF. |
| Résultat ancien ou paramètres modifiés | Nouveau dossier de sortie et réexécution des étapes concernées. |

Une figure indique où chercher ; elle ne justifie pas à elle seule un
changement de signe, de masque ou de valeur matérielle. Conserver les
mesures sources et les paramètres choisis avec les résultats pour pouvoir
reproduire le calcul.

## 10. Comprendre les limites du modèle

Une seule carte B0 mesure le champ total avant correction. Elle ne permet
pas de connaître séparément la réponse des 32 bobines. Pour calculer les
courants, il faut en plus une calibration indépendante : pour chaque canal,
la variation de fréquence par ampère, avec le signe du courant et les
coordonnées physiques.

Les cartes de `ACDC_maps` ont d'abord été utilisées pour une **étude
virtuelle** sur des acquisitions historiques sans bobine ACDC. La
confirmation Bay5 ultérieure est conservée dans les paramètres du
notebook et dans l'ancien `acdc_input/calibration.mat`.
Rééchantillonner cette calibration
confirmée sur une nouvelle matrice conserve son repère et sa provenance.
Les détails figurent dans [CALIBRATION_ACDC.md](CALIBRATION_ACDC.md) et
[ACDC_MAP_IMPORT.md](ACDC_MAP_IMPORT.md).

Pour une nouvelle expérience, la calibration doit correspondre à votre
bobine, à son placement pendant l'acquisition et à l'ordre réel des canaux.
Un indicateur `registration_confirmed` enregistre une vérification déjà
faite ; le changer ne réalise aucun recalage. De même, une limite numérique
de courant doit provenir de votre configuration matérielle.

Le solveur cherche à rapprocher l'image simulée d'une image idéale avec
B0 nul. Les limites de champ portent sur le **champ produit par l'ACDC**,
pas sur le résidu `B0 + champ ACDC`. Les contraintes sont évaluées sur les
voxels de B0 fiables du support de l'objet.

En GRE, le modèle suppose un **spoiling idéal** : chaque RF démarre une
nouvelle histoire de phase transverse. Le champ B0 et le shim accumulent
leur phase depuis la RF de la ligne concernée. Les phases RF et ADC de
Pulseq sont prises en compte ; aucun historique longitudinal T1 ni
régime stationnaire FLASH n'est simulé. Les excitations factices occupent
leur place dans le timing, sans calcul d'établissement du contraste.

Les courants GRE sont constants entre la fin du bloc RF et la fin du
bloc ADC de chaque ligne, nuls hors de ces fenêtres. Le mode EPI
historique conserve au contraire l'accumulation de phase depuis son
unique RF à travers toutes les lignes.

Dans le notebook v2, l'option de réponse temporelle EPI modélise la
transition des courants par un premier ordre et conserve leur histoire.
Quand cette option est désactivée, les transitions restent idéales.
Le modèle ne simule pas les courants de Foucault ni d'autres dynamiques
que ce premier ordre.
La réception est uniforme et T2* vaut l'infini par défaut. L'amplitude
GRE est un proxy de magnétisation, pas une mesure de densité protonique
ou une cartographie T1/T2*. Des différences de TE, TR, angle de bascule
ou épaisseur de coupe entre référence et scan peuvent changer le contraste
mesuré ; leur lecture dans les métadonnées ne calibre pas ces différences.

L'amélioration de l'image simulée ne remplace donc pas une acquisition
corrigée indépendante. Le jumeau numérique historique ne prédit pas encore
fidèlement tous les scans mesurés ; voir [README.md](README.md).

## 11. Paramètres adaptables sans compliquer le modèle

Le scan GRE pris en charge est cartésien 2D mono-écho, une RF et un ADC
par ligne, avec TE/TR constants. Des RF factices peuvent précéder les
lignes acquises. La référence GRE/FLASH a sa propre séquence et ses
paramètres peuvent différer du scan. L'EPI historique reste prise en charge
avec une excitation et zéro ou trois navigateurs.

| Paramètre modifié entre expériences | Conséquence |
| --- | --- |
| Matrice paire `Nx`, `Ny` | Taille lue automatiquement ; `Ny` détermine le nombre d'états de courant. |
| FOV | Grille physique lue automatiquement ; rester dans la couverture B0 et ACDC. |
| TE, TR, dwell | Calendrier recalculé et contrôlé ; vérifier contraste et compatibilité matérielle. |
| Résolution de référence | Grille indépendante ; cartes rééchantillonnées sans extrapolation. |
| Coupe, orientation, placement | Nouveau contrôle géométrique nécessaire ; l'adaptateur actuel suit le protocole axial existant. |

Pour commencer, varier surtout la matrice puis le FOV, en conservant
la coupe et des contrastes proches entre référence et scan. Au sein d'une
séquence GRE, la forme, l'angle et le décalage de fréquence RF doivent
rester constants ; la phase RF peut varier pour le spoiling.

Le modèle ne prend pas en charge 3D, plusieurs coupes, plusieurs échos
pour le scan à corriger, partial Fourier, imagerie parallèle, ramp sampling,
bSSFP ni plusieurs images répétées dans un même `.seq`. La B0 séparée
utilise au moins deux échos monopolaires régulièrement espacés ; les
échos bipolaires demandent une correction de phase différente.

## 12. Compatibilité avec les formats et le script antérieurs

Le notebook utilise une **référence Twix brute `.dat` mono-écho**. Les
anciens DICOM de référence restent lisibles par le lecteur de préparation
pour les usages CLI historiques ; ils ne sont pas l'entrée du parcours
notebook actuel. Son diagnostic B0 exige un **Twix brut `.dat` multi-écho** : les cellules affichent les
images complexes par écho et par canal, leurs différences de phase et
les masques avant le calcul de champ. Une carte B0 déjà reconstruite
NPZ/MAT ne contient pas ces échos et ne peut pas être substituée
directement au Twix dans le notebook 02.

Les anciennes cartes B0 NPZ/MAT restent prises en charge par le parcours
CLI secondaire, avec `b0Hz`, `mask`, `X_m` et `Y_m`. L'import de calibration
du notebook peut toujours lire une calibration 2D MAT/NPZ avec `acdc_maps`,
`X_m`, `Y_m`, `units="Hz/A"`, `registration_confirmed` et son masque valide.
Les MAT doivent être v5/v7, pas v7.3/HDF5. Les données reconstruites sans
métadonnées scanner exigent un contrôle indépendant de géométrie.

Le script [run_acdc_experiment.py](run_acdc_experiment.py) conserve le
parcours historique par dossiers et JSON. Il reste utilisable pour les
anciennes configurations ; il ne constitue plus le point d'entrée
recommandé pour comprendre les étapes. Exemple de préparation compatible :

```bash
.venv/bin/python Experiment/run_acdc_experiment.py prepare \
  --input acquisition_001 --output resultat_cli_001
```

Le notebook
[01_reconstruction_et_B0.ipynb](01_reconstruction_et_B0.ipynb), traduit en
anglais, conserve les deux anciens scans EPI et leur reconstruction B0
DICOM. Son extension au voisin le plus proche sert uniquement à l'étude
virtuelle historique. Le nouveau notebook n'étend pas une mesure B0
hors de sa couverture fiable.

## 13. Vérifications logicielles

Les tests utilisent des données temporaires et ne commandent aucun matériel :

```bash
.venv/bin/python -m unittest discover -s Experiment -p 'test_*acdc*.py'
```

Ils couvrent les lecteurs, les trajectoires et temps, les cartes, les
masques, les fenêtres GRE/EPI et les refus d'export. Les contrôles MATLAB
vérifient séparément le forward model, l'optimiseur et le passage du
notebook vers les étapes MATLAB :

```matlab
addpath('Experiment');
validate_experiment_gre();
validate_experiment_optimizer();
validate_acdc_current_response();
validate_acdc_notebook_stage( ...
    'resultat_006/prepared/experiment_input.mat', ...
    'resultat_006/prepared/calibration.mat', 1, 30);
```

Adapter les deux chemins à une entrée préparée et sa calibration.
Le dernier contrôle écrit uniquement dans un dossier temporaire.

`validate_acdc_current_response()` utilise une EPI synthétique de 32
lignes et teste notamment le point 99 % à 133 µs, la continuité après
des changements de signe et des fenêtres courtes, la décroissance après
une consigne nulle, et l'intégrale du courant contre une intégration
numérique indépendante. Il vérifie aussi l'équivalence exacte avec le
modèle historique lorsque l'option est désactivée, la limite τ → 0,
les τ différents par canal, le gradient de l'objectif par différences
finies et l'équivalence du cache CPU avec le forward model de référence.

[VALIDATION.md](VALIDATION.md) conserve les vérifications effectuées et
leurs limites. Les simulations et tests logiciels ne remplacent pas
une mesure de réponse du montage ni une acquisition corrigée indépendante.
