# Vérifications effectuées

Les premières sections conservent les vérifications historiques, dont
l'optimisation sur une ancienne ROI de 88 pixels. La dernière section
décrit le passage à une référence Twix brute. La sortie reste limitée à
un CSV, sans contrôle du FPGA. Les comptes de ROI issus du DICOM dans
les sections antérieures sont historiques.

Le notebook historique `01_reconstruction_et_B0.ipynb` a été exécuté
intégralement : **11 cellules de code, aucune erreur**. Ses exports contiennent deux images EPI 32 × 32, une grille objet
128 × 128, 32 canaux de réception et 3 navigateurs par acquisition.

La correction des navigateurs a été vérifiée sur une erreur affine connue :
erreur relative du k-espace corrigé **2,50 × 10⁻¹⁶**. Le notebook vérifie
également le nombre d'ADC, le signe des polarités, la grille cartésienne et
les TE/TR/dwell du fichier Pulseq par rapport aux Twix.

Sous MATLAB R2024b, le modèle Bloch statique a passé les contrôles suivants :

| Contrôle indépendant | Erreur relative |
|---|---:|
| Transformée de Fourier d'un objet à B0 nul | 8,84 × 10⁻¹⁶ |
| Point hors résonance, signal analytique | 0 |
| Dérivée du champ uniforme | 4,73 × 10⁻¹¹ |

Le test de position retrouve aussi le point dans la ligne et la colonne
attendues. Un export synthétique a retrouvé un gain connu de 4,2 et une
NRMSE inférieure à 10⁻¹² sur les deux conditions.

Pour l'optimiseur optionnel, avec **des données synthétiques uniquement** :

| Contrôle | Résultat |
|---|---:|
| Courants nuls : égalité au modèle statique | Erreur 0 |
| Phase cumulée, courant constant | 4,37 × 10⁻¹⁷ |
| Gradient analytique, quatre directions | Erreur maximale 7,55 × 10⁻¹⁰ |
| Deux workers, deux threads, zéro itération | Résultats sauvegardés et pool fermé |
| Pool préexistant | Refus sans modifier le pool ni créer de dossier |
| Calibration absente, unités ou géométrie incorrectes | Rejet avant démarrage |

Les fixtures et sorties synthétiques de cette validation ont été supprimées.
Le journal se trouve dans `experiment_optimizer_validation_log.txt`.

## Comparaison réelle : résultat encore provisoire

Le jumeau a été calculé avec les données exportées, la trajectoire nominale
Pulseq et un gain unique ajusté sur le baseline :

| Condition | NRMSE | Corrélation des magnitudes |
|---|---:|---:|
| Baseline, hypothèse B0 nul | 18,63 % | 0,9745 |
| Perturbé, B0 GRE total et extension | 73,63 % | 0,5117 |

L'accord sur le baseline est une calibration, pas une validation indépendante.
Le modèle ne prédit pas fidèlement le scan perturbé. Il manque notamment la
B0 baseline, une couverture fiable du champ sur l'ensemble de l'objet et
les sensibilités de réception. Le masque de confiance couvre **1777 des
8746 points actifs** du jumeau, soit 20,32 %.

Les tests numériques valident les équations implémentées, pas leur capacité
à décrire toute la physique du scanner. Les valeurs de B0 complétées hors
masque sont des hypothèses ; une ambiguïté de branche et de signe subsiste.
Les cartes ACDC 3 T en Hz/A ont ensuite été fournies. Le fantôme ayant été
scanné sans ACDC, elles sont utilisées avec un **placement virtuel documenté**,
et non comme une calibration de bobines présentes lors de l'acquisition.
Des essais courts sont maintenant possibles ; aucune optimisation complète
jusqu'à 600 itérations n'a été lancée.

## Accélération et limites de courant : R2026a, 1 octobre 2026

Les chemins accélérés restent en double précision et ont été comparés aux
fonctions originales, qui sont conservées :

| Contrôle | Résultat |
|---|---:|
| Objectif/image/gradient CPU et GPU, quatre tailles de bloc | Erreurs relatives <10⁻¹⁵ sur fixture indépendante |
| Gradient, différences directionnelles incluant première et dernière ligne | Erreur maximale 8,71 × 10⁻⁸ |
| Timing non structuré, calcul matriciel générique | Erreur 6,28 × 10⁻¹⁶ |
| Contrainte de somme absolue, Jacobien par différences directionnelles | Erreur maximale 1,07 × 10⁻⁸ |
| Courants signés dont la somme vaut zéro mais la somme absolue vaut 32 A | Rejet |
| Somme absolue égale à 30 A | Rejet avec marge stricte |
| Contraintes de champ redondantes, maxima indépendants par sommets | Erreur 0 Hz |
| Système complet/réduit sous les bornes de courant | 11400 comparaisons identiques de faisabilité |
| Frontière exacte et suppression de toutes les lignes redondantes | Vérifiés |
| Pool CPU de deux workers, ressources et threads | Nettoyage/restauration vérifiés |
| Entrée GPU virtuelle, 8125 points, un worker et six threads | Sauvegarde CPU, figure et nettoyage de `/scratch` vérifiés |
| Pipeline complet virtuel, CPU/GPU, CG, trois itérations à 100/1000 Hz | Feasible, erreurs de magnitude ≤6,78 × 10⁻¹³, ressources nettoyées |
| Objet original avec des points ACDC non couverts | Rejet, aucun champ nul inventé |
| Mode de recalage physique sur une carte virtuelle non confirmée | Rejet avant dossier/pool |

Le support virtuel est l'intersection avec les cartes 3 T, soit **8125 points**.
Les réponses sont NaN hors couverture. La référence idéale est recalculée
avec le même objet. Les 621 points exclus portent 1,109 % de la somme des
amplitudes et 0,107 % de l'énergie quadratique du proxy initial ; aucune
extension des réponses ACDC n'est faite.

Les mesures avec les vraies réponses à leur placement virtuel montrent un
objectif CPU six threads de **131,0 → 73,7 ms**, et **3,41 ms sur GPU**.
Ces nombres incluent le gradient mais excluent le solveur. Un worker avec
le sous-problème historique par factorisation, limité à deux itérations
à 100 Hz, prend **9,57 s CPU contre 9,00 s GPU** : le gain total est limité
par les contraintes sur CPU. Les résultats CPU/GPU sont identiques aux
arrondis près et respectent les contraintes.

La suppression exacte des 113728 contraintes redondantes à 1000 Hz a
réduit un essai CPU de deux itérations de **9,53 à 1,66 s**. Les itérés peuvent
différer, car le solveur intérieur n'a plus ces termes de barrière, mais
l'ensemble admissible est conservé. À 100/200/300/500 Hz, les lignes sont
toutes conservées.

Un benchmark de cinq itérations à 100 Hz compare BFGS/factorisation,
BFGS/CG, L-BFGS/auto et L-BFGS/CG. Toutes les sorties sont faisables.
**BFGS/CG**, retenu par défaut, passe de **19,82 à 15,60 s** par rapport à
la factorisation, avec une meilleure progression de l'objectif à cinq
itérations. Il ne s'agit pas d'une comparaison jusqu'à convergence.

Les preuves sont conservées dans :

- `validation/acceleration_equivalence_log.txt`
- `validation/cpu_pool_and_current_limits_log.txt`
- `validation/virtual_gpu_entry_log.txt`
- `validation/virtual_cpu_gpu_pipeline_log.txt`
- `validation/solver_options_log.txt` et `solver_options_benchmark.mat`
- `experiment_gpu_validation_log.txt` et `experiment_acceleration_benchmark.mat`

`OPTIMIZATION_SPEED.md` donne les commandes et les paramètres retenus.

## Pipeline par dossiers et export Rev-D : 2 octobre 2026

Le point d'entrée `run_acdc_experiment.py` réutilise l'optimiseur MATLAB
existant et ajoute la préparation DICOM GRE / B0 Hz / Twix et l'export
pour `rev_d_shim/software/waveform`. Voir le
[tutoriel expérimental](TUTORIEL_SHIMMING_DYNAMIQUE.md).

Une exécution complète sous Python 3.11 et MATLAB R2026a Update 2 a réussi :
préparation de trois voxels synthétiques à B0 nul, optimisation convergée,
fermeture du pool, contrôles de provenance et export des 34 lignes DAC
(zéro, 32 états, zéro). Aucun courant n'a été appliqué. Le test à B0 nul
vérifie la chaîne logicielle ; il ne mesure pas un gain de correction.

Les tests vérifient également :

- la reconstruction sur les DICOM/Twix historiques, identique au notebook
  (`rtol=1e-12`), et la trajectoire/timing identiques à l'export historique
  (`atol=1e-12`) ;
- les refus de géométrie, unités, couverture B0/calibration, calibration
  virtuelle, séquence hors modèle, solution non convergée et fichier modifié ;
- les sorties physiques, polarités, limites de courant, contraintes de
  champ après quantification DAC, arrondis d'horloge et chevauchements ;
- le début réel de fenêtre ADC à 94 µs après le début de bloc, distinct
  du centre du premier échantillon à 99,5 µs pour la séquence fournie.

Les véritables routines C de lecture CSV et de streaming du firmware au
commit `52a4235144ccddc19d1b7e6a4c50038c341f9016` ont aussi été compilées
séparément avec des interfaces matérielles simulées. Les CSV actif/nul
sont acceptés ; trois répétitions conservent les attentes trigger et la
dernière lecture ADC après le retour à zéro.

Les journaux sont dans `validation/acdc_folder_pipeline_log.txt`
(39 tests, dont MATLAB, réussis), puis `validation/acdc_folder_python_log.txt`
(régression incluant deux contrôles de séquence supplémentaires ; test
MATLAB facultatif non relancé). Commandes reproductibles dans le tutoriel.

Il n'y a eu ni transfert au contrôleur, ni essai FPGA/amplificateur/scanner.
La présence d'une calibration physique recalée, les gains analogiques,
les délais réels et le signal de trigger restent propres à l'installation.
Les cartes historiques du fantôme demeurent des données d'étude virtuelle.

## GRE classique, deux séquences et matrices variables : 5 octobre 2026

Le pipeline par dossiers accepte maintenant un scan GRE 2D mono-écho,
avec une RF par ligne, et une référence GRE/FLASH décrite par sa propre
séquence. L'optimiseur existant est conservé ; ses dimensions sont déduites
des données. Le nombre de canaux ACDC reste fixé à 32.

**78 tests réussis**, sans test ignoré, dont les deux chaînes complètes
avec le véritable optimiseur MATLAB. Journal :
`validation/gre_folder_pipeline_20261005_log.txt`. Commande exécutée :

```bash
ACDC_TEST_MATLAB=/usr/pubsw/packages/matlab/26.1/bin/matlab \
  .venv/bin/python -m unittest discover -s Experiment -p 'test_*acdc*.py' -v
```

Les contrôles Python couvrent notamment :

- la lecture de séquences 32 × 32 et 64 × 64 avec FOV, TE, TR et dwell
  différents, ainsi que des matrices rectangulaires ;
- les excitations factices initiales, le spoiling de phase RF et un ordre
  de phase non séquentiel, tout en conservant l'ordre temporel des courants ;
- le refus des RF de forme/amplitude ou fréquence variables, du multi-écho,
  des matrices impaires et des déclarations Pulseq contradictoires ;
- une référence 16 × 12 et un scan 8 × 6, avec B0 et calibration sur deux
  autres grilles ; interpolation sur le support mesuré sans extrapolation ;
- la validation séparée de chaque image contre sa séquence, les métadonnées
  DICOM/Twix historiques et la sérialisation des métadonnées NPZ/MAT ;
- les états GRE séparés par des zéros, la stabilisation avant la RF suivante,
  un trigger par image complète et la conservation des contraintes Rev-D ;
- le refus d'export après modification du fichier de référence `.seq`.

Les deux tests complets dossier → MATLAB → fichiers de commande utilisent
des objets synthétiques à B0 nul. Le test GRE comprend deux RF factices,
des grilles distinctes et un ordre de phase non séquentiel. Ces tests
vérifient la convergence et l'export, sans démontrer un gain expérimental.

Le modèle numérique a aussi été comparé à un calcul scalaire indépendant,
avec B0/T2* non uniformes et courants non nuls. Sous MATLAB R2026a Update 2 :

| Contrôle GRE | Erreur relative maximale |
| --- | --- |
| Courants nuls contre modèle statique | 2,63 × 10⁻¹⁶ |
| Signal contre calcul scalaire indépendant | 4,95 × 10⁻¹⁶ |
| Indépendance vis-à-vis des intervalles entre RF | 3,47 × 10⁻¹⁵ |
| Gradient contre différences finies | 2,49 × 10⁻⁸ |

Les validations du modèle EPI et de l'accélération CPU existants passent
également. Le journal numérique est
`validation/gre_model_20261005_log.txt`. Pour relancer ce contrôle :

```bash
matlab -batch "addpath('Experiment'); validate_experiment_gre(); validate_experiment_optimizer(false); validate_experiment_acceleration();"
```

Le périmètre expérimental reste une coupe, une image cartésienne complète,
Nx/Ny pairs et TE/TR/dwell constants dans chaque `.seq`. Le modèle GRE
suppose un spoiling idéal et ne calcule pas le régime stationnaire T1.
Les essais ne connectent aucun contrôleur. Le délai réel trigger → première
RF et les paramètres du banc restent à renseigner avant un essai matériel.
Le [tutoriel](TUTORIEL_SHIMMING_DYNAMIQUE.md) décrit ces réglages.

## Entrées B0 Twix et calibration NPZ : 6 octobre 2026

Le pipeline accepte directement `b0_map.dat` avec sa séquence GRE multi-écho,
et `ACDC_3T.npz`. La reconstruction B0 produit une carte en Hz et un masque
de fiabilité ; l'import des réponses ACDC réutilise l'interpolation existante
sur la géométrie physique de la référence. Les anciens NPZ/MAT préparés
restent acceptés. Les fichiers source ne sont pas modifiés.

La suite Python collecte **106 tests : 104 réussis, 2 tests MATLAB facultatifs
non relancés**. Les tests MATLAB complets du 5 octobre restent décrits
ci-dessus ; aucun modèle MATLAB n'a été changé pour ces nouveaux formats.
Journal : `validation/raw_inputs_20261006_log.txt`.

Les nouveaux contrôles vérifient notamment le champ connu et le signe sur
des signaux complexes indépendants, l'annulation des phases statiques des
récepteurs, le déroulement spatial, les masques, les échos manquants et les
géométries incompatibles. L'intégration par dossiers vérifie l'application
unique de `b0_sign`, le chargement NPZ, la sauvegarde des produits dérivés
et le refus après changement de séquence B0.

Le véritable NPZ Bay5 fourni reproduit le MAT confirmé avec une tolérance
de 10⁻¹², y compris masque et valeurs inconnues. Son empreinte attendue est
conservée dans la configuration ; une substitution non déclarée est refusée.

Les données réelles de `acquisition_001` ont été lues : scan EPI 32 × 32,
référence GRE 32 × 32 et B0 128 × 128 à trois échos. Les trois nouvelles
séquences correspondent aux métadonnées contrôlées, notamment TR = 1 s
pour la référence. La B0 comporte **1 454 pixels fiables** avec les seuils
par défaut. Sur la grille de référence, elle couvre **88 des 359 pixels**
du support ; la calibration ACDC couvre les 359. Il n'y a eu ni réduction
automatique du support ni optimisation sur ce dossier.

Les tableaux, le rapport et la figure de diagnostic se trouvent dans
`acdc_output/acquisition_001_input_check/`. Le signe physique et la branche
absolue de fréquence de la B0 restent à confirmer. Les réglages matériels,
dont le délai trigger → RF, demeurent inconnus ; aucun matériel n'a été commandé.

## Parcours notebook pas à pas avec l'ancienne ROI : 6 octobre 2026

La version antérieure du [notebook expérimental](02_acdc_experiment_step_by_step.ipynb)
contient **31 cellules, dont 17 cellules de code exécutées et 21 figures**,
sans erreur de cellule. Les commentaires, explications et légendes des deux
notebooks sont en anglais ; les guides Markdown restent en français.
Les paramètres sont déclarés dans les cellules, sans lecture d'un JSON.
Les instantanés et les résultats sont sauvegardés en MAT.

Le forward model et l'optimisation ont réellement été exécutés sous MATLAB
R2026a sur `acquisition_001`, avec la ROI fournie, une borne de 1 A par canal,
30 A de somme absolue et 100 Hz de champ ACDC généré. La limite d'itérations
était de 600 ; le solveur s'est arrêté auparavant.

| Contrôle sur cette exécution | Résultat |
| --- | ---: |
| Pixels de la ROI simulée | 88 |
| Signal de référence exclu par la ROI | 65,323 % |
| MSE relative de magnitude initiale | 0,8735772490 |
| MSE relative de magnitude finale | 0,3024949604 |
| Itérations rapportées par `fmincon` | 428 |
| Échantillons de l'historique | 429 |
| Temps de l'optimisation | 544,3 s |
| Maximum absolu d'un courant | 0,998482 A |
| Maximum de somme absolue des courants | 27,994167 A |
| Maximum de champ ACDC généré sur la ROI | 99,973071 Hz |
| Maximum nominal après quantification DAC | 99,975754 Hz |

`exitflag = 2` correspond à l'arrêt sur le changement relatif des paramètres,
avec les contraintes respectées. Il ne signifie pas que le critère
d'optimalité du premier ordre a été atteint ni qu'un optimum global est établi.
La comparaison concerne deux images simulées du même objet limité à la ROI ;
aucune acquisition corrigée n'a mesuré cette amélioration.

Les dernières figures de diagnostic ont été réexécutées en relisant les
résultats MATLAB terminés. Une comparaison exacte des entrées numériques,
des cartes de calibration et de leurs masques a précédé cette relecture,
pour éviter de relancer le même solveur après de simples ajouts de figures.
Le notebook livré appelle MATLAB normalement lors d'une nouvelle exécution.

Le contrôle matériel échoue sur les réglages actuels : correspondance des
canaux, stabilisation, intervalles DAC/ADC et période des triggers. Le délai
provisoire de 1 ms n'est pas une mesure ; `RF_DELAY_CONFIRMED = False` bloque
également l'export. Le statut final contient `solver_ok = true`,
`hardware_ok = false`, `hardware_exported = false`, `hardware_applied = false`.
Aucun fichier de commande pour ce résultat n'a été créé, ni courant appliqué.

La suite Python collecte **113 tests : 111 réussis et 2 tests MATLAB
facultatifs ignorés**. Elle couvre notamment les intermédiaires de
reconstruction, une erreur de phase connue et le manifeste d'export MAT.
La validation indépendante du pont MATLAB retrouve une erreur nulle sur
les images, le k-espace et l'objectif, et conserve un statut lisible en
cas d'entrée manquante. Le notebook historique a également été réexécuté :
11 cellules de code, 7 figures intégrées et aucune erreur.

Les preuves sont conservées dans :

- `validation/notebook_python_20261006_log.txt`
- `validation/notebook_matlab_bridge_20261006_log.txt`
- `validation/notebook_execution_20261006_log.txt`
- `validation/notebook_diagnostics_20261006_log.txt`
- `validation/notebook_legacy_execution_20261006_log.txt`
- `acdc_output/notebook_20261006_115330/`, pour les figures et les résultats MAT.

Le [tutoriel détaillé](TUTORIEL_SHIMMING_DYNAMIQUE.md) décrit les cellules,
les vérifications, les hypothèses et les réglages restant à mesurer.

## Définition de la ROI, version avec masque en sortie : 6 octobre 2026

L'étape **4b. Define and inspect the ROI** remplace le chargement du masque
d'entrée par un calcul sur la magnitude de référence. Avec le seuil de 3 %
et sans restriction X/Y, elle sélectionne **359 pixels**. Les figures
montrent le masque, sa superposition verte sur la GRE et les pixels exclus.
Le résultat est enregistré en sortie dans `selected_roi.npy`.

Le calcul a été vérifié sur les données réelles. Une restriction physique
X = [-40, 40] mm, Y = [-30, 30] mm sélectionne 115 pixels, tous compris
dans les limites demandées ; des limites inversées sont refusées.
La sélection par défaut diffère du masque sauvegardé de 88 pixels,
qui n'est plus lu par le notebook.

La nouvelle version contient 33 cellules. **11 cellules ont été exécutées
jusqu'au contrôle de couverture, avec 12 figures intégrées**. Le contrôle
s'arrête explicitement sur les **271 pixels de ROI sans B0 fiable**.
Il conserve la ROI demandée et ne prépare pas de nouvelle optimisation
sur des champs inconnus. Les anciennes sorties MATLAB ont été retirées
du notebook ; les résultats numériques du paragraphe précédent restent
ceux de l'ancienne ROI.

Figures et masque : `acdc_output/notebook_20261006_121841/`.
Journal : `validation/roi_definition_20261006_log.txt`.

## Sortie limitée au CSV de waveform, référence DICOM historique : 6 octobre 2026

Le parcours à sortie CSV livre uniquement **`waveform.csv`**. Les figures et
paramètres sont conservés dans le notebook ; les MAT, requêtes et journaux
MATLAB restent dans un dossier temporaire séparé. La ROI reste en mémoire.
Les scripts de lancement, manifestes, empreintes, CSV de lecture ADC,
réglages d'horloge et commandes FPGA ont été retirés de l'export.
Les modèles MATLAB et l'optimiseur sont conservés.

La suite Python collecte **121 tests : 119 réussis et 2 tests MATLAB
facultatifs ignorés**. Ces deux tests ont ensuite été exécutés avec le
véritable MATLAB R2026a et ont réussi : une chaîne EPI et une chaîne GRE
parcourent préparation temporaire → optimisation → CSV. Les données de
ces essais sont synthétiques à B0 nul ; ils vérifient la chaîne logicielle,
pas une amélioration expérimentale. Le test du dossier de sortie vérifie
qu'il ne contient que `waveform.csv` et que le dossier de travail est supprimé.

La cellule finale du notebook a également été exécutée sur deux jeux de
courants connus, EPI et GRE. Les temps, les polarités, l'ordre des sorties
et les zéros avant/après l'acquisition correspondent aux valeurs attendues.
Les états nuls entre les lignes GRE sont présents. Un dossier contenant
déjà d'autres fichiers est refusé sans modifier ces fichiers.

Lors de cette vérification avec l'ancienne référence DICOM et un seuil
GRE de 3 %, le notebook contenait 32 cellules. Sur `acquisition_001`,
11 cellules ont été exécutées avec 12 figures intégrées ; le contrôle B0
s'arrête encore sur les 271 pixels non couverts de la ROI de 359 pixels.
Aucun dossier de sortie final n'est créé lors de cet arrêt et le dossier
temporaire du noyau est supprimé. Aucun résultat de l'ancienne ROI ni
fichier de contrôle du FPGA n'est présenté comme résultat de cette exécution.

Pour reproduire les tests Python :

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s Experiment -p 'test_*acdc*.py' -v
```

Pour les deux essais MATLAB uniquement :

```bash
ACDC_TEST_MATLAB=/chemin/vers/matlab PYTHONPATH=Experiment \
  .venv/bin/python -m unittest -v \
  test_run_acdc_experiment.FolderPipelineTests.test_real_matlab_optimizer_to_waveform_csv \
  test_run_acdc_experiment.GreFolderPipelineTests.test_real_gre_matlab_optimizer_to_waveform_csv
```

## Référence GRE/FLASH Twix brute : 6 octobre 2026

Le parcours principal utilise maintenant **`gre_reference.dat`**, une
référence GRE/FLASH cartésienne 2D mono-écho, avec son propre
`sequences/reference_gre.seq`. Le lecteur réutilise la reconstruction
cartésienne : k-espace brut → transformée de Fourier par canal de réception
→ magnitude RSS. Le notebook montre le k-espace, une reconstruction
complexe de canal et la magnitude avec ses coordonnées physiques avant
la définition de ROI. Les cartes B0 et ACDC sont ensuite rééchantillonnées
sur cette grille sans extrapolation.

La matrice et le FOV de référence peuvent différer de ceux du scan. Les
contrôles portent sur chaque acquisition face à sa propre séquence, puis
sur leur coupe et leur repère physiques communs. L'adaptateur conserve le
protocole axial centré utilisé jusque-là et refuse les géométries non
prises en charge. Le forward model et l'optimiseur MATLAB sont conservés ;
la sortie finale reste uniquement `waveform.csv`, sans contrôle du FPGA.

Dans `acquisition_001`, **la nouvelle référence `gre_reference.dat` n'est
pas encore présente**. Aucune reconstruction réelle ni optimisation
complète avec cette nouvelle entrée n'est donc déclarée validée.
L'ancien DICOM n'est pas substitué au Twix ; ses sorties numériques et
ses comptes de ROI restent des résultats historiques. Les paramètres
actuellement saisis dans le notebook sont conservés et les contrôles
seront recalculés sur la nouvelle référence.

La suite Python collecte **129 tests : 127 réussis et 2 tests MATLAB
facultatifs ignorés**. Les huit nouveaux tests de référence couvrent la
reconstruction RSS, la grille physique, les contrôles de séquence et les
refus de données ou de géométrie incompatibles. Les deux essais réels
MATLAB GRE/EPI avaient réussi lors de la vérification CSV précédente ;
ils n'ont pas été relancés pour ce changement d'entrée Python.
Journal : `/tmp/acdc_twix_reference_python_tests.log`.

La cellule de reconstruction de référence et celle de définition de ROI
ont été exécutées directement sur une référence synthétique 6 × 8 à
trois canaux de réception, dont le k-espace était produit par une DFT
indépendante. La magnitude RSS, les coordonnées physiques et le masque
de ROI correspondent aux valeurs attendues ; trois figures ont été
affichées. Cela vérifie ces cellules, sans simuler une acquisition réelle.

Le notebook principal a été exécuté sur les fichiers actuels jusqu'au
contrôle des entrées : les trois premières cellules de code sont exécutées
et l'arrêt indique `Missing reference: .../acquisition_001/gre_reference.dat`.
Aucune figure de cette nouvelle référence, aucun dossier de sortie final
et aucun CSV n'ont été créés. Les anciennes sorties DICOM ont été retirées
du notebook. Le schéma du notebook et la syntaxe de toutes ses cellules
de code sont valides.


## Arrêt sur une MSE cible : 7 octobre 2026

Le notebook expose `TARGET_MSE = 0.005`. Le callback MATLAB demande
l'arrêt lorsque la MSE relative de magnitude est strictement sous ce
seuil et que les bornes de courant par canal, de somme absolue des
courants et de champ ACDC généré sont respectées. `None` désactive ce
critère ; les appels MATLAB historiques omettant `TargetMSE` gardent
leur comportement. Les autres critères et la limite d'itérations restent
actifs. Le modèle physique et l'objectif ne sont pas modifiés.

Le code `exitflag = -1` est conservé pour cet arrêt demandé. Les champs
`target_mse`, `target_reached`, `accepted` et `stopping_reason` le distinguent
d'une convergence ordinaire. Le notebook et le CLI acceptent ce résultat
pour l'export après vérification indépendante des courants, du champ,
de la quantification nominale et des paramètres du CSV.

Vérifications effectuées :

- Suite Python : **136 tests collectés, 133 réussis, 3 tests MATLAB
  facultatifs ignorés** lors de cette exécution.
- [Validation MATLAB dédiée](validate_experiment_mse_stop.m) : sur une
  acquisition synthétique, MSE **0,774644 → 0,00347445 en 4 itérations**,
  arrêt `-1`, résultat accepté et `converged = false`. Les cas de cible
  satisfaite dès le point initial, cible désactivée, limite d'itérations
  et candidats de faible MSE violant les contraintes ont aussi réussi.
- Test MATLAB réel `test_real_matlab_mse_target_stop_to_waveform_csv`
  exécuté séparément : arrêt initial à MSE nulle, code `-1` accepté,
  sortie contenant seulement `waveform.csv`.
- Cellules du notebook testées sur une acquisition synthétique avec
  le bridge MATLAB réel : cible transmise correctement, résultat relu,
  contraintes et six figures vérifiées, export d'un unique CSV réussi.
- Schéma du notebook, syntaxe Python et `git diff --check` valides.

Le journal expérimental indiqué pour `acdc_notebook_ui_ofwkd` n'était
plus présent dans le dossier temporaire au moment de cette vérification.
Ces tests ne constituent pas une nouvelle optimisation de cette acquisition.


## Notebook de reconstruction seule : 7 octobre 2026

[03_reconstruct_epi_and_reference.ipynb](03_reconstruct_epi_and_reference.ipynb)
reconstruit uniquement la référence GRE/FLASH et le scan EPI Twix, avec
leurs propres séquences. Il réutilise `read_gre`, `read_scan` et
`read_sequence`, y compris la correction des trois navigateurs EPI.
Les autres entrées du pipeline expérimental ne sont pas nécessaires.

Exécution complète sur `acquisition_005` : **8 cellules de code réussies,
7 figures intégrées, aucune erreur**. Les figures montrent les trajectoires,
la reconstruction GRE, les k-espaces EPI avant/après correction, les
navigateurs et leur ajustement de phase, les images EPI avant/après
correction et la comparaison des deux scans. Les deux derniers diagnostics
ont aussi été examinés visuellement. Les images avant/après EPI partagent
la même échelle ; la comparaison finale est normalisée séparément pour
chaque image et utilise les coordonnées physiques lorsque les géométries
correspondent. Aucun calcul de MSE entre les contrastes GRE et EPI n'est
ajouté. Aucun fichier de sortie de pipeline ni calcul MATLAB n'est lancé.

Le schéma du notebook, la syntaxe Python et `git diff --check` sont valides.


## Comparaison des trois acquisitions mesurées : 7 octobre 2026

Le notebook de reconstruction compare maintenant la référence
`acquisition_005/gre_reference.dat`, l'EPI sans ACDC `scan.dat` et
l'EPI acquise avec ACDC `results/scan_corr.dat`. Les lecteurs existants
sont conservés. Chaque EPI est lue séparément et utilise son propre
ajustement de phase par navigateur ; aucun ajustement n'est partagé entre
les acquisitions. La correction ACDC appliquée sur le scanner est
distinguée de cette étape de reconstruction.

Exécution complète avec ces trois fichiers : **8 cellules de code réussies,
11 figures intégrées, aucune erreur**. Les deux EPI ont une matrice
32 × 32, 32 canaux de réception, trois navigateurs et des paramètres
compatibles avec `sequences/scan.seq`. Le chemin `corrected_sequence`
permet de fournir une autre séquence si l'acquisition avec ACDC diffère.
Le lecteur conserve aussi son traitement sans navigateur ; ce cas n'est
pas celui de ces deux acquisitions réelles.

La figure finale et les diagnostics de navigateurs ont été examinés
visuellement. Les deux EPI utilisent le même maximum pour leur échelle
de magnitude ; la GRE utilise son propre maximum. Les trois scans ont
le même centre de coupe et la même orientation, et sont affichés en
coordonnées physiques. Les tableaux numériques reconstruits restent
inchangés. Aucun calcul de MSE GRE/EPI, aucune optimisation et aucun
export de waveform ne sont ajoutés.

Le schéma du notebook, la syntaxe de ses cellules Python et
`git diff --check` sont valides. Journal d'exécution :
`/tmp/acdc_three_scan_notebook.log`.
Les trois tests ciblés du lecteur existant ont aussi réussi :
reconstruction GRE avec matrice et nombre de canaux variables,
diagnostics GRE sans changement de reconstruction, et récupération
d'une erreur de phase connue par les navigateurs EPI.


## Comparaison mesurée et RMSE dans le notebook v2 : 7 octobre 2026

La section 14 de
[02_acdc_experiment_step_by_step_v2.ipynb](02_acdc_experiment_step_by_step_v2.ipynb)
ajoute la comparaison de `gre_reference.dat`, `scan.dat` et
`results/scan_corr.dat`. Les 34 cellules précédentes, leurs paramètres
et leurs sorties sont conservés. La GRE et le scan sans ACDC sont
réutilisés ; le scan avec ACDC est reconstruit avec ses propres
navigateurs et contrôlé face à sa séquence.

Le choix confirmé par l'utilisateur est la **ROI avec normalisation
commune**. La ROI finale de l'optimisation est intersectée avec la
couverture valide des images et affichée avant le calcul. Le gain
d'intensité est estimé uniquement sur le scan sans ACDC, puis appliqué
inchangé au scan avec ACDC. Les trois images sont divisées par le même
maximum GRE dans ce masque. La figure, la RMSE et la NRMSE utilisent
les mêmes tableaux normalisés ; les reconstructions natives restent
inchangées. Les deux erreurs portent exactement sur les mêmes pixels.

Les prérequis Python et les quatre nouvelles cellules de code ont été
exécutés sur `acquisition_005`, sans lancer MATLAB ni l'export CSV :
**3 nouvelles figures intégrées, aucune erreur dans cette exécution**.
Tous les contrôles de la séquence corrigée ont réussi. La ROI contient
561 pixels ; aucun n'est exclu pour défaut de couverture des scans.
Le gain partagé vaut environ 1,19518.

| Scan comparé à la GRE | RMSE normalisée | NRMSE |
| --- | --- | --- |
| Sans ACDC | 0,129129 | 20,408 % |
| Avec ACDC | 0,144256 | 22,798 % |

La figure finale et le masque de mesure ont été examinés visuellement.
Ces mesures décrivent la similarité des images GRE/EPI, avec leurs
différences de contraste ; elles ne sont pas la MSE du modèle MATLAB.
Le notebook complet avec optimisation n'a pas été réexécuté.

Six vérifications ciblées ont réussi : RMSE nulle pour des images
identiques après gain commun, absence de réajustement du gain corrigé,
RMSE et NRMSE en mode brut, refus d'un signal de baseline nul, absence
de fichier corrigé sans échec, et rééchantillonnage d'une grille
différente sans extrapolation. Le schéma du notebook, la syntaxe de
toutes ses cellules et `git diff --check` sont valides.
Journaux : `/tmp/acdc_v2_comparison_validation.log` et
`/tmp/acdc_v2_comparison_checks.log`.

## Réponse temporelle ACDC dans le notebook v2 : 8 octobre 2026

Le notebook v2 utilise maintenant un premier ordre activable pour l'EPI,
avec `tau = 133e-6 / log(100)` secondes par défaut, ou 32 constantes de
temps dans l'ordre des cartes de calibration. MATLAB conserve le courant
réel atteint à chaque transition et intègre analytiquement le champ
depuis le centre RF jusqu'à chaque échantillon ADC. Le forward model,
l'objectif et son gradient utilisent cette même intégrale. Les variables
optimisées et le CSV restent les consignes DAC.

La fonction [validate_acdc_current_response.m](validate_acdc_current_response.m)
a réussi les vérifications suivantes :

- Échelon de 0 à 1 A : courant de 0,99 A à 133 µs.
- Transitions rapprochées, positives et négatives, avec pauses et retour
  final à zéro : écart maximal de `2,22e-16 A` face à une récurrence
  indépendante qui utilise le courant réellement atteint.
- Intégrale de phase comparée à une quadrature numérique indépendante :
  écart maximal de `2,69e-15 A·s`.
- Gradients de l'objectif vérifiés par différences finies, avec τ commun
  puis 32 τ différents : erreur relative maximale inférieure à `5,85e-8`.
- Modèle désactivé identique au modèle historique ; limite de τ très
  petit, causalité et refus des paramètres invalides vérifiés.
- Chemins CPU de référence et accéléré : images identiques, gradients
  concordants à une erreur relative inférieure à `7e-16` sur les fixtures.

Une itération de l'optimiseur sur une fixture synthétique avec 32 τ
différents a aussi réussi. L'image sauvegardée concorde exactement avec
le diagnostic recalculé à partir des mêmes consignes. Les validations
existantes d'accélération et de GRE ont réussi. Aucun GPU n'étant détecté
par MATLAB sur cette machine, le chemin GPU n'a pas été validé numériquement.

L'arrêt anticipé existant à une MSE inférieure à `0,005` reste valide :
la fixture passe de `0,774644` à `0,00347445` en quatre itérations.
L'échange MAT a été vérifié avec configuration absente, activation seule,
τ seul et 32 τ distincts. Les prévisualisations ont 401 temps et 32 canaux,
et les images sans courant restent identiques. Un indicateur d'activation
invalide est refusé avec un statut d'erreur lisible.

Les cellules de reconstruction, la prévisualisation MATLAB de l'échelon,
la comparaison mesurée et les nouveaux diagnostics ont été exécutés sur
`acquisition_005` : **21 cellules de code et 27 figures, aucune erreur**.
Les diagnostics de réponse utilisent, pour cette vérification, le CSV
déjà acquis `notebook_20261006_211402/waveform.csv`, recalculé dans une
copie temporaire du notebook. Les figures de phase, de champ à chaque
ADC et de transition ont été examinées. Le notebook principal conserve
les sorties de reconstruction et de prévisualisation ; les anciennes
sorties d'optimisation ont été effacées. Une nouvelle optimisation avec
la réponse activée doit être lancée depuis les paramètres.

Pour ces mêmes consignes déjà acquises, la MSE du modèle passe de
`0,003692265` avec transitions instantanées à `0,003703924` avec le premier
ordre. Ce calcul conserve les instants de consigne dérivés de la séquence
et ne mesure pas le délai du contrôleur. Il ne constitue ni une
réoptimisation ni une amélioration mesurée sur le scanner. La réponse
réelle doit encore être validée expérimentalement, notamment par canal.

Les contraintes du solveur portent sur les consignes et leurs champs.
Les diagnostics contrôlent aussi les courants et champs prédits à tous
les ADC, et le notebook refuse l'export si ces contrôles échouent. Ils
ne certifient pas les limites entre échantillons quand les τ diffèrent.
Aucune commande FPGA et aucun nouveau CSV n'ont été générés durant
cette validation.

Journaux et rapport : `/tmp/acdc_response_notebook_validation.log`,
`/tmp/acdc_response_worker_validation.log`,
`/tmp/acdc_response_gpu_validation.log`,
`/tmp/acdc_first_order_real_csv.log` et
`/tmp/acdc_current_response_validation_report.mat`.

### Prévisualisation complète de la waveform avant la section 13

La nouvelle section 12c affiche les courants prédits sur toute la durée
de la waveform et un zoom sur une transition. MATLAB échantillonne chaque
changement de consigne et prolonge le calcul jusqu'à huit τ après la
consigne finale à zéro. L'échantillonnage conserve l'histoire des états
avec une récurrence analytique ; il peut utiliser une solution déjà calculée
sans relancer l'optimisation.

Les tests d'échelon, de continuité, de canaux avec τ différents, de pauses,
de retour à zéro et de désactivation ont réussi. Le calcul concorde avec
le modèle analytique existant à `1,67e-16 A` près. Sur les 32 états EPI
d'`acquisition_005`, la prévisualisation contient 2 707 instants ; la
consigne finale à zéro est à 84,840 ms et le calcul se termine à 85,071 ms.
Les deux panneaux ont été vérifiés, ainsi que les permutations de canaux,
les polarités, le décalage temporel et le choix manuel du canal du zoom.

Les cellules Markdown et code de la section 13, y compris leurs sorties,
sont conservées intégralement. La prévisualisation n'écrit aucun CSV et
ne contrôle aucun matériel. Journal : `/tmp/acdc_full_waveform_validation.log`.

### Shim statique ajouté au notebook v3 — 2026-10-08

Les sections 9b et 9c réutilisent le solveur `solve_shim_static_qp` de
`ACDC_shim_static_dyn_haotian`, avec la cible `-B0`, les cartes recalées et
la ROI déjà validée. Les 46 cellules préexistantes, y compris leurs sorties,
sont conservées intégralement. Le modèle et l'optimisation dynamiques
restent MATLAB ; leurs entrées ne sont pas modifiées par cette étape.

La reconstruction depuis les Twix d'`acquisition_005`, la préparation
des cartes et les deux nouvelles cellules statiques ont été exécutées
sans erreur. Les trois figures de champ, de courants et de CSV ont été
examinées. Dans la ROI de 561 pixels :

| Mesure du modèle statique | Valeur |
| --- | ---: |
| RMS du champ avant | 10,1249 Hz |
| RMS du champ résiduel prédit après | 1,37457 Hz |
| RMS après quantification DAC nominale | 1,37471 Hz |
| Courant maximal en valeur absolue | 0,437958 A |
| Somme des valeurs absolues des courants | 3,82939 A |

Tous les contrôles des limites de courant et du champ généré dans la ROI
ont réussi. Ces résultats décrivent le champ établi prédit, sans mesurer
l'amélioration des images ni corriger la phase accumulée avant le trigger
du premier readout d'imagerie.

Le fichier généré est
`acdc_output/notebook_20261008_163315_static/static_currents.csv` :
exactement deux lignes sans en-tête et 33 colonnes, timestamps 0 et
1 seconde, avec 32 courants identiques sur les deux lignes. Le dossier
dynamique correspondant n'a pas été créé par cette validation.
Les 35 tests de l'export statique et de l'export dynamique ont réussi.
Les cas B0 nul et calcul statique désactivé ont aussi été vérifiés.
Aucune optimisation dynamique ni commande matérielle n'a été lancée.

Journal : [static_shim_v3_20261008_log.txt](validation/static_shim_v3_20261008_log.txt).

### ACDC Dynamic B₀ Shimming ajouté au v3 — 2026-10-08

Les sections 9d à 9f reprennent le solveur NumPy
`solve_shim_waveform_fista` de Haotian. La cible est la carte B0 mesurée
constante, avec le signe opposé, répétée à chaque ADC d'imagerie. Une
consigne est maintenue par ligne. Les matrices de réponse exactes du
premier ordre utilisent les temps de la séquence et acceptent 32 τ
distincts ; le solveur accepte désormais des matrices rectangulaires
entre les états de consigne et les échantillons ADC.

La reconstruction Twix et les nouvelles cellules ont été exécutées sur
`acquisition_005`, sans optimisation d'image MATLAB ni action matérielle.
Le calcul utilise 32 canaux, 32 états, 1 024 ADC et la ROI de 561 pixels.
Les cinq figures de cible, de convergence, de champ et de courant ont
été vérifiées. Tous les contrôles des limites sur les consignes,
les valeurs quantifiées et les courants/champs prédits aux ADC réussissent.

| Mesure du modèle dynamique B0 | Valeur |
| --- | ---: |
| RMS du champ avant | 10,1249 Hz |
| RMS du champ résiduel prédit sur tous les ADC | 1,37495 Hz |
| NRMSE du champ prédit | 13,5799 % |
| Courant maximal en valeur absolue | 0,405559 A |
| Somme maximale des valeurs absolues des courants | 3,57927 A |

Le trigger est supposé au début du premier bloc de readout d'imagerie,
à 9,320 ms du centre RF. Le premier ADC est à 10,3295 ms : les courants
sont déjà établis à cet instant pour le τ choisi. Les consignes des 32
lignes sont donc identiques à environ `5,2e-14 A` près, ce qui est attendu
avec une B0 constante. Le paramètre `B0_DYNAMIC_TRIGGER_DELAY_S` permet
de renseigner un trigger plus tardif. Un calcul indépendant avec trigger
au premier échantillon ADC prédit un RMS de 1,43839 Hz et une variation
de démarrage d'environ 0,0192 A. Le courant au temps exact du trigger
reste nul dans le modèle continu.

Le CSV généré est
`acdc_output/notebook_20261008_165311_dynamic_b0/waveform.csv` :
33 lignes numériques et 33 colonnes, une première consigne à zéro seconde,
32 états d'imagerie et une dernière ligne de courants nuls à 75,520 ms.
Il conserve l'en-tête commenté du module d'export existant. Le dossier
contient uniquement ce CSV. Les 50 cellules préexistantes du v3, sorties
comprises, et les entrées de l'optimisation d'image restent inchangées.

Les 49 tests du solveur de champ et des exports statique/dynamique ont
réussi. Ils vérifient notamment l'échelon à 99 % en 133 µs, les τ par
canal, la mémoire des transitions, la désactivation, les opérateurs
rectangulaires, les contraintes et le format CSV. La projection du
solveur FISTA a aussi été corrigée lorsque les bornes par canal et la
borne totale sont simultanément actives ; la perte affichée est
recalculée aux courants effectivement retournés.

Les métriques portent sur le champ prédit, pas sur les images mesurées.
La phase accumulée avant le trigger reste présente ; la réponse physique
doit être validée expérimentalement. Le modèle et l'optimisation d'image
restent dans MATLAB avec leur export indépendant.

Journal : [dynamic_b0_shim_v3_20261008_log.txt](validation/dynamic_b0_shim_v3_20261008_log.txt).

### Initialisation statique ou nulle de l'optimisation d'image — 2026-10-08

La section 9g du notebook v3 prépare les consignes initiales du solveur
MATLAB. `IMAGE_INITIALIZATION = "static"`, valeur par défaut, répète les
courants de la section 9b sur tous les états ; `"zero"` initialise tous
les états à zéro. La matrice est transmise dans la requête MAT jusqu'à
`fmincon`. Le résultat sauvegarde les consignes initiales demandées et
leur MSE d'image. Les autres cellules du notebook restent inchangées,
en dehors des paramètres, de la requête et des contrôles du résultat.

La préparation depuis les Twix d'`acquisition_005` a réussi dans les
deux modes. Pour le mode statique, le tableau comporte 32 canaux ×
32 états, un courant maximal de 0,437958 A et une somme maximale des
valeurs absolues de 3,82939 A. La requête MAT relue contient exactement
ce tableau. Les contrôles de dimensions, de courant et de champ passent,
et la figure des consignes initiales et du champ établi a été vérifiée.
Un choix inconnu et un résultat statique absent sont refusés explicitement.
Cette vérification ne lance aucune optimisation ni aucun export CSV.

`validate_experiment_initial_currents(true)` a également réussi dans
MATLAB. Les tests synthétiques vérifient le point effectivement reçu
par `fmincon`, le comportement historique à zéro, les vecteurs et
matrices de consignes, les limites, les valeurs non finies et le modèle
temporel. Deux passages complets par la requête MAT et un worker CPU
ont sauvegardé les points de départ attendus. Le baseline sans shim
reste identique, et le courant physique initial reste nul.

Ces tests utilisent zéro itération pour observer directement le point
de départ ; ils ne prétendent pas converger ni améliorer les scans.
La MSE initiale synthétique vaut 0,774644 à zéro et 0,626903 avec le
point statique et la réponse du premier ordre. L'erreur de dépassement
de courant visible dans le journal est un test de refus volontaire,
intercepté et vérifié avant toute création de pool d'optimisation.

Journaux :
[préparation notebook](validation/image_initialization_notebook_20261008_log.txt)
et [validation MATLAB](validation/image_initialization_matlab_20261008_log.txt).

## Correction des navigateurs EPI dans v3 : 9 octobre 2026

La reconstruction commune utilise maintenant les phases séparées des
paires 2–1 et 2–3, leur moyenne symétrique, un masque fondé sur les trois
échos et un ajustement affine sur une seule région continue fiable par
récepteur. L'ancienne moyenne complexe des navigateurs positifs n'est
plus utilisée pour estimer la correction. `NAVIGATOR_PHASE_BRANCH` rend
explicite l'ambiguïté de π ; la branche par défaut suppose une petite
différence de phase au point de signal le plus fort. Les trois temps
ADC des navigateurs sont conservés depuis le `.seq` et leur espacement
symétrique est contrôlé. Aucun recalage ni déplacement d'image n'est ajouté.

Les **23 tests ciblés passent**, dont neuf nouveaux tests synthétiques :
annulation des échos positifs, changement de signe de leur moyenne
complexe, choix de branche π, exclusion du fond peu fiable, régions
discontinues, signal insuffisant, suppression d'un fantôme synthétique
à un demi-FOV et refus de navigateurs inégalement espacés. Les tests GRE
sans navigateurs et les tests Twix de référence continuent de passer.
Une séquence historique supprimée du dossier de travail a été lue depuis
Git dans `/tmp` pour son test existant ; aucun fichier de séquence de
l'utilisateur n'a été restauré ou modifié.

Les cellules Python de v3 jusqu'à la définition du support commun ont
été exécutées sur `acquisition_010`, ainsi que les cellules de comparaison
mesurée. La reconstruction EPI 64 × 64 comporte trois navigateurs espacés
de 2,710 ms et 32 récepteurs ; tous ont un ajustement valide sur 31 à
38 pixels. Le résidu de phase va de 0,00772 à 0,03016 rad. La moyenne
complexe des échos positifs ne conserverait que 14,5 % de leur amplitude
en valeur médiane sur les pixels retenus. Les nouvelles figures de
l'étape 5 sont enregistrées dans le notebook.

La similarité entre l'image et sa copie décalée de 32 pixels diminue de
0,8184 avec l'ancienne correction à 0,2672 avec la nouvelle. Ce nombre
est un diagnostic de répétition, **pas** un rapport fantôme/signal
normalisé ni une NRMSE. L'objet dépasse un demi-champ : ses deux moitiés
se recouvrent dans ce calcul, qui ne permet donc pas de conclure à la
disparition du fantôme. Le déplacement B0 et des artefacts résiduels
restent visibles. Le fichier `acquisition_010/results/scan_corr.dat`
n'est pas présent : les cellules de comparaison le signalent et ne
calculent donc pas de nouvelle RMSE ACDC sur cette acquisition.

La suite plus large de lecture des séquences comporte 16 succès,
un test ignoré faute de séquence historique et un échec sur la grille
du fichier `Mrzero/sequence_simulation/gre_64x64_fov200mm_tr1s.seq`.
Le même refus a été reproduit avec le lecteur avant l'ajout des temps
des navigateurs : cet échec préexiste au correctif. La séquence EPI 64
et la référence GRE de cette acquisition passent leurs contrôles.

Le schéma du notebook et la syntaxe de toutes ses cellules sont valides.
Les images sauvegardées qui dépendent de l'ancienne reconstruction ont
été recalculées ou vidées. Les cellules de la section 13, y compris leurs
sorties, restent exactement identiques. Aucune optimisation MATLAB ni
export CSV n'a été lancé pendant cette vérification.

Journal : [validation des navigateurs v3](validation/navigator_phase_v3_20261009_log.txt).
