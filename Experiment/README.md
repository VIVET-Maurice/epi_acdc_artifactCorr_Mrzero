# Expérience IRM : notebook ACDC pas à pas

Ouvrir [02_acdc_experiment_step_by_step.ipynb](02_acdc_experiment_step_by_step.ipynb).
C'est le parcours principal pour reconstruire les données, examiner chaque
étape et préparer le shim dynamique. Tous les paramètres à modifier sont
regroupés dans les cellules du notebook ; **aucun JSON de configuration
n'est requis**. Les notebooks et leurs figures sont en anglais. Le
[tutoriel en français](TUTORIEL_SHIMMING_DYNAMIQUE.md) détaille leur usage.

Le notebook utilise les lecteurs existants pour les Twix et
`ACDC_3T.npz`. Il montre les données brutes, les images reconstruites,
les phases inter-échos, les masques de fiabilité B0, le recalage et les
couvertures, la trajectoire et le calendrier de la séquence, les résultats
MATLAB et les contrôles de l'export. Les figures et tableaux sont conservés
dans le notebook. **La seule sortie finale est `waveform.csv`.**

Le **forward model et l'optimiseur restent dans MATLAB**. Le nouveau point
de passage [run_acdc_notebook_stage.m](run_acdc_notebook_stage.m) appelle
les modèles et le solveur existants à partir de requêtes MAT. On peut
examiner la simulation avant de décider de lancer l'optimisation, puis
examiner les résultats avant d'écrire le CSV. Les entrées, résultats et
journaux MATLAB utilisent un dossier de travail temporaire sous
`/scratch/mv1289`, distinct de la sortie finale. `SCRATCH_DIR`, dans la
première cellule de chemins, définit aussi le stockage temporaire des
workers MATLAB. Après une modification de ce chemin, réexécuter le
notebook depuis cette première cellule.

Dans la version v3, l'étape 5 corrige les navigateurs EPI à partir des
phases des deux paires séparées, en évitant l'annulation de leur moyenne
complexe. Elle montre le signal, les pixels fiables et le résidu de
l'ajustement. `NAVIGATOR_PHASE_BRANCH` rend explicite l'ambiguïté de π.
La cellule de l'étape 5 recharge le lecteur commun dans le noyau ;
réexécuter cette cellule puis la comparaison pour rafraîchir les RMSE. Le
[tutoriel](TUTORIEL_SHIMMING_DYNAMIQUE.md#vérifier-la-correction-des-navigateurs-epi-dans-la-version-v3)
détaille les hypothèses et les figures.

L'étape **8b** de v3 affiche aussi une reconstruction de diagnostic où
l'offset B0 moyen mesuré dans la ROI est retiré des données brutes et des
navigateurs. `SHOW_MEAN_B0_DIAGNOSTIC` active cet affichage. Il permet de
repérer un déplacement du phantom et son repliement aux bords ; un fantôme
pair/impair plus faible peut subsister. Cette image distincte ne remplace
pas le scan utilisé pour l'optimisation, les CSV ou les RMSE ACDC.

La version [02_acdc_experiment_step_by_step_v2.ipynb](02_acdc_experiment_step_by_step_v2.ipynb)
ajoute la **réponse temporelle du courant ACDC** à la simulation et à
l'optimisation EPI. `CURRENT_RESPONSE_ENABLED = True` active un premier
ordre avec `CURRENT_RESPONSE_TAU_US = 133 / np.log(100)`, soit environ
28,9 µs. Ce paramètre accepte aussi 32 valeurs, dans l'ordre des canaux
de la calibration. `False` retrouve le modèle à transitions instantanées.
MATLAB conserve le courant atteint lors des transitions précédentes et
intègre le champ réellement appliqué pour calculer la phase de chaque
échantillon ADC. Les figures comparent les consignes aux courants prédits
et les images simulées avec et sans cette dynamique.

Le courant initial est supposé nul. Le modèle de réponse est réservé à
l'EPI dans cette version ; la GRE conserve son modèle existant avec
l'option désactivée. Le CSV contient toujours les **consignes DAC** : les
courants filtrés servent au modèle. Réexécuter le notebook depuis les
paramètres après un changement de τ ou d'activation. Cette approximation
doit être validée expérimentalement pour les canaux utilisés ; elle ne
représente pas les courants de Foucault ni d'autres dynamiques du montage.
Le [tutoriel](TUTORIEL_SHIMMING_DYNAMIQUE.md#réponse-temporelle-des-canaux-acdc-dans-le-notebook-v2)
détaille ces réglages et les contrôles.

La version [02_acdc_experiment_step_by_step_v3.ipynb](02_acdc_experiment_step_by_step_v3.ipynb)
ajoute, aux sections **9b et 9c**, les courants pour un premier scan avec
**shim statique**. Elle reprend `solve_shim_static_qp` du dossier
`ACDC_shim_static_dyn_haotian`, avec la cible `-B0`, la ROI et les cartes
déjà vérifiées. `RUN_STATIC_SHIM` active ce calcul ;
`EXPORT_STATIC_CURRENTS` active son export. Les limites de courant
statiques reprennent par défaut celles du dynamique. Le solveur Python
demande `quadprog` et `tqdm`, installés avec `Experiment/requirements.txt`.

La sortie supplémentaire `static_currents.csv` contient **deux lignes
sans en-tête**, à 0 et `STATIC_CSV_LAST_TIME_S` secondes, avec les mêmes
32 courants : 33 colonnes avec le câblage par défaut. Elle est écrite dans
un dossier frère terminé par `_static`. Le premier readout d'imagerie,
après les navigateurs, déclenche l'application ; choisir le dernier
timestamp pour couvrir toute la durée du scan après ce trigger. Les
figures de champ montrent la prédiction une fois les courants établis ;
la phase accumulée avant le trigger reste présente. Le CSV ne comporte
pas de retour à zéro : couper les courants après ce scan avec le système
externe, puis charger les waveforms dynamiques indépendantes. Le notebook
exporte les données sans piloter le FPGA. Le parcours dynamique reste
inchangé.

Les sections **9d à 9f** du v3 ajoutent ensuite **ACDC Dynamic B₀ Shimming**,
avec `solve_shim_waveform_fista` du même dossier Haotian. Elles utilisent
la B0 mesurée, répétée à chaque ADC d’imagerie, la même ROI et les mêmes
cartes ACDC. Elles minimisent le champ résiduel à chaque échantillon ADC
d’imagerie en tenant compte de la réponse temporelle des canaux. La carte B0
est constante ; les consignes peuvent varier pour compenser les
transitions du système, ou rester presque constantes si les courants
sont déjà établis avant les ADC. Les figures distinguent les consignes DAC,
les courants prédits, le champ résiduel et les RMS/NRMSE de champ.

`RUN_DYNAMIC_B0_SHIM` et `EXPORT_DYNAMIC_B0_WAVEFORM` activent ce calcul et
son export. Les réglages `CURRENT_RESPONSE_ENABLED` et
`CURRENT_RESPONSE_TAU_US` sont partagés avec le modèle existant ;
`B0_DYNAMIC_TRIGGER_DELAY_S` définit le délai du trigger depuis le début
du premier bloc de readout d'imagerie. Le CSV est écrit dans un dossier
frère terminé par `_dynamic_b0`. Il démarre par la première consigne à
0 seconde, contient un état par ligne et revient à zéro à la fin du
dernier readout. Le notebook contrôle aussi les courants et champs prédits
à tous les ADC avant l'export. Cette étape optimise le **champ B0** ;
le forward model et l'optimisation de l'image restent dans MATLAB et
conservent leur propre export. Le
[tutoriel](TUTORIEL_SHIMMING_DYNAMIQUE.md#shimming-dynamique-du-champ-b0-dans-le-notebook-v3)
détaille le timing et les contrôles.

La section **9g** du v3 prépare le point de départ de l'optimisation
**d'image MATLAB**. La cellule de paramètres propose
`IMAGE_INITIALIZATION = "static"` par défaut ou `"zero"`.
L'option `"static"` répète les 32 courants du shim statique
de la section 9b pour chaque état de la séquence ; `"zero"` initialise
tous les états à zéro. Les figures montrent ces consignes initiales et
leur champ prédit. Le notebook vérifie les limites avant de transmettre
les courants à MATLAB. Après un changement d'option, relancer le
notebook **depuis sa première cellule** : elle
crée un nouveau dossier temporaire et une nouvelle sortie, nécessaires
pour relancer une optimisation déjà exécutée. Si la ROI, la B0 ou la
calibration change, ce parcours recalcule aussi le shim statique.
Ce choix règle seulement le départ numérique du solveur : la B0 mesurée
reste la même et le modèle conserve des courants physiques nuls avant
les premières consignes. Il ne produit aucun CSV supplémentaire et ne
pilote pas le système.

Pour **comparer la référence GRE et les EPI mesurées sans et avec ACDC**,
ouvrir [03_reconstruct_epi_and_reference.ipynb](03_reconstruct_epi_and_reference.ipynb).
Ce notebook lit `gre_reference.dat`, `scan.dat` et `results/scan_corr.dat`
dans le dossier d'acquisition choisi. Il reconstruit chaque EPI avec ses
propres navigateurs, puis compare les trois magnitudes. Les deux EPI
partagent une même échelle de magnitude. Il utilise les séquences associées
aux acquisitions ; si les deux EPI utilisent la même séquence, elles
partagent `sequences/scan.seq`. Ses figures restent dans le notebook.
Le [guide de reconstruction](TUTORIEL_RECONSTRUCTION.md) explique les
paramètres et les figures. Aucun calcul MATLAB n'est lancé.

## Démarrer

Depuis la racine du dépôt, avec Python 3.11 ou plus récent :

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r Experiment/requirements.txt
```

Ouvrir le notebook dans Jupyter ou VS Code et sélectionner ce Python.
Lire ses cellules de paramètres, choisir le dossier d'acquisition et un
nouveau dossier de sortie, puis exécuter les cellules dans l'ordre.
L'exécution non interactive conserve également les figures dans le notebook :

```bash
.venv/bin/python Experiment/execute_notebook.py \
  Experiment/02_acdc_experiment_step_by_step.ipynb
```

Le script lance un noyau avec le Python de la commande, sans installer
un noyau global. MATLAB doit être accessible pour les étapes de simulation
et d'optimisation ; les premières figures Python permettent déjà de
contrôler les entrées avant son lancement.

Les cellules de paramètres contiennent `RUN_FORWARD_MODEL`,
`RUN_OPTIMIZATION` et `EXPORT_WAVEFORM`, initialement à `True`.
Pour examiner seulement les reconstructions, désactiver les calculs MATLAB.
Pour examiner le forward model avant le solveur, activer uniquement la
simulation. L'export écrit le CSV après convergence du solveur ou arrêt
sur la cible de MSE, puis validation des courants, du champ, de la
correspondance des canaux et des temps.

`WAVEFORM_PARAMETERS` contient l'ordre des sorties, les polarités et les
bornes de courant. Vérifier `channel_map` avec les 32 sorties réellement
câblées avant l'export ; une valeur `None` bloque cet export. `RF_DELAY_S = 0.0` exprime les temps depuis
le premier centre RF. Saisir le délai trigger → RF mesuré pour les exprimer
depuis le trigger. Ce décalage n'est pas estimé à partir du `.seq`.

Les paramètres de calcul, dont `MAX_ITERATIONS`, `DEVICE` et
`THREADS_PER_WORKER`, sont ceux indiqués dans le notebook.
`TARGET_MSE = 0.005` demande un arrêt dès que la MSE relative de magnitude
est strictement inférieure à cette valeur et que les limites de courant
et de champ sont respectées. Cette erreur est normalisée par l'énergie
de l'image idéale simulée à B0 nul ; ce seuil ne porte pas sur le champ
B0 résiduel. La cible est modifiable dans la cellule de paramètres ;
`TARGET_MSE = None` désactive ce critère d'arrêt.
Les autres critères du solveur et le maximum d'itérations restent actifs :
une session peut se terminer sans atteindre la cible.
`MATLAB_EXECUTABLE` désigne l'installation R2026a de ce poste ; adapter
ce chemin sur un autre poste.
`DEVICE` sélectionne le CPU ou le GPU dans le notebook. Si le pilote GPU
déclenche `parallel:gpu:device:OldDriver`, choisir `DEVICE = "cpu"` ;
le GPU demande une installation compatible avec cette version de MATLAB.
Après un changement de paramètre, réexécuter sa cellule, puis la cellule
qui commence par `# The bridge only orchestrates the existing MATLAB functions.`
pour reconstruire la requête MAT, avant de relancer l'étape MATLAB.
Réexécuter le notebook depuis les paramètres applique aussi ce changement.

Pendant l'optimisation, la sortie de la cellule affiche les messages
MATLAB au fil du calcul : itération et maximum demandé, nombre
d'évaluations, erreur sur les magnitudes, optimalité du premier ordre,
violation des contraintes et temps écoulé. Le journal conserve les mêmes
messages. La convergence et le respect des bornes sont rapportés à la fin ;
les indicateurs affichés pendant les itérations ne les certifient pas.
L'arrêt demandé sur la cible conserve `exitflag = -1`, le code MATLAB
d'un arrêt par la fonction de suivi. Il est distingué de la convergence
du solveur et peut être accepté pour l'export si tous les contrôles
physiques réussissent.
En cas d'échec, la cellule affiche le rapport d'erreur MATLAB détaillé
et le chemin du journal pour retrouver sa cause.

Chaque exécution choisit une nouvelle sortie
`acdc_output/notebook_<date_et_heure>/waveform.csv`. Le dossier n'est créé
que si l'export peut aboutir. Si un contrôle bloque le parcours, aucun
CSV n'est créé. Les paramètres et figures exécutés restent consultables
dans le notebook.

## Données actuelles

Les entrées attendues dans [acquisition_001](../acquisition_001) sont :

- `gre_reference.dat` : référence GRE/FLASH 2D mono-écho brute, **encore à déposer**.
- `b0_map.dat` : GRE B0 128 × 128 à trois échos.
- `scan.dat` : EPI 32 × 32 avec trois navigateurs, choisie pour cet essai.
- `ACDC_3T.npz` : volume des 32 réponses de bobines Bay5, en Hz/A.
- `sequences/scan.seq`, `sequences/reference_gre.seq` et `sequences/b0_map.seq`.

Le notebook attend maintenant une référence Twix `.dat` et son propre
`reference_gre.seq`. L'ancien DICOM présent dans ce dossier ne la remplace
pas. Les figures de référence montrent son k-espace brut, la reconstruction
complexe d'un canal et la magnitude RSS avec ses coordonnées physiques.
La grille et le FOV de cette référence peuvent différer de ceux du scan.

L'étape dédiée à la ROI construit le masque sur la référence GRE, puis
affiche le masque binaire, sa superposition verte sur l'image et les
pixels exclus. `REFERENCE_THRESHOLD` fixe la fraction du maximum de
magnitude utilisée pour construire le support. Dans cette étape,
`ROI_X_LIMITS_MM` et `ROI_Y_LIMITS_MM` permettent de fixer des limites
physiques facultatives `(minimum, maximum)` ; `None` conserve toute
l'étendue de l'axe. Les deux limites valent initialement `None`.
Le masque sélectionné reste en mémoire pour les étapes suivantes.
L'ancien fichier `acquisition_001/roi_b0_reliable.npy` n'est plus lu.

Le notebook conserve la confirmation du montage Bay5 et son recalage
identité. Rééchantillonner les cartes sur la référence n'effectue pas
un nouveau recalage. Les couvertures B0 et ACDC sont recalculées sur
la grille de la référence Twix. Tant que cette référence n'est pas
présente, aucune exécution complète sur ces nouvelles entrées n'est
validée. Les anciennes mesures de couverture sur le DICOM ne prédisent
pas les comptes de pixels de la nouvelle ROI. Le notebook conserve
la ROI définie par l'utilisateur et s'arrête si elle n'est pas entièrement
couverte par une B0 fiable.

Une fois la couverture B0 validée, les calculs numériques utilisent les
bornes de courant choisies. **L'export reste bloqué** tant que la
correspondance des 32 canaux avec les sorties physiques est invalide.
Un intervalle minimal entre changements de courant peut être
renseigné dans `min_dac_interval_s` pour vérifier l'espacement du CSV ;
`None` laisse cette vérification facultative désactivée.

## Calcul historique sur une ancienne ROI de 88 pixels

Le 6 octobre 2026, le forward model et l'optimisation ont été exécutés
avec MATLAB R2026a, sur une ancienne ROI de 88 pixels. Cette ROI n'est
plus chargée par le notebook actuel. Ce calcul utilisait l'ancienne
référence DICOM ; les résultats suivants ne décrivent pas une ROI
reconstruite depuis la nouvelle référence Twix. Le calcul historique a
effectué **428 itérations** en environ **544,3 s**. L'erreur relative
quadratique sur les magnitudes simulées est passée de **0,8735772490** à
**0,3024949604**. Les courants restent sous 1 A par canal, leur somme
absolue atteint au plus **27,994167 A**, et le champ ACDC généré au plus
**99,973071 Hz**, pour des bornes de 30 A et 100 Hz.

L'arrêt `exitflag = 2` correspond à un pas relatif devenu inférieur à
la tolérance, avec les contraintes respectées. Il ne certifie pas
l'optimalité du premier ordre ni un optimum global. Ce résultat porte
sur le modèle et l'ancienne région choisie ; aucune acquisition corrigée n'a
mesuré un gain physique. Ce calcul historique n'a pas donné lieu à une
application de courant.
Les résultats et figures sont dans
[notebook_20261006_115330](acdc_output/notebook_20261006_115330).

Les figures montrent aussi les k-espaces avant et après correction,
les masques B0 avant et après filtrage des petites régions, ainsi que
le champ ACDC et le champ instantané `B0 + champ ACDC`. La borne de champ
concerne uniquement le champ ACDC généré. La mise à jour des figures a
relu les résultats numériques après vérification des entrées ; elle n'a
pas relancé une seconde optimisation complète.

## Formats et portée

Les matrices, FOV, TE, TR et dwell sont lus dans les acquisitions et leurs
séquences. La grille de référence, la matrice du scan et les 32 canaux
ACDC restent distincts. Les cartes déjà alignées sont interpolées dans
leurs coordonnées physiques sans extrapolation de B0 ni de calibration.
Le notebook 02 demande une référence Twix GRE/FLASH brute mono-écho
et un Twix B0 brut multi-écho pour afficher les images
complexes et les étapes de phase. Les anciennes B0 déjà reconstruites
NPZ/MAT restent acceptées par le script CLI secondaire ; elles ne
remplacent pas directement ces échos dans le notebook. Les calibrations
MAT/NPZ restent lisibles par l'import de calibration.

Le scan à corriger peut être une GRE cartésienne 2D mono-écho, une RF par
ligne, ou une EPI à une excitation. Une coupe, une image complète et des
matrices paires sont pris en charge. En GRE, le modèle suppose un spoiling
idéal, des RF de forme et d'angle constants dans chaque séquence et aucun
régime stationnaire T1. Le shim revient à zéro pendant les RF et entre
les lignes GRE. Un trigger externe démarre l'image complète, excitations
factices initiales comprises.

Le notebook produit uniquement `waveform.csv`. Il ne pilote aucun
contrôleur, ne se connecte pas au FPGA et n'écrit aucun script de commande.

## Compatibilité avec le pipeline par dossiers

Le script [run_acdc_experiment.py](run_acdc_experiment.py) et les anciennes
configurations JSON restent disponibles pour les usages existants. Leur
parcours `init` → `prepare` → `run` → `export` est secondaire ; la nouvelle
interface conserve les paramètres dans le notebook.

Le notebook [01_reconstruction_et_B0.ipynb](01_reconstruction_et_B0.ipynb)
est désormais en anglais et conserve l'analyse historique d'origine.
Son extension de B0 au voisin le plus proche reste une hypothèse de
simulation virtuelle ; elle n'est pas utilisée pour préparer la nouvelle
expérience. Les sections suivantes documentent cette étude distincte.

## Données et analyses EPI historiques

Les sections suivantes décrivent les anciennes acquisitions EPI 32 × 32,
leur référence 128 × 128 et l'étude virtuelle associée. Ces tailles
décrivent ces données, pas les limites du nouveau pipeline GRE.

Le notebook **01_reconstruction_et_B0.ipynb** est exécuté :
les images, les nombres et les explications sont déjà visibles.

Il reconstruit les deux EPI 32 × 32 depuis les Twix, utilise les trois
navigateurs pour une correction affine pair/impair par canal, puis combine
les 32 canaux de réception en RSS. Les trois échos GRE DICOM 128 × 128
permettent de calculer une carte B0 en Hz avec un masque de confiance.

Les images mesurées sont `figures/epi_scans_32x32.png`. La correction des
navigateurs est illustrée dans `figures/epi_navigator_correction.png`.
La B0 mesurée et son extension pour le modèle sont distinguées dans
`figures/b0_map_quality.png`.

## Réexécuter le notebook

Python **3.11 ou plus récent** est nécessaire pour le lecteur Pulseq utilisé.
Depuis `Experiment` :

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python execute_notebook.py 01_reconstruction_et_B0.ipynb
```

L'environnement de validation de cette session est
`/tmp/experiment_mri_py311`. Il est temporaire ; la commande ci-dessus
installe un environnement réutilisable. Le notebook peut aussi être ouvert
dans Jupyter ou VS Code avec ce Python. Le script utilise un noyau local et
enregistre les sorties dans le notebook.

Les fichiers `.seq` ont été copiés depuis `~/scanner/sequence` dans
`Experiment/sequence`. La trajectoire ADC et les temps sont extraits du
fichier EPI, et sa signature SHA256 est sauvegardée avec les résultats.
Ses TE/TR/dwell sont contrôlés contre les Twix. Le fichier Pulseq déclare
90° d'excitation, contre 15° dans le header Siemens ; cette différence est
documentée. La version du format `.seq` est 1.5.1, contre 1.5.0 pour le
lecteur : les dimensions, la grille k-espace et les temps calculés sont
vérifiés explicitement.

## Comparer les scans à la simulation Bloch dans MATLAB

Depuis le dossier parent du projet :

```matlab
addpath('Experiment');
[run_dir, report] = run_experiment_digital_twin();
```

La figure `digital_twin_comparison.png` et sa version `.fig` sont sauvegardées
dans un nouveau dossier sous `Experiment/results`. Elle contient :

1. Le scan sans perturbation A22 ajoutée.
2. Le scan avec perturbation A22.
3. Le jumeau simulé avec B0=0, hypothèse de référence.
4. Le jumeau simulé avec la B0 totale de l'acquisition GRE perturbée.

Un gain positif est ajusté uniquement sur le baseline et conservé pour la
condition perturbée. Les quatre panneaux ont la même échelle. Le rapport
texte et le MAT donnent les erreurs et les hypothèses.

L'essai historique du **1er octobre 2026** sur ces données donne une NRMSE de **18,6 %** sur le
baseline et **73,6 %** sur le perturbé. Le modèle conditionnel ne prédit
donc pas encore fidèlement le scan perturbé. Ces erreurs ne sont pas des
preuves de validation du jumeau et ne doivent pas servir à approuver une
correction physique.

Le modèle résout la précession transverse aux ADC de la séquence Pulseq.
Il utilise les coordonnées physiques et des temps depuis le centre RF.
L'entrée est une amplitude transverse effective au TE, interpolée depuis
le baseline EPI, et non une mesure de densité protonique. T2* vaut Inf par
défaut. Une hypothèse explicite peut être essayée, en secondes :

```matlab
[run_dir, report] = run_experiment_digital_twin([], 'T2starSeconds', 0.08);
```

Cette valeur d'exemple n'est pas une mesure du fantôme. Le modèle conserve
une réception uniforme ; les effets RF, le profil de coupe, le régime
stationnaire TR/T1 et les sensibilités des 32 canaux ne sont pas identifiés
par les données disponibles.

## Ce qui manque pour une validation physique complète

Le scan sans A22 ajouté n'est pas à B0 nul : sa carte B0 n'a pas été fournie.
La GRE mesure seulement l'état perturbé. Les réglages de shim scanner et la
fréquence émetteur changent aussi entre les EPI. On ne peut donc pas séparer
le seul effet A22 par soustraction de ces images.

Le masque de phase conserve 1778 pixels. Avec les réglages actuels, il
couvre environ 20 % du support de simulation. En dehors, le modèle utilise
le B0 du pixel fiable le plus proche ; cette extension est une hypothèse,
distincte des valeurs mesurées conservées avec des NaN hors masque.
Le déroulement de phase laisse également une ambiguïté de branche absolue
et de convention de signe à vérifier avec un champ de signe connu.

L'accord sur le baseline est une calibration, puisqu'il fournit l'amplitude
du jumeau. L'erreur sur le perturbé évalue le modèle conditionnel avec ses
hypothèses ; elle ne certifie pas encore un jumeau validé.

Pour améliorer cette étape, acquérir une GRE multi-écho du baseline avec
les mêmes réglages que son EPI, améliorer le signal et la couverture B0,
puis ajouter une estimation indépendante de l'objet et des sensibilités
de réception si une prédiction quantitative est recherchée.

## Optimisation virtuelle ACDC sur les mesures IRM

Le fantôme a été scanné **sans bobine ACDC**. L'optimisation emploie donc les
images et la B0 mesurées, avec une bobine ACDC virtuelle décrite par les
réponses Hz/A de `ACDC_maps/ACDC_3T.npz`. Le placement virtuel correspond à
l'identité entre les coordonnées patient et les coordonnées des cartes.
Voir **ACDC_MAP_IMPORT.md** pour la géométrie et la couverture.

Depuis le dossier parent du projet, commencer par un essai court :

```matlab
addpath('Experiment');
[run_dir, summary] = run_experiment_virtual_acdc('cpu', ...
    'MaxIterations',3,'FieldLimits',[100 1000]);
```

Pour les cinq limites et un maximum de **600 itérations** :

```matlab
[run_dir, summary] = run_experiment_virtual_acdc('cpu');
```

Les limites sont **±1 A par canal** et **somme des valeurs absolues <30 A à
chaque ligne EPI**. Le CPU utilise jusqu'à **5 workers × 6 threads**, plus un
thread client : 31 <32. Les caches sont préparés une seule fois par worker ;
le stockage temporaire du pool est placé dans `/scratch` et supprimé à la fin.

Le backend GPU est disponible sous MATLAB **R2026a** :

```matlab
[run_dir, summary] = run_experiment_virtual_acdc('gpu', ...
    'MaxIterations',3,'FieldLimits',1000);
```

Il emploie **un worker et six threads CPU**, avec un cache GPU persistant en
double précision. Les différentes limites passent successivement sur le GPU.
Le choix CPU/GPU et les temps mesurés sont expliqués dans
**OPTIMIZATION_SPEED.md** et **GPU_ACCELERATION.md** : accélérer l'objectif
ne suffit pas à accélérer les contraintes du solveur.

L'entrée séparée `prepared/acdc_calibration/experiment_input_acdc_support.mat`
contient les **8125 points couverts**. La référence idéale est recalculée sur
ce même support ; l'entrée originale du notebook conserve ses 8746 points.
Les cartes inconnues hors couverture restent `NaN`. Aucun fichier du run4
historique n'est modifié.

`summary.status='complete'` indique que les fichiers ont été sauvegardés.
Vérifier aussi `summary.all_converged`, `summary.all_current_limits_met` et
`summary.all_field_limits_met`. Un essai limité à trois itérations ne certifie
pas la convergence. Le modèle du scan perturbé conserve les limites décrites
plus haut ; l'accélération numérique ne les supprime pas.

Dans ces calculs historiques, les courants sont supposés nuls pendant la RF et les navigateurs, puis
constants sur chaque bloc d'imagerie. Le modèle cumule leur phase dans le
temps ; il ne suppose pas que la magnétisation repart à zéro à chaque ligne.
Les transitions de courant sont idéales, sans modèle de slew matériel.
Les contraintes de champ portent sur les points de B0 de confiance.

## Vérifications

```matlab
validate_experiment_digital_twin();
validate_experiment_optimizer();       % gradient et physique, données synthétiques
validate_acdc_current_response();      % réponse temporelle, phase, adjoint et cache CPU
validate_experiment_optimizer(true);   % ajoute les contrôles du pool sur des cas synthétiques
validate_experiment_acceleration([], 'Devices',{'cpu','gpu'}); % équivalence et gradient
validate_experiment_current_constraint(); % somme absolue et dérivée
validate_experiment_field_constraints();  % suppression exacte des contraintes redondantes
```

Ce contrôle compare le modèle à une transformée de Fourier connue,
vérifie un point hors résonance, l'orientation des axes et une dérivée de
champ. Le notebook vérifie les dimensions, la phase de navigateur et les
métadonnées de séquence avant d'exporter.

La validation de l'optimiseur utilise des cartes synthétiques uniquement
comme fixtures temporaires : elles sont supprimées et ne sont jamais
exportées comme calibration de l'expérience. Le gradient a une erreur
relative inférieure à 10⁻⁹ sur quatre directions ; le modèle à courant nul
reproduit exactement le modèle statique. Deux workers ont aussi été
contrôlés avec fermeture du pool et restauration des threads du client.

Les références des lecteurs et des méthodes de B0 figurent à la fin du
notebook. Les données brutes restent dans `scan_twix` et `scan_dcm`.
