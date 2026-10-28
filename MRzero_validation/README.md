# Dynamic ACDC shimming with MRzero

### 05/06 — calibration partagée et exports waveform

Les sept notebooks 05/06, copies, V1 et versions ClosedLoop chargent maintenant
`CALIBRATION_PATH=PROJECT_ROOT/"data/ACDC_3T.npz"`. Dans cette installation,
ce fichier est celui demandé sous `/homes/7/mv1289/MAURICE/to_be_sorted/epi_acdc_artifactCorr_MrZero_pc/epi_acdc_artifactCorr_Mrzero/data`.
Il n'est ni recréé ni recopié dans chaque acquisition. Les contrôles d'empreinte,
d'unités et de placement de la calibration restent actifs.

`WAVEFORM_OUTPUT_DIR=INPUT_DIR/"waveform"` reçoit les CSV :

- `waveform1.csv`, `waveform2.csv`, etc. : optimisation image, première commande non nulle à `t=0` ;
- `waveform1_zero_time.csv`, etc. : copie de chaque CSV image, seule la colonne temps devient zéro ;
- `waveform_adc.csv` : temps de mesure des courants, une colonne en secondes remise à zéro à chaque trigger ;
- `waveform_static.csv` : shim de champ statique, QP du dépôt ACDC_Optim ;
- `waveform_dynamic.csv` : shim de champ avec FISTA CUDA et SIRF du même dépôt.

Les fichiers existants sont préservés : un nouvel export statique/dynamique prend
le suffixe `2`, `3`, etc. Chaque export de champ possède un JSON de provenance
et un NPZ de commandes, temps et valeurs DAC vérifiées. Les rapports/images de
simulation restent dans leur dossier de résultats. Les propositions fermées des
nouvelles sessions sont sous `waveform/closed_loop/<session>_<identifiant>/round_N`.
Les sessions existantes gardent leurs chemins archivés.

Les sept notebooks proposent une cellule de mesure des courants après la lecture
des acquisitions, avant la construction de MRzero. `EXPORT_ADC_WAVEFORM=True`
active cet export indépendamment des optimisations ; `WAVEFORM_ADC_INTERVAL_S`
fixe le pas, avec `0.000050` s (50 µs) par défaut. Le fichier est numérique, sans
en-tête : chaque trigger vérifié démarre un segment `0`, `0.000050`, `0.000100`,
etc., jusqu'au trigger suivant exclu. Le dernier segment se termine à la fin du
dernier readout image. `WAVEFORM_ADC_DURATION_S=None` conserve cette fenêtre ;
une valeur positive remplace sa durée totale, mesurée depuis le premier trigger.
Le JSON associé conserve les déclenchements, les bornes et les indices des
segments pour retrouver les temps absolus. Une paire CSV/JSON identique est
réutilisée ; des paramètres différents créent `waveform_adc2.csv`, etc.
Avec la séquence v6 actuelle à 64 triggers espacés de 760 µs, le pas de 50 µs
donne 1 024 lignes : 64 segments de 16 échantillons allant de 0 à 750 µs.

Les exports acceptent une séquence EPI avec un seul trigger au début de
l'imagerie ou un trigger par ligne image, comme les 64 triggers `ext1` de la
séquence `gre_epi_64x64_fov200mm_nav3_cimax_snr_t2star40ms_line_triggers_v6.seq`.
Dans ce second cas, chaque événement doit coïncider avec le début de son état
d'imagerie. Le premier fixe l'origine `first_imaging_trigger` ; le rapport
`trigger` conserve le mode, le nombre, les temps et les indices de tous les
événements vérifiés. Cette vérification porte sur la séquence ; elle ne détermine
pas le mode de lecture des CSV par le contrôleur.

Pour les exports image, les états initiaux entièrement nuls sont retirés et les
temps restants sont recalés sur la première commande non nulle, sans ajout de
ligne. Les intervalles et le retour final à zéro sont conservés. Les métadonnées
enregistrent le décalage de l'horloge et les états retirés ; l'aperçu et l'import
en boucle fermée les utilisent pour retrouver les temps du modèle. Un résultat
entièrement nul conserve ses états, avec le premier temps à zéro. Le compagnon
`_zero_time.csv` conserve à l'identique l'en-tête et les colonnes de courant du CSV
recalé. Les fichiers historiques sont conservés.

`EXPORT_FIELD_SHIM_WAVEFORMS=True` génère les deux solutions de champ, indépendamment
de `RUN_OPTIMIZATION`, de l'initialisation image ou du résultat du solveur image.
`FIELD_SHIM_RASTER_S=10e-6` et `FIELD_SHIM_MAX_ITERATIONS=80` règlent le calcul
dynamique. Le champ cible est le B0 constant mesuré/modélisé, de signe opposé ;
la dynamique sert à préaccentuer la réponse des amplificateurs. Aucune carte
B0(t) n'est inventée à partir des 32 échos.

Les solveurs du [dépôt ACDC_Optim](https://github.com/bughht/ACDC_Optim) sont inclus
sans modification au commit `703af140b6cb05df45a05f8d49bc4417a8be9d56`, avec sa licence MIT.
Leur dépendance QP `quadprog` est ajoutée aux fichiers d'installation ; PyTorch
et `tqdm` sont également requis pour les solveurs.
Une réduction QR exacte évite de créer une matrice temps × tous les voxels.
Le noyau exponentiel normalisé est construit avec le tau configuré du modèle.
La fenêtre exportée et son origine viennent des événements EPI et du trigger
Pulseq vérifiés. Les amplitudes, somme des courants, champs aux coins des voxels
et limites électriques configurées sont vérifiés en temps continu, avant et
après DAC. Une contraction uniforme éventuelle est documentée, sans écrasement
silencieux des courants. Les limites non mesurées restent désactivées.

### 06 — canaux hors service

Dans `06_acdc_echo32_b0_s0_t2star_MRzero.ipynb`, `DISABLED_CHANNELS=[]`
conserve toutes les voies. Renseigner par exemple `[7]` ou `[7, 12]` pour
désactiver une ou plusieurs **sorties physiques**, numérotées de 1 à 32 dans
la configuration habituelle. La sortie 7 correspond à `ch6_A` dans l'en-tête
CSV, qui commence à zéro. Si `channel_map` est permuté, les numéros désignent
toujours les sorties physiques, pas les colonnes des cartes de calibration.

Le notebook convertit ce choix en bornes de courant nulles, communes au modèle,
au shim initial et aux exports. L'optimisation image contraint ces voies à zéro
à chaque itération ; les ajustements statique et dynamique utilisent uniquement
les voies actives avant de réinsérer les colonnes nulles. Les autres courants sont
donc réoptimisés. Les exports refusent toute commande non nulle sur une sortie
désactivée. Les numéros des voies, leurs bornes et la configuration sont conservés
dans les résultats JSON/NPZ.

Après installation de cette modification, redémarrer le noyau et relancer le
notebook. Pour changer la liste ensuite, relancer depuis la cellule de paramètres
afin de reconstruire le modèle ; modifier seulement une cellule d'export ne
recalcule pas les autres courants.

### 06 — objet GRE + 32 échos, optimisation vers la GRE mesurée

Les deux notebooks 06 distinguent maintenant la construction du modèle et sa
cible d'optimisation. `OBJECT_AMPLITUDE_SOURCE="joint"` combine la structure GRE
avec l'enveloppe et la dynamique des 32 échos. Une correction positive et lisse
du contraste entre acquisitions est estimée sur les échos d'entraînement, à
l'échelle fixée par `OBJECT_CONTRAST_SIGMA_MM=10`. Les six échos réservés choisissent
le poids du candidat ; sans amélioration, l'enveloppe multi-échos est conservée.
Les cartes de contraste et la décision sont archivées. Cette correction effective
ne constitue pas une mesure de T1, PD ou sensibilité de réception.

À l'étape 6, la cible est toujours **la GRE mesurée sélectionnée**, indépendamment
du poids retenu pour construire l'objet. Sa matrice peut différer de l'EPI : elle
est interpolée par ses coordonnées physiques sur la grille EPI, sans extrapolation
ni création de détails acquis. Le coût est :

`J(I) = sum_M((g * abs(EPI_MRzero(I_ACDC)) - GRE_mesurée)**2) / sum_M(GRE_mesurée**2)`.

Le gain `g` est fixé avant optimisation, par les deux étapes simulation sans ACDC
→ EPI mesurée sans ACDC → GRE. Le masque `M` et la GRE ne changent jamais avec les
courants. `OPTIMIZATION_REFERENCE_THRESHOLD=0` conserve toute la couverture commune,
y compris les pixels sombres. Une valeur positive choisit une ROI GRE fixe.
Les gradients, affichages, coûts après DAC et diagnostics de réponse utilisent
cette même cible. La simulation B0 nul reste un diagnostic physique distinct.

`measured_image_target.py` installe ce coût sans modifier l'objet ou la GRE.
Les notebooks antérieurs gardent leur coût initial tant qu'aucune cible mesurée
n'est installée. Redémarrer le noyau après mise à jour des modules ; créer une
nouvelle session fermée pour employer le nouveau modèle d'entrée.

### 06 — fortes distorsions et repliement EPI

La reconstruction Fourier représente déjà le signal qui dépasse un bord du FOV
et réapparaît de l'autre côté. Une MSE de magnitude seule peut toutefois rester
dans un minimum correspondant à une mauvaise correction physique. Le notebook
`06_acdc_echo32_b0_s0_t2star_MRzero.ipynb` ajoute un guidage optionnel :

```python
IMAGE_INITIALIZATION="static"
FIELD_GUIDANCE_WEIGHT=0.05
FIELD_GUIDANCE_SCALE_HZ=100.0
```

Le solveur minimise `MSE_image + weight * mean((B0_residuel / scale_Hz)**2)`.
Le résidu conserve le B0 absolu en Hz, y compris sa moyenne ; il n'est jamais
réduit modulo 2π. Il est calculé dans le masque de qualité du modèle, aux positions
physiques des sous-spins. Le champ correcteur est moyenné entre les centres
temporels des readouts successifs à partir des intégrales exactes de réponse des
amplificateurs. Ce guide de pente de phase complète le modèle image complet,
qui continue à calculer tous les ADC et les effets de voxel fini.

Le poids `0.05` est un point de départ à comparer expérimentalement, pas une
valeur validée pour chaque fantôme. `0` conserve exactement l'objectif image seul.
L'échelle `100 Hz` normalise le coût ; elle ne limite pas le champ. Les limites
matérielles et `DISABLED_CHANNELS` restent appliquées à chaque candidat.

`fval`, `best_fval` et la NRMSE restent des métriques image ; `best_fval` est le
minimum image rencontré. Le candidat retourné minimise le **coût total**, suivi
séparément par `objective` et `best_objective`. `field_penalty` et
`residual_field_rms_Hz` sont enregistrés dans les historiques et les résultats.
Avec un poids positif, `TARGET_MSE` reste un diagnostic et ne déclenche plus
l'arrêt : une faible MSE peut encore correspondre à une mauvaise branche.
Après DAC, les limites matérielles, la MSE image et le coût total sont revérifiés.
Les deux coûts doivent être au plus ceux du point initial pour autoriser l'export.

Ce guide suppose une carte B0 correctement déroulée et référencée à la fréquence
de l'EPI. Un plafond de champ trop bas ou une combinaison de bobines incapable
de reproduire le champ demandé reste une limite physique ; augmenter le poids
ne la supprime pas. La récupération de tout le signal perdu n'est pas garantie.
Redémarrer le noyau après mise à jour des modules, puis relancer depuis les
paramètres pour comparer un essai guidé à un essai avec un poids nul.

### 06 — choix de la référence GRE et de la cartographie B0

Les notebooks [06](06_acdc_echo32_b0_s0_t2star_MRzero.ipynb) et
[06 boucle fermée](06_ACDC_ClosedLoop_RealFeedback.ipynb) proposent les mêmes
sélections indépendantes :

```python
GRE_EPI_SEQUENCE_NAME="gre_epi_128x128_fov200mm_nav3_dwell5us_rise240us_ext1_single_trigger_first_readout_v5.seq"
GRE_EPI_SCAN_PATH=None
REFERENCE_SCAN_PATH="meas_MID00758_FID48501_reference128.dat"
REFERENCE_SEQUENCE_NAME=None
B0_SEQUENCE_NAME="megre_b0_s0_t2star_128x128_fov200mm_32echo.seq"
B0_SCAN_PATH=None
```

Les chemins `.dat` sont libres, relatifs à `INPUT_DIR` ou absolus. Pour la
référence 64² disponible, choisir `meas_MID00757_FID48500_reference64.dat`.
Avec `None`, les motifs `*reference*.dat`, `*b0_map.dat` et `*scan.dat` doivent
donner chacun un fichier unique ; les références/B0 sélectionnées sont exclues
de la recherche EPI. Le notebook fermé conserve ses choix synthétiques initiaux.

Une référence conserve sa matrice, son FOV et ses coordonnées physiques. Son
propre `.seq`, lorsqu'il est fourni, est vérifié strictement. Sans séquence
compatible, le lecteur vérifie une acquisition mono-écho cartésienne complète
dans le Twix et reconstruit sur la grille nominale de son en-tête. Le rapport
signale que la trajectoire Pulseq exacte n'est alors pas vérifiée ; aucun fichier
de séquence d'une autre résolution n'est présenté comme correspondant.

Le mode historique `OBJECT_AMPLITUDE_SOURCE="auto"` évalue la référence comme candidat
d'amplitude MRzero : transfert de contraste TE avec les pertes ajustées,
gain global estimé sur les échos d'entraînement, puis choix du mélange sur les
échos complexes réservés. Le candidat doit améliorer leur erreur d'au moins
`REFERENCE_PRIOR_MIN_VALIDATION_IMPROVEMENT=0.02`. `"echo32"` conserve
l'amplitude multi-échos et `"reference"` impose la référence sur sa couverture
valide, avec les diagnostics de validation. La décroissance et les zéros de
cohérence restent ceux des 32 échos. Une seule GRE ne mesure ni PD ni T1 ; un
gain global ne corrige pas les différences spatiales de saturation entre TR.

Les chemins réellement utilisés, leurs empreintes, la validation de géométrie
et la décision sur l'amplitude sont archivés. Une session fermée existante
conserve ses données et son modèle figés ; choisir un nouveau dossier de session
pour intégrer une autre référence ou changer la modélisation.

### 06 — signal complexe des 32 échos et calcul CUDA

Le notebook 06 ajuste désormais les échos complexes par récepteur avec la
covariance du bruit. La fréquence centrale, l'enveloppe au premier TE et la
décroissance résiduelle sont séparées des gradients B0 dans le plan. Les
modèles exponentiel, dispersion uniforme (sinc signé) et dispersion gaussienne
sont comparés sur les six derniers échos réservés ; le modèle retenu est
ensuite réajusté aux 32 échos. Un zéro ou une courbe non exponentielle ne
déclenche plus le repli sans décroissance du modèle historique.

`ECHO_SIGNAL_MODEL="auto"` active ce modèle pour les acquisitions mesurées ;
`"legacy"` conserve la comparaison historique. Les tableaux et les planches
MEGRE comparent les pertes mesurées et prédites, notamment à l'écho 15. Les
erreurs de sélection sur échos réservés sont distinctes des erreurs après
réajustement de tous les échos. Les masques, paramètres, prédictions et
diagnostics sont enregistrés avec l'optimisation.

`complex_echo_fit.py` utilise une variable projection des amplitudes complexes
et Levenberg–Marquardt par blocs sur le périphérique sélectionné.
`multiecho_object.py` prépare cet objet ; `experiment_multiecho.py` applique
sa cohérence au signal natif MRzero avec les gradients exacts des courants.
La dispersion ajustée reste sans direction spatiale identifiée : elle est
conservée dans le diagnostic B0 nul et ne crée pas de restauration ACDC supposée.
Les gradients dans le plan sont ceux fixés pendant le fit, appliqués une seule
fois. L'extrapolation au-delà du dernier écho reste une hypothèse à vérifier.

La reconstruction GRE/EPI et des 32 échos accepte maintenant `device` et
`dtype`. Le notebook utilise le GPU pour la reconstruction, le fit complexe
et le modèle MRzero. `GPU_MEMORY_FRACTION=0.60` favorise le cache natif sur
GPU ; la taille des blocs reste bornée par la mémoire disponible. L'affichage
toutes les 20 itérations réduit les interruptions CPU. `GPU_REQUIRED=True`
empêche une exécution CPU silencieuse ; le mode CPU volontaire reste disponible.

### Optimisation ACDC avec retours d'acquisitions

Les nouveaux notebooks [Version 05](05_ACDC_ClosedLoop_RealFeedback.ipynb)
et [Version 06](06_ACDC_ClosedLoop_RealFeedback.ipynb) conservent leurs
préparations physiques respectives et ajoutent une recherche sur l'amplitude
des commandes historiques, calibrée par les acquisitions. Deux nouveaux scans
au maximum sont importés manuellement, avec une session persistante et les
contrôles matériels existants. Les originaux 05/06 sont conservés.
Le [guide de démarrage et des rounds](CLOSED_LOOP_README.md) décrit les associations
scan/CSV, la normalisation fixe, les reprises et les limites de cette recherche.

### Pertes de signal réelles et déficits ajoutés par la simulation

Les diagnostics des notebooks 05/06 et des deux scripts de validation mesurent
désormais les écarts dans les **deux sens** : une simulation trop sombre malgré
un signal mesuré, mais aussi une simulation trop lumineuse là où l'EPI réelle
perd du signal. L'ancien compteur `holes`, conservé pour la compatibilité des
archives, ne regardait que les pixels mesurés au-dessus de 20 % du maximum : il
ne pouvait donc pas évaluer correctement les creux sombres de l'EPI réelle.

`signal_disagreement.py` fournit la différence signée simulation − mesure et
décompose son erreur quadratique en excès et déficits de signal. Le score sur
tout le champ de vue inclut les pixels mesurés sombres ou nuls, sans seuillage.
Les compteurs aux rapports 25 % et 50 % sont des indicateurs complémentaires,
avec un seuil de luminosité annoncé et un gain figé. Une enveloppe dérivée de
la référence B0 nul sert uniquement à situer ces écarts ; elle ne change ni
l'objet ni le score global et ne constitue pas une vérité terrain. Deux images
sous le seuil peuvent encore présenter un désaccord important.

La MSE de calibration pénalisait déjà les deux sens. Ces diagnostics ne
modifient pas l'ajustement physique. Dans les notebooks 06, l'optimisation ACDC
compare désormais directement la simulation à la GRE mesurée.
Une baisse des déficits artificiels ne démontre pas que les pertes réelles sont
reproduites : leur position, leur profondeur et leur étendue restent à vérifier.
Les écarts observés seuls n'identifient pas leur cause physique et ne doivent
pas être recopiés comme des trous dans l'objet GRE.

`report_signal_disagreement.py` réévalue les archives de comparaison et de
calibration sans relancer MRzero ni ajuster de gain. Il vérifie les coordonnées,
annonce les gains historiques et fournit les résidus signés, les compteurs aux
seuils de luminosité 10 % et 20 %, ainsi que la provenance des fichiers :

```bash
python MRzero_validation/report_signal_disagreement.py --calibration-archive <calibration.npz> --legacy-archive <comparison.npz> --output <nouveau_dossier>
```

### Calibration bornée de la baseline EPI

Les notebooks 05 et 06 ajoutent une étape avant l'initialisation des courants :
`BASELINE_CALIBRATION="auto"` l'active sur les acquisitions mesurées et la
désactive pour les deux formats synthétiques connus ; `"off"` conserve le modèle
sans ajustement EPI. Les réglages de l'optimisation restent indépendants.

`baseline_metadata_audit.py` compare les fréquences déclarées dans les en-têtes
GRE/B0/EPI, examine le résidu des navigateurs déjà corrigés et quantifie la
quasi-dégénérescence entre fréquence globale et déplacement en direction de phase.
`baseline_calibration.py` ajuste seulement trois paramètres possibles :

- un décalage fréquentiel global, borné à ±30 Hz ;
- un déplacement de l'objet dans la direction de lecture, borné à ±2 mm ;
- une phase constante négatif-moins-positif résiduelle, bornée à ±0,15 rad.

Le déplacement en direction de phase et la rotation sont fixes. Aucune carte B0
voxel par voxel, amplitude, relaxation ou déformation d'image n'est ajustée.
La phase résiduelle ne réapplique pas la correction des trois navigateurs.
Le signal complexe simulé est multiplié par le facteur exact correspondant aux
paramètres, puis reconstruit avec les coordonnées ADC réelles. La comparaison
porte sur la magnitude RSS : le récepteur virtuel ne fournit pas les phases
individuelles des 32 récepteurs mesurés.

Les blocs spatiaux de 8×8 pixels, séparés par des bandes non utilisées,
définissent les ensembles d'ajustement, de sélection et de test. Le gain
d'intensité est estimé seulement sur l'ensemble d'ajustement, puis figé. Ajouter
un paramètre doit améliorer la NRMSE de sélection d'au moins 0,05 point. Le test
n'intervient ni dans les paramètres ni dans le choix du nombre de paramètres ;
il sert ensuite à autoriser ou refuser leur application. Un ajustement en butée,
mal conditionné ou sans amélioration admissible est refusé. Ce contrôle reste
**une validation spatiale interne d'une acquisition**, déjà utilisée pour le
développement du modèle ; il ne remplace pas une nouvelle EPI indépendante.

Après acceptation, le notebook reconstruit le modèle MRzero depuis les entrées
originales : B0 reçoit l'offset, les positions sont déplacées, les cartes ACDC
(centres, gradients et coins) sont rééchantillonnées. La phase ADC résiduelle
reste dans la cible B0 nul, tandis que l'offset fréquentiel y est supprimé avec
B0. Une parité complexe vérifie ce nouveau calcul natif contre le facteur
analytique utilisé pendant l'ajustement. Les courants et leur gradient utilisent
donc le modèle calibré, pas seulement une image corrigée pour l'affichage.
Les paramètres, masques, en-têtes, gains et images avant/après sont archivés.
Le déplacement de lecture est interprété comme un recalage rigide de l'objet
GRE et de son B0. Une baseline RSS seule ne permet pas de le distinguer d'une
erreur instrumentale d'origine ou de rampe de phase ; elle ne valide donc pas
indépendamment le placement relatif de l'objet et des bobines ACDC.

Pour reproduire la calibration et vérifier le modèle natif, sans export CSV :

```bash
python MRzero_validation/validate_baseline_calibration.py --profile auto --subspins-z 1 --smoke-iterations 2
```

Les résultats sont écrits sous `results_baseline_calibration/`. Le test du solveur
est limité à deux itérations ; il ne démontre pas la convergence ni l'efficacité
d'une acquisition corrigée au scanner.
Le script utilise par défaut des blocs natifs de 128 ADC. Les notebooks réutilisent
la taille effective du premier modèle pour la reconstruction calibrée : changer
le découpage modifie les erreurs d'arrondi des historiques MRzero float32 et peut
fausser le contrôle de parité. Les tailles effectives sont archivées.

### Fidélité EPI — B0 pondéré et intégration des voxels

Les notebooks 05 et 06 proposent `SIMULATION_MODEL="auto"` : les acquisitions
réelles utilisent `FiniteVoxelEchoACDCModel`, tandis que les acquisitions
synthétiques des générateurs ponctuels conservent le modèle qui les a générées.
`"point"` et `"finite_voxel"` permettent une sélection explicite. Les anciens
appels Python gardent leurs valeurs par défaut et leurs résultats ponctuels.

`B0_WEIGHTING="auto"` sélectionne, pour les mesures réelles, une régularisation
locale pondérée par la puissance des produits complexes entre échos précoces.
Les pixels de phase faible contribuent moins ; la couverture B0 n'est pas
étendue et cette régularisation n'ajuste aucun paramètre sur l'image EPI. Le mode `"uniform"`
reproduit la pondération précédente. L'échelle spatiale reste fixée en pixels natifs.

Le modèle fini conserve les aimantations natives MRzero au centre des sous-spins
suivant z. Il intègre analytiquement la surface du pixel natif avec un champ
localement linéaire : `sinc(q_row) * sinc(q_column)`, où chaque phase inclut les
moments d'encodage, le gradient B0 et l'intégrale des courants ACDC. Le gradient
de l'optimisation inclut les dérivées de ces deux facteurs. Le cache MRzero
n'est pas multiplié par une grille supplémentaire dans le plan. Les contraintes
de champ couvrent les quatre coins linéarisés de chaque sous-spin de qualité.
Les gradients inconnus et les stencils de calibration incomplets sont exclus
explicitement, sans comblement des données manquantes ni renormalisation du signal.

Les figures comparent l'ancien modèle et le modèle révisé au scan réel sur les
mêmes parents retenus. Le contrôle des voxels finis utilise une quadrature
Gauss-Legendre indépendante ; le contrôle V4 ponctuel reste actif en mode ponctuel.
Les métadonnées, gradients et métriques sont archivés avec les résultats.
Un R2* effectif non nul peut déjà inclure du déphasage intravoxel ; cette limite
est enregistrée et ne permet pas d'interpréter automatiquement ce taux comme T2 intrinsèque.

Pour reproduire la comparaison sans lancer d'optimisation ni exporter de CSV :

```bash
python MRzero_validation/validate_epi_fidelity.py --subspins-z 1 --profile auto
```

Un point suivant z est équivalent aux huit à courant nul sous l'hypothèse actuelle
de B0/amplitude constants dans l'épaisseur et kz nul. Le notebook conserve huit
positions pour les courants, car les champs ACDC varient avec z. Le script produit
un rapport JSON, une archive NPZ et une figure avant/après sous
`results_epi_fidelity/`. `--smoke-iterations 2` vérifie aussi le passage au solveur
sur deux mises à jour, sans revendiquer une convergence.

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
