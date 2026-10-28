# ACDC : boucle manuelle guidée par les acquisitions

Les notebooks 05/06 et leurs variantes chargent la calibration unique
`PROJECT_ROOT/data/ACDC_3T.npz`. Les nouveaux exports sont regroupés dans
`INPUT_DIR/waveform` : optimisation image, `waveform_static.csv` et
`waveform_dynamic.csv`. Les deux solutions de champ utilisent respectivement
le QP et le FISTA/SIRF du dépôt ACDC_Optim, avec contrôle du timing et des limites
avant/après DAC. Elles peuvent être calculées sans optimisation image.

La cellule « Mesurer les courants » produit aussi `INPUT_DIR/waveform/waveform_adc.csv`,
indépendamment des courants proposés à chaque round. Ce fichier contient une
colonne de secondes, sans en-tête, avec un pas `WAVEFORM_ADC_INTERVAL_S=0.000050`
et un retour à zéro à chaque trigger de la séquence. La fenêtre se termine au
dernier readout ; `WAVEFORM_ADC_DURATION_S` permet de régler la durée totale
depuis le premier trigger. Les segments et les triggers sont conservés dans le
JSON associé. Une reprise de session saute cette cellule et conserve les
fichiers déjà produits.

Pour une nouvelle session fermée, `waveform_output_dir` fixe le dossier de sortie.
Chaque proposition et ses métadonnées sont sous
`INPUT_DIR/waveform/closed_loop/<session>_<identifiant>/round_N` ; le notebook
affiche le chemin CSV exact. Les dossiers de session gardent l'état et les scores.
Les sessions historiques reprennent leurs fichiers et chemins déjà archivés,
sans déplacement automatique ni modification des empreintes.

La vérification de synchronisation accepte un trigger au début de l'imagerie
ou un train comportant exactement un trigger par ligne image. Chaque événement
du train est vérifié contre les fenêtres du modèle ; le premier définit
l'origine temporelle et le train complet est archivé dans `trigger`.

Chaque nouvel export image comprend `waveformN.csv` et
`waveformN_zero_time.csv`. Le premier retire les commandes initiales entièrement
nulles et recale les temps sur la première commande non nulle (`t=0`), sans
dupliquer de ligne. Son compagnon conserve les mêmes commandes et remplace
uniquement la colonne temps par des zéros. Les métadonnées enregistrent le
décalage et les états retirés pour reconstruire les commandes du modèle lors de
l'import ; les deux fichiers sont archivés et contrôlés par leurs empreintes.
Le CSV horodaté reste le fichier associé aux retours d'acquisition.

Notebooks indépendants :

- `05_ACDC_ClosedLoop_RealFeedback.ipynb` conserve la préparation trois échos du 05.
- `06_ACDC_ClosedLoop_RealFeedback.ipynb` utilise le modèle complexe des 32 échos du 06, ses pertes de cohérence et ses calculs CUDA.

Dans le 06, `GRE_EPI_SCAN_PATH`, `REFERENCE_SCAN_PATH` et `B0_SCAN_PATH`
permettent de choisir indépendamment les acquisitions, avec des noms `.dat`
libres ou des chemins absolus. `REFERENCE_SEQUENCE_NAME` peut préciser la
séquence propre à la référence 64²/128² ; sans séquence compatible, la lecture
cartésienne nominale Twix est contrôlée et signalée. `OBJECT_AMPLITUDE_SOURCE`
active le candidat d'amplitude issu de cette référence, avec contraste lisse
inter-acquisitions et sélection sur les échos réservés en mode `"joint"`.
Les mêmes contrôles et diagnostics sont présents
dans le notebook 06 de simulation.

L'étape 6 du 06 utilise maintenant la GRE mesurée comme cible, ainsi que les
rounds fermés. La construction de l'objet GRE + 32 échos ne modifie jamais cette
image cible. Gains et masque restent fixes pendant la recherche et après DAC ;
la simulation B0 nul sert uniquement de diagnostic. En mode fermé, l'étape 6
utilise `CLOSED_LOOP_REFERENCE_THRESHOLD_FRACTION` et `CLOSED_LOOP_ROI_MASK`, comme
le Round 0. Hors mode fermé, elle utilise `OPTIMIZATION_REFERENCE_THRESHOLD`.

Une nouvelle référence ou de nouvelles hypothèses physiques demandent un nouveau
`CLOSED_LOOP_SESSION_DIR`. En reprise, les acquisitions, la ROI, le gain et le
modèle archivés restent figés, y compris pour une session plus ancienne que ces
options ; aucun réajustement MRzero n'est exécuté.

Les originaux 05/06 et le V1 ne sont pas modifiés par cette implémentation.
Les nouvelles versions gardent les fonctionnalités antérieures, avec des dossiers de sortie propres.
`RUN_CLOSED_LOOP=False` conserve leur fonctionnement habituel. En mode fermé,
`RUN_SIMULATION_OPTIMIZATION=False` évite de relancer l'ancienne optimisation complète et son export ; ils restent activables.

## Round 0

1. Ouvrir le nouveau notebook adapté à la cartographie disponible.
2. Dans `CLOSED_LOOP_CONFIGURATION`, choisir un dossier `CLOSED_LOOP_SESSION_DIR` fixe, propre à l'expérience. Il ne dépend pas de la date d'exécution.
3. Renseigner `PREVIOUS_ACDC_SCAN`, `PREVIOUS_ACDC_CSV` et/ou `PREVIOUS_OPTIMIZATION_NPZ`. Le JSON associé au NPZ fournit la provenance des commandes, le mapping et la synchronisation. Pour un CSV seul, renseigner `PREVIOUS_METADATA_JSON` avec le JSON historique correspondant ; le programme refuse de deviner ces paramètres.
4. Vérifier que le scan historique a bien été acquis avec ces commandes et la même séquence, puis mettre `PREVIOUS_ASSOCIATION_CONFIRMED=True` et `RUN_CLOSED_LOOP=True`.
5. Exécuter jusqu'au Round 0. La première exécution prépare le modèle propre au notebook, reconstruit les scans, simule les amplitudes et sauvegarde une banque de candidats DAC.

Le 05 préremplit les fichiers retrouvés dans le V1 :

- `acquisition_019/results/meas_MID00088_FID47694_scan.dat` ;
- `MRzero_validation/results_experiment_echo/mrzero05_20261021_181441_314849/waveform1.csv` ;
- le `mrzero05_optimization.npz` du même dossier.

Ces chemins sont des références à vérifier : l'archive V1 déclarait explicitement que l'association entre le scan et le CSV n'avait pas été certifiée. Les en-têtes Twix ne prouvent pas les commandes exécutées par un contrôleur externe.

Le 06 pointe vers le dossier fourni `acquisition_020_synthetic`, et ses résultats sont étiquetés **synthétiques**. Les GRE/EPI/B0 y portent des protocoles `MRZERO_SYNTHETIC`, avec 32 échos B0. Le MID88 dans `results` est une copie du scan corrigé réel du 019 : la boucle refuse de le mélanger avec la baseline synthétique. Il faut fournir une acquisition corrigée cohérente et ses commandes, ou basculer vers un jeu entièrement mesuré avec sa vraie cartographie 32 échos.

## Importer les deux retours possibles

Le Round 0 exporte éventuellement `round_1/waveform1.csv` sous le dossier de waveforms propre à la session. Le notebook affiche le chemin exact et les diagnostics. Appliquer ce fichier manuellement au contrôleur avec la séquence et l'origine de temps vérifiées, puis acquérir le scan.

- Renseigner `NEW_ACQUISITION_1`, confirmer `NEW_ACQUISITION_1_ASSOCIATION_CONFIRMED=True`, puis exécuter **Round 1**. Celui-ci importe le scan, mesure son coût, archive l'erreur de prédiction et propose éventuellement `round_2/waveform1.csv`.
- Renseigner `NEW_ACQUISITION_2` et sa confirmation, puis exécuter **Round 2**. Aucun troisième essai n'est proposé.
- Laisser un chemin à `None` n'invente aucune acquisition. `MAX_ADDITIONAL_SCANS` accepte 0, 1 ou 2.

Après réouverture, exécuter les imports/configurations et le round voulu. Un « Run All » en mode fermé avec `session.json` présent saute la préparation physique, la simulation et le contrôle CUDA initial. Les courbes et le contexte matériel sont chargés depuis les fichiers persistants. Un round déjà importé avec le même fichier est idempotent ; un autre fichier pour ce même round est refusé. Les CSV existants ne sont jamais remplacés.

## Critère expérimental et recherche

Toutes les EPI utilisent la reconstruction continue existante, la correction des trois navigateurs et le RSS multicoil. La grille, la ROI issue de la GRE et son seuil, le gain baseline→GRE et l'échelle d'intensité sont figés au Round 0. Les pixels EPI sombres dans la ROI restent dans le coût :

`J = sum_ROI((EPI - GRE)**2) / sum_ROI(GRE**2)`, `NRMSE = 100*sqrt(J)`.

L'échelle MRzero est ajustée une seule fois à l'EPI sans ACDC, puis utilise le même gain expérimental vers la GRE. Aucun gain n'est réajusté par candidat ou par nouvelle acquisition. La comparaison reste en magnitude ; aucun k-space MRzero n'est soustrait au k-space réel multicoil.

Les commandes sont limitées à `alpha * commandes_historiques_appliquées`, avec `0 <= alpha <= 1`. La banque comporte par défaut 21 amplitudes ; chaque simulation utilise les commandes **après quantification**, avec la réponse temporelle existante. Les candidats identiques après DAC sont dédupliqués.

Avec les deux acquisitions initiales, la correction du coût est affine et passe exactement par leurs deux coûts. Après un retour, la différence coût acquis−coût simulé est interpolée linéairement par morceaux entre les acquisitions. Cette méthode ajoute une seule valeur de désaccord par retour, conserve les points observés et évite un polynôme libre oscillant ; elle ne corrige pas la physique de MRzero.

La recherche reste dans une fenêtre de rayon `CLOSED_LOOP_TRUST_RADIUS=0.25` autour de la meilleure amplitude acquise. Un export ordinaire demande au moins `MIN_PREDICTED_GAIN_PP=0.5` point de NRMSE prédit. Une prédiction de coût négatif est signalée, jamais convertie artificiellement en NRMSE nul. Si rien n'est prometteur, aucun CSV n'est produit. `EXPORT_EXPLORATORY_CANDIDATE=True` autorise explicitement un essai local exploratoire, sans gain promis.

La sélection finale conserve séparément :

- la meilleure configuration **effectivement acquise**, `best_measured_id` et `best_measured_currents_A` ;
- une recommandation conservatrice, `incumbent_id` et `retained_currents_A`, qui ne change qu'au-delà de `UNCERTAINTY_FLOOR_PP`.

Ce seuil pratique ne remplace pas une estimation de variabilité par acquisitions répétées. Une nouvelle acquisition dégradée ne remplace jamais le meilleur résultat précédent. Une prévision MRzero seule ne peut jamais être sélectionnée comme résultat acquis.

## Matériel, provenance et fichiers

`closed_loop_commands.py` réutilise les contrôles existants : limites par canal et totale, domaine de champ complet du modèle (y compris les coins des voxels finis), réponse des amplificateurs, timing, cycle TR et limites électriques éventuellement configurées. Ils sont vérifiés sur la cible, les commandes CSV et les courants DAC. Les CSV suivent exactement l'export existant : ordre et polarité des canaux, déclenchement EPI, origine temporelle et retour à zéro.

Un encodage de quelques ULP évite qu'une reconversion flottante d'une valeur DAC lui retire un code par troncature. `alpha=1` reproduit exactement les codes historiques ; le CSV est relu et sa conversion DAC vérifiée avant publication atomique du dossier. Aucune communication avec le scanner ou le contrôleur n'est effectuée.

La session contient :

- `session.json`, pointeur vers la dernière révision complète ;
- `state_NNNN.json/.npz`, révisions des métriques, images, ROI, gains, candidats, commandes retenues et associations ;
- `command_context.npz`, contraintes, timing, cartes et réponse matérielle nécessaires sans modèle vivant ;
- `round_1/` et éventuellement `round_2/`, chacun avec CSV et `commands.npz/.json` ;
- les quatre figures par round : coût, images, résidus et courants.

Les empreintes des sources, scans, séquences, commandes et archives sont contrôlées à la reprise. Modifier la géométrie, la ROI, le modèle ou les paramètres matériels demande une nouvelle session explicitement choisie. Le mode exploratoire et les seuils de décision peuvent évoluer, mais un CSV déjà proposé reste associé à son round jusqu'au retour.

## Limites

La méthode cherche une amplitude d'une seule forme d'onde historique. Elle ne réoptimise pas librement les 32 canaux ni leurs milliers de commandes. Les deux points initiaux ne permettent pas d'estimer une incertitude statistique ou de garantir une amélioration. La GRE et l'EPI ont des contrastes différents : le critère reste celui demandé, sans prétendre identifier toutes les causes physiques des résidus.

Les tests synthétiques vérifient les décisions, la persistance et les refus ; ils ne démontrent pas une amélioration au scanner. Les fichiers historiques disponibles permettent un contrôle de reconstruction/provenance, tandis que les deux nouveaux retours restent à acquérir manuellement.
