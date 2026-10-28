# Préparation ISMRM 2027 — ACDC EPI

Version du 27 octobre 2026. Note de travail pour préparer la décision du PI ; aucun résultat quantitatif final n'est inventé. Les acquisitions finales sont encore sur le scanner. Aucun message n'a été envoyé et aucune soumission effectuée.

## 1. Positionnement et décision à préparer

Le projet a une base crédible pour un abstract de faisabilité expérimentale sur fantôme à 3 T : optimisation de courants ACDC variables pendant le train EPI, guidée par une fonction objectif dans le domaine image contre une référence GRE mesurée, puis application prospective au scanner.

Dans son courriel du 27 octobre, Bastien reconnaît l'amélioration mais relève le ghosting N/2, demande si la référence est une GRE et comment l'inhomogénéité B0 a été créée. Il propose de décider à partir de 3 à 5 figures. La priorité est donc une comparaison expérimentale identifiable, quantifiée et reproductible, avec un texte déjà structuré.

Les réponses apportées par Maurice sont : inhomogénéité créée par les réglages de shim du scanner ; carte B0 obtenue par acquisition multi-écho ; acquisitions disponibles sans ACDC, avec shim statique B0 et avec correction dynamique à objectif image ; quatre heures de scanner encore possibles. Les fichiers finaux et leur association aux CSV restent à récupérer.

Titre proposé : **Simulation-guided image-domain AC/DC shimming for prospective EPI distortion correction at 3 T**.

La contribution à défendre est l'optimisation de la fidélité géométrique de l'image par des commandes ACDC variables dans le temps, avec un modèle informé par les mesures et une réponse amplificateur. Le GPU rend cette approche pratique ; il ne constitue pas à lui seul la nouveauté. « Image-domain objective » est plus précis que « image-based shimming », expression qui peut aussi désigner simplement un shimming utilisant une carte B0 mesurée par imagerie.

## 2. Ce que le travail accumulé permet de raconter

| Élément | Rôle scientifique | Limite actuelle |
|---|---|---|
| Reconstruction Twix/Pulseq, trajectoire, temps ADC, orientation et polarité des champs ACDC | Rendre cohérents données, simulation et commandes | Associer les paramètres exacts au résultat final ; ne pas reprendre automatiquement ceux du dernier notebook |
| Acquisition GRE multi-écho à 32 échos | Estimer B0, amplitude et comportement de décroissance/cohérence, y compris aux échos tardifs | Une série d'échos ne mesure pas à elle seule B0(t) ; elle n'identifie pas de façon unique une distribution de champ 3D |
| Référence GRE | Fournir la cible mesurée pour l'optimisation image et une référence géométrique | Son contraste diffère de celui de l'EPI ; une erreur d'intensité n'est pas une mesure pure de distorsion |
| MRzero et modèle ACDC GPU | Relier objet, encodage et commandes à l'image prédite | MRzero fournit des aimantations mises en cache ; les itérations appliquent ensuite une phase ACDC et des facteurs de voxel fini analytiques, avec gradients analytiques explicites |
| Réponse amplificateur et contraintes de courant/champ | Produire des commandes matériellement réalisables dans le modèle | Un modèle de réponse du premier ordre ne prouve pas que les courants réels le suivent ; préciser hypothèse ou mesure |
| Exports CSV synchronisés aux triggers | Permettre l'application prospective des commandes | Conserver le CSV effectivement chargé, les canaux actifs et les règles de synchronisation |
| Amélioration expérimentale rapportée par Maurice et reconnue visuellement par le PI | Justifier un abstract préliminaire sur fantôme | Le gain final reste à quantifier sur les bons fichiers ; les captures seules ne suffisent pas |
| Shim statique B0 déjà acquis | Comparateur expérimental essentiel | Vérifier mêmes réglages scanner, géométrie et reconstruction |

Deux distinctions doivent rester explicites :

- La référence GRE est la cible image. Les 32 échos renseignent le modèle de l'objet. Dans le dernier résultat archivé inspecté, une fusion effective de l'amplitude GRE avec le modèle multi-écho n'était pas démontrée : ne pas la revendiquer sans vérifier le run appliqué.
- Le modèle réel utilise une trajectoire et des temps ADC issus de Pulseq, des aimantations MRzero en cache, une intégration analytique du champ ACDC et un modèle de déphasage de voxel fini. Il ne réalise pas une nouvelle simulation Bloch complète, différentiée à travers MRzero, à chaque itération. La dispersion résiduelle ajustée sur les échos reste indépendante des courants.

Formulation anglaise fidèle au code :

> A GPU-accelerated image-domain optimizer combined cached MRzero magnetization with analytical shim-field phase integration, finite-voxel dephasing, and a first-order amplifier-response model.

## 3. Récupérer d'abord les résultats déjà obtenus

Constituer un tableau d'association avec une ligne par acquisition :

| Condition | Fichier .dat / identifiant scanner | .seq | CSV réellement chargé | Heure | Fréquence centrale | Réglages de shim scanner | Reconstruction | Répétition |
|---|---|---|---|---|---|---|---|---|
| Sans ACDC | À renseigner | À renseigner | Courants nuls | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner |
| Statique B0 | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner |
| Dynamique à objectif image | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner |
| Dynamique B0, si acquis | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner | À renseigner |

Exporter également GRE de référence, multi-écho utilisé, paramètres scanner, cartes/reconstruction associées, configuration et résultat de l'optimisation, liste des canaux actifs et éventuels enregistrements des courants. Garder les DICOM pour documenter les images scanner ; utiliser les bruts pour une reconstruction et une quantification homogènes.

Attention au rapport local `results_experiment_echo32/mrzero06_20261026_172348_985378/measured_comparison.json` : il compare MID01631 à MID01696, sans association waveform vérifiée, et ne correspond pas à toutes les captures finales. Les en-têtes de cette ancienne paire diffèrent de 63 Hz en fréquence centrale. Ses métriques ne permettent donc pas de conclure sur le résultat final encore sur le scanner. Les fenêtres d'affichage des captures diffèrent aussi : refaire les figures depuis les données.

## 4. Plan priorisé pour quatre heures de scanner

Les durées sont des budgets de travail incluant vérifications, transferts et éventuelle réoptimisation, pas des durées d'acquisition garanties.

| Créneau | Travail | Critère de sortie |
|---|---|---|
| 0–30 min | Récupérer et identifier le triplet déjà acquis ; vérifier les CSV et les paramètres ; choisir une configuration de fantôme et de shim scanner qui restera fixe | Association acquisition–commande sans ambiguïté |
| 30–60 min | Vérifier GRE et 32 échos compatibles avec cette configuration, les réacquérir si nécessaire ; contrôler l'EPI sans ACDC et sa reconstruction, notamment le ghost N/2 | Référence et modèle correspondant au même état du scanner |
| 60–150 min | Acquérir au moins trois répétitions techniques par condition : sans ACDC, statique B0, dynamique à objectif image. Intercaler les conditions et revenir aux courants nuls | Effet reproductible et réversible sur un même fantôme |
| 150–200 min | Si le triplet est acquis et exploitable : ajouter le dynamique fondé sur B0 avec réponse amplificateur, si déjà opérationnel. Sinon renforcer les répétitions. Un deuxième réglage B0 ou un transfert de contraste vient ensuite | Comparaison plus informative, sans compromettre le résultat principal |
| 200–240 min | Mesurer les courants si la chaîne est déjà prête ; sinon consacrer ce temps au contrôle qualité, à une répétition utile et au transfert complet. Calculer un premier indicateur géométrique avant de quitter le scanner | Données exploitables et sauvegardées pour le dossier PI |

À maintenir identiques : position du fantôme, coupe et FOV, matrice, TE/TR, bande passante et sens de phase, trajectoire EPI, fréquence centrale, shims scanner, gain de réception si configurable et procédure de reconstruction. Éviter qu'un ajustement automatique entre conditions change la fréquence ou les shims ; conserver les valeurs réellement utilisées. Le shim ACDC statique est une condition expérimentale distincte du réglage des shims du scanner qui crée l'inhomogénéité de départ.

Utiliser la même procédure de correction de phase impaire/paire pour toutes les conditions. Cela peut inclure l'estimation de coefficients propres à chaque acquisition selon les mêmes règles ; il ne faut pas imposer artificiellement des coefficients identiques si la procédure exige des navigateurs propres à chaque scan. Vérifier si le gain attribué à ACDC subsiste lorsque cette reconstruction est contrôlée.

Figer la waveform avant les acquisitions de validation. Les répétitions évaluent sa reproductibilité ; ne pas sélectionner seulement la meilleure acquisition après inspection. Trois répétitions techniques restent une étude sur un seul fantôme, pas trois sujets indépendants.

Le dynamique B0 est un comparateur utile pour distinguer l'intérêt d'une réponse amplificateur prise en compte de celui de l'objectif image. Employer les mêmes cartes de bobine, canaux et contraintes pertinentes. Si les paramétrisations diffèrent, cette comparaison ne suffit pas, à elle seule, à isoler strictement l'effet de la fonction objectif.

## 5. Mesures à privilégier

**Critère principal : erreur géométrique par rapport à la GRE.** Définir quelques repères du fantôme, contours ou profils selon l'axe de phase avant de regarder les nouvelles répétitions. Mesurer les déplacements en millimètres et pixels. Harmoniser les grilles à partir de la géométrie d'acquisition ; si un recalage est nécessaire, appliquer une transformation commune justifiée. Ne pas réaligner séparément chaque EPI pour faire disparaître une translation due à B0 et ne pas appliquer de recalage déformable pour l'évaluation principale.

**Critère secondaire indépendant : ghosting N/2.** Définir des régions fixes pour le fantôme, le ghost déplacé de FOV/2 et le bruit de fond, en évitant les recouvrements. Documenter précisément la formule du rapport ghost/signal et sa correction de fond éventuelle. Montrer le FOV complet pour que le ghost reste visible. Si la géométrie du fantôme empêche une ROI propre, choisir une mesure adaptée et expliciter cette limite au lieu d'utiliser un masque contaminé.

**NRMSE/SSIM : compléments.** Leur amélioration par rapport à la GRE est intéressante mais dépend du contraste, de la normalisation, du masque et du traitement du ghost. Définir ces choix une fois pour toutes et les appliquer à toutes les conditions ; éviter une renormalisation libre de chaque image. Le score d'entraînement simulé ne remplace pas une métrique sur les acquisitions de validation.

Présenter les points individuels des répétitions et leur dispersion. Ne pas utiliser les voxels comme des sujets indépendants pour produire artificiellement une significativité statistique. Une géométrie améliorée ou un signal apparent plus élevé ne démontre pas une récupération générale de signal perdu par déphasage intravoxel. Une image unique ne justifie pas non plus une revendication de gain de SNR.

Si des courants sont enregistrés, comparer commande, prédiction de la réponse amplificateur et mesure avec une synchronisation documentée. `waveform_adc.csv` est un calendrier d'échantillonnage ; ce fichier ne constitue pas une mesure de courant.

## 6. Les figures à proposer au PI

1. **Méthode et entrées mesurées.** GRE cible, carte B0 en Hz, quelques échos et courbe de décroissance incluant un écho tardif, schéma modèle → optimisation → commandes → scanner. Séparer clairement modèle de l'objet et cible image.
2. **Résultat expérimental principal.** GRE, EPI sans ACDC, statique B0, dynamique à objectif image ; ajouter dynamique B0 uniquement si réellement acquis. Même géométrie et échelle d'affichage entre EPI ; contraste GRE signalé. Contours ou profils et agrandissement d'une région déformée, avec le FOV complet accessible pour voir le ghost.
3. **Quantification et répétabilité.** Erreur géométrique et ghost/signal, points par répétition et dispersion. Donner les améliorations contre l'absence d'ACDC et contre le statique, en unités absolues et éventuellement relatives.
4. **Validation du modèle ou argument complémentaire.** Images prédites et mesurées pour la même waveform, ou courants effectivement mesurés, ou transfert à un autre contraste avec courants fixés. Choisir la preuve la plus solide disponible ; une figure de commande seule ne prouve pas son suivi matériel.

Trois figures solides peuvent suffire à la discussion avec Bastien. Ne pas remplir cinq emplacements avec des figures faibles. Les simulations de transfert entre contrastes présentées par Bastien en septembre peuvent servir de contexte avec attribution, mais ne doivent pas être présentées comme un transfert expérimental déjà acquis.

## 7. Trame de l'abstract et limites des revendications

Le fichier Word compagnon fournit une trame anglaise provisoire. Les paramètres et nombres entre crochets doivent être remplacés après récupération des données ; aucune ligne de résultats n'est prête à soumettre.

- **Introduction :** distorsion EPI, objectif de shimming classique fondé sur B0, intérêt d'optimiser directement une image reconstruite.
- **Methods :** fantôme à 3 T, perturbation par shims scanner, GRE de référence, acquisition 32 échos et modèle, optimisation GPU, réponse amplificateur, contraintes et synchronisation, conditions réellement acquises et mesures.
- **Results :** gains géométriques contre les deux comparateurs, ghosting résiduel, répétitions, éventuellement fidélité simulation–mesure et temps réellement chronométré. Séparer simulation et acquisition.
- **Discussion :** pourquoi les résultats soutiennent l'approche, ce que le statique ou le dynamique B0 permet de conclure, limites d'un seul fantôme, différence de contraste et simplifications du modèle.
- **Conclusion :** faisabilité prospective à 3 T et portée exacte des comparaisons, sans généralisation clinique.

Éviter : « first », « real-time », « closed-loop experimentally validated », « end-to-end differentiable Bloch optimization », « full sequence digital twin », « recovery of irreversible signal loss » et « superior to conventional shimming » sans les preuves nécessaires. Si le seul comparateur acquis est statique, écrire explicitement « compared with static B0-based AC/DC shimming ».

Le mécanisme n'est pas le suivi d'un B0(t) mesuré : le B0 de départ est considéré constant, tandis que les commandes varient au cours du train EPI et que la réponse des amplificateurs est modélisée.

## 8. Conditions pour un dossier convaincant demain

L'accord du PI sera plus facile à envisager si le dossier contient : une association vérifiable entre données et commandes ; une comparaison à paramètres constants ; un gain géométrique quantifié et reproductible ; le statique B0 comme comparateur ; une réponse explicite au ghosting N/2 ; trois à cinq figures lisibles ; un texte anglais complet et une liste d'auteurs/remerciements à valider.

Si le gain contre le statique est faible ou absent, présenter honnêtement une faisabilité sans revendiquer sa supériorité. Si l'association aux commandes ou les réglages scanner restent ambigus, il faut refaire le triplet avant d'avancer un gain causal. Si le ghost empêche toute mesure fiable, il demeure une limite décisive à résoudre ou à exposer au PI. L'in vivo, une nouvelle boucle fermée complète et l'optimisation multicontraste conjointe ne sont pas des prérequis explicitement demandés dans son dernier courriel.

## 9. Format officiel et antériorités

La soumission actuelle concerne **ISMRM 2027, Vancouver, 8–13 mai 2027**. Clôture : **28 octobre 2026, 23 h 59 UTC**, soit 19 h 59 à Boston et 00 h 59 le 29 octobre à Paris. Sources : [appel officiel](https://www.ismrm.org/27m/call/) et [heure de clôture](https://www.ismrm.org/mrm-letter-from-the-editor-october-2026/).

Standard Abstract : anglais ; titre ≤125 caractères ; synopsis ≤100 mots au total, Motivation/Goals/Approach/Results ; impact ≤40 mots ; corps ≤750 mots hors références ; jusqu'à cinq figures, légendes ≤500 caractères chacune. Une preview figure supplémentaire est prévue pour le programme. Vérifier le contenu final dans le portail. [Consignes officielles](https://www.ismrm.org/27m/call/standard/)

Catégories suggérées : Physics & Engineering / Shims (1318), puis Acquisition & Reconstruction / Artifacts and Correction Strategies (103). [Liste officielle](https://www.ismrm.org/27/docs/27m_categories_and_keywords.pdf)

Antériorités à intégrer au positionnement, sans prétendre à une recherche exhaustive :

1. Stockmann et al. A 32-channel combined RF and B0 shim array for 3T brain imaging. MRM 2016;75:441–451. [doi:10.1002/mrm.25587](https://doi.org/10.1002/mrm.25587). L'AC/DC et son intérêt pour l'EPI existent déjà.
2. Loktyushin et al. MRzero—Automated discovery of MRI sequences using supervised learning. MRM 2021;86:709–724. [doi:10.1002/mrm.28727](https://doi.org/10.1002/mrm.28727). Cadre de simulation et optimisation guidée par une cible.
3. Endres et al. Phase distribution graphs for fast, differentiable, and spatially encoded Bloch simulations of arbitrary MRI sequences. MRM 2024;92:1189–1204. [doi:10.1002/mrm.30055](https://doi.org/10.1002/mrm.30055). Moteur de simulation.
4. West et al. MR sequence design to account for nonideal gradient performance. MRM 2026;95:1135–1145. [doi:10.1002/mrm.70093](https://doi.org/10.1002/mrm.70093). Optimisation tenant compte d'une réponse matérielle, sur les gradients.
5. Liao et al. A Dynamic Shim Approach for Correcting Eddy Current Effects in Diffusion-Prepared MRI Acquisition Using a Multi-Coil AC/DC Shim-Array. MRM 2026;95:3594–3607. [doi:10.1002/mrm.70294](https://doi.org/10.1002/mrm.70294). Correction dynamique AC/DC déjà démontrée dans un autre contexte.
6. [ACDC_Optim](https://github.com/bughht/ACDC_Optim). Source des méthodes de comparaison fondées sur le champ ; préciser la version effectivement utilisée si nécessaire.

## 10. Sources locales consultées

- `Re_ First experimental success with image-based ACDC shimming.eml` : priorité aux figures, réserves sur ghosting/référence/B0.
- `ISMRM abstracts draft due Friday, October 2nd.eml` : attente d'un document complet et versionné ; échéance interne passée.
- `Some more tests of ACDC dynamic correction.eml` : simulations de transfert de contraste proposées par le PI en septembre.
- `Messagerie de l’ENS PARIS-SACLAY - (474) Inbox.pdf` : motivation initiale et questions scientifiques de long terme.
- Les deux DOCX `ISMRM2026_XCAT_CS_simulations_v1` sont des copies identiques d'un exemple sur un autre sujet ; ils ne fournissent ni résultats ni auteurs ni financements pour ce projet.
- Implémentations et archives dans `MRzero_validation`, notamment `experiment_mrzero.py`, `experiment_voxel.py`, `experiment_multiecho.py`, `results_experiment_echo32`, et captures dans `acquisition_024_(06)/problem-orientation/good results`.

Cette note distingue les acquisitions rapportées par Maurice, les archives locales et les expériences proposées. Elle ne remplace pas la vérification des fichiers finaux.
