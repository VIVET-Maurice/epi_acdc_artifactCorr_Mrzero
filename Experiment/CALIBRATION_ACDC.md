# Cartes ACDC pour l'optimisation virtuelle des données expérimentales

**Le fantôme a été scanné sans bobines ACDC.** La perturbation B0 provient du
scanner. L'objectif est maintenant de simuler une correction ACDC sur cette
B0 mesurée, en utilisant les réponses des bobines fournies comme une base
virtuelle. Aucun courant n'a été appliqué pendant les acquisitions disponibles.

Les réponses des **32 canaux en Hz/A** sont dans `ACDC_maps/ACDC_3T.npz` et
`ACDC_maps/ACDC_7T 1.npz`. L'utilisateur a confirmé les unités et indique que
les cartes 3 T correspondent à **Bay5**, les cartes 7 T à **Bay2**. La base
**3 T** est retenue pour l'expérience, dont la fréquence d'émission est autour
de 123 MHz ; les données 7 T restent séparées. La source fournie est le dépôt
[ACDC_Optim](https://github.com/bughht/ACDC_Optim).

## Placement défini pour cette simulation

Les cartes sont placées virtuellement dans le repère du fantôme avec la
transformation patient → source **identité**, enregistrée dans la provenance.
Sur la coupe GRE à `patient z = 0`, elles sont interrogées en
`(xx, yy, zz) = (−Y_m, X_m, 0)` car les axes du simulateur sont
`X_m = patient y` et `Y_m = −patient x`. Le script vérifie les coordonnées
DICOM et l'ordre réel des axes du NPZ ; aucun retournement implicite n'est fait.

Ce placement constitue une hypothèse de l'étude virtuelle. Il ne décrit pas
une position de bobines présentes lors du scan. Une confirmation de recalage
physique n'est donc **pas nécessaire** pour réaliser l'optimisation demandée.
Un autre placement virtuel peut être étudié avec une transformation explicite,
et sa couverture est alors recalculée.

## Couverture et fichiers préparés

`prepare_acdc_calibration.py --restrict-support` interpole la base 3 T sur
la grille GRE 128 × 128 et conserve les **8 125 des 8 746 points de simulation
couverts (92,90 %)**. Les **1 778 points B0 fiables** sont tous couverts.
Les 621 points exclus représentent **1,109 % de la somme des amplitudes du
modèle** et **0,107 % de leur énergie quadratique**. Les réponses hors
couverture restent inconnues (`NaN`) ; elles ne sont pas extrapolées.

Les fichiers dérivés se trouvent dans `prepared/acdc_calibration/` :

| Fichier | Contenu |
| --- | --- |
| `acdc_calibration_candidate.mat` | Réponses `[128,128,32]`, unités, coordonnées, masque valide et placement virtuel. |
| `experiment_input_acdc_support.mat` | Entrée distincte avec support et amplitude restreints à la couverture. |
| `acdc_calibration_provenance.json` | Sources, empreintes SHA-256, transformation et métriques de couverture. |
| `acdc_calibration_geometry.png` | Support couvert et exemples des réponses des canaux. |

L'entrée initiale, les images acquises et les sources NPZ restent intactes.
Le simulateur recalcule sa référence idéale avec le même objet restreint.
**`virtual_optimization_ready = true`** désigne cette préparation numérique.
Les indicateurs **`registration_confirmed = false`** et **`optimization_ready
= false`** du mode physique historique restent distincts. Ils n'empêchent
pas le lancement explicite en mode virtuel.

## Lancer l'étude virtuelle

```matlab
addpath('Experiment');
[run_dir, summary] = run_experiment_virtual_acdc('cpu', 'MaxIterations', 3);
```

Le point d'entrée choisit `CalibrationMode = 'virtual'`. Le maximum par défaut
reste **600 itérations** ; l'exemple ci-dessus sert à un essai court. Le
[guide d'import](ACDC_MAP_IMPORT.md) présente les coordonnées et la préparation.
Les limites fournies pour cette simulation sont **1 A par canal** et **somme
des valeurs absolues des courants strictement inférieure à 30 A par ligne
EPI**. Le modèle optimise **32 canaux × 32 états de ligne**, malgré la grille
GRE 128 × 128. Les navigateurs sont modélisés avec des courants ACDC nuls.

`A22_1000` est un coefficient de shim du scanner de **1000 µT/m²**, selon
l'information fournie. Il est inclus dans le champ total mesuré par la GRE.
Ce coefficient n'est ni un courant ACDC ni une réponse individuelle de bobine.
Les courants optimisés sont des résultats de simulation ; la carte B0 mesurée,
le signal acquis et les réglages passés du scanner ne sont pas modifiés.

## Limites du jumeau numérique des deux états


La carte GRE actuelle correspond à l'état perturbé. Une **carte B0 non perturbée** dans les mêmes conditions que l'EPI non perturbé est nécessaire pour une comparaison mesurée des deux états. Leurs réglages de shim scanner diffèrent et la fréquence émetteur passe de **123 220 997 Hz** à **123 221 002 Hz**, soit **+5 Hz**. Mettre B0 à zéro pour la simulation de référence représenterait un champ idéal, pas le champ mesuré du scan non perturbé. Soustraire uniquement une perturbation A22 idéale ne tient pas compte de tous les autres réglages qui ont changé.

La séquence EPI a été retrouvée ici :

```text
/autofs/homes/007/mv1289/scanner/sequence/gre_epi_32x32_fov200mm_nav3_mintr.seq
```

Ses définitions donnent TE = **12,7645 ms**, TR = **23,060 ms**, espacement entre échos EPI = **0,540 ms**, et ADC = **32 échantillons à 11 µs**. La définition `FlipAngle` annonce **90°**, alors que les headers Twix annoncent **15°** : vérifier l'impulsion RF effective du fichier `.seq` et son utilisation sur le scanner avant d'interpréter quantitativement le contraste Bloch. L'intensité obtenue à partir de l'image peut servir de proxy de magnétisation transverse, mais elle ne constitue pas à elle seule une mesure de densité protonique, T1 ou T2.

La séquence GRE est également disponible :

```text
/autofs/homes/007/mv1289/scanner/sequence/megre_b0_128x128_fov200mm_3echo.seq
```

