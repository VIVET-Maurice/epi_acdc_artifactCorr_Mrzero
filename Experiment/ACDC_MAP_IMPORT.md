# Import des cartes ACDC fournies

`prepare_acdc_calibration.py` prépare les réponses des 32 canaux sur la grille
GRE de l'expérience. Il ne lance aucune optimisation et ne commande aucun
matériel. **Les scans du fantôme ont été acquis sans bobines ACDC.** L'étude
cherche une correction **virtuelle** de la B0 mesurée : les cartes 3 T de Bay5
fournissent une base de réponses et un placement virtuel explicite. Les cartes
7 T de Bay2 ne sont pas utilisées.

## Géométrie vérifiée dans les fichiers

Le fichier `ACDC_maps/ACDC_3T.npz` contient `fieldmap[112,112,112,32]`, un
masque et trois volumes de coordonnées. L'utilisateur a confirmé les unités
**Hz/A**. Les coordonnées sont en mètres :

| Axe du tableau NPZ | Coordonnée | Étendue |
| --- | --- | --- |
| 0 | `xx` | −112 à +110 mm |
| 1 | `yy` | −112 à +110 mm |
| 2 | `zz` | −91 à +131 mm |
| 3 | canal | ordre fourni, 1 à 32 |

Les directions et la position DICOM déterminent les coordonnées patient de
chaque pixel GRE. La coupe actuelle est à `patient z = 0`. Les axes du
simulateur sont `X_m = patient y` et `Y_m = −patient x` ; avec le
placement virtuel identité, les cartes sont donc interrogées en
`(xx, yy, zz) = (−Y_m, X_m, 0)`. Il ne s'agit pas d'une transposition arbitraire.
Le script vérifie cet ordre et refuse une géométrie incohérente.

**Le placement identité est une définition de la simulation.** Il ne prétend
pas retrouver une position de bobines présentes lors du scan, puisqu'il n'y
en avait pas. Une confirmation de recalage physique n'est donc pas nécessaire
pour cet objectif virtuel. On peut étudier un autre placement avec une
transformation affine explicite. Pour une application ultérieure sur des
bobines réelles, un recalage mesuré serait une étape distincte.

## Audit du placement virtuel

Depuis le projet, dans l'environnement Python de reconstruction :

```bash
python Experiment/prepare_acdc_calibration.py
```

Les fichiers dérivés sont enregistrés dans `prepared/acdc_calibration/` :

- `acdc_calibration_candidate.mat` : cartes `[128,128,32]`, `X_m`, `Y_m`,
  unités, masque spatial valide, placement virtuel et statuts distincts pour
  l'optimisation virtuelle et un éventuel usage physique ;
- `acdc_calibration_provenance.json` : empreintes SHA-256, transformation,
  couverture et raisons bloquant l'utilisation ;
- `acdc_calibration_geometry.png` : couverture et trois canaux illustratifs.

L'interpolation est trilinéaire. Le masque source doit couvrir **tout le
voisinage d'interpolation** ; aucune extrapolation n'est faite. Les réponses
inconnues sont exportées comme `NaN`, jamais comme un champ nul inventé.

Avec le placement virtuel identité, la couverture est de **8 125 / 8 746
points du support de simulation, soit 92,90 %**. Les **1 778 points B0 fiables**
sont tous couverts. Les 621 points exclus représentent **1,109 % de la somme
des amplitudes du modèle** et **0,107 % de leur énergie quadratique**. Ces
mesures portent sur le proxy d'amplitude actuel, pas sur un volume anatomique.

## Choisir explicitement un support restreint

Si l'on décide de simuler uniquement l'objet couvert par la calibration :

```bash
python Experiment/prepare_acdc_calibration.py --restrict-support
```

Cela crée **un nouveau** `experiment_input_acdc_support.mat`. Son masque de
simulation est l'intersection du support initial et de la couverture ; son
proxy d'amplitude est nul en dehors. Les images mesurées et la carte B0 fiable
sont conservées. `prepared/experiment_input.mat` reste intact. Le simulateur
doit recalculer son image idéale et sa référence à partir de **ce même support**.

Pour cette entrée restreinte, **`virtual_optimization_ready = true`**. Les
indicateurs **`registration_confirmed = false`** et **`optimization_ready =
false`** concernent le mode physique historique et restent distincts. Aucune
estimation de réponse par remplissage du masque n'est appliquée automatiquement.

Dans MATLAB, le point d'entrée de cette étude sélectionne explicitement le
mode virtuel :

```matlab
addpath('Experiment');
[run_dir, summary] = run_experiment_virtual_acdc('cpu', 'MaxIterations', 3);
```

Le lancement complet conserve le maximum par défaut de 600 itérations. Les
contraintes de la simulation sont **1 A par canal** et **somme des courants
absolus strictement inférieure à 30 A par ligne**. Les champs et les courants
obtenus sont des résultats de simulation, sans application aux acquisitions
déjà réalisées ni commande de bobines.

## Étudier un autre placement virtuel

Une transformation peut être fournie avec `--transform registration.json`.
Ce JSON contient un tableau affine 4 × 4, ou un objet avec la clé
`patient_to_source_m`. Il transforme les coordonnées patient **en mètres**
vers les coordonnées source **en mètres**, translation comprise. La dernière
ligne vaut `[0,0,0,1]`. Cette transformation définit alors le placement
virtuel des cartes dans le fantôme et est tracée dans la provenance. La
couverture est recalculée : elle reste nécessaire même pour une simulation.

L'option historique `--confirm-registration` sert uniquement à tracer une
éventuelle confirmation physique indépendante. Elle n'est pas utilisée et
n'est pas requise pour cette étude virtuelle.

Le modèle final garde les 32 états de courant correspondant aux **32 lignes
EPI**, malgré le rééchantillonnage des cartes sur une grille GRE 128 × 128.
