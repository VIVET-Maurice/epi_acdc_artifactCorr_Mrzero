# Comparer la référence GRE et les EPI sans et avec ACDC

Ouvrir [03_reconstruct_epi_and_reference.ipynb](03_reconstruct_epi_and_reference.ipynb).
Ce notebook compare trois acquisitions mesurées sur le scanner :
la référence GRE, l'EPI sans correction ACDC et l'EPI acquise avec la
correction ACDC. Il reprend les fonctions de reconstruction du notebook
expérimental [02_acdc_experiment_step_by_step.ipynb](02_acdc_experiment_step_by_step.ipynb).
Les paramètres se modifient dans le notebook. Les commentaires, tableaux
et figures sont en anglais ; ce guide est en français.

La reconstruction utilise `read_gre`, `read_scan` et `read_sequence`
déjà présents dans le dépôt. Elle affiche les données brutes, les
reconstructions par canal de réception, les navigateurs EPI et les
magnitudes combinées. Elle ne prépare pas de ROI et ne lance ni le
modèle MATLAB, ni l'optimisation, ni un export de waveform. Les figures
restent dans le notebook sauvegardé ; aucun dossier de sortie n'est créé.

## 1. Rassembler les scans et leurs séquences

Le dossier d'acquisition choisi contient :

```text
acquisition_005/
├── gre_reference.dat
├── scan.dat
├── results/
│   └── scan_corr.dat
└── sequences/
    ├── reference_gre.seq
    └── scan.seq
```

`gre_reference.dat` est une référence GRE/FLASH 2D mono-écho brute.
`scan.dat` est le scan EPI acquis sans ACDC.
`results/scan_corr.dat` est le scan EPI acquis sur le scanner avec ACDC.
La correction ACDC a donc déjà été appliquée pendant cette acquisition ;
le notebook reconstruit ces données mesurées.

Dans le dossier actuel, le sous-dossier s'appelle `results`.
Si vous choisissez `resultat` dans une autre acquisition, modifier
`RESULTS_DIR` dans le notebook pour utiliser ce nom.

Chaque `.seq` doit correspondre au Twix réellement acquis : renommer
une séquence différente ne rend pas ses paramètres compatibles.
Si les deux EPI utilisent la même séquence, `epi_sequence` et
`corrected_sequence` pointent vers `sequences/scan.seq`. Si la séquence
a changé pour l'acquisition avec ACDC, fournir son propre `.seq` dans
`FILES["corrected_sequence"]`. La B0 et la calibration ACDC ne sont pas
nécessaires pour ce notebook.

Le lecteur traite une image cartésienne complète, avec une seule coupe,
un seul écho et une seule répétition. Les grilles partielles et le
suréchantillonnage en lecture ne sont pas supprimés automatiquement.
La référence utilise le repère axial centré déjà pris en charge par le
pipeline. La géométrie est vérifiée avant la comparaison finale.

## 2. Ouvrir le notebook et choisir le dossier

Utiliser le même environnement Python que pour le notebook principal.
S'il n'est pas encore installé, depuis la racine du dépôt :

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r Experiment/requirements.txt
```

Ouvrir le notebook dans Jupyter ou VS Code et sélectionner ce Python.
Dans les premières cellules, relire :

- `INPUT_DIR` : dossier d'acquisition, initialement `acquisition_005`.
- `RESULTS_DIR` : dossier du Twix acquis avec ACDC, initialement
  `INPUT_DIR / "results"`.
- `FILES` : chemins des trois Twix et de leurs séquences.
- `RECEIVER_INDEX` : canal affiché pour les figures de reconstruction
  complexe et de navigateurs, initialement `0`.

`RECEIVER_INDEX` commence à zéro. Il change seulement le canal examiné
dans les figures : la combinaison RSS utilise toujours tous les canaux.
Dans `FILES`, `reference` et `reference_sequence` désignent la GRE ;
`epi` et `epi_sequence` désignent l'EPI sans ACDC ; `epi_corrected` et
`corrected_sequence` désignent l'EPI avec ACDC.
Après un changement de dossier ou de fichier, réexécuter les cellules
depuis le début pour reconstruire les trois scans avec ces entrées.

## 3. Examiner la référence GRE

La référence est reconstruite par une transformée de Fourier inverse
centrée dans les deux directions spatiales, pour chaque canal de réception.
La magnitude finale est une combinaison RSS : racine carrée de la somme
des carrés des magnitudes des canaux.

Les figures permettent d'examiner le k-espace, la magnitude et la phase
du canal choisi, puis l'image RSS. Le k-espace est affiché avec une
compression logarithmique pour rendre visibles ses valeurs faibles.
La phase d'un canal est peu informative dans les pixels sans signal.
Les grilles et FOV sont lus depuis les acquisitions et les séquences ;
la référence n'est pas ramenée artificiellement à la matrice de l'EPI.

## 4. Examiner les navigateurs de chacune des deux EPI

Le notebook distingue deux corrections : l'ACDC appliquée pendant
l'acquisition de `scan_corr.dat`, et la correction de phase pair/impair
appliquée pendant la reconstruction EPI. La comparaison ACDC finale
utilise les deux EPI après leur propre reconstruction par navigateur.
Les figures avant/après navigateurs servent à examiner cette étape de
reconstruction, pour chacun des deux scans.

Le lecteur identifie les navigateurs avec le drapeau Twix `PHASCOR` et
leur polarité avec `REFLECT`. Leur nombre et les polarités des lignes
doivent être compatibles avec le `.seq` associé. Si trois navigateurs
sont présents, leurs polarités attendues sont `+`, `−`, `+`.
Une EPI sans navigateur est également lisible ; le notebook indique
alors qu'aucune correction par navigateur n'est appliquée. Il utilise
les navigateurs propres à chaque fichier, car leurs phases peuvent
différer entre les acquisitions.

Le lecteur remet d'abord les échantillons des lectures inversées dans
le même ordre spatial à partir de `REFLECT`. Il applique ensuite une
transformée inverse dans la direction de lecture aux navigateurs.
La moyenne des deux navigateurs positifs sert de référence pour mesurer
la différence de phase du navigateur négatif. Pour chaque canal, une
droite ajuste cette différence dans les zones où le signal est suffisant :
une phase constante et une pente selon la position de lecture.
Cette phase ajustée est soustraite aux lignes EPI de polarité négative.
Les points conservés ont un poids supérieur à 10 % du maximum du poids,
calculé avec les magnitudes des navigateurs. Il faut au moins quatre
points ; sinon, le lecteur conserve une correction nulle pour ce canal.

Les figures montrent les navigateurs, la différence de phase et la
correction ajustée, puis les images RSS avant et après correction de
phase. Pour chaque EPI, ces images avant/après partagent une même échelle
de magnitude. Une diminution du fantôme pair/impair permet d'apprécier
visuellement l'effet des navigateurs. Cette étape ne corrige pas les
distorsions EPI dues à B0. Les images finales utilisent les lignes ainsi
reconstruites, une transformée inverse 2D par canal, puis la même
combinaison RSS que la GRE.

## 5. Comparer les trois images et sauvegarder

La figure finale présente côte à côte :

1. La référence GRE reconstruite depuis `gre_reference.dat`.
2. L'EPI sans ACDC reconstruite depuis `scan.dat`.
3. L'EPI avec ACDC reconstruite depuis `results/scan_corr.dat`.

Pour l'affichage, la référence GRE est divisée par son propre maximum.
Les deux EPI sont divisées par un même maximum, pris sur les deux images,
et utilisent la même échelle de gris. Cette échelle commune conserve
leurs différences relatives de magnitude. Les tableaux reconstruits ne
sont pas modifiés par cet affichage. Des changements de réglages du
scanner ou de gain de réception peuvent aussi modifier le signal :
comparer les images avec les conditions d'acquisition en tête.

Les tailles de matrices GRE et EPI peuvent différer. Les axes en
millimètres permettent de lire leur étendue physique lorsque les trois
scans ont la même position de coupe et la même orientation. Sinon, la
figure utilise les coordonnées de tableau et signale cette différence.
Le notebook ne réalise pas de nouveau recalage. Les contrastes GRE et EPI
peuvent différer ; aucune MSE entre ces deux types d'image n'est utilisée
pour juger la correction ACDC.

Enregistrer le notebook pour conserver les figures et les paramètres
utilisés. On peut aussi l'exécuter depuis la racine du dépôt :

```bash
.venv/bin/python Experiment/execute_notebook.py \
  Experiment/03_reconstruct_epi_and_reference.ipynb
```

Le script conserve les sorties dans ce même fichier `.ipynb`. Si un
contrôle échoue, relire le message et la dernière figure produite :
un fichier absent se règle dans les chemins ; une matrice ou un temps
incompatible demande de vérifier le `.seq` associé ; une géométrie
différente demande de vérifier la coupe et son orientation avant toute
comparaison spatiale.
