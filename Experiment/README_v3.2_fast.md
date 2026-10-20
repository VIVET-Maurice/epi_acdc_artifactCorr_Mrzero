# ACDC v3.2 fast

Ouvrir `02_acdc_experiment_step_by_step_v3.2_fast.ipynb`, adapter `INPUT_DIR`
et les fichiers de séquence, puis exécuter de haut en bas. Cette version reprend
la v3 avec trois solveurs Python/PyTorch : shim statique, shim dynamique du B0,
et optimisation de l'image. Le modèle, les gradients et les mises à jour des
courants utilisent CUDA. Les lecteurs et figures restent en Python/NumPy.

## Installation

Utiliser un environnement Python 3.11, puis :

```bash
python -m pip install --upgrade pip
python -m pip install -r Experiment/requirements-fast.txt
```

Installer ensuite **une** version de PyTorch adaptée au GPU et au pilote.
Les commandes suivantes épinglent des versions dont les distributions sont
répertoriées dans les [archives officielles PyTorch](https://pytorch.org/get-started/previous-versions/).

| Machine | Commande PyTorch proposée | Profil du notebook |
|---|---|---|
| V100S de cette machine, pilote 535.54.03 | `python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118` | `"auto"` ou `"v100"` |
| GTX 1660 Ti avec pilote compatible CUDA 11.8 | Même commande `cu118` | `"auto"` ou `"gtx1660ti"` |
| RTX PRO 6000 Blackwell avec pilote compatible CUDA 12.8 | `python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128` | `"auto"` ou `"blackwell"` |
| Vérification sans GPU | `python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu` | `"cpu"` |

Blackwell est pris en charge à partir de PyTorch 2.7 avec la distribution CUDA
12.8, selon les [notes de version officielles](https://pytorch.org/blog/pytorch-2-7/).
Les distributions CUDA sont distinctes : changer le profil du notebook ne change
pas les bibliothèques installées. Redémarrer le noyau après l'installation.
Le pilote 535 de la V100S a été vérifié avec `nvidia-smi` lors de cette modification.

Vérifier l'environnement :

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Réglages utiles

Les paramètres se trouvent dans la cellule 2 du notebook :

```python
INPUT_ROTATION_DEG = 180   # 180 : correction des scans ; 0 : orientation native
GPU_PROFILE = "auto"       # auto, v100, gtx1660ti, blackwell, cpu
GPU_PRECISION = "auto"     # float32 sur GPU, float64 sur CPU
GPU_MEMORY_FRACTION = 0.35 # fraction de mémoire libre pour les caches/blocs
SAMPLE_CHUNK = None        # taille automatique des blocs ADC
MAX_ITERATIONS = 150
IMAGE_LEARNING_RATE_A = 0.01
IMAGE_PATIENCE = 40
```

`INPUT_ROTATION_DEG = 180` tourne réellement les données GRE, scan et échos
complexes du B0 avant la ROI et l'optimisation. Le scan corrigé suit la même
convention. Les grilles physiques et la calibration ACDC sont conservées comme
repère ; les deux axes spatiaux des images sont retournés, sans permuter les
récepteurs ou les échos. Relancer depuis le début après un changement d'angle.

Le tableau du notebook affiche les réglages effectifs et la carte utilisée. Les
blocs par défaut sont plafonnés à 256 ADC sur GTX 1660 Ti, 4096 sur V100 et
4096 sur Blackwell ; un budget mémoire plus faible les réduit automatiquement.
Le calcul conserve tous les voxels de la ROI et tous les ADC. Réduire la taille
des blocs change la mémoire et le temps de calcul, sans sous-échantillonner le modèle.
`GPU_PRECISION = "float64"` permet de comparer la précision sur une petite exécution.
La précision simple est le choix rapide sur les trois cartes ; FP16 et TF32 ne
sont pas utilisés pour les phases.

Avec `"auto"`, CUDA est choisi lorsqu'il est disponible, sinon le tableau affiche
CPU. Un profil GPU explicite échoue si CUDA est indisponible. Le budget dépend
de la mémoire libre au moment de la préparation ; libérer le modèle et réexécuter
depuis le début si la charge de la carte change fortement.

## Calcul et limites

- Le modèle conserve les conventions de la v3 : coordonnées physiques, temps
  réels ADC, déphasage RF/ADC, décroissance T2* relative au TE et FFT centrée.
  La phase GRE redémarre à chaque RF. La réponse temporelle reste activée pour
  EPI uniquement, comme dans la v3. Pour un scan GRE, désactiver
  `CURRENT_RESPONSE_ENABLED` et `RUN_DYNAMIC_B0_SHIM`. Les commandes initiales sont indépendantes
  de la condition physique de courant nul avant la première commande.
- Les contributions fixes sont mises en cache lorsque la mémoire le permet.
  Le gradient analytique réutilise les contributions complexes du calcul direct.
  L'optimisation d'image utilise Adam avec contraintes et garde le meilleur candidat
  évalué. Elle peut donner un résultat différent de `fmincon` et ne garantit pas
  un optimum global. Les courbes et critères d'arrêt sont affichés.
- Les shims de champ utilisent `B.T @ B` et `B.T @ cible`, puis un gradient
  accéléré sur GPU. Cela évite un produit voxels × ADC à chaque itération.
  `B0_DYNAMIC_RIDGE` multiplie la moyenne des commandes au carré et s'exprime
  en Hz²/A² ; cette unité diffère du paramètre spectral de la v3.
- La projection courant impose les limites par canal et la somme des valeurs
  absolues. Une réduction radiale impose le champ nominal si nécessaire.
  Cette dernière opération donne un candidat admissible, sans constituer une
  projection euclidienne sur l'intersection de toutes les contraintes.
- Les commandes et les courants réels sont revérifiés avant export, y compris
  après quantification nominale du DAC. Les limites réelles sont vérifiées aux
  ADC. Avec des constantes de temps différentes, cela ne certifie pas les valeurs
  entre deux ADC. Les CSV contiennent les commandes DAC.

Le benchmark du notebook mesure un objectif et son gradient après échauffement
et synchronisation CUDA. Le temps dépend aussi de l'occupation du GPU. Aucun
facteur d'accélération par rapport à MATLAB n'est présumé sans comparaison mesurée.

## Fichiers et validation

`acdc_gpu.py` contient le modèle, le gradient, l'optimisation d'image et la réponse
temporelle. `acdc_gpu_field.py` contient les deux ajustements de champ.
Les fonctions d'export existantes sont réutilisées. Chaque branche possède son
dossier CSV ; aucune commande matérielle n'est exécutée. Le shim statique et le
shim de B0 écrivent `waveform.csv`. La branche image conserve les options de la
v3 actuelle : fichiers numérotés `waveform1.csv`, `waveform2.csv`, etc., et
`CSV_ZERO_TIME_COLUMN = True` par défaut. Mettre cette option à False pour
exporter les vrais temps en première colonne. Le graphique utilise les temps
réels dans les deux cas.

Exécuter les tests indépendants sur petits problèmes :

```bash
python -m unittest discover -s Experiment -p test_acdc_gpu.py -v
```

Les tests vérifient le signal avec une somme NumPy indépendante, le gradient par
différences finies, les historiques EPI/GRE, les contraintes et la réduction de
l'objectif. La comparaison CUDA s'exécute si un GPU est accessible.


## Validation effectuée le 14 octobre 2026

Les **27 tests passent sur la V100S**, dont la comparaison CPU/CUDA. Les données
locales d'`acquisition_014` ont aussi été utilisées pour exécuter toutes les étapes
calculables, avec export des trois CSV dans un dossier temporaire. La comparaison
avec un scan corrigé a été ignorée puisque `results/scan_corr.dat` est absent.

Mesures ci-dessous réalisées avec l’orientation native (`INPUT_ROTATION_DEG = 0`),
avant la correction de 180° des entrées.

Configuration mesurée : Tesla V100S-PCIE-32GB, pilote 535.54.03, PyTorch 2.7.1
avec CUDA 11.8, float32, 4096 ADC et 2202 points dans la ROI, réponse temporelle
activée avec tau commun, blocs de 4096 ADC, pas Adam 0.01.

| Mesure | Résultat |
|---|---:|
| Shim statique | 0,24 s |
| Shim dynamique du B0 | 0,15 s |
| Boucle d'optimisation d'image, 150 itérations | 2,55 s |
| MSE relative sans ACDC | 0,10885 |
| MSE relative au point initial statique | 0,02648 |
| Meilleure MSE relative après optimisation | 0,01659 |
| Écart relatif L2 image float32 / float64, commandes de test identiques | 2,22 × 10⁻⁶ |
| Écart relatif L2 gradient float32 / float64 | 3,55 × 10⁻⁶ |

Les erreurs concernent les **images simulées**. Le solveur atteint la limite de
150 itérations ; il ne déclare pas une convergence. Les contrôles de courant,
champ et quantification passent. Les temps dépendent de la charge de cette carte
partagée ; ces mesures ne préjugent pas des performances sur GTX ou Blackwell.


Le notebook a également été exécuté dans un noyau Jupyter, avec ses figures,
et les trois exports ont été vérifiés. Les résultats exécutés sont des artefacts
de validation temporaires ; le notebook livré reste sans sorties enregistrées,
prêt pour une nouvelle acquisition.
