# Générations 08 et 09

Ces trois notebooks reprennent la configuration **active** de
`MRzero_validation/06_acdc_echo32_b0_s0_t2star_MRzeroV3.ipynb` : acquisition
`acquisition_029_(06)`, 32 échos MEGRE, alignement 90°, repère commun 90°,
sortie 9 désactivée, réponse matérielle et cible GRE mesurée.

| Notebook | Déphasage dans le voxel | ACDC | Reconstruction |
| --- | --- | --- | --- |
| `08_acdc_echo32_b0_s0_t2star_MRzero.ipynb` | Intégrale analytique x/y/z, un centre MRzero par voxel | Phase intégrée après MRzero, comme 06 | Opérateur existant : IFFT ou sommes selon la trajectoire |
| `08_V2_acdc_echo32_b0_s0_t2star_MRzero.ipynb` | Identique à 08 | Identique à 08 | Sommes DFT explicites, sans accélération FFT/IFFT |
| `09_acdc_echo32_b0_s0_t2star_MRzero.ipynb` | Taille physique et noyaux de voxel MRzero dans le simulateur | Historique B0 total dans le simulateur ; PDG recalculé à chaque évaluation | Même opérateur que 08 |

## Utilisation

Ouvrir un notebook dans ce dossier et redémarrer le noyau avant de changer de
génération. Modifier les chemins et paramètres en tête, puis exécuter les cellules
dans l'ordre. Les acquisitions et `data/ACDC_3T.npz` restent dans le dossier parent.
Les fonctions nécessaires sont copiées ici pour isoler cette génération des anciens
notebooks ; seuls les modules de fonctionnement importés sont inclus, sans anciens
notebooks, tests ou scripts de génération.

Les paramètres GPU et les exports locaux de 06 sont conservés. Pour une exécution
CPU : `GPU_REQUIRED=False`, `GPU_PROFILE="cpu"`. Pour examiner les diagnostics sans
optimisation : `RUN_OPTIMIZATION=False`. Les exports de champ restent indépendants ;
leurs trois interrupteurs `EXPORT_*WAVEFORM*` sont explicités dans les paramètres.
Les résultats propres à chaque lancement sont dans `results/mrzero08_*`,
`results/mrzero08_V2_*` ou `results/mrzero09_*`. Les CSV restent dans
`INPUT_DIR/waveform`, comme dans 06.

## Ce qui change

- **Un voxel dans l'épaisseur entière**, avec poids 1. Il n'y a plus huit sous-coupes.
- Les variations ACDC sur la largeur complète des trois axes et les huit coins
  proviennent de la calibration 3D, dans la géométrie de coupe mesurée. Un voxel
  sans couverture complète est exclu, sans renormaliser le signal restant.
- Le B0 mesuré étant 2D, sa dérivée normale est inconnue et supposée nulle. La
  dispersion résiduelle des 32 échos reste sans direction spatiale identifiée.
- 08 multiplie les trois intégrales sinc une fois et différentie les trois facteurs
  pendant l'optimisation. La cohérence résiduelle de 06 est conservée une seule fois.
- V2 calcule directement `sum(signal * exp(+2πi k·r))/N_ADC`. Le signe positif
  conserve la convention inverse/adjoint et l'orientation de 06. Le drapeau local
  `continuous_reconstruction.FORCE_DIRECT_DFT` impose les mêmes sommes explicites
  aux données mesurées, aux cartes d'entrée, à la calibration et aux comparaisons
  finales ; l'optimisation utilise l'adjoint hermitien correspondant. Ce calcul
  est plus lent et ne constitue pas une inversion générale d'une trajectoire non uniforme.

## MRzero dans 09

L'API standard de MRzero utilise un B0 statique par voxel et un déphasage de voxel
commun à tous les voxels. `native_pdg.py` reprend son exécuteur avec deux extensions
locales, marquées `GEN09` : B0 effectif à chaque ADC et trajectoire locale par voxel
pour le noyau natif. Le paquet installé n'est pas modifié.

Après l'excitation idéale unique, le champ donné à la rotation native est
`B0_effectif(t)=B0 + intégrale_RF→t(champ_ACDC)/(t−RF)`. L'intégrale conserve toute
l'histoire des commandes et leur réponse matérielle. Les trajectoires locales
contiennent aussi les intégrales des gradients ACDC et B0. MRzero applique la
phase, la relaxation, la réception et ses propres noyaux `box`/`gauss`. Aucun
facteur de phase ACDC ni de déphasage n'est appliqué au signal en sortie.
La cohérence résiduelle non résolue utilise ces noyaux natifs à l'intérieur de
l'exécuteur ; elle reste indépendante des courants, comme dans 08.

Cette représentation est exacte, à la précision numérique native près, pour le
modèle hérité : spins stationnaires, une excitation idéale non sélective à 90°,
pas de diffusion ni de refocalisation. Les autres histoires RF sont refusées.
T1, profil de coupe et état stationnaire ne sont pas mesurés/simulés ; l'amplitude
effective vient des données. Les gradients sont localement linéaires dans chaque
voxel. 09 recalcule et exécute le PDG à chaque évaluation et lors de la dérivation
par blocs : prévoir un temps nettement supérieur à 08.

L'exécuteur local est adapté de **MRzeroCore 1.2.0**, sous AGPL v3 ; sa licence
est conservée dans `LICENSE_MRzeroCore.txt`. Le code a été vérifié avec MRzeroCore
1.2.0 et PyTorch 2.8.0. Les dépendances du projet parent restent nécessaires.

## Rapport de première simulation

La cellule « Paramètres de la première simulation » est placée avant la calibration
EPI et l'optimisation. Elle affiche notamment RF, TE/TR, ADC, PD effective, B0,
T1/T2/T2dash, B1, diffusion, taille/nombre de voxels, précision, version, paramètres
PDG, et gain/NRMSE de comparaison au scan EPI. Les différences entre les hypothèses
simulées et les valeurs réellement acquises sont explicites.

`first_simulation.json` et `first_simulation.npz` conservent ce premier résultat.
Dans 09, les statistiques du premier bloc natif B0 activé sont identifiées comme
telles ; la comparaison d'image porte sur tous les ADC de la baseline.

## Validation

Les tests associés sont dans `../MRzero_validation/test_generations08_09.py` :
quadrature 3D indépendante, somme de spins déplacés, gradients des courants,
contraintes aux coins, couverture/calibration, DFT et adjoint, parité 08/09 avec
réponse matérielle et dispersions uniforme/gaussienne, dérivation native et rapport
JSON. Exécution depuis la racine du projet :

```bash
python -m unittest discover -s MRzero_validation -p 'test_generations08_09.py'
```

La validation disponible ici est CPU ; aucune optimisation complète sur CUDA
ni nouvelle mesure scanner n'est revendiquée.

Le premier passage 08 sur l'acquisition 029, avec les 60 itérations de fit
configurées et avant calibration EPI, a simulé 7 785 voxels, chacun sur toute
l'épaisseur de 5 mm. B0 allait de −87,11 à 250,76 Hz ; le TE simulé était de
61,3275 ms contre 61,328 ms dans le scan. La NRMSE de magnitude, après un gain
global de comparaison, était de 45,98 % : ce contrôle valide l'exécution, sans
établir une concordance quantitative avec le scan. Le rapport montre aussi
l'excitation idéale simulée à 90° et le flip mesuré à 15° ; la convention
d'amplitude effective de 06 est conservée, sans inventer une carte T1.

Les 12 tests CPU passent. Un passage de toutes les cellules de 08 a aussi
réussi avec un fit raccourci, optimisation et exports de waveform désactivés.
La première simulation de 09, sur le même objet réel préparé, concorde avec
08 à `1,25e-7` d'erreur relative dans le signal complexe.
