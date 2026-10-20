# Accélération CPU/GPU de l'objectif expérimental

Les fonctions `prepare_experiment_runtime` et `obj_fun_experiment_accelerated`
conservent le modèle et son gradient analytique en **double précision**.
Elles peuvent garder les tableaux sur le CPU ou sur le GPU. Le GPU calcule
les phases complexes, les sommes, les FFT et les projections de gradient ;
le solveur `fmincon` reste sur le CPU. Seuls la valeur scalaire et le gradient
reviennent au CPU à chaque évaluation. Copier les grandes matrices à chaque
itération annulerait une partie du gain : le cache est transféré une fois.

## Compatibilité constatée sur la machine

Sonde indépendante le 1 octobre 2026, sans arrêt de processus ni reset :

- MATLAB R2026a, Parallel Computing Toolbox disponible.
- NVIDIA RTX PRO 6000 Blackwell **Server Edition**, compute capability 12.0.
- Pilote 580.159.03 ; toolkit employé par MATLAB : **CUDA 12.8**.
- GPU déclaré `DeviceSupported=true` par MATLAB.
- Mémoire totale annoncée par MATLAB : 101,97 Go (environ 95 Gio).
- Mémoire disponible lors de la sonde : 80,99 Go ; elle varie avec les autres
  calculs sur la machine et ne constitue pas une réservation.
- `SingleDoubleRatio=64` : le matériel contient beaucoup plus d'unités de
  calcul en simple précision qu'en double. La performance en IA ou en rendu
  ne permet donc pas de prédire celle de cette optimisation en double.

MATLAB R2026a prend en charge les GPU NVIDIA de compute capability 5.0 à
12.x. Il n'est pas nécessaire d'installer un toolkit CUDA 13 pour utiliser
`gpuArray` : MATLAB emploie son propre toolkit pris en charge. Voir les
[exigences officielles R2026a](https://www.mathworks.com/help/releases/R2026a/parallel-computing/gpu-computing-requirements.html).

## Mesurer avant de choisir

Dans une session MATLAB distincte, préparer `ctx` avec les données et le
placement documenté des cartes ACDC virtuelles, puis :

```matlab
addpath('Experiment');
report = benchmark_experiment_acceleration(ctx, ...
    'Threads',[1 2 4 6], 'UseGPU',true);
```

Ce benchmark ne lance **aucune optimisation** et ne crée aucun pool.
Il compare l'ancien objectif et l'objectif accéléré en calculant aussi leur
gradient. Il vérifie les résultats numériques, restaure le réglage de
threads initial et distingue le transfert initial du temps par évaluation.
Le rapport `timing_scope` rappelle que les contraintes et les opérations du
solveur ne sont pas incluses : un gain sur l'objectif n'est pas le gain du
temps total d'optimisation.

Dans cette expérience les scans proviennent du phantom réel et du B0 du
scanner ; les bobines ACDC sont un dispositif **virtuel** dans le modèle.
La carte bibliothèque 3T et sa pose supposée ne sont donc pas une calibration
mesurée de bobines présentes pendant le scan. Le benchmark peut employer
ces cartes virtuelles sans transformer cette hypothèse en mesure physique.

Le GPU est asynchrone ; un `tic/toc` qui oublie d'attendre la fin des kernels
peut donner un faux gain. Ici `gputimeit(...,2)` inclut les deux sorties et
attend la fin des opérations. Voir
[Mesurer la performance GPU](https://www.mathworks.com/help/parallel-computing/measure-and-improve-gpu-performance.html).

## Utilisation des ressources

Sur le CPU, les cinq limites de champ constituent au plus cinq travaux
indépendants. La contrainte choisie précédemment reste
`nombre_de_workers * threads_par_worker + 1 < 32`.

La machine ne possède qu'un GPU visible : réserver un seul worker au GPU et
mettre les différentes limites en file d'attente est le point de départ
prévu. Cinq workers sur le même GPU ne donnent pas cinq GPU et peuvent se
concurrencer pour la mémoire et les kernels. Les performances dépendent
aussi des autres utilisateurs. Aucun `reset(gpuDevice)` n'est nécessaire.

La simple précision ou les tensor cores demanderaient une validation séparée
des phases, des gradients, des contraintes et de la convergence. Ils ne sont
pas activés automatiquement. Le
[guide MathWorks](https://www.mathworks.com/help/releases/R2026a/parallel-computing/run-matlab-functions-on-a-gpu.html)
explique l'emploi de `gpuArray` et les différences d'arrondi entre CPU et GPU.

## Mesure sur le fantôme et les cartes virtuelles 3T fournies

Benchmark du 1 octobre 2026, 8 125 voxels dans le support couvert, 1 024
échantillons ADC, 32 canaux × 32 états, courants de test non nuls. Les scans,
le B0 et les réponses Hz/A sont ceux du dossier `Experiment` ; le placement
des bobines virtuelles est l'hypothèse documentée de l'importeur. Les temps
ci-dessous incluent l'objectif **et son gradient**, sans lancer de solveur.

| Configuration | Ancien objectif + gradient | Objectif + gradient accélérés |
|---|---:|---:|
| CPU, 1 thread | 394,17 ms | 225,77 ms |
| CPU, 2 threads | 238,97 ms | 142,74 ms |
| CPU, 4 threads | 158,84 ms | 89,94 ms |
| CPU, 6 threads | 131,00 ms | 73,68 ms |
| GPU, double précision | — | 3,41 ms |

Le GPU est ici **21,6 fois plus rapide que le CPU accéléré à six threads**
pour cette partie du calcul. Le transfert initial du cache a pris 0,106 s.
L'écart CPU/GPU vérifié sur l'objectif était 3,33 × 10⁻¹⁶ et sur le gradient
6,60 × 10⁻¹⁷, avec normalisation par `max(1,norm(...))`.

Une validation séparée sur des réponses synthétiques a aussi vérifié quatre
directions de gradient (erreur maximale 3,25 × 10⁻⁸) et le fallback exact pour
un timing non structuré. Les résultats sont dans
`experiment_gpu_validation_log.txt`.

**21,6 n'est pas un facteur d'accélération de l'optimisation complète.** Le
solveur, les contraintes et les sorties restent à comptabiliser. Sur CPU,
cinq limites peuvent tourner simultanément ; sur le GPU unique elles sont
traitées successivement. Les gains mesurés sur une session partagée peuvent
aussi varier avec sa charge.

Deux tests courts du worker, à **deux itérations**, six threads CPU et les
mêmes contraintes de courant/champ, avec `interior-point` et l'approximation
de Hessienne BFGS standard, donnent :

| Limite | Worker CPU | Worker GPU | Gain de temps du GPU |
|---|---:|---:|---:|
| 100 Hz | 9,568 s | 8,996 s | 6,0 % |
| 1 000 Hz | 1,664 s | 1,487 s | 10,7 % |

Ces temps incluent la préparation du cache, la construction des contraintes,
le solveur et le calcul des sorties ; ils excluent le démarrage d'un pool.
Chaque solveur a fait cinq évaluations de l'objectif et s'est arrêté à la
limite des deux itérations (`exitflag=0`). Les valeurs finales concordent et
les bornes sont respectées ; ces essais ne prouvent pas une convergence sur
600 itérations. L'écart entre CPU/GPU sur les courants est inférieur à
1,4 × 10⁻¹⁴, avec la même normalisation.

Le coût des contraintes et du solveur CPU domine donc déjà ces essais.
Le filtrage **exact** des contraintes redondantes garde les 113 728 lignes
pour 100/200/300/500 Hz, et les supprime toutes pour 1 000 Hz : avec un ampère
par canal et un budget absolu total de 30 A, le champ maximal possible sur le
masque de qualité est 840,383 Hz. Cinq travaux CPU simultanés restent une
configuration pertinente ; le gain GPU doit être évalué sur le temps total,
avec amortissement de la préparation sur des optimisations plus longues.

Pour 1 000 Hz, le même essai CPU avec les 113 728 contraintes redondantes
conservées a pris **9,530 s**, contre **1,664 s** après filtrage, soit **5,73×**
sur ce test. Les deux variantes gardent le même ensemble admissible et les
bornes sont respectées. Le chemin suivi par la méthode de barrière change :
après seulement deux itérations leurs objectifs sont respectivement 0,3445
et 0,3322 ; on ne doit pas attendre des courants intermédiaires identiques.

Le rapport structuré du benchmark de l'objectif avec les cartes virtuelles
est sauvegardé dans `experiment_acceleration_benchmark.mat` (variable
`report`). Les tests courts CPU/GPU et de filtrage sont également consignés
dans `experiment_gpu_validation_log.txt`.

Le worker utilise désormais BFGS avec `SubproblemAlgorithm='cg'`.
Les chiffres des workers ci-dessus concernent la factorisation antérieure.
Le contrôle final CPU/GPU par le point d’entrée, sur trois itérations à
100 et 1000 Hz, respecte les limites de courant et de champ ; l’écart de
magnitude à 100 Hz est 6,78 × 10⁻¹³. La comparaison des réglages du solveur
et le retrait exact des contraintes sont dans `OPTIMIZATION_SPEED.md`.
