# Optimisation virtuelle et vitesse sur la nouvelle machine

Le point d'entrée est `run_experiment_virtual_acdc.m`. Les scans du fantôme
et la carte B0 sont mesurés ; la bobine ACDC est **virtuelle**, car elle
n'était pas présente lors des scans. L'import des réponses 3 T utilise le
placement identité dans le repère patient, puis les axes réels de lecture
et de phase. Une autre pose de bobine peut être étudiée avec l'option de
transformation de `prepare_acdc_calibration.py`.

## Lancer

Depuis le projet, dans une session MATLAB distincte sans pool préexistant :

```matlab
addpath('Experiment');
% Essai court : deux contraintes, trois itérations au maximum.
[folder, summary] = run_experiment_virtual_acdc('cpu', ...
    'FieldLimits',[100 1000], 'MaxIterations',3);

% Étude complète : cinq contraintes, 600 itérations au maximum par cas.
[folder, summary] = run_experiment_virtual_acdc('cpu');
```

Les contraintes de champ `[100 200 300 500 1000]` Hz port
ent sur le **champ
généré par l'ACDC** dans les voxels de B0 fiables, pour chaque état. Elles
ne bornent pas le champ total B0+ACDC. La cible de l'objectif est l'image
idéale à B0 nul du jumeau, et non une carte B0 baseline mesurée.

La configuration CPU utilise au plus cinq workers de processus, chacun
avec six threads MATLAB. `workers*threads+1=31<32` respecte le budget
précédemment choisi. Les 128 cœurs physiques ne sont pas tous utilisés par
ces cinq problèmes indépendants. Six threads ne donnent pas six fois le
gain : le calcul accéléré de l'objectif mesuré passe de 226 ms à 74 ms entre
un et six threads, soit environ **3,1 fois** pour cette partie.

Le GPU est une option explicite sous R2026a :

```matlab
[folder, summary] = run_experiment_virtual_acdc('gpu', ...
    'FieldLimits',1000,'MaxIterations',3);
```

Un seul worker utilise le GPU, avec six threads CPU pour le solveur.
Les cinq contraintes sont mises en file d'attente si toutes sont demandées.
Le programme transfère le cache une seule fois et reste en double précision.
Le résultat rassemble les tableaux sur le CPU avant sauvegarde.

## Changements et raison

| Fichier | Changement |
|---|---|
| `prepare_acdc_calibration.py` | Interpolation des réponses 3 T en Hz/A sur la grille GRE ; masque de couverture, provenance et entrée restreinte séparée. |
| `prepare_experiment_optimizer.m` | Contrôle des réponses uniquement sur le support actif couvert ; les NaN hors couverture restent tracés. Recalcul de la référence sur ce support. |
| `prepare_experiment_runtime.m` | Blocs contigus CPU/GPU et cache persistant. Vérification du timing avant réduction des calculs. |
| `forward_model_experiment_accelerated.m` | Phase cumulative calculée avec les durées des lignes précédentes et la durée partielle courante. Conservation du résultat intermédiaire. |
| `obj_fun_experiment_accelerated.m` | Réutilisation de ce résultat pour l'adjoint ; un seul passage direct avec exponentielles, au lieu de deux. |
| `prepare_experiment_field_constraints.m` | Retrait des contraintes dont la satisfaction est déjà prouvée par les bornes de courant ; aucune réduction du nombre de points par échantillonnage. |
| `experiment_current_constraint.m` | Budget absolu de courant par ligne et Jacobien analytique sparse, en plus des bornes ±1 A. |
| `run_experiment_fast_parallel_worker.m` | Appel de ces fonctions ; `fmincon` et BFGS conservés, sous-problèmes résolus par CG, diagnostic de faisabilité et de convergence. |
| `run_experiment_fast_parallel.m` | Cache préparé une fois par worker, backend explicite, jobs temporaires locaux dans `/scratch`, nettoyage des seules ressources créées. |
| `run_experiment_virtual_acdc.m` | Paramètres de cette expérience : cartes 3 T, pose virtuelle, support couvert, 1 A/canal, total <30 A, six threads. |

Les anciennes fonctions de modèle direct et d'objectif restent disponibles
pour vérifier l'équivalence. Les fichiers des simulations historiques
`run_code3` et `run_code4_fast_parallel` ne sont pas modifiés.

Le timing vérifié permet le calcul exact suivant :

```text
phase(v,s) = somme des champs(v,j)*durée(j) pour j avant la ligne courante
             + champ(v,ligne courante)*temps écoulé sur cette ligne.
```

L'adjoint regroupe les échantillons par ligne, puis cumule leurs effets sur
les états antérieurs. La magnétisation conserve donc toute son histoire de
phase. Si le timing ne respecte pas cette structure, la fonction utilise
le calcul matriciel générique exact.

## Limites de courant

La somme pertinente est `sum(abs(currents_A),1)` : des courants de signes
opposés ne peuvent pas annuler leur consommation dans cette limite.

`fmincon` demande des contraintes différentiables. Pour éviter 1024 variables
auxiliaires, le code utilise la majoration conservative :

```text
sum(sqrt(I.^2 + (1e-6 A)^2)) <= 30 A - 1e-4 A.
```

Puisque `sqrt(I²+epsilon²) >= abs(I)`, cette contrainte implique le budget
absolu. La marge de 0,1 mA garde une somme strictement inférieure à 30 A ;
l'excès de majoration est au plus 32 µA par ligne. La tolérance du solveur
est 1e-8. Les valeurs absolues réelles sont sauvegardées et revérifiées
après calcul. Les contraintes matérielles de commutation ne sont pas
modélisées dans cette étude virtuelle.

## Mesures et limites du gain

Mesures R2026a sur les 8125 voxels couverts, 1024 ADC, 1024 variables :

| Partie mesurée | CPU, six threads | GPU, double |
|---|---:|---:|
| Objectif accéléré + gradient | 73,68 ms | 3,41 ms |
| Deux itérations, limite 100 Hz | 9,57 s | 9,00 s |
| Deux itérations, limite 1000 Hz | 1,66 s | 1,49 s |

Ce tableau mesure le sous-problème historique par **factorisation**, avant
le changement vers CG. Les temps de deux itérations incluent la préparation du runtime, les
contraintes, le solveur et le résultat final du worker ; ils excluent le
démarrage d'un pool et les écritures/figures. Ces essais sont limités et
ne permettent pas de prédire le temps jusqu'à convergence sur 600 itérations.

L'objectif CPU est environ **1,8 fois plus rapide** qu'avant, à six threads.
Le GPU est **21,6 fois plus rapide** pour cet objectif, mais le premier essai
complet de deux itérations à 100 Hz ne réduit le temps que de **6 %**.
Les contraintes et les systèmes linéaires du solveur dominent sur CPU.
Pour les cinq limites, la configuration CPU permet leur traitement simultané ;
c'est le réglage par défaut. Tester le GPU séparément est pertinent pour
une limite unique.

Les limites 100, 200, 300 et 500 Hz conservent ici leurs 113728 inégalités
de champ. À 1000 Hz, toutes sont redondantes : le maximum possible avec les
bornes 1 A/canal et 30 A vaut 840,38 Hz sur ces voxels. Elles sont retirées,
mais les bornes individuelles et la somme absolue restent imposées.

Un second essai, à 100 Hz et sur **cinq itérations**, compare les méthodes
de résolution du sous-problème, avec exactement le même objectif et les
mêmes contraintes (objectif initial 0,344979) :

| Réglage CPU, six threads | Temps | Évaluations | Objectif final |
|---|---:|---:|---:|
| BFGS + factorisation | 19,82 s | 11 | 0,338014 |
| **BFGS + CG, retenu par défaut** | **15,60 s** | **9** | **0,272356** |
| L-BFGS automatique | 17,44 s | 11 | 0,343105 |
| L-BFGS + CG | 15,47 s | 11 | 0,276687 |

Les courants et champs sont faisables dans les quatre essais. CG réduit
ici le temps d'environ **21 %** et améliore la progression après cinq
itérations. Il peut changer le parcours et le nombre d'itérations ; cet essai
ne garantit pas un gain identique jusqu'à convergence. BFGS reste utilisé.
Le réglage historique peut être choisi avec
`'SubproblemAlgorithm','factorization'`. Voir les
[options officielles de fmincon](https://www.mathworks.com/help/optim/ug/fmincon.html).

À 1000 Hz, le retrait exact des 113728 contraintes redondantes a réduit un
essai CPU de deux itérations de **9,53 s à 1,66 s** avec factorisation, soit
environ **5,7 fois**. Retirer des termes de barrière redondants peut changer
les itérés, tout en conservant le même ensemble de courants admissibles.

Le gain dépend de la charge de la machine partagée. La grande RAM permet de
garder les caches en mémoire ; le stockage local `/scratch` sert au pool.
Les résultats finaux sont petits et restent sous `Experiment/results` par
défaut. `OutputDir` permet de choisir un nouveau dossier local si nécessaire.

## Mesurer et vérifier

```matlab
base = fullfile('Experiment','prepared','acdc_calibration');
data = load(fullfile(base,'experiment_input_acdc_support.mat'),'experiment');
cal = load(fullfile(base,'acdc_calibration_candidate.mat'));
ctx = prepare_experiment_optimizer(data.experiment,cal,1);
report = benchmark_experiment_acceleration(ctx,'UseGPU',true);
```

Ce benchmark ne lance pas de solveur. Les tests indépendants sont décrits
dans `VALIDATION.md`, et les informations GPU dans `GPU_ACCELERATION.md`.

Dans un résultat, `optimization_info.exitflag>0` et `converged=true`
indiquent la convergence numérique avec les vérifications de faisabilité.
`summary.status='complete'` indique seulement que tous les fichiers ont été
sauvegardés. Une sortie après la limite d'itérations conserve les résultats
intermédiaires avec `converged=false`.
