# epi_acdc_artifactCorr_Mrzero

Code de reconstruction GRE/EPI, estimation B0 et optimisation dynamique des courants ACDC avec MRzero/PyTorch.

Le notebook principal est [05_acdc_echo128_relaxation_orientation_MRzero.ipynb](MRzero_validation/05_acdc_echo128_relaxation_orientation_MRzero.ipynb). Il utilise les trois échos GRE natifs, les champs B0/ACDC, les rotations et miroirs, les contraintes matérielles, la réponse temporelle, le suivi des images et l'export CSV.

## Exécuter sur un serveur

Suivre **[le guide serveur, étape par étape](docs/SERVER_SETUP.md)** : clonage, environnement Python, installation CUDA/PyTorch, transfert des acquisitions, configuration des fichiers, exécution et récupération des résultats.

```bash
git clone https://github.com/VIVET-Maurice/epi_acdc_artifactCorr_Mrzero.git
cd epi_acdc_artifactCorr_Mrzero
python3.11 -m venv .venv
source .venv/bin/activate
# Installer PyTorch adapté au GPU suivant le guide, puis :
python -m pip install -r requirements-server.txt
python scripts/check_server.py
```

Pour exécuter le notebook avec l'interpréteur de cet environnement :

```bash
python MRzero_validation/execute_notebook.py \
  MRzero_validation/05_acdc_echo128_relaxation_orientation_MRzero.ipynb \
  --timeout 3600
```

Configurer les données avant cette commande. Le lanceur enregistre les sorties dans le notebook ; les résultats et CSV sont placés dans le dossier de session défini par `OUTPUT_DIR`.

## Organisation

| Dossier | Contenu |
|---|---|
| `MRzero_validation/` | Notebooks 01–05 et variantes, modèles MRzero, matériel, sous-spins, géométrie, lecteurs et tests |
| `Experiment/` | Lecteurs Siemens/Pulseq, reconstruction, modèles d'origine et notebooks expérimentaux |
| `Mrzero_simulation/` | Génération d'acquisitions synthétiques et comparaison aux vérités terrain |
| `acquisition_*/sequences/` | Fichiers Pulseq `.seq` disponibles pour les acquisitions |
| Racine et `eval_other_subj/` | Scripts MATLAB historiques |

Tous les notebooks présents lors de la publication sont conservés, y compris les copies de travail. Leurs sorties et états de noyau sont retirés dans la version GitHub ; les versions exécutées locales sont conservées.

Le pipeline Python principal 05 et la variante `v3.2_fast` ne nécessitent pas MATLAB. Certains anciens notebooks expérimentaux dépendent de MATLAB et d'un module externe `ACDC_Optim` ; leurs exigences sont décrites dans le guide. Les publier ne fournit pas ces logiciels externes.

## Données d'acquisition

Les scans `.dat`, calibrations `.npz`, fichiers MATLAB `.mat`, images reconstruites, résultats, CSV, environnements Python et PDF ne sont pas inclus. Les acquisitions nécessaires sont à transférer directement sur le serveur. Les noms de fichiers et les contrôles de géométrie/matrice doivent rester cohérents avec les `.seq`.

Dans le notebook 05, régler `INPUT_DIR` et `GRE_EPI_SEQUENCE_NAME` en tête du notebook. La référence GRE et la cartographie B0 sont choisies dans `MRzero_validation/twin_inputs.py`. Pour changer de configuration, relancer depuis les imports.

Les séquences proviennent des fichiers de la session ; leur présence dans le dépôt ne constitue pas une validation d'exécution sur un scanner.
