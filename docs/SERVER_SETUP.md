# Exécuter les notebooks sur un serveur Linux avec GPU

Le notebook principal est
[`05_acdc_echo128_relaxation_orientation_MRzero.ipynb`](../MRzero_validation/05_acdc_echo128_relaxation_orientation_MRzero.ipynb).
Il utilise MRzeroCore et PyTorch pour la simulation et l'optimisation. Les
lecteurs Twix/Pulseq, la préparation des cartes et les figures utilisent aussi
le CPU. Une seule carte GPU est utilisée par processus.

Les commandes ci-dessous se lancent depuis la racine du dépôt. Sur un serveur
de calcul, réserver d'abord un GPU et la mémoire RAM avec le gestionnaire du
site. Vérifier la carte disponible dans cette allocation avec `nvidia-smi`.

## 1. Créer un environnement Linux

Utiliser Python 3.11 pour rester proche de l'environnement vérifié du projet.
Un environnement Windows `.venv-gpu-win` ne se transfère pas vers Linux.

```bash
git clone https://github.com/VIVET-Maurice/epi_acdc_artifactCorr_Mrzero.git
cd epi_acdc_artifactCorr_Mrzero
python3.11 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install --upgrade pip
```

Installer ensuite **une**
distribution de PyTorch correspondant au GPU et au pilote, avant les autres
dépendances.

### V100 / V100S : configuration connue du projet

La V100 est une carte de capacité de calcul 7.0 (`sm_70`), selon la
[table NVIDIA des cartes historiques](https://developer.nvidia.com/cuda/gpus/legacy).
La combinaison PyTorch 2.6.0 / CUDA 11.8 est celle retenue pour retrouver
l'environnement MRzero du projet. La commande est publiée dans les
[archives officielles PyTorch](https://pytorch.org/get-started/previous-versions/#v260).

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

### Blackwell et autres cartes récentes

Ne pas réutiliser automatiquement la distribution CUDA 11.8 ci-dessus.
PyTorch a introduit le support Blackwell avec les distributions CUDA 12.8 de
la version 2.7 ; voir les
[notes officielles de PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/).
Les capacités des modèles Blackwell diffèrent, par exemple B200 et RTX PRO
6000 ; vérifier le modèle dans la
[table NVIDIA](https://developer.nvidia.com/cuda/gpus).

Choisir la commande Linux/Pip/CUDA adaptée dans le
[sélecteur officiel PyTorch](https://pytorch.org/get-started/locally/), ou une
version compatible dans les archives. Vérifier également le pilote avec
l'administrateur du serveur. Le paramètre `GPU_PROFILE="blackwell"` change
les tailles de blocs du programme ; il n'installe pas une autre distribution
de PyTorch. Une nouvelle combinaison GPU/PyTorch doit être contrôlée sur ce
serveur avant un calcul long.

Pour la **RTX PRO 6000 Blackwell (`sm_120`)**, la **GTX 1660 Ti (`sm_75`)**
et la **V100 (`sm_70`)**, une distribution commune est disponible :

```bash
python -m pip install --upgrade torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Dans Jupyter, remplacer `python -m pip` par `%pip` pour viser le Python du noyau,
puis **redémarrer le noyau**. Le notebook peut conserver `GPU_PROFILE="auto"`
sur les trois cartes. Cette commande remplace la variante CUDA 11.8 de PyTorch
dans l'environnement choisi ; elle ne modifie pas le pilote NVIDIA, qui doit
être compatible CUDA 12.8.
Les [architectures de la distribution Linux PyTorch 2.8.0 / CUDA 12.8](https://github.com/pytorch/pytorch/blob/v2.8.0/.ci/manywheel/build_cuda.sh)
comprennent `sm_70`, `sm_75` et `sm_120`. La variante CUDA 12.8 de PyTorch 2.7.1
excluait `sm_70` ; la version 2.8 rétablit cette architecture. Une V100 peut aussi
conserver l'environnement CUDA 11.8 décrit plus haut sur sa machine. Les
machines peuvent utiliser le même notebook avec une distribution PyTorch adaptée
à leur carte et à leur pilote.

### Exécution CPU uniquement

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Pour une grande simulation, le CPU peut être beaucoup plus lent. Choisir
explicitement `GPU_PROFILE="cpu"` dans le notebook pour ce mode.

### Dépendances communes

Après le choix de PyTorch :

```bash
python -m pip install -r requirements-server.txt
python -m pip check
```

`requirements-server.txt` regroupe les dépendances des pipelines Python du
dépôt, notamment MRzeroCore, les lecteurs Twix/Pulseq et les outils Jupyter.
Les anciens fichiers `MRzero_validation/requirements*.txt` conservent leur
épinglage historique `torch==2.6.0` : ne pas les réinstaller après avoir choisi
une autre version pour un GPU récent. Ni `torchvision` ni `torchaudio` ne sont
utilisés par les notebooks du projet.

Contrôler l'installation dans l'allocation GPU :

```bash
python - <<'PY'
import importlib.metadata as metadata
import torch

for name in ("MRzeroCore", "torch", "numpy", "scipy", "pypulseq", "twixtools"):
    print(name, metadata.version(name))
print("Runtime CUDA :", torch.version.cuda)
print("CUDA disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Carte :", torch.cuda.get_device_name(0))
    print("Capacité :", torch.cuda.get_device_capability(0))
    print("Architectures compilées :", torch.cuda.get_arch_list())
    z = torch.ones((8, 8), device="cuda", dtype=torch.complex64)
    print("Calcul complexe CUDA :", float((z @ z).abs().mean()))
    torch.cuda.synchronize()
PY
```

Un `True` pour CUDA et le petit calcul complexe vérifient le chargement du GPU.
Le notebook affiche ensuite son GPU, sa précision et ses tailles de blocs
effectifs. `GPU_PROFILE="v100"` ou `"blackwell"` impose CUDA ; `"auto"` revient
au CPU si CUDA est indisponible ou si le test d'exécution échoue. Dans ce dernier
cas, un avertissement affiche la carte, sa capacité, la version PyTorch, le
runtime CUDA et les architectures compilées. Le motif reste enregistré dans
`model.runtime_info["cpu_fallback_reason"]`. La précision `"auto"` devient
float64 sur CPU ; la simulation native MRzero reste float32.

L'erreur `no kernel image is available for execution on the device` peut donc
survenir même si `torch.cuda.is_available()` vaut `True` : les kernels de la
distribution installée ne conviennent pas au GPU. Choisir une distribution
compatible dans **l'interpréteur utilisé par le notebook**, puis redémarrer le
noyau. Changer seulement `GPU_PROFILE` n'installe pas de nouveaux kernels.
Le repli CPU permet de poursuivre, avec un temps de calcul potentiellement
beaucoup plus long.

Le notebook **06** impose par défaut `GPU_REQUIRED=True`, tout en conservant
`GPU_PROFILE="auto"` pour sélectionner GTX 1660 Ti, V100 ou Blackwell. Son contrôle
initial et son modèle utilisent `require_cuda=True` : un GPU absent ou une
distribution incompatible arrête donc le calcul avant l'optimisation. Pour un
calcul CPU volontaire, choisir `GPU_REQUIRED=False` et `GPU_PROFILE="cpu"`.
Après un changement de distribution PyTorch, redémarrer le noyau puis reconstruire
le modèle en relançant le notebook depuis le début.

À l'étape 6, le périphérique de calcul (`model.device`) est affiché séparément du
stockage du cache MRzero. `Cache = CPU` signifie que le signal fixe est conservé
en RAM ; le calcul peut rester sur CUDA avec transfert des blocs vers le GPU.
`use_cuda_graph=False` désactive la capture des opérations, pas CUDA. Ce réglage
conserve la projection matérielle du notebook 04.

L'environnement Windows audité le 20 octobre 2026 contenait Python 3.11.9,
MRzeroCore 1.1.1, PyTorch 2.6.0+cu118, NumPy 2.4.6, SciPy 1.17.1,
Matplotlib 3.11.2, PyPulseq 1.5.0.post1 et twixtools 0.24. Cette liste décrit
l'environnement local ; elle ne remplace pas une vérification du serveur.

## 2. Transférer les données séparément

Les fichiers d'acquisition privés et les sorties volumineuses ne sont pas
distribués avec le code. Copier le dossier d'acquisition nécessaire par le
canal privé habituel, par exemple `scp` ou `rsync`, en conservant son arborescence :

```text
acquisition_019/
    *scan.dat
    *b0_map.dat
    *reference.dat
    ACDC_3T.npz
    sequences/
        b0_map.seq
        gre_32*.seq
        gre_epi_....seq
    results/                   # facultatif : scans corrigés pour comparaison
```

Il faut exactement un fichier de chaque type Twix à la racine du dossier.
Les noms de séquence doivent correspondre aux fichiers effectivement acquis.
`ACDC_3T.npz` est nécessaire au 05 pour les réponses des bobines ; la calibration
actuelle occupe environ 396 Mo. Elle doit rester cohérente avec le hash, les
unités, le placement, les canaux et les polarités configurés dans le notebook.

Exemple à adapter, depuis une machine possédant déjà les données :

```bash
scp -r acquisition_019 utilisateur@serveur:/chemin/prive/epi_acdc_artifactCorr_Mrzero/
```

Le générateur dans `Mrzero_simulation` ne requiert pas les signaux réels pour
créer ses Twix synthétiques. Il utilise cependant les fichiers `.seq` de
`acquisition_015/sequences` comme modèles, et copie sa calibration lorsque
`copy_calibration=True`. Pour les deux notebooks synthétiques, transférer ces
séquences ; la copie de calibration peut être désactivée pour les tests V4.
Les notebooks de validation 01 et 02 ont également besoin de leurs cartes et
références numériques, absentes d'une publication contenant seulement du code.

## 3. Paramétrer puis lancer le 05

Dans les premières cellules, adapter `INPUT_DIR` et
`GRE_EPI_SEQUENCE_NAME`. La version actuellement publiée peut pointer vers
`acquisition_019` et une séquence EPI 128 × 128 ; le nom de fichier doit être
celui présent dans le dossier transféré. Modifier les paramètres dans une copie
du notebook si l'on souhaite garder l'original intact.

Sur V100, les réglages de départ conseillés sont :

```python
GPU_PROFILE = "v100"
GPU_PRECISION = "auto"
GPU_MEMORY_FRACTION = 0.35
SAMPLE_CHUNK = None
RECONSTRUCTION_CHUNK = 256
```

Les angles et miroirs décrivent la géométrie de l'expérience, pas celle du
serveur. Les conserver lors du transfert. Les limites de courant et la réponse
matérielle restent également des paramètres physiques de l'expérience.
`EXPORT_WAVEFORM` commande seulement l'écriture d'un CSV local ; le notebook
ne pilote pas le matériel.

```bash
python MRzero_validation/execute_notebook.py \
    MRzero_validation/05_acdc_echo128_relaxation_orientation_MRzero.ipynb \
    --timeout 3600
```

Le délai est **par cellule**. Le lanceur démarre son propre interpréteur comme
noyau temporaire et travaille depuis le dossier du notebook. Il fonctionne
donc même si les métadonnées enregistrées mentionnent un noyau Windows.
Il enregistre les sorties dans le notebook donné, y compris en cas d'erreur :
utiliser une copie si cette modification n'est pas souhaitée.
Les résultats du 05 sont créés dans
`MRzero_validation/results_experiment_echo/<session>/`.

Pour une session interactive, VS Code peut sélectionner `.venv-server/bin/python`.
On peut aussi installer JupyterLab et un noyau nommé :

```bash
python -m pip install jupyterlab
python -m ipykernel install --user --name acdc-server --display-name "Python (ACDC serveur)"
```

## 4. Mémoire, blocs et matrice 512 × 512

`SAMPLE_CHUNK=512` signifie 512 échantillons ADC par bloc de calcul. Cela ne
transforme pas une acquisition 128 × 128 en 512 × 512. Les dimensions de
l'acquisition viennent du Twix et de sa séquence ; leur cohérence est vérifiée.
La grille native de l'objet vient des échos B0 et reste un autre paramètre.

Le profil V100 autorise jusqu'à 4096 ADC par bloc avant application du budget
mémoire. La taille effective peut être inférieure. Commencer avec `None`, puis
réduire les blocs si nécessaire ; cela ne retire aucun spin ni aucun ADC.
Le changement de précision ne change pas la précision native float32 de MRzero.

La mise en cache du signal fixe nécessite approximativement
`nombre_de_spins × nombre_ADC × octets_complexes`. Si le cache ne tient pas dans
le budget GPU, le modèle actuel le conserve en **RAM CPU**. Réduire la taille
des blocs réduit les allocations temporaires, pas la somme de ce cache.
À titre de dimensionnement, 34 384 spins et une EPI 128² représentent environ
4,2 Gio pour le seul cache complex64 (8,4 Gio en complex128, utilisé par défaut en CPU) ; avec les mêmes spins et 512² ADC, ce
serait environ 67,2 Gio, avant les autres allocations. Ce dernier chiffre est
une estimation, pas un résultat mesuré sur une acquisition 512².

La reconstruction exacte utilise un FFT quand les coordonnées le permettent,
sinon un opérateur séparable par lignes, ou un calcul direct par blocs. Ses
noyaux mis en cache sont plafonnés à 64 Mio ; une grande matrice n'entraîne pas
le stockage systématique de sa matrice de Fourier complète. Consulter
`model.runtime_info["reconstruction"]["method"]` : `direct_chunked` peut être
beaucoup plus lent que `row_separable`. `RECONSTRUCTION_CHUNK` contrôle ces
blocs séparément de `SAMPLE_CHUNK`. Les rotations communes du 05 conservent
la grille de reconstruction cohérente avec les coordonnées ADC.

## 5. Notebooks historiques

Les notebooks Python rapides `Experiment/*v3.2_fast.ipynb` et les notebooks
MRzero 03–05 n'ont pas besoin de MATLAB. Les anciennes variantes de
`Experiment/02_acdc_experiment_step_by_step*.ipynb`, hors `v3.2_fast`, ainsi que
`simple_navigator_v3.1`, contiennent encore des étapes MATLAB et des chemins
propres au serveur initial (`/usr/pubsw/...`, `/scratch/mv1289`). Ils demandent
MATLAB et ses outils correspondants, des chemins adaptés et les fichiers
de données adéquats.

Les variantes v3 et v3.1 utilisent aussi le paquet externe `ACDC_Optim` dans
`Experiment/ACDC_shim_static_dyn_haotian`, absent du checkout audité.
Installer seulement `quadprog` ne remplace pas ce code manquant. Ces notebooks
sont conservés pour leur historique ; leur présence dans le dépôt ne signifie
pas que toutes leurs dépendances externes sont distribuées. Le point de départ
pour une exécution Python/MRzero complète est le 05.


## 6. Garder un calcul actif et récupérer ses résultats

Dans une allocation de calcul persistante, on peut enregistrer un journal :

```bash
mkdir -p logs
nohup python MRzero_validation/execute_notebook.py \
  MRzero_validation/05_acdc_echo128_relaxation_orientation_MRzero.ipynb \
  --timeout 3600 > logs/mrzero05.log 2>&1 < /dev/null &
tail -f logs/mrzero05.log
```

Sur un cluster, suivre les règles du site et utiliser son ordonnanceur ; `nohup`
ne réserve ni GPU ni RAM. Pour une session interactive, lancer sur le serveur :

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Puis, sur sa machine locale, ouvrir le tunnel et utiliser dans le navigateur
l'adresse avec le jeton affichée par Jupyter :

```bash
ssh -L 8888:127.0.0.1:8888 utilisateur@serveur
```

La récupération des résultats peut se faire depuis la machine locale :

```bash
scp -r utilisateur@serveur:/chemin/prive/epi_acdc_artifactCorr_Mrzero/MRzero_validation/results_experiment_echo ./results_serveur
```

Le chemin exact du CSV est affiché à la section 11 du 05. Le dossier de session
contient aussi les images, courants, diagnostics et fichiers JSON/NPZ. L'export
CSV ne programme pas le contrôleur ACDC.
