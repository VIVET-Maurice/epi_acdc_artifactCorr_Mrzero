# Mettre à jour la publication

Le dossier de travail d'origine conserve son ancien dépôt Git. La publication
vers `epi_acdc_artifactCorr_Mrzero` est préparée dans un checkout séparé, pour
ne pas envoyer l'ancien historique contenant des fichiers de données.

Après modification du code local, exporter à nouveau depuis la racine :

```bash
python scripts/export_source.py --destination tmp/github_publish
```

Ce dossier doit être le checkout du **nouveau** dépôt. Vérifier avant tout envoi :

```bash
git -C tmp/github_publish remote -v
git -C tmp/github_publish status --short
git -C tmp/github_publish diff --stat
```

Après revue des changements :

```bash
git -C tmp/github_publish add --all
git -C tmp/github_publish commit -m "Update MRzero pipeline"
git -C tmp/github_publish push origin main
```

L'exporteur copie les sources et les séquences, retire les sorties des notebooks
et normalise leur noyau. Il conserve les fichiers `.seq` octet pour octet. Il
n'efface aucun fichier du checkout : une suppression volontaire de source doit
être reportée explicitement dans celui-ci. Les notebooks exécutés locaux ne
sont pas modifiés par l'export.

Le manifeste `publication_manifest.json` indique les fichiers publiés, leurs
SHA-256, les 20 notebooks de la première publication et leurs adaptations de
portabilité. Les scans, calibrations et résultats ne sont pas copiés.

Sur le serveur, faire `git pull` après avoir conservé ses modifications et
résultats locaux. Les sorties exécutées des notebooks sont des modifications
Git ; garder une copie de travail si elles doivent être conservées.
