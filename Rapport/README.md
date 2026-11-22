# Rapport de projet — Projet 07

Source LaTeX du rapport *Communication sémantique apprise pour la transmission d'images et de vidéo sur
canal sans fil simulé* — **39 pages** (11 pt), en français.

Structure et charte reprises du rapport Genow (`Rapport_Final`) : classe `report`, géométrie, en-têtes
`fancyhdr`, titres colorés en `insatblue`, tableaux `booktabs`, bibliographie `ieeetr`. Les éléments propres
à un stage en entreprise (chapitre société, page de validation, logo et encadrant d'entreprise) ont été
retirés : il s'agit d'un projet académique.

## Compilation

```
latexmk -pdf main.tex
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Organisation

| Chemin | Contenu |
|---|---|
| `main.tex`, `preamble.tex` | squelette, paquets, styles, couleurs |
| `frontmatter/` | page de garde (INSAT / Université de Carthage), résumé bilingue, abréviations et notations |
| `chapters/` | introduction générale, 8 chapitres, conclusion |
| `appendices/` | reproduction, fichiers de résultats, paramètres de simulation |
| `figures/` | figures du rapport, en français (plus les deux logos) |
| `scripts/figures_fr.py` | régénère les figures en français à partir des CSV de `results/eval/` |
| `references.bib` | bibliographie (27 entrées) |

Chapitres : 1 Fondements et état de l'art · 2 Problème, exigences et architecture · 3 Canal simulé et chaîne
séparée · 4 Codage conjoint source-canal appris · 5 Protocole d'évaluation et métriques ·
**6 Résultats et analyse** · 7 Extensions · 8 Simulateur interactif.

## Figures

Les figures du projet (`../figures/`) sont en anglais, pour l'article IEEE. Celles du rapport sont les mêmes,
régénérées en français :

```
python Rapport/scripts/figures_fr.py     # depuis la racine du projet
```

Le script importe `scripts/make_figures.py` et `scripts/demo.py` sans les modifier, redirige leur dossier de
sortie vers `Rapport/figures/` et traduit les textes au moment où matplotlib les pose : il n'existe donc
qu'une seule implémentation des figures. Les chiffres ne sont pas recalculés, ils sont relus dans
`results/eval/*.csv`. La régénération de `visual.png` nécessite en plus les données Kodak et les modèles
entraînés ; sans eux, le script laisse la figure existante en place.

## Chiffres cités

Tous les tableaux du rapport proviennent de `results/tables/summary.md` et de `results/eval/*.csv`, produits
par `scripts/evaluate.py`. Aucun chiffre n'est saisi à la main à partir d'une autre source.
