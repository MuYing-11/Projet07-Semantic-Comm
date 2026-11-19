# Projet 07 — Communication sémantique apprise pour l'image et la vidéo

**Rapport de projet.** Codage conjoint source-canal profond (deep JSCC), canaux AWGN et Rayleigh simulés,
images CIFAR-10 et Kodak, séquences vidéo CIF.

---

## 1. Résumé

Une image est transmise sur un canal radio bruité simulé par trois systèmes construits et mesurés dans les mêmes
conditions : la chaîne séparée classique de Shannon (compression puis protection, simulée bit à bit), un JSCC
syntaxique appris de bout en bout sur une perte pixel, et un **JSCC sémantique** dont la seule différence est la
fonction de perte — elle ajoute un terme de caractéristiques et un terme de tâche calculés à travers un
classifieur gelé.

Le résultat central : à architecture, budget et canal identiques, le système sémantique **sacrifie environ 0,5 dB
de PSNR et gagne 10 à 14 points de précision** chez un juge indépendant. Les pixels perdus n'étaient pas ceux qui
portaient le sens. À 0 dB, là où la chaîne séparée réelle tombe au niveau du hasard (9,8 % sur 10 classes), les
deux JSCC restent exploitables.

Deux extensions prolongent le travail : une transmission **orientée tâche** qui atteint 70,5 % de précision avec
32 fois moins de bande que le JSCC complet, et un **renouvellement conditionnel** des latents en vidéo qui divise
la bande par trois sur une scène statique.

Tous les résultats proviennent d'un canal simulé, pas d'une liaison radio réelle.

---

## 2. Problème

La chaîne de transmission d'aujourd'hui repose sur le **théorème de séparation de Shannon** : on peut, sans perte
d'optimalité, compresser la source d'un côté (JPEG, H.264) et protéger les bits de l'autre (code correcteur,
modulation). C'est l'architecture de JPEG + Wi-Fi, de H.264 + 4G.

Ce théorème est vrai *asymptotiquement* : blocs infiniment longs, canal dont on connaît la statistique. En
pratique les images sont courtes et le SNR réel diffère du SNR prévu. Deux conséquences visibles :

- **L'effet falaise.** Au-dessus du SNR de conception, la chaîne sature sans exploiter la marge. En dessous, le
  décodeur décroche et l'image est *perdue* d'un coup — pas dégradée, perdue.
- **Le critère optimisé n'est pas le bon.** On optimise la fidélité pixel, alors que le récepteur veut souvent
  seulement *comprendre* l'image.

La question du projet : **à budget radio égal, vaut-il mieux transmettre des pixels fidèles ou transmettre le
sens ?**

---

## 3. Méthode

### 3.1 Contraintes communes à tous les systèmes

Sans budget commun, aucune comparaison n'a de sens. Les trois systèmes partagent :

| Contrainte | Valeur |
|---|---|
| Rapport de bande `k/n` | `k` symboles complexes pour `n = 3·H·W` valeurs source ; point principal **1/12** |
| Puissance | moyenne normalisée à 1 par symbole complexe (`power_normalize`) |
| Canal | AWGN `y = x + n`, ou Rayleigh par blocs `y = h·x + n`, `h ~ CN(0,1)`, CSI parfaite et égalisation zero-forcing |
| Plage de SNR | 0 à 20 dB |

Pour le JSCC, `k` est fixé par le nombre de canaux latents émis : `k/n = c/96`, avec `c ∈ {2, 4, 8, 12, 16}`
soit `k/n` de 1/48 à 1/6. L'image, elle, ne change jamais : c'est bien le budget qui varie, pas la source.

### 3.2 Les trois systèmes

**1 — Chaîne séparée (référence).** Simulée bit à bit, sans approximation : codage de source, code convolutif
`K=7` de polynômes (133,171) poinçonné aux rendements 1/2, 2/3 et 3/4 selon les motifs 802.11a, QAM de Gray
(QPSK, 16-QAM, 64-QAM), canal, LLR max-log, décodage de Viterbi sur treillis à 64 états, décodage de source. Les
paquets font au plus 1024 bits utiles, terminés par 6 bits de queue.

Le codeur de source dépend de la taille d'image. Sur Kodak et en vidéo c'est **JPEG**, à la plus haute qualité
qui tient dans le budget. Sur CIFAR-10 c'est un **auto-encodeur quantifié sur 4 bits** : à 32×32, le plus petit
fichier JPEG possible pèse environ 300 octets alors que le budget au point 1/12 n'en offre que 142 au meilleur
MCS. Aucun JPEG ne tiendrait, et la référence serait une image grise à tous les SNR — l'auto-encodeur est le
choix qui *avantage* la chaîne séparée.

Deux variantes servent de bornes : **meilleur MCS**, un génie qui connaît le SNR et choisit d'avance la meilleure
modulation et le meilleur rendement ; et **borne de capacité**, où l'on suppose `k·log2(1+SNR)` bits transmis
sans aucune erreur — le mieux que pourrait faire n'importe quel système séparé avec ce codeur de source.

**2 — JSCC syntaxique.** Il n'y a plus de bits. Un encodeur convolutif produit un tenseur latent au stride 4 ;
des paires de valeurs réelles deviennent directement des symboles complexes analogiques normalisés en
puissance ; le canal est inséré **comme une couche du réseau** ; un décodeur convolutif reconstruit l'image. Le
bruit traverse la rétropropagation, donc le réseau apprend un code adapté au canal. Perte : MSE pixel.

L'entraînement tire le SNR uniformément dans [0, 20] dB **et la bande au hasard**. Les canaux latents
s'ordonnent donc par importance, et un seul modèle couvre les cinq rapports `k/n` sans réentraînement.

**3 — JSCC sémantique (proposition).** Architecture strictement identique au système 2. Seule la perte change :

```
perte = MSE(x̂, x) + 0,01 · perte_caractéristiques(clsA) + 0,003 · entropie_croisée(clsA(x̂), étiquette)
```

Le terme de caractéristiques compare les activations internes d'un classifieur A **gelé** pour l'image reçue et
l'originale ; le terme de tâche demande que la classe reste décodable. Les poids sont faibles : le sens est un
ajout à la fidélité, pas un remplacement. Comme l'architecture et le budget sont identiques, **tout écart de
résultat vient de la perte seule.**

### 3.3 Mesurer le sens sans tricher

Deux familles de mesures. **Fidélité pixel** : PSNR et SSIM. **Sens préservé** : précision d'un classifieur sur
l'image reçue, similarité cosinus des caractéristiques d'un ResNet-18 ImageNet, et accord top-1 des étiquettes
avant/après transmission.

Le point méthodologique décisif : **les juges ne servent jamais à l'entraînement**. La perte sémantique utilise
le classifieur A (ResNet) ; le juge est le classifieur B (VGG, architecture différente) et un ResNet-18
ImageNet. Sans cette séparation, le système apprendrait à flatter son propre correcteur et les chiffres ne
voudraient rien dire.

### 3.4 Protocole

- **CIFAR-10** : 50 000 images d'entraînement, évaluation sur les 5 000 premières images de test, pas de 1 dB.
- **Kodak** : 24 images 768×512, **jamais vues à l'entraînement** — les modèles ne sont entraînés que sur
  CIFAR-10 32×32. Pas de 2 dB.
- **Vidéo** : 30 premières images de trois séquences CIF (352×288) — `akiyo` (statique), `foreman` (mouvement
  modéré), `bus` (panoramique rapide). Elles remplacent les clips UVG du cahier des charges, dont les fichiers
  1080p bruts pèsent plusieurs gigaoctets.
- Les points numériques sont simulés **exactement** : chaque paquet est codé, modulé, bruité et décodé par
  Viterbi. Un fichier JPEG que le décodeur refuse devient une image grise uniforme.

---

## 4. Résultats

### 4.1 CIFAR-10, `k/n = 1/12`, AWGN — le résultat central

Référence : le juge B atteint **88,4 %** sur les images originales non transmises.

| SNR | Système | PSNR (dB) | SSIM | Précision (juge B) |
|---|---|---:|---:|---:|
| 0 dB | JSCC syntaxique | 20,08 | 0,669 | 42,0 % |
| 0 dB | **JSCC sémantique** | 19,68 | 0,677 | **56,4 %** |
| 0 dB | Séparé — borne de capacité | 19,65 | 0,600 | 15,3 % |
| 0 dB | Séparé — meilleur MCS | 12,63 | 0,132 | 9,8 % |
| 10 dB | JSCC syntaxique | 25,99 | 0,874 | 67,6 % |
| 10 dB | **JSCC sémantique** | 25,48 | 0,874 | **78,2 %** |
| 10 dB | Séparé — borne de capacité | 24,51 | 0,808 | 29,7 % |
| 10 dB | Séparé — meilleur MCS | 19,65 | 0,600 | 15,3 % |
| 20 dB | JSCC syntaxique | 27,72 | 0,906 | 70,6 % |
| 20 dB | **JSCC sémantique** | 27,15 | 0,903 | **80,2 %** |
| 20 dB | Séparé — borne de capacité | 27,33 | 0,896 | 67,4 % |
| 20 dB | Séparé — meilleur MCS | 25,71 | 0,850 | 40,5 % |

Trois lectures :

1. **Le compromis sémantique est net et constant.** Environ 0,5 dB de PSNR cédé, 10 à 14 points de précision
   gagnés, à tous les SNR. Le SSIM, lui, ne se dégrade pas — la structure locale est préservée.
2. **L'effet falaise est massif.** À 0 dB la chaîne séparée réelle donne 9,8 %, soit le niveau du hasard sur
   10 classes : l'image est perdue, pas dégradée. Les JSCC restent à 42 % et 56 %.
3. **La borne de capacité n'est pas atteignable.** Même cette borne idéale reste sous les deux JSCC jusqu'à
   20 dB en précision.

### 4.2 Canal de Rayleigh — le gain tient

À 10 dB et `k/n = 1/12`, sous évanouissement par blocs :

| Système | PSNR (dB) | Précision |
|---|---:|---:|
| JSCC syntaxique | 24,32 | 60,3 % |
| **JSCC sémantique** | 23,84 | **72,4 %** |

Même structure de compromis qu'en AWGN : 0,5 dB cédé, 12 points gagnés. Le bénéfice ne dépend pas du modèle de
canal.

### 4.3 Kodak et vidéo — le résultat défavorable, et pourquoi

| Jeu | SNR | JSCC synt. | JSCC sém. | Séparé (capacité) | Séparé (meilleur MCS) |
|---|---|---:|---:|---:|---:|
| Kodak | 0 dB | 22,97 | 22,73 | **26,70** | 12,82 |
| Kodak | 10 dB | 27,11 | 27,34 | **32,27** | 29,77 |
| Vidéo | 0 dB | 21,93 | 21,77 | **25,28** | 10,41 |
| Vidéo | 10 dB | 26,59 | 26,62 | **31,95** | 29,11 |

(PSNR en dB, `k/n = 1/12`, AWGN.)

Sur les grandes images, **la borne de capacité de la chaîne séparée l'emporte**, et il faut le dire franchement.
Deux raisons :

- Les modèles JSCC sont entraînés **uniquement sur CIFAR-10 en 32×32** et appliqués tels quels à des images
  768×512 jamais vues. Ils sont entièrement convolutifs, donc ils *fonctionnent*, mais ils n'ont jamais vu les
  statistiques d'une image naturelle de cette taille. JPEG, lui, est à son terrain de jeu.
- La borne de capacité n'est pas un système réalisable : c'est un idéal sans erreurs.

Face au **vrai** système séparé, le JSCC reste très supérieur dans les mauvaises conditions : à 0 dB sur Kodak,
12,82 dB de PSNR et 0 % d'accord top-1 pour la chaîne séparée, contre ~23 dB pour les JSCC.

### 4.4 Désadaptation train/test du SNR

Trois modèles entraînés chacun à un seul SNR, testés partout (CIFAR-10, précision du juge B) :

| Entraîné à | testé 0 dB | testé 10 dB | testé 20 dB |
|---|---:|---:|---:|
| 0 dB | 28,9 % | 30,2 % | 30,5 % |
| 10 dB | 39,6 % | 69,5 % | 73,7 % |
| 20 dB | 33,9 % | 67,3 % | 75,1 % |

Le modèle entraîné à 0 dB plafonne à 30 % même en excellentes conditions : il a appris à se méfier du canal et
n'exploite plus la marge. Celui entraîné à 20 dB s'effondre quand le canal se dégrade. **Entraîner sur une plage
de SNR, et non à un point fixe, est ce qui rend un modèle unique utilisable de 0 à 20 dB.**

### 4.5 Extension 1 — transmission orientée tâche

Si le récepteur ne veut que l'étiquette, pourquoi reconstruire l'image ? Un encodeur au stride 8 alimente
directement une tête de classification, sans décodeur image.

| `k/n` | Précision à 0 dB | Précision à 10 dB |
|---|---:|---:|
| 1/384 | 57,4 % | **70,5 %** |
| 1/48 | 72,4 % | 76,9 % |

À 1/384, soit **32 fois moins de bande** que le JSCC complet au point 1/12, la précision reste comparable
(70,5 % contre 78,2 %). Quand la tâche est connue d'avance, le gain est spectaculaire : c'est le point extrême
de la communication sémantique.

### 4.6 Extension 2 — renouvellement conditionnel en vidéo

Deux images vidéo consécutives se ressemblent. On envoie une image clé complète, puis seulement les blocs
latents qui ont changé, plus une signalisation (12,7 % du débit ici).

À `frac = 0,2` et `c = 8`, contre la transmission image par image à `k/n = 1/12` :

| Séquence | `k/n` avec CR | PSNR CR | PSNR image par image (`k/n` = 1/12) |
|---|---:|---:|---:|
| `akiyo` (statique) | 1/37 | 26,40 | 26,48 |
| `foreman` (mouvement modéré) | 1/37 | 25,15 | 28,06 |
| `bus` (panoramique rapide) | 1/37 | 18,50 | 25,32 |

Sur une scène statique, **la même qualité pour trois fois moins de bande**. Sur un panoramique rapide, le gain
disparaît : la redondance temporelle n'existe plus, et forcer le renouvellement coûte 6,8 dB. Le résultat est
donc conditionnel au contenu, ce qui est exactement ce qu'on attend d'une méthode fondée sur la redondance
temporelle.

---

## 5. Mise en œuvre

| Fichier | Contenu |
|---|---|
| `semcom/channel.py` | Normalisation de puissance, AWGN, Rayleigh par blocs (CSI parfaite, ZF), troncature de bande |
| `semcom/models.py` | Encodeur/décodeur JSCC, auto-encodeur numérique 4 bits, classifieurs A (ResNet) et B (VGG), modèle orienté tâche |
| `semcom/digital.py` | Code convolutif K=7 (133,171) et poinçonnage 802.11a, QAM de Gray, LLR max-log, Viterbi vectorisé, ajustement du débit JPEG |
| `semcom/metrics.py` | PSNR, SSIM, métriques sémantiques ResNet-18 ImageNet |
| `semcom/data.py` | Chargeurs CIFAR-10, Kodak et Y4M |
| `semcom/train.py` | Boucles d'entraînement et liste des 9 modèles (`JOBS`) |
| `semcom/video.py` | Transmission image par image et renouvellement conditionnel des latents |
| `scripts/` | `download_data.py`, `run_all.py`, `evaluate.py`, `make_figures.py`, `demo.py`, `explore.py` |
| `explorer/` | Simulateur interactif : `trace.py`, `server.py`, `catalog.py`, `encode.py`, `static/` |
| `article/` | Article (Word + LaTeX IEEEtran) et présentation de soutenance |
| `results/` | `models/`, `eval/*.csv`, `tables/summary.{csv,md}`, `logs/` |

Tous les modèles sont **entièrement convolutifs** : un réseau entraîné sur 32×32 tourne sur 768×512 sans
modification, pourvu que les dimensions soient divisibles par 4.

### 5.1 Simulateur interactif

`python scripts/explore.py` ouvre une page locale qui déroule **une transmission réelle, étape par étape**.
L'émetteur, le canal et le récepteur sont dessinés comme une carte ; on avance d'étape en étape (lecture,
pause, pas à pas, curseur) et chaque étape montre ce qui a réellement traversé ce point : le tenseur latent en
pile 3D (canaux émis en couleur, canaux coupés en gris), les symboles dans le plan I/Q avant et après le bruit,
les bits codés avec les bits faux en rouge, les LLR signés, l'image reçue, le verdict des juges.

Rien n'y est rejoué ni pré-enregistré : chaque étape appelle le code du projet (`semcom.channel`,
`semcom.digital`, `semcom.models`, `semcom.metrics`), donc ce que la page montre est ce que `evaluate.py`
mesure. Quatre onglets : *Explorer*, *Comparer* (plusieurs systèmes sur la même image, calculés en direct),
*Résultats mesurés* (courbes lues dans `results/eval/*.csv`) et *Concepts* (les notions de cours, chacune
renvoyant au fichier source). L'état tient dans l'URL, donc une étape précise d'une transmission précise se
partage par lien.

Le serveur est `http.server` et la page est du HTML/CSS/JS sans bibliothèque : aucune dépendance au-delà de
`requirements.txt`.

---

## 6. Limites

Ce qu'il faut annoncer avant qu'on le demande :

1. **Le canal est simulé.** AWGN et Rayleigh par blocs sont des modèles mathématiques. Aucune mesure ne vient
   d'une liaison radio réelle, d'un SDR ou d'un banc d'essai.
2. **La CSI est parfaite** sous Rayleigh : le récepteur connaît exactement `h`. Un vrai récepteur l'estime, avec
   une erreur.
3. **Les JSCC ne sont entraînés que sur CIFAR-10.** Les résultats Kodak et vidéo sont donc des résultats de
   *transfert*, et c'est la principale raison pour laquelle la borne séparée les dépasse sur grandes images.
4. **La borne de capacité n'est pas un système.** Elle suppose zéro erreur ; aucun code réel ne l'atteint sur
   des paquets courts.
5. **Les séquences vidéo ne sont pas celles du cahier des charges** : trois clips CIF remplacent les clips UVG
   1080p, pour une raison de taille de fichiers.
6. **Le juge sémantique sur images naturelles est un ResNet-18 ImageNet**, qui n'est pas un juge de qualité
   perceptuelle : il mesure la préservation du contenu reconnaissable, pas l'agrément visuel.

---

## 7. Reproduction

```bash
python -m venv .venv
.venv\Scripts\activate              # Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py     # CIFAR-10, Kodak, 3 clips Xiph, poids ResNet-18 (~250 Mo)

python scripts/run_all.py --workers 4 --threads 4   # entraîne les 9 modèles (relançable)
python scripts/evaluate.py all --workers 6          # -> results/eval/*.csv
python scripts/make_figures.py                      # tableaux et figures
python scripts/explore.py                           # simulateur interactif
```

Sur un CPU 16 threads, la campagne d'entraînement prend environ 2 h 30 et l'évaluation environ 30 min. Les
données ne sont pas versionnées (391 Mo, dont des fichiers au-dessus de la limite GitHub) : `download_data.py`
les régénère.

---

## 8. Conclusion

Le codage conjoint source-canal appris supprime les bits et place le canal à l'intérieur du réseau, ce qui
supprime l'effet falaise : la dégradation devient progressive. Y ajouter un terme sémantique dans la perte — et
rien d'autre — déplace le compromis dans la direction qui compte quand le récepteur est une machine : environ
0,5 dB de PSNR contre 10 à 14 points de précision, en AWGN comme en Rayleigh.

La limite est tout aussi claire : sur de grandes images jamais vues à l'entraînement, la chaîne séparée idéale
reste devant, et seule une campagne d'entraînement sur images naturelles permettrait de trancher. Les deux
extensions montrent où le gain est le plus grand : quand la tâche du récepteur est connue (1/384 de bande), et
quand le contenu est temporellement redondant (bande divisée par trois sur une scène statique).
