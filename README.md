# Téléphone arabe NLP, Robustesse des modèles NLLB en back-translation

Projet du cours de Traitement du Langage Naturel (EPITA).

## Cas d'usage

On reprend le principe du **« téléphone arabe »** appliqué à la traduction automatique :
une phrase anglaise est traduite en chaîne à travers plusieurs langues intermédiaires,
puis **re-traduite vers l'anglais** (back-translation) à chaque étape. En comparant la
re-traduction à la phrase de départ, on mesure la **dérive sémantique** accumulée, donc
la **robustesse** du modèle de traduction.

Le sujet étant volontairement simple, le projet est construit comme un **benchmark
comparatif** de plusieurs modèles pré-entraînés, conformément aux consignes.

## Modèle pré-entraîné

[**NLLB-200**](https://huggingface.co/docs/transformers/model_doc/nllb) (*No Language Left
Behind*, Meta), utilisé en **inférence** via HuggingFace Transformers. Deux tailles sont
comparées :

| Modèle | Paramètres |
|---|---|
| `facebook/nllb-200-distilled-600M` | 600 M |
| `facebook/nllb-200-1.3B` | 1.3 B |

(Le `nllb-200-3.3B` est exclu pour son coût mémoire/calcul.)

## Architecture

```
phrase EN ──translate──▶ langue 1 ──translate──▶ langue 2 ──▶ … (N étapes)
   │                         │
   │  à chaque étape : back-translation vers l'anglais
   ▼                         ▼
 référence  ◀── métriques ── re-traduction EN
```

- **`MTA`** (`nlp.py`) encapsule le modèle NLLB (chargement GPU/CPU, traduction beam search).
- **`run_MTA`** exécute la chaîne et calcule les métriques à chaque étape.
- **`run_benchmark`** rejoue ce processus pour chaque modèle sur tout un corpus et agrège.

## Métriques

Chaque re-traduction (anglais) est comparée à la phrase initiale (anglais) avec :

| Métrique | Échelle | Mesure |
|---|---|---|
| **BLEU** (sacrebleu) | 0–100 | recouvrement de n-grammes de mots |
| **chrF** (sacrebleu) | 0–100 | n-grammes de caractères (robuste à la morphologie) |
| **METEOR** (nltk) | 0–1 | tient compte des synonymes/stemming (WordNet) |
| **BERTScore F1** | 0–1 | similarité sémantique par embeddings |
| Similarité lexicale (Ratcliff/Obershelp) | 0–1 | similarité de surface (complément) |
| Similarité TF-IDF + cosinus | 0–1 | proximité thématique (complément) |

Plus le score est **élevé**, mieux le sens est **conservé** après la chaîne de traductions.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ressources WordNet pour METEOR (une seule fois) :
python -m nltk.downloader wordnet omw-1.4
```

## Données

Sous-ensemble de **FLORES-200** (Meta), partition `devtest`, phrases anglaises.
Généré par un script (la consigne autorise un script de téléchargement) :

```bash
python download_flores.py --n 100
```

## Utilisation

### Démo interactive (une phrase)

```bash
python nlp.py
python nlp.py -v
```

Options : `-v` verbeux · `-c` choisir les langues (depuis `languages.csv`) ·
`-m` choisir le modèle · `-r` langues aléatoires · `-t` phrase personnalisée ·
`-h` aide. Le récapitulatif final (phrases + toutes les métriques) s'affiche toujours,
et les résultats sont écrits dans `results/results.csv`.

### Benchmark comparatif

```bash
python download_flores.py --n 25 
python nlp.py -b
```

Produit :
- un **tableau comparatif** dans le terminal,
- `results/benchmark.csv` (une ligne par modèle, une colonne par métrique),
- `results/benchmark.png` (graphe des métriques normalisées par modèle).

> **Coût** : le benchmark fait `N_phrases × N_étapes × 2 traductions × 2 modèles`

## Résultats

Les artefacts sont écrits dans `results/` : `results.csv`, `benchmark.csv`,
`benchmark.png` (+ graphes du run interactif). Le tableau `benchmark.csv` constitue
le livrable de comparaison des modèles.

## Structure du dépôt

```
nlp.py              # programme principal (traduction, métriques, benchmark, CLI)
download_flores.py  # téléchargement du sous-ensemble FLORES-200
languages.csv       # table des 200 codes de langues NLLB
requirements.txt    # dépendances épinglées
data/               # corpus généré (flores_eng.txt)
results/            # sorties (CSV + figures)
```

## Limites & perspectives

- Pas de fine-tuning : comparaison de modèles **pré-entraînés** en inférence.
- BLEU/chrF au niveau corpus restent bruités sur de petits corpus → élargir le jeu de
  données pour plus de robustesse statistique.
- Pistes : comparer d'autres systèmes (M2M100, OPUS-MT), étudier l'effet du **chemin de
  langues** et de la **longueur de chaîne**, ajouter COMET (métrique neuronale MT).
