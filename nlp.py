import argparse
import csv
import os
import random
import re
import sys
import time
from difflib import SequenceMatcher

# ============================================================
# VÉRIFICATION DES DÉPENDANCES
# ============================================================

IMPORT_NAMES = {"scikit-learn": "sklearn", "bert-score": "bert_score"}

def check_dependencies(verbose=True):
    """Vérifie que les paquets de requirements.txt sont importables, sinon quitte."""
    with open("requirements.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    missing_packages = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=~!]", line, maxsplit=1)[0].strip()
        try:
            __import__(IMPORT_NAMES.get(name, name))
        except ImportError:
            missing_packages.append(name)

    if missing_packages:
        print("Librairies manquantes :")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstalle-les avec :")
        print(f"   pip install {' '.join(missing_packages)}")
        sys.exit(1)
    else:
        if verbose:
            print("Toutes les librairies sont disponibles.")

# ============================================================
check_dependencies(verbose=False)
# ============================================================

import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score as nltk_meteor

# Références : https://huggingface.co/docs/transformers/model_doc/nllb
#             https://gist.github.com/glowinthedark/887e715b23473d0054cf5299a8328f8d
#             https://gist.github.com/tos-kamiya/033d016094d0570e514a867fd37c21a9

# ============================================================
# 1. CONFIGURATION BASIQUE
# ============================================================

# Version basique
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Version moyenne :
# MODEL_NAME = "facebook/nllb-200-1.3B"

# Version lourde :
# MODEL_NAME = "facebook/nllb-200-3.3B"

# NLLB supporte 200 langues (qualité variable selon la langue). Les textes > 512 tokens
# peuvent être mal traduits ; ici on limite la sortie à 80 tokens (max_new_tokens).

SOURCE_LANG_CODE = "eng_Latn"
SOURCE_LANG_NAME = "English"

N_STEPS = 5

# Languages basique
LANGUAGES = {
    "English": "eng_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Spanish": "spa_Latn",
    # "Breton": "bre_Latn",
    # "Welsh": "cym_Latn",
    # "Irish": "gle_Latn",
    # "Scottish Gaelic": "gla_Latn",
    # "Esperanto": "epo_Latn",
}

INITIAL_SENTENCE = "Our NLP's teacher is the best teacher at EPITA !"

RANDOM_SEED = 42

# --- Benchmark comparatif (mode -b) ---
# Modèles comparés (le 3.3B ~17 Go est volontairement exclu pour le coût).
BENCHMARK_MODELS = [
    "facebook/nllb-200-distilled-600M",
    "facebook/nllb-200-1.3B",
]
BENCHMARK_N_STEPS = 4              # nombre d'étapes du téléphone arabe par phrase
BENCHMARK_DATA = "data/flores_eng.txt"  # corpus généré par download_flores.py

# ============================================================
# 2. FONCTIONS DE GESTION DES LANGUES ET DU MENU
# ============================================================

def import_languages_from_csv(filepath):

    languages = {}

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            _, language_name, code, *_ = row
            languages[language_name] = code.strip()

    return languages

def clear_previous_block(n):
    """Efface les n-1 dernières lignes affichées dans le terminal."""
    for _ in range(n - 1):
        print("\033[1A\r\033[2K", end="", flush=True)  # remonte d'1 ligne + efface

def build_language_menu(languages_items, selected, page, page_size):
    total_pages = (len(languages_items) + page_size - 1) // page_size
    start = page * page_size
    end = start + page_size
    current_page_items = languages_items[start:end]

    lines = []
    lines.append("=" * 80)
    lines.append(f"Choix des langues - Page {page + 1}/{total_pages}")
    lines.append("=" * 80)

    for i, (name, code) in enumerate(current_page_items, start=(start+1)):
        mark = "[X]" if name in selected else "[ ]"
        lines.append(f"{i:3d}. {mark} {name} ({code})")

    lines.append("")
    lines.append("Languages sélectionnées :")

    if (selected):
        selected_text = ", ".join([f"{name} ({code})" for name, code in selected.items()])
        lines.append(selected_text)
    else:
        lines.append("Aucune langue sélectionnée.")

    lines.append("")
    lines.append("Commandes :")
    lines.append("  numéro       -> ajouter / retirer une langue")
    lines.append("  n            -> page suivante")
    lines.append("  p            -> page précédente")
    lines.append("  s            -> afficher la sélection")
    lines.append("  q            -> valider et quitter")
    lines.append("  clear        -> vider la sélection")

    menu_text = "\n".join(lines)
    return menu_text, len(lines)

def choose_language_terminal(languages, page_size=10, min_selection=2, max_selection=None):

    language_items = sorted(languages.items(), key=lambda x: x[0].lower())
    selected_languages = {}
    page = 0

    total_pages = (len(language_items) + page_size - 1) // page_size
    printed_lines = 0

    while True:
        if (printed_lines > 0):
            clear_previous_block(printed_lines)

        menu_text, printed_lines = build_language_menu(language_items, selected_languages, page, page_size)
        print(menu_text, end="")

        choice = input().strip().lower()
        printed_lines += 1

        if (choice == "n"):
            if (page < total_pages - 1):
                page += 1

        elif (choice == "p"):
            if (page > 0):
                page -= 1

        elif (choice == "clear"):
            selected_languages.clear()

        elif (choice == "q"):
            if (len(selected_languages) == 0):
                continue

            if (len(selected_languages) < min_selection):
                print(f"Veuillez sélectionner au moins {min_selection} langue(s).")
                continue

            if (printed_lines > 0):
                clear_previous_block(printed_lines)

            print("Langues sélectionnées :")
            for name, code in selected_languages.items():
                print(f"- {name} ({code})")

            return selected_languages

        elif (choice.isdigit()):
            index = int(choice)

            if (1 <= index <= len(language_items)):
                name, code = language_items[index - 1]

                if (name in selected_languages):
                    del selected_languages[name]
                else:
                    selected_languages[name] = code
                    if (max_selection is not None and len(selected_languages) >= max_selection):
                        if (printed_lines > 0):
                            clear_previous_block(printed_lines)
                        print("Langues sélectionnées :")
                        for n, c in selected_languages.items():
                            print(f"- {n} ({c})")
                        return selected_languages

def prompt_until_valid(message, parse, error):
    """Répète la saisie jusqu'à ce que parse(entrée) renvoie une valeur non None."""
    while True:
        value = parse(input(message).strip())
        if value is not None:
            return value
        print(error)

def choose_number_of_steps():
    """Demande le nombre de traductions (entier > 1)."""
    return prompt_until_valid(
        "Nombre de traductions (n) : ",
        lambda c: int(c) if c.isdigit() and int(c) > 1 else None,
        "Veuillez entrer un nombre entier supérieur à 1.",
    )

def choose_model():
    """Affiche le menu des modèles et renvoie l'identifiant du modèle choisi."""
    models = {
        "1": ("facebook/nllb-200-distilled-600M", "Version basique"),
        "2": ("facebook/nllb-200-1.3B", "Version moyenne"),
        "3": ("facebook/nllb-200-3.3B", "Version lourde"),
    }

    print("Choix du modèle de traduction :")
    for key, (model_name, description) in models.items():
        print(f"{key}. {description} ({model_name})")

    return prompt_until_valid(
        "Votre choix (1/2/3) : ",
        lambda c: models[c][0] if c in models else None,
        "Veuillez entrer 1, 2 ou 3.",
    )

def choose_text_lang(languages_available):
    """Choisit la phrase initiale : phrase par défaut (1) ou personnalisée (2)."""
    print("Choix de la phrase initiale :")
    print(f"1. {INITIAL_SENTENCE}")
    print("2. Une phrase personnalisée")

    choice = prompt_until_valid(
        "Votre choix (1/2) : ",
        lambda c: c if c in ("1", "2") else None,
        "Veuillez entrer 1 ou 2.",
    )

    if choice == "1":
        return INITIAL_SENTENCE, SOURCE_LANG_CODE, SOURCE_LANG_NAME

    source = choose_language_terminal(languages_available, page_size=15, min_selection=1, max_selection=1)
    name, code = list(source.items())[0]
    print(f"{name} ({code})")
    text = input(f"Entrez votre phrase initiale {name}: ").strip()
    return text, code, name

# ============================================================
# 3. FONCTIONS DE PRE-PROCESSING ET DE SIMILARITE
# ============================================================

def pre_process_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def lexical_similarity(text1, text2):
    """Similarité de surface (caractères) via SequenceMatcher, dans [0.0, 1.0]."""
    a = pre_process_text(text1)
    b = pre_process_text(text2)
    return SequenceMatcher(None, a, b).ratio()

def tfidf_cosine_similarity(text1, text2):
    """Similarité TF-IDF (uni+bigrammes) + cosinus, dans [0.0, 1.0]."""
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]

    return float(score)

def bleu_score(reference, hypothesis):
    """Score BLEU (sacrebleu) au niveau phrase, dans [0, 100]."""
    return sacrebleu.sentence_bleu(hypothesis, [reference]).score

def chrf_score(reference, hypothesis):
    """Score chrF (sacrebleu, n-grammes de caractères), dans [0, 100]."""
    return sacrebleu.sentence_chrf(hypothesis, [reference]).score

_NLTK_READY = False

def _ensure_nltk():
    """Vérifie/(re)télécharge (une seule fois, visiblement) les ressources WordNet de METEOR."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    for res, path in [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
        force = False
        try:
            nltk.data.find(path)
            continue
        except LookupError:
            print(f"[NLTK] Téléchargement de la ressource METEOR « {res} »...")
        except Exception:
            force = True
            print(f"[NLTK] Ressource « {res} » corrompue — re-téléchargement forcé...")
        if not nltk.download(res, force=force):
            raise RuntimeError(
                f"Échec du téléchargement de la ressource NLTK « {res} ». "
                f"Réessayez manuellement : python -m nltk.downloader -f wordnet omw-1.4"
            )
    _NLTK_READY = True

def meteor_score_fn(reference, hypothesis):
    """Score METEOR (nltk, synonymes/stemming via WordNet), dans [0.0, 1.0]."""
    _ensure_nltk()
    return float(nltk_meteor([reference.split()], hypothesis.split()))

_BERT_SCORER = None

def get_bert_scorer():
    """Instancie (une seule fois) le scorer BERTScore anglais et le réutilise."""
    global _BERT_SCORER
    if _BERT_SCORER is None:
        from bert_score import BERTScorer
        _BERT_SCORER = BERTScorer(lang="en", rescale_with_baseline=False)
    return _BERT_SCORER

def bertscore_f1(reference, hypothesis):
    """F1 BERTScore (similarité sémantique par embeddings), dans [0.0, 1.0]."""
    _, _, f1 = get_bert_scorer().score([hypothesis], [reference])
    return float(f1[0])

# ============================================================
# 4. MODELE DE TRADUCTION
# ============================================================

class MTA: # Multiple Traduction Assistant
    """
    Encapsule le modèle de traduction NLLB (Meta) via HuggingFace Transformers :
    chargement sur GPU/CPU, langues supportées, et traduction source -> cible.
    """

    def __init__(self, model_name, verbose=True):
        """Charge le tokenizer et le modèle NLLB sur le dispositif disponible (CUDA sinon CPU)."""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if verbose:
            print(f'Chargement du modèle: {model_name}')
            print(f'Dispositif utilisé: {self.device}')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # .to(device) : déplace les poids vers le dispositif cible (VRAM si GPU).
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        # On pilote la longueur via max_new_tokens dans translate() ; on retire le
        # max_length par défaut du modèle pour éviter l'avertissement de conflit.
        self.model.generation_config.max_length = None

        self.supported_languages = self.get_supported_languages()

    def get_supported_languages(self):
        """Renvoie l'ensemble des codes de langues acceptés par le tokenizer."""
        if hasattr(self.tokenizer, "lang_code_to_id"):
            return set(self.tokenizer.lang_code_to_id.keys())

        vocab = self.tokenizer.get_vocab()
        return set(vocab.keys())

    def check_language(self, lang_code):
        """Vrai si lang_code (ex: "fra_Latn") est reconnu par le tokenizer."""
        return lang_code in self.supported_languages

    def translate(self, text, source_lang, target_lang):
        """
        Traduit `text` de source_lang vers target_lang (codes NLLB, ex: "eng_Latn").

        Pipeline : tokenisation (src_lang) -> generate en beam search avec premier
        token forcé à la langue cible (forced_bos_token_id) -> décodage du texte.
        """
        # Étape 1 : informer le tokenizer de la langue source
        self.tokenizer.src_lang = source_lang

        # Tokenisation + conversion en tenseurs PyTorch + envoi sur device
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # Étape 2 : obtenir l'ID entier du token de langue cible (contrainte de décodage)
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_lang)

        # Étape 3 : inférence sans calcul de gradient (économie mémoire + vitesse)
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,                               # input_ids + attention_mask
                forced_bos_token_id=forced_bos_token_id,# force la langue cible
                num_beams=4,                            # beam search : 4 hypothèses
                max_new_tokens=80,                      # limite la longueur de la traduction
                no_repeat_ngram_size=3,                 # empêche la répétition de n-grammes de taille 3
                repetition_penalty=1.2,                 # pénalise légèrement la répétition de tokens
                early_stopping=True,                    # arrête la génération dès que tous les faisceaux ont généré un token de fin (EOS)
            )

        # Étape 4 : décodage des indices de tokens en texte lisible
        translated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,  # retire les tokens spéciaux (BOS, EOS, langue...)
        )[0]  # batch_size = 1, on prend la première (et seule) hypothèse

        return translated_text

# ============================================================
# 5. TELEPHONE ARABE
# ============================================================

def filter_supported_language(translator, languages, verbose=True):
    """Ne garde que les langues {nom: code} reconnues par le tokenizer du modèle."""
    valid_language = {}

    for name, code in languages.items():
        if (translator.check_language(code)):
            valid_language[name] = code
            if verbose:
                print(f"OK - {name}: {code}")
        else:
            if verbose:
                print(f"ERREUR - {name}: {code}")
    return valid_language

def run_MTA(translator, initial_sequence, languages, n_step, source_lang_code, source_lang_name, random_seed=None, random_selection=False, verbose=True):
    """
    Jeu du "téléphone arabe" : traduit la phrase à travers n_step langues, et à chaque
    étape la retraduit vers la source (back-translation) pour mesurer la dérive via
    les scores de similarité. Langues parcourues en cycle, ou tirées au hasard si
    random_selection. Renvoie une liste de dict (un par étape) ; voir les clés ci-dessous.
    """
    if (random_seed is not None):
        random.seed(random_seed)

    results = []

    current_text = initial_sequence
    current_lang_code = source_lang_code
    current_lang_name = source_lang_name

    language_items = list(languages.items())

    # Langues disponibles comme cibles (excluant la langue source)
    non_source_items = [(name, code) for name, code in language_items if code != source_lang_code]
    if not non_source_items:
        raise ValueError(
            f"Aucune langue cible disponible : toutes les langues sélectionnées "
            f"sont la langue source ({source_lang_name}). "
            f"Veuillez sélectionner au moins une autre langue."
        )

    if (random_selection):
        if verbose:
            print("Sélection aléatoire des langues activée.")

    for step in range(1, n_step + 1):

        if (random_selection):
            possible_target = [(name, code) for name, code in non_source_items if code != current_lang_code]
            if not possible_target:
                possible_target = [(name, code) for name, code in language_items if code != current_lang_code]
            target_lang_name, target_lang_code = random.choice(possible_target)
        else:
            possible_target = non_source_items
            if (possible_target[(step - 1) % len(possible_target)] == (current_lang_name, current_lang_code)):
                target_lang_name, target_lang_code = possible_target[(step if step < len(possible_target) else 0) % len(possible_target)]
            else:
                target_lang_name, target_lang_code = possible_target[(step - 1) % len(possible_target)]

        translated_text = translator.translate(
            text=current_text,
            source_lang=current_lang_code,
            target_lang=target_lang_code,
        )

        back_translated_text = translator.translate(
            text=translated_text,
            source_lang=target_lang_code,
            target_lang=source_lang_code,
        )

        # Métriques : back-traduction (anglais) comparée à la phrase initiale.
        lex_score = lexical_similarity(initial_sequence, back_translated_text)
        tfidf_score = tfidf_cosine_similarity(initial_sequence, back_translated_text)
        bleu = bleu_score(initial_sequence, back_translated_text)
        chrf = chrf_score(initial_sequence, back_translated_text)
        meteor = meteor_score_fn(initial_sequence, back_translated_text)
        bertscore = bertscore_f1(initial_sequence, back_translated_text)

        result = {
            "step": step,
            "source_language_code": current_lang_code,
            "source_language_name": current_lang_name,
            "target_language_code": target_lang_code,
            "target_language_name": target_lang_name,
            "translated_text": translated_text,
            "back_translated_text": back_translated_text,
            "lexical_similarity": lex_score,
            "tfidf_cosine_similarity": tfidf_score,
            "bleu": bleu,
            "chrf": chrf,
            "meteor": meteor,
            "bertscore_f1": bertscore,
        }

        results.append(result)

        current_text = translated_text
        current_lang_code = target_lang_code
        current_lang_name = target_lang_name

    return results


# ============================================================
# 6. Affichage resultat - Textuel
# ============================================================

def print_results(initial_sequence, results, source_lang_name):
    """Affiche le récapitulatif final (phrases + scores) dans le terminal."""
    print("=" * 80)
    print("\tTÉLÉPHONE ARABE")
    print("=" * 80)

    print(f'Phrase initiale : {initial_sequence}')

    for r in results:
        print("-" * 80)
        print(f"Étape {r['step']}")

        print(
            f"{r['source_language_name']} ({r['source_language_code']}) "
            f"-> {r['target_language_name']} ({r['target_language_code']})"
        )

        print(f"Phrase traduite : {r['translated_text']}")

        print(f"Retraduction en {source_lang_name} : {r['back_translated_text']}")

        print("Scores par rapport à la phrase initiale :")
        print(f"Similarité lexicale         : {r['lexical_similarity']:.4f}  (0-1)")
        print(f"Similarité TF-IDF cosinus   : {r['tfidf_cosine_similarity']:.4f}  (0-1)")
        print(f"BLEU                        : {r['bleu']:.2f}    (0-100)")
        print(f"chrF                        : {r['chrf']:.2f}    (0-100)")
        print(f"METEOR                      : {r['meteor']:.4f}  (0-1)")
        print(f"BERTScore (F1)              : {r['bertscore_f1']:.4f}  (0-1)")

    print("=" * 80)

# ============================================================
# 7. Affichage resultat - Graph
# ============================================================

def ensure_dir(filepath):
    """Crée le répertoire parent de filepath s'il n'existe pas."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

def finalize_plot(save_path, verbose):
    """Sauvegarde (si save_path) puis affiche la figure si le backend est interactif."""
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight")
        if verbose:
            print(f"Graphique sauvegardé dans : {save_path}")
    # En headless (backend Agg, ex: terminal WSL), plt.show() ne fait qu'avertir.
    if plt.rcParams["backend"].lower() != "agg":
        plt.show()

def save_results_to_csv(results, filepath, verbose=True):
    if not results:
        return

    ensure_dir(filepath)

    fieldnames = list(results[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    if verbose:
        print(f"Résultats sauvegardés dans : {filepath}")

def plot_scores(results, save_path=None, verbose=True):
    steps = [r["step"] for r in results]
    lexical_scores = [r["lexical_similarity"] for r in results]
    tfidf_scores = [r["tfidf_cosine_similarity"] for r in results]

    plt.figure(figsize=(10, 6))

    plt.plot(steps, lexical_scores, marker="o", label="Similarité lexicale")
    plt.plot(steps, tfidf_scores, marker="o", label="TF-IDF + cosinus")

    plt.xlabel("Nombre de traductions")
    plt.ylabel("Score")
    plt.title("Évolution de la similarité après plusieurs traductions")

    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()

    finalize_plot(save_path, verbose)

def print_result_graph(results, save_path=None, verbose=True):

    labels = []
    drops = []

    previous_score = 1.0

    for r in results:
        current_score = r['tfidf_cosine_similarity']
        drop = previous_score - current_score

        labels.append(f"{r['step']} - {r['target_language_name']}")
        drops.append(drop)

        previous_score = current_score

    plt.figure(figsize=(12, 6))
    plt.bar(labels, drops)

    plt.xlabel("Langue cible")
    plt.ylabel("Baisse du score TF-IDF")
    plt.title("Impact approximatif de chaque langue sur la perte de similarité")

    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")

    finalize_plot(save_path, verbose)


# ============================================================
# 8. BENCHMARK COMPARATIF
# ============================================================

def load_sentences(path):
    """Lit un corpus (une phrase par ligne) ; erreur claire si le fichier manque."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus introuvable : {path}. "
            f"Lancez d'abord : python download_flores.py --n 25"
        )
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def run_benchmark(model_names, sentences, languages, n_step, verbose=True):
    """
    Compare plusieurs modèles NLLB sur le jeu du téléphone arabe.

    Pour chaque modèle : exécute run_MTA sur chaque phrase du corpus, puis agrège
    les métriques finales — BLEU/chrF en corpus-level (plus robuste), le reste en
    moyenne. Renvoie une liste de dicts (une ligne par modèle).
    """
    rows = []
    total_models = len(model_names)
    n = len(sentences)
    for mi, model_name in enumerate(model_names, start=1):
        print(f"\n[Modèle {mi}/{total_models}] Chargement de {model_name} ...", flush=True)
        translator = MTA(model_name, verbose=verbose)
        valid_languages = filter_supported_language(translator, languages, verbose=False)
        print(f"  Prêt sur {translator.device}. "
              f"Traitement de {n} phrases x {n_step} étapes ")

        refs, hyps = [], []
        sums = {"meteor": 0.0, "bertscore_f1": 0.0,
                "lexical_similarity": 0.0, "tfidf_cosine_similarity": 0.0}

        t0 = time.time()
        for i, sentence in enumerate(sentences, start=1):
            print(f"\r    Phrase {i}/{n} en cours...", end="", flush=True)
            results = run_MTA(
                translator=translator,
                initial_sequence=sentence,
                languages=valid_languages,
                n_step=n_step,
                source_lang_code=SOURCE_LANG_CODE,
                source_lang_name=SOURCE_LANG_NAME,
                random_seed=RANDOM_SEED,
                random_selection=False,
                verbose=False,
            )
            final = results[-1]
            refs.append(sentence)
            hyps.append(final["back_translated_text"])
            for key in sums:
                sums[key] += final[key]

            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            print(f"\r    Phrase {i}/{n}  |  écoulé {elapsed:4.0f}s  |  ETA {eta:4.0f}s     ",
                  end="", flush=True)
        print()

        row = {
            "model": model_name,
            "n_sentences": n,
            "n_steps": n_step,
            "bleu_corpus": sacrebleu.corpus_bleu(hyps, [refs]).score,
            "chrf_corpus": sacrebleu.corpus_chrf(hyps, [refs]).score,
            "meteor_mean": sums["meteor"] / n,
            "bertscore_f1_mean": sums["bertscore_f1"] / n,
            "lexical_mean": sums["lexical_similarity"] / n,
            "tfidf_mean": sums["tfidf_cosine_similarity"] / n,
        }
        rows.append(row)
        print(f"  {model_name.split('/')[-1]} terminé en {time.time() - t0:.0f}s — "
              f"BLEU {row['bleu_corpus']:.1f} | chrF {row['chrf_corpus']:.1f} | "
              f"METEOR {row['meteor_mean']:.3f} | BERTScore {row['bertscore_f1_mean']:.3f}",
              flush=True)
    return rows

def print_benchmark_table(rows):
    """Affiche le tableau comparatif des modèles dans le terminal."""
    cols = ["bleu_corpus", "chrf_corpus", "meteor_mean",
            "bertscore_f1_mean", "lexical_mean", "tfidf_mean"]
    print("\n" + "=" * 100)
    print("\tBENCHMARK COMPARATIF")
    print("=" * 100)
    header = f"{'modèle':38s} " + " ".join(f"{c:>13s}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:38s} " + " ".join(f"{r[c]:13.3f}" for c in cols))
    print("=" * 100)

def plot_benchmark(rows, save_path=None, verbose=True):
    """Graphe comparatif : métriques normalisées dans [0, 1] par modèle."""
    metrics = [
        ("bleu_corpus", "BLEU", 100.0),
        ("chrf_corpus", "chrF", 100.0),
        ("meteor_mean", "METEOR", 1.0),
        ("bertscore_f1_mean", "BERTScore", 1.0),
        ("lexical_mean", "Lexical", 1.0),
        ("tfidf_mean", "TF-IDF", 1.0),
    ]
    x = range(len(metrics))
    width = 0.8 / max(len(rows), 1)

    plt.figure(figsize=(12, 6))
    for i, r in enumerate(rows):
        heights = [r[key] / scale for key, _, scale in metrics]
        offsets = [xi + i * width for xi in x]
        plt.bar(offsets, heights, width=width, label=r["model"].split("/")[-1])

    plt.xticks([xi + width * (len(rows) - 1) / 2 for xi in x],
               [label for _, label, _ in metrics])
    plt.ylabel("Score normalisé [0, 1]")
    plt.title("Benchmark NLLB — robustesse en back-translation (plus haut = mieux)")
    plt.ylim(0, 1.05)
    plt.grid(axis="y")
    plt.legend()
    finalize_plot(save_path, verbose)

# ============================================================
# 9. MAIN
# ============================================================

def parse_args():
    """Définit et parse les options en ligne de commande (-h/-v/-c/-m/-r/-t)."""
    parser = argparse.ArgumentParser(description="Téléphone arabe de traduction (NLLB).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Affiche des informations détaillées pendant l'exécution.")
    parser.add_argument("-c", "--choice", action="store_true",
                        help="Choisir les langues depuis un fichier CSV (languages.csv).")
    parser.add_argument("-m", "--model", action="store_true",
                        help="Choisir le modèle de traduction à utiliser.")
    parser.add_argument("-r", "--random", action="store_true",
                        help="Sélection aléatoire des langues (sinon ordre de sélection / défaut).")
    parser.add_argument("-t", "--text", action="store_true",
                        help="Spécifier une phrase initiale personnalisée avec sa langue.")
    parser.add_argument("-b", "--benchmark", action="store_true",
                        help="Mode benchmark : compare les modèles NLLB sur le corpus FLORES.")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    VERBOSE = args.verbose

    _ensure_nltk()

    if args.benchmark:
        sentences = load_sentences(BENCHMARK_DATA)
        if VERBOSE:
            print(f"Benchmark : {len(sentences)} phrases, {BENCHMARK_N_STEPS} étapes, "
                  f"modèles : {', '.join(BENCHMARK_MODELS)}\n")
        rows = run_benchmark(BENCHMARK_MODELS, sentences, LANGUAGES,
                             BENCHMARK_N_STEPS, verbose=VERBOSE)
        print_benchmark_table(rows)
        save_results_to_csv(rows, os.path.join("results", "benchmark.csv"), verbose=True)
        plot_benchmark(rows, os.path.join("results", "benchmark.png"), verbose=VERBOSE)
        sys.exit(0)

    if VERBOSE:
        print("\nInitialisation du Multiple Traduction Assistant...\n")

    if args.model:
        MODEL_NAME = choose_model()

    if VERBOSE:
        print(f"Modèle sélectionné : {MODEL_NAME}\n")

    translator = MTA(MODEL_NAME, VERBOSE)

    print()

    file = "languages.csv"

    all_languages = {}

    if os.path.exists(file) and args.choice:
        if VERBOSE:
            print(f"Importation des langues depuis {file}...")
        all_languages = import_languages_from_csv(file)
    else:
        all_languages = LANGUAGES

    all_languages = filter_supported_language(translator, all_languages, verbose=VERBOSE)

    if (len(all_languages) == 0):
        raise ValueError("Aucune langue valide trouvée.")

    languages_selected = all_languages

    if args.choice:
        languages_selected = choose_language_terminal(all_languages, page_size=15)

        if len(languages_selected) == 0:
            raise ValueError("Aucune langue sélectionnée.")

    if args.text:
        INITIAL_SENTENCE, SOURCE_LANG_CODE, SOURCE_LANG_NAME = choose_text_lang(all_languages)
        if VERBOSE:
            print()

    N_STEPS = choose_number_of_steps()

    if VERBOSE:
        print("\nLangues sélectionnées :")
        for name, code in languages_selected.items():
            print(f"{name} ({code})")

    results = run_MTA(
        translator=translator,
        initial_sequence=INITIAL_SENTENCE,
        languages=languages_selected,
        n_step=N_STEPS,
        source_lang_code=SOURCE_LANG_CODE,
        source_lang_name=SOURCE_LANG_NAME,
        random_seed=RANDOM_SEED,
        random_selection=args.random,
        verbose=VERBOSE,
    )

    print_results(INITIAL_SENTENCE, results, source_lang_name=SOURCE_LANG_NAME)

    results_dir = "results"

    save_results_to_csv(results, os.path.join(results_dir, "results.csv"), verbose=True)
    print_result_graph(results, os.path.join(results_dir, "bar_chart.png"), verbose=VERBOSE)
    plot_scores(results, os.path.join(results_dir, f"line_chart.png"), verbose=VERBOSE)