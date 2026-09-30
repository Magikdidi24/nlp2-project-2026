"""
Télécharge un sous-ensemble de phrases anglaises de FLORES-200 (Meta) et l'écrit
dans un fichier texte (une phrase par ligne), utilisé ensuite par le mode
benchmark de nlp.py (`python nlp.py -b`).

On récupère l'archive officielle du projet NLLB (téléchargement direct, sans
compte HuggingFace ni dépendance `datasets`).

Usage :
    python download_flores.py --n 25
    python download_flores.py --n 50 --split dev --out data/flores_eng.txt
"""

import argparse
import os
import tarfile
import tempfile
import urllib.request

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def _progress(block_num, block_size, total_size):
    if total_size > 0:
        pct = min(100, block_num * block_size * 100 // total_size)
        print(f"\rTéléchargement FLORES-200 : {pct:3d}%", end="", flush=True)


def english_sentences(split, n):
    """Télécharge l'archive FLORES-200 et renvoie les n premières phrases anglaises."""
    suffix = f"/{split}/eng_Latn.{split}"
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "flores200_dataset.tar.gz")
        urllib.request.urlretrieve(FLORES_URL, archive, reporthook=_progress)
        print()
        with tarfile.open(archive, "r:gz") as tar:
            member = next((m for m in tar.getmembers() if m.name.endswith(suffix)), None)
            if member is None:
                raise SystemExit(f"Fichier *{suffix} introuvable dans l'archive FLORES.")
            with tar.extractfile(member) as f:
                lines = [line.decode("utf-8").strip() for line in f]
    sentences = [line for line in lines if line]
    return sentences[:n]


def main():
    parser = argparse.ArgumentParser(
        description="Télécharge un sous-ensemble FLORES-200 (anglais)."
    )
    parser.add_argument("--n", type=int, default=25,
                        help="Nombre de phrases à extraire (défaut : 25).")
    parser.add_argument("--split", default="devtest", choices=["dev", "devtest"],
                        help="Partition FLORES à utiliser (défaut : devtest).")
    parser.add_argument("--out", default="data/flores_eng.txt",
                        help="Fichier de sortie (défaut : data/flores_eng.txt).")
    args = parser.parse_args()

    sentences = english_sentences(args.split, args.n)
    if not sentences:
        raise SystemExit("Aucune phrase extraite — vérifiez la partition / l'archive.")

    dirpath = os.path.dirname(args.out)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences) + "\n")

    print(f"{len(sentences)} phrases écrites dans {args.out}")


if __name__ == "__main__":
    main()
