"""Construye data/processed/corpus.json a partir de data/pdfs/*.pdf.

Uso (desde la raíz del repositorio):
    python -m src.ingestion.build_corpus

Si algún campo sale mal, se corrige en data/overrides.json:
    {"archivo.pdf": {"title": "...", "authors": "...", "abstract": "..."}}
"""
import json
import sys
from pathlib import Path

from src.ingestion.pdf_reader import (
    find_abstract,
    guess_authors,
    guess_title,
    read_first_pages,
)

PDF_DIR = Path("data/pdfs")
OUT_FILE = Path("data/processed/corpus.json")
OVERRIDES_FILE = Path("data/overrides.json")


def build_corpus(pdf_dir: Path = PDF_DIR, overrides_file: Path = OVERRIDES_FILE) -> list[dict]:
    overrides = {}
    if overrides_file.exists():
        overrides = json.loads(overrides_file.read_text(encoding="utf-8"))

    corpus = []
    for idx, pdf in enumerate(sorted(pdf_dir.glob("*.pdf")), start=1):
        review = []
        try:
            text, meta_title = read_first_pages(str(pdf))
        except Exception as exc:  # PDF dañado o ilegible
            text, meta_title = "", None
            review.append(f"no se pudo leer el PDF ({exc})")

        abstract, abs_pos = find_abstract(text)
        title, title_src = guess_title(text, meta_title)
        authors = guess_authors(text[:abs_pos] if abs_pos > 0 else text[:800], title)

        ov = overrides.get(pdf.name, {})
        title = ov.get("title", title)
        authors = ov.get("authors", authors)
        abstract = ov.get("abstract", abstract)
        if "title" in ov:
            title_src = "override"

        words = len(abstract.split()) if abstract else 0
        if not abstract:
            review.append("abstract no encontrado")
        elif words < 50:
            review.append(f"abstract muy corto ({words} palabras)")
        elif words > 400:
            review.append(f"abstract muy largo ({words} palabras)")
        if title_src == "primera linea":
            review.append("título tomado de la primera línea")
        if not authors:
            review.append("autores vacíos")

        corpus.append({
            "id": f"A{idx:02d}",
            "file": pdf.name,
            "title": title,
            "authors": authors,
            "abstract": abstract or "",
            "abstract_words": words,
            "needs_review": review,
        })
    return corpus


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    corpus = build_corpus()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(corpus)} PDFs procesados -> {OUT_FILE}\n")
    for doc in corpus:
        estado = "OK " if not doc["needs_review"] else "REV"
        print(f"[{estado}] {doc['id']}  {doc['abstract_words']:>4} palabras  {doc['file']}")
        print(f"        título : {doc['title'][:90]}")
        print(f"        autores: {doc['authors'][:90]}")
        for motivo in doc["needs_review"]:
            print(f"        ! {motivo}")


if __name__ == "__main__":
    main()