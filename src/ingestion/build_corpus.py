"""Construye data/processed/corpus.json a partir de los PDFs de data/pdfs/.

Uso (desde la raíz del repositorio):
    python -m src.ingestion.build_corpus

Ajustes por artículo en data/overrides.json. La clave puede ser el nombre exacto
del archivo o su prefijo numérico ("7."). Campos admitidos:
    title, authors, abstract          -> reemplazan lo detectado
    abstract_end_phrase               -> frase con la que empieza el texto que sigue al abstract
    abstract_start_trim               -> último fragmento previo al abstract (p. ej. la última keyword)
"""
import json
import re
import sys
from pathlib import Path

from src.ingestion.pdf_reader import (
    find_abstract,
    guess_authors,
    guess_title,
    normalize_text,
    read_first_pages,
)

PDF_DIR = Path("data/pdfs")
OUT_FILE = Path("data/processed/corpus.json")
OVERRIDES_FILE = Path("data/overrides.json")
EXPECTED_PDFS = 20


def _natural_key(name: str) -> list:
    """Orden natural: '2. a.pdf' va antes que '10. b.pdf'."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name.lower())]


def list_pdfs(pdf_dir: Path = PDF_DIR) -> list[Path]:
    """PDFs de la carpeta (cualquier combinación de mayúsculas en la extensión), en orden natural."""
    if not pdf_dir.is_dir():
        return []
    pdfs = [p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(pdfs, key=lambda p: _natural_key(p.name))


def _override_for(overrides: dict, pdf: Path) -> dict:
    prefix = pdf.name.split(" ")[0]
    return {**overrides.get(prefix, {}), **overrides.get(pdf.name, {})}


def build_corpus(pdf_dir: Path = PDF_DIR, overrides_file: Path = OVERRIDES_FILE) -> list[dict]:
    overrides = {}
    if overrides_file.exists():
        overrides = json.loads(overrides_file.read_text(encoding="utf-8"))

    corpus = []
    for idx, pdf in enumerate(list_pdfs(pdf_dir), start=1):
        review = []
        ov = _override_for(overrides, pdf)
        try:
            text, meta_title = read_first_pages(str(pdf))
        except Exception as exc:  # PDF dañado o ilegible
            text, meta_title = "", None
            review.append(f"no se pudo leer el PDF ({exc})")

        text = normalize_text(text)
        abstract, abs_pos = find_abstract(
            text, ov.get("abstract_end_phrase"), ov.get("abstract_start_trim")
        )
        title, title_src = guess_title(text, meta_title)
        authors = guess_authors(text[:abs_pos] if abs_pos > 0 else text[:800], title)

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

    n_pdfs = len(list_pdfs())
    if n_pdfs == 0:
        print(f"No hay PDFs en '{PDF_DIR}'. Copia allí los artículos del corpus y vuelve a ejecutar.")
        sys.exit(1)
    if n_pdfs != EXPECTED_PDFS:
        print(f"AVISO: se esperaban {EXPECTED_PDFS} PDFs y hay {n_pdfs} en '{PDF_DIR}'.\n")

    corpus = build_corpus()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(corpus)} PDFs procesados -> {OUT_FILE}\n")
    for doc in corpus:
        estado = "OK " if not doc["needs_review"] else "REV"
        ab = doc["abstract"]
        print(f"[{estado}] {doc['id']}  {doc['abstract_words']:>4} palabras  {doc['file']}")
        print(f"        título : {doc['title'][:90]}")
        print(f"        autores: {doc['authors'][:90]}")
        if ab:
            print(f"        inicio : {ab[:70]}")
            print(f"        final  : ...{ab[-70:]}")
        for motivo in doc["needs_review"]:
            print(f"        ! {motivo}")


if __name__ == "__main__":
    main()