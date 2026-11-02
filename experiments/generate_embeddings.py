"""Genera los embeddings del corpus con dos modelos de vectores de palabras.

Requiere gensim (solo para generar; la aplicación luego solo lee los JSON):
    pip install -r requirements-embeddings.txt

Uso (desde la raíz del repositorio):
    python -m experiments.generate_embeddings              # ambos modelos
    python -m experiments.generate_embeddings glove        # solo uno

La primera ejecución descarga los modelos a la carpeta gensim-data del usuario
(GloVe ~130 MB, FastText ~1 GB aprox.), fuera del repositorio.
"""
import json
import sys
from pathlib import Path

from src.preprocessing.text_preprocessing import load_corpus, preprocess_corpus
from src.similarity.embeddings import embed_corpus

MODELS = {
    "glove": "glove-wiki-gigaword-100",
    "fasttext": "fasttext-wiki-news-subwords-300",
}
OUT_DIR = Path("data/processed")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import gensim.downloader as api  # import tardío: solo se necesita al generar

    keys = sys.argv[1:] or list(MODELS)
    tokens_by_id = preprocess_corpus(load_corpus())

    for key in keys:
        name = MODELS[key]
        print(f"Cargando {name} (puede tardar la primera vez)...")
        kv = api.load(name)
        result = embed_corpus(tokens_by_id, kv, kv.vector_size)
        result["model"] = name

        out = OUT_DIR / f"embeddings_{key}.json"
        out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

        total = sum(o["tokens"] for o in result["oov"].values())
        missing = sum(o["missing"] for o in result["oov"].values())
        print(f"  -> {out}  (dim={kv.vector_size})")
        print(f"  palabras fuera de vocabulario: {missing}/{total} ({100 * missing / max(total, 1):.1f} %)")


if __name__ == "__main__":
    main()