"""Muestra el antes y el después del preprocesamiento (insumo para el documento).

Uso: python -m experiments.preview_preprocessing
"""
import sys

from src.preprocessing.text_preprocessing import load_corpus, preprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

corpus = load_corpus()
print(f"{'id':<5}{'palabras':>9}{'tokens':>8}{'vocab':>7}")
for doc in corpus:
    toks = preprocess(doc["abstract"])
    print(f"{doc['id']:<5}{doc['abstract_words']:>9}{len(toks):>8}{len(set(toks)):>7}")

first = corpus[0]
print("\nANTES  :", first["abstract"][:300])
print("DESPUÉS:", " ".join(preprocess(first["abstract"])[:40]))