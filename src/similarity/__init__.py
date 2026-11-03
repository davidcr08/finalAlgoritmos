"""Exports públicos del módulo similarity."""
from src.similarity.edit_distance import levenshtein_words, LevenshteinTrace
from src.similarity.embeddings import average_vector, embed_corpus
from src.similarity.jaccard import jaccard_tokens, JaccardTrace
from src.similarity.metrics import cosine_similarity, dot, euclidean_distance, norm
from src.similarity.needleman_wunsch import needleman_wunsch_words, NeedlemanWunschTrace
from src.similarity.tfidf import (
    TFIDFTrace,
    compute_idf,
    load_idf,
    save_idf,
    tfidf_cosine,
    tfidf_vector,
    tf_vector,
)

__all__ = [
    # Métricas base
    "dot",
    "norm",
    "cosine_similarity",
    "euclidean_distance",
    # Embeddings (infra)
    "average_vector",
    "embed_corpus",
    # Clásicos R1
    "levenshtein_words",
    "LevenshteinTrace",
    "needleman_wunsch_words",
    "NeedlemanWunschTrace",
    "tfidf_cosine",
    "TFIDFTrace",
    "tf_vector",
    "tfidf_vector",
    "compute_idf",
    "save_idf",
    "load_idf",
    "jaccard_tokens",
    "JaccardTrace",
]