"""Tests de referencia para validar implementaciones R1 contra SciPy/sklearn.

Estos tests SÍ usan librerías prohibidas en src/ (scipy, sklearn).
Solo se ejecutan en tests/ y requieren requirements-dev.txt.
"""
import pytest
import numpy as np

# Estos imports están permitidos SOLO en tests/
try:
    from scipy.spatial.distance import cosine as scipy_cosine
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    from sklearn.feature_extraction.text import TfidfVectorizer
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.similarity import (
    levenshtein_words,
    needleman_wunsch_words,
    tfidf_cosine,
    jaccard_tokens,
    cosine_similarity,
    euclidean_distance,
    compute_idf,
    tf_vector,
)
from src.preprocessing.text_preprocessing import load_corpus, preprocess_corpus


pytestmark = pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy/sklearn no instalados")


class TestLevenshteinReference:
    """Valida Levenshtein contra implementación de referencia (programación dinámica simple)."""

    def test_levenshtein_basic(self):
        a = ["algoritmo", "genetico", "optimizacion"]
        b = ["algoritmo", "evolutivo", "optimizacion"]
        # Distancia esperada: 1 sustitución (genetico -> evolutivo)
        # max(len) = 3, sim = 1 - 1/3 = 0.666...
        sim = levenshtein_words(a, b)
        assert abs(sim - 2/3) < 1e-6

    def test_levenshtein_empty(self):
        assert levenshtein_words([], []) == 1.0
        assert levenshtein_words(["a"], []) == 0.0
        assert levenshtein_words([], ["a"]) == 0.0

    def test_levenshtein_identical(self):
        a = ["hola", "mundo", "test"]
        assert levenshtein_words(a, a) == 1.0

    def test_levenshtein_no_overlap(self):
        a = ["a", "b", "c"]
        b = ["x", "y", "z"]
        # 3 sustituciones, max=3 -> sim = 0
        assert levenshtein_words(a, b) == 0.0

    def test_levenshtein_insertion_deletion(self):
        a = ["a", "b", "c"]
        b = ["a", "b", "c", "d", "e"]
        # 2 inserciones, max=5 -> sim = 1 - 2/5 = 0.6
        sim = levenshtein_words(a, b)
        assert abs(sim - 0.6) < 1e-6


class TestNeedlemanWunschReference:
    """Valida Needleman-Wunsch contra referencia."""

    def test_nw_basic(self):
        a = ["algoritmo", "genetico", "optimizacion"]
        b = ["algoritmo", "evolutivo", "optimizacion"]
        # Match: +1, Mismatch: -1, Gap: -1
        # Alineamiento óptimo: match, mismatch, match = 1 - 1 + 1 = 1
        # sim = (1 + 3) / (2 * 3) = 4/6 = 0.666...
        sim = needleman_wunsch_words(a, b)
        assert abs(sim - 2/3) < 1e-6

    def test_nw_empty(self):
        assert needleman_wunsch_words([], []) == 1.0
        assert needleman_wunsch_words(["a"], []) == 0.0

    def test_nw_identical(self):
        a = ["hola", "mundo"]
        assert needleman_wunsch_words(a, a) == 1.0


class TestTFIDFCosineReference:
    """Valida TF-IDF + Coseno contra sklearn."""

    @pytest.fixture(scope="class")
    def corpus_data(self):
        corpus = load_corpus()
        tokens_by_id = preprocess_corpus(corpus)
        idf = compute_idf(tokens_by_id)
        return tokens_by_id, idf

    def test_tfidf_cosine_vs_sklearn(self, corpus_data):
        tokens_by_id, idf = corpus_data
        ids = list(tokens_by_id.keys())[:5]  # primeros 5 docs

        # Nuestros vectores TF-IDF
        from src.similarity.tfidf import tfidf_vector, tf_vector
        vectors = {}
        for doc_id in ids:
            tf = tf_vector(tokens_by_id[doc_id])
            vectors[doc_id] = tfidf_vector(tf, idf)

        # Vocabulario común
        vocab = sorted(set().union(*[set(v.keys()) for v in vectors.values()]))
        our_matrix = np.array([[v.get(t, 0.0) for t in vocab] for v in vectors.values()])

        # sklearn TF-IDF (mismos docs, mismo corpus para IDF)
        docs_text = [" ".join(tokens_by_id[doc_id]) for doc_id in ids]
        # Usar nuestro IDF fijo en sklearn
        vectorizer = TfidfVectorizer(vocabulary={t: i for i, t in enumerate(vocab)})
        sklearn_matrix = vectorizer.fit_transform(docs_text).toarray()

        # Comparar cosenos pairwise
        for i in range(len(ids)):
            for j in range(i, len(ids)):
                our_cos = cosine_similarity(our_matrix[i], our_matrix[j])
                sk_cos = sklearn_cosine([our_matrix[i]], [our_matrix[j]])[0, 0]
                assert abs(our_cos - sk_cos) < 1e-5, f"Mismatch {ids[i]}-{ids[j]}: {our_cos} vs {sk_cos}"

    def test_tfidf_cosine_self(self, corpus_data):
        tokens_by_id, idf = corpus_data
        tokens = tokens_by_id["A01"]
        sim = tfidf_cosine(tokens, tokens, idf)
        assert abs(sim - 1.0) < 1e-6


class TestJaccardReference:
    """Valida Jaccard contra referencia simple."""

    def test_jaccard_basic(self):
        a = ["a", "b", "c", "d"]
        b = ["c", "d", "e", "f"]
        # Intersección: {c,d}=2, Unión: {a,b,c,d,e,f}=6 -> 2/6 = 1/3
        sim = jaccard_tokens(a, b)
        assert abs(sim - 1/3) < 1e-6

    def test_jaccard_empty(self):
        assert jaccard_tokens([], []) == 1.0
        assert jaccard_tokens(["a"], []) == 0.0

    def test_jaccard_identical(self):
        a = ["x", "y", "z"]
        assert jaccard_tokens(a, a) == 1.0


class TestMetricsReference:
    """Valida métricas base (coseno, euclidiana) contra SciPy/NumPy."""

    def test_cosine_vs_scipy(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 3.0, 4.0, 5.0]
        our = cosine_similarity(a, b)
        sp = 1 - scipy_cosine(a, b)
        assert abs(our - sp) < 1e-10

    def test_cosine_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_euclidean_vs_numpy(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        our = euclidean_distance(a, b)
        np_dist = np.linalg.norm(np.array(a) - np.array(b))
        assert abs(our - np_dist) < 1e-10


class TestIDFComputation:
    """Valida cálculo de IDF contra fórmula esperada."""

    def test_idf_formula(self):
        # Corpus simple: 3 docs
        tokens = {
            "d1": ["a", "b", "c"],
            "d2": ["a", "b", "d"],
            "d3": ["a", "e", "f"],
        }
        idf = compute_idf(tokens, expected_size=None)
        N = 3
        # df(a)=3 -> idf = ln((1+3)/(1+3)) + 1 = ln(1) + 1 = 1
        assert abs(idf["a"] - 1.0) < 1e-6
        # df(b)=2 -> idf = ln(4/3) + 1
        import math
        assert abs(idf["b"] - (math.log(4/3) + 1)) < 1e-6
        # df(c)=1 -> idf = ln(4/2) + 1 = ln(2) + 1
        assert abs(idf["c"] - (math.log(2) + 1)) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])