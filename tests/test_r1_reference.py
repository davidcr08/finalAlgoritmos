"""Tests de referencia para validar implementaciones R1 contra SciPy/sklearn.

Estos tests SÍ usan librerías prohibidas en src/ (scipy, sklearn).
Solo se ejecutan en tests/ y requieren requirements-dev.txt.
"""
import pytest
import numpy as np

# Estos imports están permitidos SOLO en tests/
# Manejo independiente de cada dependencia externa
try:
    from scipy.spatial.distance import cosine as scipy_cosine
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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


# Sin pytestmark global: cada test decide su dependencia


# ============================================================================
# TESTS PROPIOS (sin dependencias externas)
# ============================================================================

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


# ============================================================================
# TESTS CONTRA DEPENDENCIAS EXTERNAS
# ============================================================================

class TestMetricsReference:
    """Valida métricas base (coseno, euclidiana) contra SciPy/NumPy."""

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="Requiere SciPy")
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


class TestTFIDFCosineReference:
    """Valida TF-IDF + Coseno contra sklearn con MISMO vocabulario, IDF y normalización."""

    @pytest.fixture(scope="class")
    def corpus_data(self):
        corpus = load_corpus()
        tokens_by_id = preprocess_corpus(corpus)
        idf = compute_idf(tokens_by_id)
        return tokens_by_id, idf

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Requiere scikit-learn")
    def test_tfidf_cosine_vs_sklearn(self, corpus_data):
        """Compara vectores TF-IDF propios vs sklearn con MISMO vocab e IDF."""
        tokens_by_id, idf = corpus_data
        ids = list(tokens_by_id.keys())[:5]  # primeros 5 docs

        # Nuestros vectores TF-IDF
        from src.similarity.tfidf import tfidf_vector
        our_vectors = {}
        for doc_id in ids:
            tf = tf_vector(tokens_by_id[doc_id])
            our_vectors[doc_id] = tfidf_vector(tf, idf)

        # Vocabulario común (ordenado para consistencia)
        vocab = sorted(set().union(*[set(v.keys()) for v in our_vectors.values()]))
        our_matrix = np.array([[v.get(t, 0.0) for t in vocab] for v in our_vectors.values()])

        # sklearn TF-IDF con MISMO vocabulario y MISMO IDF
        # Nuestra fórmula: idf = ln((1+N)/(1+df)) + 1 con N=20 (corpus completo)
        # sklearn con smooth_idf=True: idf = log((n_docs + 1) / (df + 1)) + 1
        # Son IDÉNTICAS cuando n_docs = N (corpus completo)
        # Para que coincida exactamente, forzamos el IDF de sklearn a ser el nuestro
        docs_text = [" ".join(tokens_by_id[doc_id]) for doc_id in ids]
        vectorizer = TfidfVectorizer(
            vocabulary={t: i for i, t in enumerate(vocab)},
            use_idf=True,
            smooth_idf=True,   # Coincide con nuestra fórmula: ln((1+N)/(1+df)) + 1
            norm='l2',         # Normalización L2 (coseno)
            sublinear_tf=False, # TF bruto (conteo)
            tokenizer=str.split,  # Respetar tokens tal como vienen
            token_pattern=None,   # No filtrar por regex
            lowercase=False       # No convertir a minúsculas
        )
        vectorizer.fit(docs_text)
        # CRÍTICO: Sobrescribir idf_ de sklearn con nuestro IDF precomputado (N=20)
        vectorizer.idf_ = np.array([idf.get(t, 1.0) for t in vocab])
        sklearn_matrix = vectorizer.transform(docs_text).toarray()

        # Comparar cosenos pairwise (lo que realmente importa)
        # Nota: our_matrix tiene pesos TF-IDF crudos, sklearn_matrix tiene vectores L2-normalizados
        # El coseno debe coincidir porque ambas normalizan internamente
        for i in range(len(ids)):
            for j in range(i, len(ids)):
                # Coseno entre nuestros vectores
                our_cos = cosine_similarity(our_matrix[i], our_matrix[j])
                # Coseno entre vectores de sklearn
                sk_cos = sklearn_cosine([sklearn_matrix[i]], [sklearn_matrix[j]])[0, 0]
                assert abs(our_cos - sk_cos) < 1e-5,                     f"Mismatch {ids[i]}-{ids[j]}: our={our_cos:.6f} vs sklearn={sk_cos:.6f}"

        # Comparar matrices normalizadas L2 (pesos equivalentes tras normalización)
        # Normalizar filas de our_matrix (L2)
        our_norm = np.linalg.norm(our_matrix, axis=1, keepdims=True)
        our_matrix_norm = np.divide(our_matrix, our_norm, out=np.zeros_like(our_matrix), where=our_norm != 0)
        # sklearn_matrix ya está normalizada L2 por fila (norm='l2')
        # Verificar dimensiones
        assert our_matrix_norm.shape == sklearn_matrix.shape, "Matrices deben tener mismas dimensiones"
        # Comparar usando tolerancia numérica justificada
        assert np.allclose(our_matrix_norm, sklearn_matrix, atol=1e-5),             f"Matrices TF-IDF normalizadas difieren: max_diff={np.max(np.abs(our_matrix_norm - sklearn_matrix)):.6f}"

    def test_tfidf_cosine_self(self, corpus_data):
        tokens_by_id, idf = corpus_data
        tokens = tokens_by_id["A01"]
        sim = tfidf_cosine(tokens, tokens, idf)
        assert abs(sim - 1.0) < 1e-6

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Requiere scikit-learn")
    def test_tfidf_sklearn_single_char_tokens(self):
        """Verifica que sklearn conserva tokens de un solo carácter usando corpus controlado."""
        # Corpus pequeño controlado con tokens de 1 carácter
        corpus_controlado = {
            "d1": ["a", "i"],
            "d2": ["a"],
        }
        idf_controlado = compute_idf(corpus_controlado, expected_size=None)
        # Verificar que los tokens están en el IDF
        assert "a" in idf_controlado
        assert "i" in idf_controlado

        docs_text = ["a i", "a"]
        vocab = ["a", "i"]
        vectorizer = TfidfVectorizer(
            vocabulary={t: i for i, t in enumerate(vocab)},
            use_idf=True,
            smooth_idf=True,
            norm='l2',
            sublinear_tf=False,
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False
        )
        vectorizer.fit(docs_text)
        vectorizer.idf_ = np.array([idf_controlado.get(t, 1.0) for t in vocab])
        sklearn_matrix = vectorizer.transform(docs_text).toarray()
        # Verificar que ambos tokens aparecen (columna > 0)
        assert sklearn_matrix[0, 0] > 0  # "a" en doc 0
        assert sklearn_matrix[0, 1] > 0  # "i" en doc 0
        assert sklearn_matrix[1, 0] > 0  # "a" en doc 1

    def test_tfidf_own_single_char_tokens(self):
        """Verifica que nuestra implementación conserva tokens de un solo carácter con IDF real."""
        # Corpus pequeño controlado con tokens de 1 carácter
        corpus_controlado = {
            "d1": ["a", "i"],
            "d2": ["a"],
        }
        idf_controlado = compute_idf(corpus_controlado, expected_size=None)
        # Verificar que los tokens están en el IDF
        assert "a" in idf_controlado
        assert "i" in idf_controlado

        a = ["a", "i"]
        b = ["a"]
        # Usar IDF calculado del corpus controlado
        sim = tfidf_cosine(a, b, idf_controlado)
        # "a" en común, "i" solo en a -> similitud > 0 y < 1
        assert 0.0 < sim < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])