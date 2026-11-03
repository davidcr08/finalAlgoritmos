"""Tests de referencia para clustering jerárquico contra SciPy.

Estos tests SÍ usan SciPy (prohibido en src/, permitido en tests/).
Validan que nuestros 3 enlaces producen resultados equivalentes a scipy.cluster.hierarchy.
"""
import pytest
import numpy as np

try:
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.cluster.hierarchy import cophenet as scipy_cophenet
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.clustering.hierarchical import (
    agglomerative_clustering,
    cophenetic_correlation,
    LinkageMatrix,
    get_clusters_at_k,
)


pytestmark = pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy no instalado")


def _make_test_distance_matrix(n: int = 10, seed: int = 42) -> np.ndarray:
    """Genera matriz de distancias válida para tests."""
    np.random.seed(seed)
    X = np.random.rand(n, 5)
    # Usar distancia euclidiana y convertir a similitud coseno-like
    from scipy.spatial.distance import pdist, squareform
    D_condensed = pdist(X, metric="cosine")
    D = squareform(D_condensed)
    return D


class TestLinkageVsScipy:
    """Compara nuestros 3 enlaces contra scipy.cluster.hierarchy.linkage."""

    @pytest.fixture
    def distance_matrix(self):
        return _make_test_distance_matrix(12, seed=123)

    def _compare_linkage_matrices(self, our: LinkageMatrix, scipy_mat: np.ndarray, tol: float = 1e-6):
        """Compara matrices de enlace (pueden diferir en orden de fusiones por desempates)."""
        # Las matrices deben tener misma forma
        assert our.matrix.shape == scipy_mat.shape, f"Shape mismatch: {our.matrix.shape} vs {scipy_mat.shape}"
        
        # Las alturas (columna 2) deben ser muy similares (mismas distancias de fusión)
        our_heights = np.sort(our.matrix[:, 2])
        scipy_heights = np.sort(scipy_mat[:, 2])
        assert np.allclose(our_heights, scipy_heights, atol=tol), \
            f"Alturas difieren: {our_heights} vs {scipy_heights}"

    def test_single_linkage_vs_scipy(self, distance_matrix):
        """Single linkage debe coincidir con scipy method='single'."""
        our = agglomerative_clustering(distance_matrix, "single")
        scipy_mat = scipy_linkage(squareform(distance_matrix), method="single")
        self._compare_linkage_matrices(our, scipy_mat)

    def test_complete_linkage_vs_scipy(self, distance_matrix):
        """Complete linkage debe coincidir con scipy method='complete'."""
        our = agglomerative_clustering(distance_matrix, "complete")
        scipy_mat = scipy_linkage(squareform(distance_matrix), method="complete")
        self._compare_linkage_matrices(our, scipy_mat)

    def test_average_linkage_vs_scipy(self, distance_matrix):
        """Average linkage (UPGMA) debe coincidir con scipy method='average'."""
        our = agglomerative_clustering(distance_matrix, "average")
        scipy_mat = scipy_linkage(squareform(distance_matrix), method="average")
        self._compare_linkage_matrices(our, scipy_mat)


class TestCopheneticCorrelation:
    """Valida coeficiente de correlación cofenética contra SciPy."""

    def test_cophenetic_vs_scipy(self):
        np.random.seed(42)
        n = 10
        X = np.random.rand(n, 4)
        from scipy.spatial.distance import pdist, squareform
        D_condensed = pdist(X, metric="euclidean")
        D = squareform(D_condensed)
        
        for method in ["single", "complete", "average"]:
            our = agglomerative_clustering(D, method)
            our_coph = cophenetic_correlation(D, our)
            
            # SciPy
            scipy_mat = scipy_linkage(D_condensed, method=method)
            scipy_coph, _ = scipy_cophenet(scipy_mat, D_condensed)
            
            # Deben ser muy cercanos (diferencias por desempates)
            assert abs(our_coph - scipy_coph) < 0.02, \
                f"{method}: our={our_coph:.4f}, scipy={scipy_coph:.4f}"


class TestClusteringProperties:
    """Pruebas de propiedades básicas del clustering."""

    def test_linkage_matrix_shape(self):
        """Matriz de enlace debe tener n-1 filas y 4 columnas."""
        for n in [5, 10, 20]:
            D = _make_test_distance_matrix(n)
            for method in ["single", "complete", "average"]:
                linkage = agglomerative_clustering(D, method)
                assert linkage.matrix.shape == (n - 1, 4)
                assert linkage.n_leaves == n

    def test_linkage_monotonic_heights(self):
        """Las alturas en la matriz de enlace deben ser monótonas no decrecientes."""
        D = _make_test_distance_matrix(15)
        for method in ["single", "complete", "average"]:
            linkage = agglomerative_clustering(D, method)
            heights = linkage.matrix[:, 2]
            # En clustering aglomerativo, las alturas de fusión son no decrecientes
            assert np.all(np.diff(heights) >= -1e-10), f"{method}: alturas no monótonas"

    def test_diagonal_zero_symmetry(self):
        """Matriz de distancia de entrada: diagonal=0, simétrica."""
        D = _make_test_distance_matrix(8)
        assert np.allclose(np.diag(D), 0)
        assert np.allclose(D, D.T)

    def test_cluster_sizes_sum_to_n(self):
        """Tamaños en linkage (columna 3) deben sumar correctamente."""
        D = _make_test_distance_matrix(10)
        for method in ["single", "complete", "average"]:
            linkage = agglomerative_clustering(D, method)
            sizes = linkage.matrix[:, 3]
            # El último cluster debe tener tamaño n
            assert sizes[-1] == 10
            # Los tamaños deben ser enteros positivos
            assert np.all(sizes >= 1)
            assert np.allclose(sizes, np.round(sizes))

    def test_get_clusters_at_k(self):
        """get_clusters_at_k debe retornar k clusters que particionan el conjunto."""
        D = _make_test_distance_matrix(12)
        for method in ["single", "complete", "average"]:
            linkage = agglomerative_clustering(D, method)
            for k in [2, 3, 4, 6, 10]:
                clusters = get_clusters_at_k(linkage, k)
                assert len(clusters) == k
                # Todos los elementos 0..n-1 deben aparecer exactamente una vez
                all_elements = []
                for c in clusters:
                    all_elements.extend(c)
                assert sorted(all_elements) == list(range(12))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])