"""Agrupamiento jerárquico aglomerativo (desde cero).

Implementa tres criterios de enlace:
- Single Linkage: distancia mínima entre pares de elementos
- Complete Linkage: distancia máxima entre pares de elementos
- Average Linkage: promedio ponderado por tamaño de cluster (UPGMA)

Regla de desempate: menor distancia; ante empate, menores ids (id_a, id_b).
Matriz de enlaces (linkage matrix): n-1 filas [id_a, id_b, altura, tamaño].
Hojas: 0..n-1 (correspondientes a A01..A20). Clusters nuevos: n, n+1, ...

Métricas internas:
- Coeficiente de correlación cofenética
- Silueta para k=2..6 (cortando el árbol deshaciendo últimas k-1 fusiones; singletons -> silueta 0)
"""
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

LinkageMethod = Literal["single", "complete", "average"]


@dataclass
class LinkageMatrix:
    """Matriz de enlaces estilo SciPy: n-1 filas x 4 columnas [id_a, id_b, height, size]."""
    matrix: np.ndarray  # shape (n-1, 4)
    n_leaves: int
    method: str

    def to_list(self) -> list[list[float]]:
        return self.matrix.tolist()


@dataclass
class ClusterResult:
    """Resultado completo de clustering para un método."""
    linkage: LinkageMatrix
    cophenetic_corr: float
    silhouettes: dict[int, float]  # k -> avg silhouette
    best_k: int  # k con mayor silueta promedio (2..6)


def _validate_distance_matrix(D: np.ndarray) -> None:
    """Valida que D es cuadrada, simétrica, diagonal=0, no-negativa."""
    n = D.shape[0]
    assert D.shape == (n, n), "Matriz debe ser cuadrada"
    assert np.allclose(np.diag(D), 0), "Diagonal debe ser 0"
    assert np.allclose(D, D.T), "Matriz debe ser simétrica"
    assert np.all(D >= -1e-10), "Distancias no pueden ser negativas"


def _init_clusters(n: int) -> list[set[int]]:
    """Inicializa n clusters, cada uno con un elemento (hoja)."""
    return [{i} for i in range(n)]


def _cluster_distance(
    D: np.ndarray,
    cluster_a: set[int],
    cluster_b: set[int],
    method: LinkageMethod,
    sizes: dict[int, int] | None = None,
) -> float:
    """
    Calcula distancia entre dos clusters según el método de enlace.
    
    Args:
        D: Matriz de distancias original (n x n)
        cluster_a, cluster_b: Conjuntos de índices de hojas
        method: "single", "complete", "average"
        sizes: Tamaños de clusters (para average ponderado). Si None, usa len(cluster).
    
    Returns:
        Distancia entre clusters.
    """
    # Todas las distancias entre pares (a in A, b in B)
    distances = []
    for a in cluster_a:
        for b in cluster_b:
            distances.append(D[a, b])
    
    distances = np.array(distances)
    
    if method == "single":
        return float(np.min(distances))
    elif method == "complete":
        return float(np.max(distances))
    elif method == "average":
        # UPGMA: promedio ponderado por tamaño
        if sizes is not None:
            size_a = sizes.get(min(cluster_a), len(cluster_a))
            size_b = sizes.get(min(cluster_b), len(cluster_b))
            # En UPGMA real, se usa el tamaño de cada sub-cluster
            # Pero como podemos tener clusters internos, usamos el tamaño total de hojas
            size_a = len(cluster_a)
            size_b = len(cluster_b)
            # Promedio simple (ya que distances incluye todos los pares)
            return float(np.mean(distances))
        return float(np.mean(distances))
    else:
        raise ValueError(f"Método desconocido: {method}")


def _find_closest_clusters(
    D: np.ndarray,
    clusters: list[set[int]],
    active: list[bool],
    method: LinkageMethod,
    sizes: dict[int, int],
) -> tuple[int, int, float]:
    """
    Encuentra el par de clusters activos más cercanos.
    Regla de desempate: menor distancia -> menor id_a -> menor id_b.
    """
    n = len(clusters)
    best_dist = float("inf")
    best_i = -1
    best_j = -1
    
    for i in range(n):
        if not active[i]:
            continue
        for j in range(i + 1, n):
            if not active[j]:
                continue
            dist = _cluster_distance(D, clusters[i], clusters[j], method, sizes)
            if dist < best_dist - 1e-12:
                best_dist = dist
                best_i, best_j = i, j
            elif abs(dist - best_dist) < 1e-12:
                # Desempate: menores ids
                if i < best_i or (i == best_i and j < best_j):
                    best_i, best_j = i, j
    
    return best_i, best_j, best_dist


def agglomerative_clustering(
    D: np.ndarray,
    method: LinkageMethod = "average",
) -> LinkageMatrix:
    """
    Clustering aglomerativo jerárquico desde cero.
    
    Args:
        D: Matriz de distancias n x n (simétrica, diagonal=0)
        method: "single", "complete", "average"
    
    Returns:
        LinkageMatrix con n-1 filas [id_a, id_b, height, size]
    """
    _validate_distance_matrix(D)
    n = D.shape[0]
    
    # Clusters actuales (conjuntos de índices de hojas)
    clusters = _init_clusters(n)
    active = [True] * (2 * n - 1)  # Espacio para hojas + clusters internos
    
    # Tamaños de clusters (número de hojas)
    sizes = {i: 1 for i in range(n)}
    
    # Mapeo: índice de cluster actual -> id en linkage matrix
    # Inicialmente, hojas 0..n-1 tienen id 0..n-1
    cluster_to_id = {i: i for i in range(n)}
    
    # Matriz de enlaces: n-1 filas
    linkage_rows = []
    next_id = n  # Primer cluster interno
    
    for step in range(n - 1):
        # Encontrar par más cercano
        i, j, height = _find_closest_clusters(D, clusters, active, method, sizes)
        
        # IDs en la linkage matrix
        id_a = cluster_to_id[i]
        id_b = cluster_to_id[j]
        size = sizes[i] + sizes[j]
        
        linkage_rows.append([float(id_a), float(id_b), height, float(size)])
        
        # Fusionar clusters
        new_cluster = clusters[i] | clusters[j]
        clusters.append(new_cluster)
        active.append(True)
        sizes[next_id] = size
        cluster_to_id[len(clusters) - 1] = next_id
        
        # Desactivar clusters fusionados
        active[i] = False
        active[j] = False
        
        next_id += 1
    
    linkage_matrix = np.array(linkage_rows, dtype=float)
    return LinkageMatrix(linkage_matrix, n, method)


def _build_tree_from_linkage(linkage: LinkageMatrix) -> dict:
    """
    Construye árbol de clusters a partir de linkage matrix.
    Retorna diccionario: cluster_id -> {left, right, height, size, leaves}
    """
    n = linkage.n_leaves
    tree = {}
    
    # Hojas
    for i in range(n):
        tree[i] = {
            "left": None,
            "right": None,
            "height": 0.0,
            "size": 1,
            "leaves": {i},
        }
    
    # Clusters internos
    for row_idx, row in enumerate(linkage.matrix):
        id_a, id_b, height, size = row
        id_a, id_b = int(id_a), int(id_b)
        new_id = n + row_idx
        
        leaves_a = tree[id_a]["leaves"]
        leaves_b = tree[id_b]["leaves"]
        
        tree[new_id] = {
            "left": id_a,
            "right": id_b,
            "height": height,
            "size": int(size),
            "leaves": leaves_a | leaves_b,
        }
    
    return tree


def _get_clusters_at_k(tree: dict, n_leaves: int, k: int, linkage: LinkageMatrix) -> list[set[int]]:
    """
    Obtiene k clusters cortando el árbol (deshaciendo últimas k-1 fusiones).
    Retorna lista de conjuntos de hojas.
    """
    if k <= 1:
        root_id = n_leaves + len(tree) - n_leaves - 1
        return [tree[root_id]["leaves"]]

    if k >= n_leaves:
        return [{i} for i in range(n_leaves)]

    # El árbol tiene n_leaves - 1 fusiones
    # Para k clusters, necesitamos deshacer k-1 fusiones
    # Es decir, tomar el estado después de (n_leaves - k) fusiones
    n_merges = n_leaves - k

    # Clusters activos después de n_merges fusiones
    active_clusters = list(range(n_leaves))

    for row_idx in range(n_merges):
        row = linkage.matrix[row_idx]
        id_a, id_b = int(row[0]), int(row[1])
        new_id = n_leaves + row_idx
        
        # Remover clusters fusionados, agregar nuevo
        active_clusters = [c for c in active_clusters if c not in (id_a, id_b)]
        active_clusters.append(new_id)
    
    return [tree[c]["leaves"] for c in active_clusters]


def cophenetic_correlation(D: np.ndarray, linkage: LinkageMatrix) -> float:
    """
    Coeficiente de correlación cofenética.
    Correlación entre distancias originales y distancias cofenéticas (altura de fusión).
    """
    n = linkage.n_leaves
    tree = _build_tree_from_linkage(linkage)
    
    # Matriz de distancias cofenéticas
    D_coph = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            # Encontrar LCA (lowest common ancestor) de i y j
            # La altura de su fusión es la distancia cofenética
            # Buscar el primer cluster interno que contiene a ambos
            for cluster_id in range(n, n + len(linkage.matrix)):
                if i in tree[cluster_id]["leaves"] and j in tree[cluster_id]["leaves"]:
                    D_coph[i, j] = D_coph[j, i] = tree[cluster_id]["height"]
                    break
    
    # Correlación de Pearson entre partes superiores triangulares
    mask = np.triu_indices(n, k=1)
    x = D[mask]
    y = D_coph[mask]
    
    if np.std(x) == 0 or np.std(y) == 0:
        return 1.0 if np.allclose(x, y) else 0.0
    
    return float(np.corrcoef(x, y)[0, 1])


def _silhouette_for_clustering(D: np.ndarray, clusters: list[set[int]]) -> float:
    """
    Calcula silueta promedio para una partición en clusters.
    Singletons aportan 0.
    """
    n = D.shape[0]
    if len(clusters) <= 1:
        return 0.0
    
    # Mapear cada punto a su cluster
    point_to_cluster = {}
    for idx, cluster in enumerate(clusters):
        for point in cluster:
            point_to_cluster[point] = idx
    
    silhouettes = []
    
    for i in range(n):
        ci = point_to_cluster[i]
        cluster_i = clusters[ci]
        
        if len(cluster_i) == 1:
            # Singleton -> silueta 0
            silhouettes.append(0.0)
            continue
        
        # a(i): distancia promedio a otros puntos en mismo cluster
        a_i = np.mean([D[i, j] for j in cluster_i if j != i])
        
        # b(i): menor distancia promedio a puntos en otro cluster
        b_i = float("inf")
        for cj, cluster_j in enumerate(clusters):
            if cj == ci:
                continue
            if len(cluster_j) == 0:
                continue
            avg_dist = np.mean([D[i, j] for j in cluster_j])
            b_i = min(b_i, avg_dist)
        
        if b_i == float("inf"):
            s_i = 0.0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)
        
        silhouettes.append(s_i)
    
    return float(np.mean(silhouettes))


def silhouette_scores(
    D: np.ndarray,
    linkage: LinkageMatrix,
    k_range: range = range(2, 7),
) -> dict[int, float]:
    """
    Silueta promedio para k = 2..6.
    Corta el árbol deshaciendo últimas k-1 fusiones.
    """
    n = linkage.n_leaves
    tree = _build_tree_from_linkage(linkage)
    linkage_matrix = linkage.matrix
    
    results = {}
    
    for k in k_range:
        if k > n:
            results[k] = 0.0
            continue
        
        # Obtener clusters para este k
        clusters = _get_clusters_at_k(tree, n, k, linkage)
        results[k] = _silhouette_for_clustering(D, clusters)
    
    return results


def evaluate_clustering(D: np.ndarray, linkage: LinkageMatrix) -> ClusterResult:
    """
    Evaluación completa: cofenética + siluetas k=2..6 + mejor k.
    """
    coph = cophenetic_correlation(D, linkage)
    sil_scores = silhouette_scores(D, linkage)
    
    # Mejor k = mayor silueta promedio
    best_k = max(sil_scores, key=sil_scores.get) if sil_scores else 2
    
    return ClusterResult(
        linkage=linkage,
        cophenetic_corr=coph,
        silhouettes=sil_scores,
        best_k=best_k,
    )


def get_clusters_at_k(linkage: LinkageMatrix, k: int) -> list[list[int]]:
    """
    Retorna los clusters (como listas de índices de hojas 0..n-1) para un k dado.
    """
    n = linkage.n_leaves
    tree = _build_tree_from_linkage(linkage)
    clusters = _get_clusters_at_k(tree, n, k, linkage)
    return [sorted(list(c)) for c in clusters]


def _demo():
    """Demo con matriz de distancias aleatoria."""
    np.random.seed(42)
    n = 10
    # Generar matriz de distancias válida
    X = np.random.rand(n, 5)
    D = 1 - np.corrcoef(X)
    D = np.clip(D, 0, 1)
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    
    for method in ["single", "complete", "average"]:
        print(f"\n=== {method.upper()} LINKAGE ===")
        linkage = agglomerative_clustering(D, method)
        print(f"Linkage matrix shape: {linkage.matrix.shape}")
        print(f"First 3 rows:\n{linkage.matrix[:3]}")
        
        result = evaluate_clustering(D, linkage)
        print(f"Cophenetic corr: {result.cophenetic_corr:.4f}")
        print(f"Silhouettes: {result.silhouettes}")
        print(f"Best k: {result.best_k}")
        
        for k in [2, 3]:
            clusters = get_clusters_at_k(linkage, k)
            print(f"  k={k}: {clusters}")


if __name__ == "__main__":
    _demo()