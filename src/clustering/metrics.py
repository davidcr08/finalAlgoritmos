"""Métricas de evaluación de clustering (desde cero)."""
import numpy as np


def _labels_to_clusters(labels: list[int]) -> list[set[int]]:
    """Convierte lista de etiquetas a lista de conjuntos de índices por cluster."""
    clusters_dict: dict[int, set[int]] = {}
    for idx, label in enumerate(labels):
        clusters_dict.setdefault(label, set()).add(idx)
    # Ordenar clusters por etiqueta para consistencia
    return [clusters_dict[label] for label in sorted(clusters_dict.keys())]


def silhouette_score(dist: list[list[float]], labels: list[int]) -> float:
    """
    Coeficiente de silueta promedio para una partición dada.

    Args:
        dist: Matriz de distancias n x n (simétrica, diagonal=0).
        labels: Lista de longitud n con etiquetas de cluster (enteros >= 0).

    Returns:
        Silueta promedio en [-1, 1].

    Raises:
        ValueError: si hay menos de 2 clusters, o dimensiones incompatibles.
    """
    n = len(dist)
    if n == 0:
        raise ValueError("Matriz de distancias vacía")
    if len(labels) != n:
        raise ValueError(f"labels debe tener longitud {n}, tiene {len(labels)}")
    if any(len(row) != n for row in dist):
        raise ValueError("dist debe ser cuadrada")

    # Validar al menos 2 clusters
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        raise ValueError("Se requieren al menos 2 clusters para calcular silueta")

    D = np.array(dist, dtype=float)
    clusters = _labels_to_clusters(labels)

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
        a_i = float(np.mean([D[i, j] for j in cluster_i if j != i]))

        # b(i): menor distancia promedio a puntos en otro cluster
        b_i = float("inf")
        for cj, cluster_j in enumerate(clusters):
            if cj == ci:
                continue
            if len(cluster_j) == 0:
                continue
            avg_dist = float(np.mean([D[i, j] for j in cluster_j]))
            b_i = min(b_i, avg_dist)

        if b_i == float("inf"):
            s_i = 0.0
        else:
            denom = max(a_i, b_i)
            s_i = (b_i - a_i) / denom if denom != 0 else 0.0

        silhouettes.append(s_i)

    return float(np.mean(silhouettes))