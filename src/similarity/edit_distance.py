"""Levenshtein distance (word-level) con programación dinámica.

Definición (sobre secuencias de tokens/palabras):
    d[i][j] = distancia entre tokens_a[:i] y tokens_b[:j]
    d[0][j] = j, d[i][0] = i
    d[i][j] = min(
        d[i-1][j] + 1,      # eliminación
        d[i][j-1] + 1,      # inserción
        d[i-1][j-1] + cost  # sustitución (cost=0 si igual, 1 si distinto)
    )

Similitud normalizada: sim = 1 - d / max(len_a, len_b)
Dos secuencias vacías -> sim = 1.0

Complejidad: O(n*m) tiempo y espacio (n, m = longitudes en palabras).
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class LevenshteinTrace:
    """Estructuras intermedias para caso de estudio."""
    matrix: list[list[int]]
    path: list[tuple[int, int, Literal["del", "ins", "sub", "match"]]]
    distance: int
    similarity: float
    len_a: int
    len_b: int


def levenshtein_words(
    tokens_a: list[str],
    tokens_b: list[str],
    trace: bool = False,
) -> float | LevenshteinTrace:
    """
    Distancia de Levenshtein a nivel de palabras (tokens).

    Args:
        tokens_a: Lista de tokens del abstract A (preprocesados).
        tokens_b: Lista de tokens del abstract B (preprocesados).
        trace: Si True, devuelve LevenshteinTrace con matriz y camino óptimo.

    Returns:
        float: similitud en [0, 1] si trace=False.
        LevenshteinTrace: si trace=True.
    """
    n, m = len(tokens_a), len(tokens_b)

    # Casos base: secuencias vacías
    if n == 0 and m == 0:
        if trace:
            return LevenshteinTrace(
                matrix=[[0]],
                path=[],
                distance=0,
                similarity=1.0,
                len_a=0,
                len_b=0,
            )
        return 1.0
    if n == 0 or m == 0:
        dist = max(n, m)
        if trace:
            # Matriz trivial
            matrix = [[i + j for j in range(m + 1)] for i in range(n + 1)]
            path = []
            if n == 0:
                path = [(0, j, "ins") for j in range(1, m + 1)]
            else:
                path = [(i, 0, "del") for i in range(1, n + 1)]
            return LevenshteinTrace(
                matrix=matrix,
                path=path,
                distance=dist,
                similarity=0.0,
                len_a=n,
                len_b=m,
            )
        return 0.0

    # Matriz DP (n+1) x (m+1)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Llenado
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if tokens_a[i - 1] == tokens_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # eliminación
                dp[i][j - 1] + 1,      # inserción
                dp[i - 1][j - 1] + cost,  # sustitución / match
            )

    distance = dp[n][m]
    similarity = 1.0 - distance / max(n, m)

    if not trace:
        return similarity

    # Reconstrucción del camino óptimo (backtracking)
    # Prioridad: diagonal (match/sub) > arriba (del) > izquierda (ins)
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if tokens_a[i - 1] == tokens_b[j - 1] else 1):
            op = "match" if tokens_a[i - 1] == tokens_b[j - 1] else "sub"
            path.append((i, j, op))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            path.append((i, j, "del"))
            i -= 1
        else:
            path.append((i, j, "ins"))
            j -= 1
    path.reverse()

    return LevenshteinTrace(
        matrix=dp,
        path=path,
        distance=distance,
        similarity=similarity,
        len_a=n,
        len_b=m,
    )


def _demo():
    """Demo rápido para verificación manual."""
    a = ["algoritmo", "genetico", "optimizacion", "funcion", "objetivo"]
    b = ["algoritmo", "evolutivo", "optimizacion", "multiobjetivo"]
    print("A:", a)
    print("B:", b)
    result = levenshtein_words(a, b, trace=True)
    print(f"Distancia: {result.distance}")
    print(f"Similitud: {result.similarity:.4f}")
    print("Matriz:")
    for row in result.matrix:
        print("  ", row)
    print("Camino:", result.path)


if __name__ == "__main__":
    _demo()