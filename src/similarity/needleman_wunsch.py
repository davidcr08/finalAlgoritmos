"""Needleman-Wunsch (alineamiento global) a nivel de palabras.

Definición (sobre secuencias de tokens/palabras):
    S[i][j] = score óptimo alineando tokens_a[:i] y tokens_b[:j]
    S[0][j] = j * gap_penalty
    S[i][0] = i * gap_penalty
    S[i][j] = max(
        S[i-1][j-1] + (match_score si tokens_a[i-1]==tokens_b[j-1] else mismatch_score),
        S[i-1][j] + gap_penalty,
        S[i][j-1] + gap_penalty
    )

Parámetros fijos (según especificación):
    match_score = +1
    mismatch_score = -1
    gap_penalty = -1

Similitud normalizada:
    sim = (S[n][m] + max(n, m)) / (2 * min(n, m))   si min(n,m) > 0
    vacía contra no vacía -> 0
    ambas vacías -> 1

Retroceso (backtracking) con prioridad:
    1. Diagonal (match/mismatch)
    2. Arriba (gap en B / eliminación en A)
    3. Izquierda (gap en A / inserción en B)

Complejidad: O(n*m) tiempo y espacio.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class NeedlemanWunschTrace:
    """Estructuras intermedias para caso de estudio."""
    matrix: list[list[int]]
    alignment_a: list[str | None]  # tokens o None para gap
    alignment_b: list[str | None]
    score: int
    similarity: float
    len_a: int
    len_b: int


def needleman_wunsch_words(
    tokens_a: list[str],
    tokens_b: list[str],
    trace: bool = False,
) -> float | NeedlemanWunschTrace:
    """
    Alineamiento global Needleman-Wunsch a nivel de palabras.

    Args:
        tokens_a: Lista de tokens del abstract A (preprocesados).
        tokens_b: Lista de tokens del abstract B (preprocesados).
        trace: Si True, devuelve NeedlemanWunschTrace con matriz y alineamiento.

    Returns:
        float: similitud en [0, 1] si trace=False.
        NeedlemanWunschTrace: si trace=True.
    """
    n, m = len(tokens_a), len(tokens_b)

    # Parámetros fijos
    MATCH = 1
    MISMATCH = -1
    GAP = -1

    # Casos base
    if n == 0 and m == 0:
        if trace:
            return NeedlemanWunschTrace(
                matrix=[[0]],
                alignment_a=[],
                alignment_b=[],
                score=0,
                similarity=1.0,
                len_a=0,
                len_b=0,
            )
        return 1.0
    if n == 0 or m == 0:
        if trace:
            matrix = [[i * GAP + j * GAP for j in range(m + 1)] for i in range(n + 1)]
            if n == 0:
                alignment_a = [None] * m
                alignment_b = tokens_b[:]
            else:
                alignment_a = tokens_a[:]
                alignment_b = [None] * n
            return NeedlemanWunschTrace(
                matrix=matrix,
                alignment_a=alignment_a,
                alignment_b=alignment_b,
                score=-max(n, m),
                similarity=0.0,
                len_a=n,
                len_b=m,
            )
        return 0.0

    # Matriz DP (n+1) x (m+1)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * GAP
    for j in range(1, m + 1):
        dp[0][j] = j * GAP

    # Llenado
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = MATCH if tokens_a[i - 1] == tokens_b[j - 1] else MISMATCH
            dp[i][j] = max(
                dp[i - 1][j - 1] + match,  # diagonal (match/mismatch)
                dp[i - 1][j] + GAP,        # arriba (gap en B)
                dp[i][j - 1] + GAP,        # izquierda (gap en A)
            )

    score = dp[n][m]
    similarity = (score + max(n, m)) / (2 * min(n, m))

    if not trace:
        return similarity

    # Backtracking con prioridad: diagonal > arriba > izquierda
    alignment_a = []
    alignment_b = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match = MATCH if tokens_a[i - 1] == tokens_b[j - 1] else MISMATCH
            if dp[i][j] == dp[i - 1][j - 1] + match:
                alignment_a.append(tokens_a[i - 1])
                alignment_b.append(tokens_b[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + GAP:
            alignment_a.append(tokens_a[i - 1])
            alignment_b.append(None)  # gap en B
            i -= 1
            continue
        # izquierda (gap en A)
        alignment_a.append(None)
        alignment_b.append(tokens_b[j - 1])
        j -= 1

    alignment_a.reverse()
    alignment_b.reverse()

    return NeedlemanWunschTrace(
        matrix=dp,
        alignment_a=alignment_a,
        alignment_b=alignment_b,
        score=score,
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
    result = needleman_wunsch_words(a, b, trace=True)
    print(f"Score: {result.score}")
    print(f"Similitud: {result.similarity:.4f}")
    print("Matriz:")
    for row in result.matrix:
        print("  ", row)
    print("Alineamiento A:", [x if x else "-" for x in result.alignment_a])
    print("Alineamiento B:", [x if x else "-" for x in result.alignment_b])


if __name__ == "__main__":
    _demo()