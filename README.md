# Análisis de algoritmos en el contexto de la bibliometría

Proyecto final — Análisis de Algoritmos, Universidad del Quindío.

## Descripción
[Pendiente: resumen del proyecto]

## Requerimientos implementados
1. Similitud textual (algoritmos clásicos y con embeddings)
2. Agrupamiento jerárquico y dendrogramas
3. Búsqueda de patrones (KMP, Boyer-Moore, Rabin-Karp)
4. Despliegue y documentación

## Estructura
- `src/`: código fuente por módulo
- `tests/`: pruebas
- `experiments/`: scripts que generan tablas y gráficos
- `data/`: corpus y datos procesados
- `docs/`: documentación

## Instalación
Write-File "src\string_matching\kmp.py" @'
"""Algoritmo Knuth-Morris-Pratt (KMP).

Preprocesamiento: tabla LPS (longest proper prefix which is also suffix), O(m).
Búsqueda: O(n), sin retroceder nunca en el texto.
"""
from dataclasses import dataclass, field


@dataclass
class KMPResult:
    positions: list[int]          # índices (base 0) donde empieza cada ocurrencia
    comparisons: int              # comparaciones texto-patrón durante la búsqueda
    lps: list[int]                # tabla de prefijos calculada
    steps: list[tuple] = field(default_factory=list)  # traza: (i, j, evento)


def build_lps(pattern: str) -> list[int]:
    """lps[i] = longitud del prefijo propio más largo de pattern[0..i]
    que también es sufijo de pattern[0..i]."""
    m = len(pattern)
    lps = [0] * m
    k = 0
    for i in range(1, m):
        while k > 0 and pattern[i] != pattern[k]:
            k = lps[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        lps[i] = k
    return lps


def kmp_search(text: str, pattern: str, trace: bool = False) -> KMPResult:
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return KMPResult([], 0, build_lps(pattern) if m else [])

    lps = build_lps(pattern)
    positions: list[int] = []
    steps: list[tuple] = []
    comparisons = 0
    i = 0  # posición en el texto
    j = 0  # posición en el patrón

    while i < n:
        comparisons += 1
        if text[i] == pattern[j]:
            if trace:
                steps.append((i, j, "coincide"))
            i += 1
            j += 1
            if j == m:
                positions.append(i - m)
                if trace:
                    steps.append((i - 1, j - 1, "ocurrencia"))
                j = lps[j - 1]          # permite ocurrencias solapadas
        else:
            if trace:
                steps.append((i, j, "falla"))
            if j > 0:
                j = lps[j - 1]          # reutiliza el prefijo ya coincidente
            else:
                i += 1

    return KMPResult(positions, comparisons, lps, steps)