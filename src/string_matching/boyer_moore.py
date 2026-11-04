"""Algoritmo de Boyer-Moore.

Compara el patrón de derecha a izquierda y, ante un fallo, avanza usando el
mayor desplazamiento entre dos heurísticas:
  - regla del carácter malo (bad character)
  - regla del sufijo bueno (good suffix, versión fuerte)

Preprocesamiento: O(m + |alfabeto|). Búsqueda: puede ser sublineal
en el mejor caso; peor caso O(n*m), sin optimización de Galil.
"""
from dataclasses import dataclass, field


@dataclass
class BMResult:
    positions: list[int]
    comparisons: int
    bad_char: dict[str, int]       # carácter -> última posición en el patrón
    good_suffix: list[int]         # desplazamiento por sufijo bueno, índices 0..m
    steps: list[tuple] = field(default_factory=list)  # (s, j, evento, desplazamiento)


def build_bad_char(pattern: str) -> dict[str, int]:
    """Última posición (base 0) de cada carácter dentro del patrón."""
    return {ch: i for i, ch in enumerate(pattern)}


def build_good_suffix(pattern: str) -> list[int]:
    """Tabla de desplazamientos de la regla del sufijo bueno (versión fuerte).

    shift[j] es cuánto avanzar cuando el fallo ocurre en la posición j-1 del
    patrón (es decir, ya coincidió el sufijo pattern[j:]). shift[0] es el
    desplazamiento a aplicar tras encontrar una ocurrencia completa.
    """
    m = len(pattern)
    shift = [0] * (m + 1)
    border = [0] * (m + 1)   # border[i]: inicio del borde más ancho del sufijo pattern[i:]

    # Caso 1: el sufijo bueno aparece de nuevo en el patrón
    i, j = m, m + 1
    border[i] = j
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = border[j]
        i -= 1
        j -= 1
        border[i] = j

    # Caso 2: solo un prefijo del patrón coincide con parte del sufijo bueno
    j = border[0]
    for i in range(m + 1):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = border[j]

    return shift


def boyer_moore_search(text: str, pattern: str, trace: bool = False, bad_char: dict[str, int] | None = None, good_suffix: list[int] | None = None) -> BMResult:
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return BMResult([], 0, build_bad_char(pattern), build_good_suffix(pattern) if m else [])

    if bad_char is None:
        bad_char = build_bad_char(pattern)
    if good_suffix is None:
        good_suffix = build_good_suffix(pattern)
    positions: list[int] = []
    steps: list[tuple] = []
    comparisons = 0
    s = 0  # desplazamiento del patrón respecto al texto

    while s <= n - m:
        j = m - 1
        while j >= 0:
            comparisons += 1
            if pattern[j] != text[s + j]:
                break
            j -= 1

        if j < 0:
            positions.append(s)
            jump = good_suffix[0]
            if trace:
                steps.append((s, -1, "ocurrencia", jump))
            s += jump
        else:
            bc = j - bad_char.get(text[s + j], -1)
            gs = good_suffix[j + 1]
            jump = max(1, bc, gs)
            if trace:
                steps.append((s, j, "falla", jump))
            s += jump

    return BMResult(positions, comparisons, bad_char, good_suffix, steps)