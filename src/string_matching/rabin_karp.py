"""Algoritmo de Rabin-Karp.

Calcula un hash de la ventana del texto y lo compara con el hash del patrón.
El hash se actualiza en O(1) al deslizar la ventana (rolling hash):

    h(i+1) = ((h(i) - ord(texto[i]) * B^(m-1)) * B + ord(texto[i+m])) mod Q

Si los hashes coinciden se verifica carácter a carácter, porque dos cadenas
distintas pueden tener el mismo hash (colisión o "falso positivo").

Preprocesamiento: O(m). Búsqueda: O(n + m) en promedio; O(n*m) en el peor caso
(muchas colisiones).
"""
from dataclasses import dataclass, field


@dataclass
class RKResult:
    positions: list[int]
    comparisons: int          # comparaciones carácter a carácter (solo en verificación)
    hash_comparisons: int     # comparaciones de hash (una por ventana)
    collisions: int           # hashes iguales pero cadenas distintas
    pattern_hash: int
    base: int
    mod: int
    steps: list[tuple] = field(default_factory=list)  # (i, hash_ventana, evento)


def power_mod(base: int, exp: int, mod: int) -> int:
    """base^exp mod mod, con un bucle explícito."""
    result = 1
    for _ in range(exp):
        result = (result * base) % mod
    return result


def hash_of(s: str, base: int, mod: int) -> int:
    """Hash polinómico de s: sum(ord(s[k]) * base^(len-1-k)) mod mod (regla de Horner)."""
    h = 0
    for ch in s:
        h = (h * base + ord(ch)) % mod
    return h


def rabin_karp_search(
    text: str,
    pattern: str,
    base: int = 256,
    mod: int = 1_000_000_007,
    trace: bool = False,
) -> RKResult:
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return RKResult([], 0, 0, 0, hash_of(pattern, base, mod) if m else 0, base, mod)

    high = power_mod(base, m - 1, mod)          # B^(m-1) mod Q
    p_hash = hash_of(pattern, base, mod)
    w_hash = hash_of(text[:m], base, mod)

    positions: list[int] = []
    steps: list[tuple] = []
    comparisons = 0
    hash_comparisons = 0
    collisions = 0

    for i in range(n - m + 1):
        hash_comparisons += 1
        if w_hash == p_hash:
            # Verificación carácter a carácter
            k = 0
            while k < m:
                comparisons += 1
                if text[i + k] != pattern[k]:
                    break
                k += 1
            if k == m:
                positions.append(i)
                if trace:
                    steps.append((i, w_hash, "ocurrencia"))
            else:
                collisions += 1
                if trace:
                    steps.append((i, w_hash, "colision"))
        elif trace:
            steps.append((i, w_hash, "hash distinto"))

        if i < n - m:
            w_hash = ((w_hash - ord(text[i]) * high) * base + ord(text[i + m])) % mod

    return RKResult(positions, comparisons, hash_comparisons, collisions, p_hash, base, mod, steps)