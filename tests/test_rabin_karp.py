import random
import re

from src.string_matching.rabin_karp import hash_of, rabin_karp_search


def reference(text, pattern):
    """Referencia (solo para pruebas): todas las ocurrencias, incluidas las solapadas."""
    return [m.start() for m in re.finditer(f"(?={re.escape(pattern)})", text)]


def test_hash_polinomico_conocido():
    # "ab" con base 10 y mod 1000: (97 * 10 + 98) % 1000 = 68
    assert hash_of("ab", 10, 1000) == (97 * 10 + 98) % 1000


def test_ocurrencias_solapadas():
    assert rabin_karp_search("aaaa", "aa").positions == [0, 1, 2]


def test_sin_coincidencia():
    assert rabin_karp_search("abcdef", "xyz").positions == []


def test_patron_vacio_o_mas_largo():
    assert rabin_karp_search("abc", "").positions == []
    assert rabin_karp_search("ab", "abc").positions == []


def test_colisiones_con_modulo_pequeno():
    # Con un módulo muy pequeño casi seguro hay colisiones, pero el resultado sigue siendo correcto
    r = rabin_karp_search("abcabcabdabc", "abd", base=256, mod=7)
    assert r.positions == [6]
    assert r.collisions > 0


def test_contra_referencia_aleatoria():
    rnd = random.Random(3)
    for mod in (7, 101, 1_000_000_007):
        for _ in range(500):
            text = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 50)))
            pattern = "".join(rnd.choice("abc") for _ in range(rnd.randint(1, 6)))
            assert rabin_karp_search(text, pattern, mod=mod).positions == reference(text, pattern)