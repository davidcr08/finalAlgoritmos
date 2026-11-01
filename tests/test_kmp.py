import random
import re

from src.string_matching.kmp import build_lps, kmp_search


def reference(text, pattern):
    """Referencia (solo para pruebas): todas las ocurrencias, incluidas las solapadas."""
    return [m.start() for m in re.finditer(f"(?={re.escape(pattern)})", text)]


def test_lps_clasico():
    assert build_lps("ABABCABAB") == [0, 0, 1, 2, 0, 1, 2, 3, 4]


def test_ocurrencias_solapadas():
    assert kmp_search("aaaa", "aa").positions == [0, 1, 2]


def test_sin_coincidencia():
    assert kmp_search("abcdef", "xyz").positions == []


def test_patron_vacio_o_mas_largo():
    assert kmp_search("abc", "").positions == []
    assert kmp_search("ab", "abc").positions == []


def test_contra_referencia_aleatoria():
    rnd = random.Random(42)
    for _ in range(500):
        text = "".join(rnd.choice("ab") for _ in range(rnd.randint(0, 40)))
        pattern = "".join(rnd.choice("ab") for _ in range(rnd.randint(1, 5)))
        assert kmp_search(text, pattern).positions == reference(text, pattern)