import random
import re

from src.string_matching.boyer_moore import (
    boyer_moore_search,
    build_bad_char,
)


def reference(text, pattern):
    """Referencia (solo para pruebas): todas las ocurrencias, incluidas las solapadas."""
    return [m.start() for m in re.finditer(f"(?={re.escape(pattern)})", text)]


def test_tabla_caracter_malo():
    assert build_bad_char("abcab") == {"a": 3, "b": 4, "c": 2}


def test_ocurrencias_solapadas():
    assert boyer_moore_search("aaaa", "aa").positions == [0, 1, 2]
    assert boyer_moore_search("ABABABAB", "ABAB").positions == [0, 2, 4]


def test_sin_coincidencia():
    assert boyer_moore_search("abcdef", "xyz").positions == []


def test_patron_vacio_o_mas_largo():
    assert boyer_moore_search("abc", "").positions == []
    assert boyer_moore_search("ab", "abc").positions == []


def test_contra_referencia_aleatoria():
    rnd = random.Random(7)
    for alphabet in ("ab", "abc"):
        for _ in range(700):
            text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 50)))
            pattern = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 6)))
            assert boyer_moore_search(text, pattern).positions == reference(text, pattern)