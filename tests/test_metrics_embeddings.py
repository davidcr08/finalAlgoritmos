import math

import pytest

from src.similarity.embeddings import average_vector, embed_corpus
from src.similarity.metrics import cosine_similarity, dot, euclidean_distance, norm


def test_producto_punto_y_norma():
    assert dot([1, 2, 3], [4, 5, 6]) == 32
    assert norm([3, 4]) == 5


def test_coseno_casos_basicos():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([1, 1], [2, 2]) == pytest.approx(1.0)   # no depende de la magnitud


def test_coseno_vector_nulo():
    assert cosine_similarity([0, 0], [1, 2]) == 0.0


def test_euclidiana():
    assert euclidean_distance([0, 0], [3, 4]) == 5


def test_dimensiones_distintas():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_relacion_euclidiana_coseno_en_vectores_unitarios():
    a = [3 / 5, 4 / 5]
    b = [1.0, 0.0]
    assert euclidean_distance(a, b) ** 2 == pytest.approx(2 - 2 * cosine_similarity(a, b))


LOOKUP = {"ai": [1.0, 0.0], "model": [0.0, 1.0], "data": [1.0, 1.0]}


def test_promedio_de_vectores():
    vec, used, missing = average_vector(["ai", "model"], LOOKUP, 2)
    assert vec == [0.5, 0.5]
    assert used == 2 and missing == []


def test_palabras_fuera_de_vocabulario_se_omiten_y_se_registran():
    vec, used, missing = average_vector(["ai", "xyzzy", "data"], LOOKUP, 2)
    assert vec == [1.0, 0.5]
    assert used == 2 and missing == ["xyzzy"]


def test_todas_fuera_de_vocabulario_da_vector_nulo():
    vec, used, missing = average_vector(["foo", "bar"], LOOKUP, 2)
    assert vec == [0.0, 0.0] and used == 0 and missing == ["foo", "bar"]


def test_embed_corpus():
    res = embed_corpus({"A01": ["ai", "foo"], "A02": ["model"]}, LOOKUP, 2)
    assert res["dim"] == 2
    assert res["vectors"]["A01"] == [1.0, 0.0]
    assert res["oov"]["A01"]["missing"] == 1
    assert res["oov"]["A02"]["missing"] == 0