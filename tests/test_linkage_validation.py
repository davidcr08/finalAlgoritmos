import pytest

from src.clustering.hierarchical import cophenetic_matrix, cut_tree


# Matriz Z válida para 3 elementos:
# Primero fusiona 0 y 1 (nuevo ID 3).
# Después fusiona 3 y 2 (nuevo ID 4).
Z_VALID = [
    [0, 1, 1.0, 2],
    [3, 2, 2.0, 3],
]


@pytest.mark.parametrize("function", [cophenetic_matrix])
@pytest.mark.parametrize("invalid_z", [
    # Identificador inexistente.
    [[0, 3, 1.0, 2], [3, 2, 2.0, 3]],

    # Fusión de un elemento consigo mismo.
    [[0, 0, 1.0, 1], [3, 2, 2.0, 3]],

    # Reutiliza el elemento 0 después de haber sido fusionado.
    [[0, 1, 1.0, 2], [0, 2, 2.0, 3]],

    # Tamaño incorrecto: fusionar dos elementos debe producir tamaño 2.
    [[0, 1, 1.0, 3], [3, 2, 2.0, 3]],
])
def test_cophenetic_matrix_rejects_invalid_linkage(function, invalid_z):
    with pytest.raises(ValueError):
        function(invalid_z, 3)


@pytest.mark.parametrize("invalid_z", [
    [[0, 3, 1.0, 2], [3, 2, 2.0, 3]],
    [[0, 0, 1.0, 1], [3, 2, 2.0, 3]],
    [[0, 1, 1.0, 2], [0, 2, 2.0, 3]],
    [[0, 1, 1.0, 3], [3, 2, 2.0, 3]],
])
def test_cut_tree_rejects_invalid_linkage(invalid_z):
    with pytest.raises(ValueError):
        cut_tree(invalid_z, 3, 2)


@pytest.mark.parametrize("k", [1, 3])
def test_cut_tree_validates_linkage_even_with_early_returns(k):
    invalid_z = [
        [0, 3, 1.0, 2],
        [3, 2, 2.0, 3],
    ]

    with pytest.raises(ValueError):
        cut_tree(invalid_z, 3, k)


def test_valid_linkage_is_accepted():
    result = cophenetic_matrix(Z_VALID, 3)
    assert len(result) == 3
    assert all(len(row) == 3 for row in result)

    clusters = cut_tree(Z_VALID, 3, 2)
    assert len(set(clusters)) == 2