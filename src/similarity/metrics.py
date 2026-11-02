"""Medidas entre vectores, implementadas desde cero (sin NumPy ni SciPy)."""
import math


def _check(a: list[float], b: list[float]) -> None:
    if len(a) != len(b):
        raise ValueError(f"Los vectores deben tener la misma dimensión ({len(a)} != {len(b)})")


def dot(a: list[float], b: list[float]) -> float:
    _check(a, b)
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """cos(theta) = (a . b) / (|a| |b|). Devuelve 0.0 si algún vector es nulo."""
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    _check(a, b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))