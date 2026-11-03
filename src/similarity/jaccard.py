"""Coeficiente de Jaccard sobre conjuntos de tokens.

Definición:
    J(A, B) = |A ∩ B| / |A ∪ B|

Donde A y B son los conjuntos de tokens únicos de cada abstract.
Rango: [0, 1]. 1 = idénticos, 0 = sin tokens en común.

Complejidad: O(n + m) para construir sets + O(min(|A|,|B|)) para intersección.
"""
from dataclasses import dataclass


@dataclass
class JaccardTrace:
    """Estructuras intermedias para caso de estudio."""
    set_a: set[str]
    set_b: set[str]
    intersection: set[str]
    union: set[str]
    intersection_size: int
    union_size: int
    jaccard: float


def jaccard_tokens(
    tokens_a: list[str],
    tokens_b: list[str],
    trace: bool = False,
) -> float | JaccardTrace:
    """
    Coeficiente de Jaccard entre dos listas de tokens.

    Args:
        tokens_a: tokens preprocesados del abstract A.
        tokens_b: tokens preprocesados del abstract B.
        trace: si True, devuelve JaccardTrace con conjuntos e intersección.

    Returns:
        float: Jaccard en [0, 1] si trace=False.
        JaccardTrace: si trace=True.
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    inter = set_a & set_b
    union = set_a | set_b

    if not union:
        jac = 1.0  # ambos vacíos
    else:
        jac = len(inter) / len(union)

    if not trace:
        return jac

    return JaccardTrace(
        set_a=set_a,
        set_b=set_b,
        intersection=inter,
        union=union,
        intersection_size=len(inter),
        union_size=len(union),
        jaccard=jac,
    )


def _demo():
    a = ["algoritmo", "genetico", "optimizacion", "funcion", "objetivo"]
    b = ["algoritmo", "evolutivo", "optimizacion", "multiobjetivo"]
    result = jaccard_tokens(a, b, trace=True)
    print(f"Jaccard: {result.jaccard:.4f}")
    print(f"Intersección ({result.intersection_size}): {sorted(result.intersection)}")
    print(f"Unión ({result.union_size}): {sorted(result.union)}")


if __name__ == "__main__":
    _demo()