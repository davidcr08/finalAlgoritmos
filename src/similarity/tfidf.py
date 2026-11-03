"""TF-IDF Vectorizer + Cosine Similarity (desde cero).

Definición:
    tf(t, d) = conteo bruto del término t en documento d
    df(t) = número de documentos del corpus que contienen t
    idf(t) = ln((1 + N) / (1 + df(t))) + 1     (suavizado, N = 20 corpus completo)
    peso(t, d) = tf(t, d) * idf(t)

Similitud coseno entre vectores TF-IDF:
    cos(d1, d2) = (v1 · v2) / (|v1| * |v2|)   en [0, 1] para TF-IDF no negativo

Requisitos:
    - df e idf se calculan sobre TODO el corpus (N=20), no sobre la selección.
    - idf precomputado y persistido en data/processed/idf.json

Complejidad: O(V) por documento para vectorizar, O(V) para coseno (V = vocabulario).
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.similarity.metrics import cosine_similarity, norm


@dataclass
class TFIDFTrace:
    """Estructuras intermedias para caso de estudio (2 docs)."""
    tf_a: dict[str, int]
    tf_b: dict[str, int]
    df: dict[str, int]
    idf: dict[str, float]
    weights_a: dict[str, float]
    weights_b: dict[str, float]
    dot_product: float
    norm_a: float
    norm_b: float
    cosine: float
    angle_degrees: float
    vocab_size: int


IDF_PATH = Path("data/processed/idf.json")
CORPUS_SIZE = 20  # N fijo según especificación


def compute_idf(corpus_tokens: dict[str, list[str]], expected_size: int | None = CORPUS_SIZE) -> dict[str, float]:
    """
    Calcula IDF para todos los términos del corpus completo.

    Args:
        corpus_tokens: {doc_id: [tokens]} del corpus.
        expected_size: Tamaño esperado del corpus (None para desactivar check).

    Returns:
        dict[termino] -> idf (float)
    """
    N = len(corpus_tokens)
    if expected_size is not None:
        assert N == expected_size, f"Corpus debe tener {expected_size} docs, tiene {N}"

    # df: en cuántos docs aparece cada término
    df: dict[str, int] = {}
    for tokens in corpus_tokens.values():
        unique = set(tokens)
        for term in unique:
            df[term] = df.get(term, 0) + 1

    # idf = ln((1+N)/(1+df)) + 1
    idf = {}
    for term, df_val in df.items():
        idf[term] = math.log((1 + N) / (1 + df_val)) + 1.0

    return idf


def save_idf(idf: dict[str, float], path: Path = IDF_PATH) -> None:
    """Persiste IDF a JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(idf, ensure_ascii=False, indent=2), encoding="utf-8")


def load_idf(path: Path = IDF_PATH) -> dict[str, float]:
    """Carga IDF desde JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def tf_vector(tokens: list[str]) -> dict[str, int]:
    """TF bruto: conteo de cada token."""
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


def tfidf_vector(tf: dict[str, int], idf: dict[str, float]) -> dict[str, float]:
    """Peso TF-IDF = tf * idf (solo términos con idf conocido)."""
    return {term: count * idf[term] for term, count in tf.items() if term in idf}


def tfidf_cosine(
    tokens_a: list[str],
    tokens_b: list[str],
    idf: dict[str, float],
    trace: bool = False,
) -> float | TFIDFTrace:
    """
    Similitud coseno TF-IDF entre dos documentos.

    Args:
        tokens_a, tokens_b: tokens preprocesados de cada abstract.
        idf: diccionario idf precomputado (corpus completo N=20).
        trace: si True, devuelve TFIDFTrace con tabla completa.

    Returns:
        float: coseno en [0, 1] si trace=False.
        TFIDFTrace: si trace=True.
    """
    tf_a = tf_vector(tokens_a)
    tf_b = tf_vector(tokens_b)

    # Unión de términos presentes en A o B
    terms = set(tf_a.keys()) | set(tf_b.keys())

    weights_a = {t: tf_a.get(t, 0) * idf.get(t, 0.0) for t in terms}
    weights_b = {t: tf_b.get(t, 0) * idf.get(t, 0.0) for t in terms}

    # Vectores densos para métricas
    vec_a = [weights_a[t] for t in terms]
    vec_b = [weights_b[t] for t in terms]

    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    na = norm(vec_a)
    nb = norm(vec_b)
    cosine = dot / (na * nb) if na > 0 and nb > 0 else 0.0
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    if not trace:
        return cosine

    # df para la tabla (solo términos en A o B)
    df = {t: 0 for t in terms}
    # Nota: df real requeriría corpus completo; aquí solo para visualización
    # En caso de estudio real, se pasaría df completo

    return TFIDFTrace(
        tf_a=tf_a,
        tf_b=tf_b,
        df=df,  # placeholder, se llena en service
        idf={t: idf[t] for t in terms if t in idf},
        weights_a=weights_a,
        weights_b=weights_b,
        dot_product=dot,
        norm_a=na,
        norm_b=nb,
        cosine=cosine,
        angle_degrees=angle_deg,
        vocab_size=len(terms),
    )


def _demo():
    """Demo con IDF ficticio."""
    idf = {"algoritmo": 1.5, "genetico": 2.0, "optimizacion": 1.8, "evolutivo": 2.2}
    a = ["algoritmo", "genetico", "optimizacion"]
    b = ["algoritmo", "evolutivo", "optimizacion"]
    result = tfidf_cosine(a, b, idf, trace=True)
    print(f"Coseno: {result.cosine:.4f}")
    print(f"Ángulo: {result.angle_degrees:.2f}°")
    print("Tabla términos:")
    for t in sorted(result.weights_a.keys()):
        print(f"  {t}: tf_a={result.tf_a.get(t,0)}, tf_b={result.tf_b.get(t,0)}, "
              f"idf={result.idf.get(t,0):.3f}, w_a={result.weights_a[t]:.3f}, w_b={result.weights_b[t]:.3f}")


if __name__ == "__main__":
    _demo()