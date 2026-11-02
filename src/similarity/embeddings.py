"""Embeddings de un abstract como promedio de los vectores de sus palabras.

Dado un modelo preentrenado de vectores de palabras (GloVe, Word2Vec, FastText):

    v(abstract) = (1 / k) * sum(v(palabra_i))   para las k palabras con vector

Las palabras que el modelo no conoce (fuera de vocabulario, OOV) se omiten y se
registran, porque su proporción es un dato importante para el análisis crítico.
"""


def average_vector(tokens: list[str], lookup, dim: int) -> tuple[list[float], int, list[str]]:
    """Devuelve (vector promedio, palabras usadas, palabras omitidas).

    `lookup` es cualquier objeto que soporte `palabra in lookup` y `lookup[palabra]`
    (un dict o los KeyedVectors de gensim).
    """
    total = [0.0] * dim
    used = 0
    missing: list[str] = []
    for token in tokens:
        if token in lookup:
            vec = lookup[token]
            for k in range(dim):
                total[k] += float(vec[k])
            used += 1
        else:
            missing.append(token)
    vector = [x / used for x in total] if used else total
    return vector, used, missing


def embed_corpus(tokens_by_id: dict[str, list[str]], lookup, dim: int) -> dict:
    """Calcula el vector de cada artículo y las estadísticas de palabras omitidas."""
    vectors: dict[str, list[float]] = {}
    oov: dict[str, dict] = {}
    for doc_id, tokens in tokens_by_id.items():
        vec, used, missing = average_vector(tokens, lookup, dim)
        vectors[doc_id] = vec
        oov[doc_id] = {
            "tokens": len(tokens),
            "used": used,
            "missing": len(missing),
            "missing_tokens": sorted(set(missing)),
        }
    return {"dim": dim, "vectors": vectors, "oov": oov}