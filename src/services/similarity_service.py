"""Servicio de similitud textual (R1): orquesta 6 algoritmos + caso de estudio / matriz."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.preprocessing.text_preprocessing import load_corpus, preprocess_corpus
from src.similarity import (
    LevenshteinTrace,
    NeedlemanWunschTrace,
    TFIDFTrace,
    JaccardTrace,
    compute_idf,
    cosine_similarity,
    euclidean_distance,
    jaccard_tokens,
    levenshtein_words,
    load_idf,
    needleman_wunsch_words,
    save_idf,
    tfidf_cosine,
)

# Archivos de embeddings pregenerados
EMBEDDINGS_DIR = Path("data/processed")
GLOVE_PATH = EMBEDDINGS_DIR / "embeddings_glove.json"
FASTTEXT_PATH = EMBEDDINGS_DIR / "embeddings_fasttext.json"
IDF_PATH = EMBEDDINGS_DIR / "idf.json"
CORPUS_SIZE = 20


@dataclass
class SimilarityResult:
    """Resultado unificado por algoritmo."""
    algorithm: str
    similarity: float
    trace: dict | None = None  # estructuras intermedias si trace=True


@dataclass
class PairwiseResult:
    """Resultado para un par de artículos."""
    doc_a: str
    doc_b: str
    results: dict[str, SimilarityResult]  # algorithm -> SimilarityResult


@dataclass
class MatrixResult:
    """Matriz k×k para >2 artículos."""
    doc_ids: list[str]
    matrices: dict[str, list[list[float]]]  # algorithm -> matrix[k][k]


class SimilarityService:
    """Orquesta los 6 algoritmos de similitud sobre el corpus."""

    def __init__(self):
        self.corpus = load_corpus()
        self.tokens_by_id = preprocess_corpus(self.corpus)
        self.idf = self._ensure_idf()
        self.embeddings_glove = self._load_embeddings(GLOVE_PATH)
        self.embeddings_fasttext = self._load_embeddings(FASTTEXT_PATH)

    def _ensure_idf(self) -> dict[str, float]:
        """Carga IDF o lo computa y guarda si no existe."""
        if IDF_PATH.exists():
            return load_idf(IDF_PATH)
        print("Calculando IDF sobre corpus completo...")
        idf = compute_idf(self.tokens_by_id)
        save_idf(idf, IDF_PATH)
        print(f"IDF guardado en {IDF_PATH} ({len(idf)} términos)")
        return idf

    def _load_embeddings(self, path: Path) -> dict[str, list[float]]:
        """Carga vectores precomputados: {doc_id: vector}."""
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("vectors", {})

    def _get_tokens(self, doc_id: str) -> list[str]:
        return self.tokens_by_id.get(doc_id, [])

    def _get_abstract(self, doc_id: str) -> str:
        doc = next((d for d in self.corpus if d["id"] == doc_id), None)
        return doc["abstract"] if doc else ""

    def _get_embedding(self, embeddings: dict, doc_id: str) -> list[float] | None:
        return embeddings.get(doc_id)

    # --- 6 Algoritmos ---

    def _levenshtein(self, id_a: str, id_b: str, trace: bool) -> SimilarityResult:
        sim = levenshtein_words(self._get_tokens(id_a), self._get_tokens(id_b), trace=trace)
        trace_data = asdict(sim) if trace and isinstance(sim, LevenshteinTrace) else None
        return SimilarityResult("Levenshtein", sim if not trace else sim.similarity, trace_data)

    def _needleman_wunsch(self, id_a: str, id_b: str, trace: bool) -> SimilarityResult:
        sim = needleman_wunsch_words(self._get_tokens(id_a), self._get_tokens(id_b), trace=trace)
        trace_data = asdict(sim) if trace and isinstance(sim, NeedlemanWunschTrace) else None
        return SimilarityResult("Needleman-Wunsch", sim if not trace else sim.similarity, trace_data)

    def _tfidf_cosine(self, id_a: str, id_b: str, trace: bool) -> SimilarityResult:
        sim = tfidf_cosine(self._get_tokens(id_a), self._get_tokens(id_b), self.idf, trace=trace)
        trace_data = asdict(sim) if trace and isinstance(sim, TFIDFTrace) else None
        return SimilarityResult("TF-IDF Coseno", sim if not trace else sim.cosine, trace_data)

    def _jaccard(self, id_a: str, id_b: str, trace: bool) -> SimilarityResult:
        sim = jaccard_tokens(self._get_tokens(id_a), self._get_tokens(id_b), trace=trace)
        trace_data = asdict(sim) if trace and isinstance(sim, JaccardTrace) else None
        return SimilarityResult("Jaccard", sim if not trace else sim.jaccard, trace_data)

    def _embeddings_glove(self, id_a: str, id_b: str, trace: bool) -> SimilarityResult:
        vec_a = self._get_embedding(self.embeddings_glove, id_a)
        vec_b = self._get_embedding(self.embeddings_glove, id_b)
        if vec_a is None or vec_b is None:
            return SimilarityResult("Embeddings GloVe", 0.0, {"error": "vectors not found"})
        cos = cosine_similarity(vec_a, vec_b)
        euc = euclidean_distance(vec_a, vec_b)
        trace_data = None
        if trace:
            # stats OOV ya vienen en embeddings_glove.json
            trace_data = {
                "dim": len(vec_a),
                "cosine": cos,
                "euclidean": euc,
                "vec_a_preview": vec_a[:5],
                "vec_b_preview": vec_b[:5],
            }
        return SimilarityResult("Embeddings GloVe", cos, trace_data)

    def _embeddings_fasttext(self, id_a: str, id_b: str, trace: bool) -> SimilarityResult:
        vec_a = self._get_embedding(self.embeddings_fasttext, id_a)
        vec_b = self._get_embedding(self.embeddings_fasttext, id_b)
        if vec_a is None or vec_b is None:
            return SimilarityResult("Embeddings FastText", 0.0, {"error": "vectors not found"})
        cos = cosine_similarity(vec_a, vec_b)
        euc = euclidean_distance(vec_a, vec_b)
        trace_data = None
        if trace:
            trace_data = {
                "dim": len(vec_a),
                "cosine": cos,
                "euclidean": euc,
                "vec_a_preview": vec_a[:5],
                "vec_b_preview": vec_b[:5],
            }
        return SimilarityResult("Embeddings FastText", cos, trace_data)

    # --- API Pública ---

    def compare_pair(self, doc_id_a: str, doc_id_b: str, trace: bool = False) -> PairwiseResult:
        """
        Compara 2 artículos con los 6 algoritmos.
        Con trace=True -> modo caso de estudio (devuelve estructuras intermedias).
        """
        algorithms = [
            self._levenshtein,
            self._needleman_wunsch,
            self._tfidf_cosine,
            self._jaccard,
            self._embeddings_glove,
            self._embeddings_fasttext,
        ]
        results = {}
        for algo in algorithms:
            res = algo(doc_id_a, doc_id_b, trace)
            results[res.algorithm] = res
        return PairwiseResult(doc_id_a, doc_id_b, results)

    def compare_matrix(self, doc_ids: list[str]) -> MatrixResult:
        """
        Compara k artículos (k>=3) -> matriz k×k por algoritmo.
        Solo valores de similitud (sin trace).
        """
        k = len(doc_ids)
        if k < 3:
            raise ValueError("compare_matrix requiere al menos 3 documentos")

        algorithms = [
            ("Levenshtein", lambda a, b: self._levenshtein(a, b, False).similarity),
            ("Needleman-Wunsch", lambda a, b: self._needleman_wunsch(a, b, False).similarity),
            ("TF-IDF Coseno", lambda a, b: self._tfidf_cosine(a, b, False).similarity),
            ("Jaccard", lambda a, b: self._jaccard(a, b, False).similarity),
            ("Embeddings GloVe", lambda a, b: self._embeddings_glove(a, b, False).similarity),
            ("Embeddings FastText", lambda a, b: self._embeddings_fasttext(a, b, False).similarity),
        ]

        matrices = {}
        for name, fn in algorithms:
            mat = [[0.0] * k for _ in range(k)]
            for i in range(k):
                for j in range(k):
                    if i == j:
                        mat[i][j] = 1.0
                    elif i < j:
                        mat[i][j] = fn(doc_ids[i], doc_ids[j])
                    else:
                        mat[i][j] = mat[j][i]  # simétrica
            matrices[name] = mat

        return MatrixResult(doc_ids, matrices)

    def list_documents(self) -> list[dict]:
        """Devuelve lista de documentos disponibles: id, title, authors, abstract_words."""
        return [
            {
                "id": d["id"],
                "title": d["title"],
                "authors": d["authors"],
                "abstract_words": d["abstract_words"],
            }
            for d in self.corpus
        ]


# Instancia global para uso sencillo
_service_instance: SimilarityService | None = None


def get_service() -> SimilarityService:
    global _service_instance
    if _service_instance is None:
        _service_instance = SimilarityService()
    return _service_instance


def _demo():
    """Demo rápido."""
    svc = get_service()
    docs = svc.list_documents()
    print("Documentos disponibles:")
    for d in docs[:3]:
        print(f"  {d['id']}: {d['title'][:60]}... ({d['abstract_words']} palabras)")

    # Caso de estudio con 2
    if len(docs) >= 2:
        a, b = docs[0]["id"], docs[1]["id"]
        print(f"\nCaso de estudio: {a} vs {b}")
        pair = svc.compare_pair(a, b, trace=True)
        for name, res in pair.results.items():
            print(f"  {name}: {res.similarity:.4f}")

    # Matriz con 3+
    if len(docs) >= 3:
        ids = [d["id"] for d in docs[:3]]
        print(f"\nMatriz 3x3: {ids}")
        mat = svc.compare_matrix(ids)
        for name, m in mat.matrices.items():
            print(f"  {name}:")
            for row in m:
                print(f"    {[f'{x:.3f}' for x in row]}")


if __name__ == "__main__":
    _demo()