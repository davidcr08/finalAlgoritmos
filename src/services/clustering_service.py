"""Servicio de clustering jerárquico (R2): orquesta fuentes de distancia, 3 enlaces, evaluación, dendrogramas."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np

from src.preprocessing.text_preprocessing import load_corpus, preprocess_corpus
from src.similarity import load_idf, cosine_similarity
from src.similarity.tfidf import tf_vector, tfidf_vector
from src.clustering.hierarchical import (
    LinkageMatrix,
    LinkageMethod,
    agglomerative_clustering,
    evaluate_clustering,
    get_clusters_at_k,
    ClusterResult,
)
from src.clustering.dendrogram import plot_all_dendrograms

DistanceSource = Literal["tfidf", "glove", "fasttext"]


@dataclass
class ClusteringOutput:
    """Salida completa de la página de clustering."""
    distance_source: str
    distance_matrix: list[list[float]]  # n x n
    linkages: dict[str, list[list[float]]]  # method -> linkage matrix
    dendrograms: dict[str, str]  # method -> path to PNG
    metrics: dict[str, dict]  # method -> {cophenetic, silhouettes, best_k}
    best_linkage: str
    clusters_at_k: dict[str, list[list[int]]]  # method -> clusters for chosen k
    doc_labels: list[str]  # ["A01", "A02", ...]


class ClusteringService:
    """Orquesta clustering jerárquico sobre el corpus."""

    def __init__(self):
        self.corpus = load_corpus()
        self.tokens_by_id = preprocess_corpus(self.corpus)
        self.idf = load_idf()
        self.embeddings_glove = self._load_embeddings("embeddings_glove.json")
        self.embeddings_fasttext = self._load_embeddings("embeddings_fasttext.json")
        self.doc_ids = [d["id"] for d in self.corpus]
        self.doc_labels = [f"A{i+1:02d}" for i in range(len(self.corpus))]

    def _load_embeddings(self, filename: str) -> dict[str, list[float]]:
        path = Path("data/processed") / filename
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("vectors", {})

    # --- Matrices de distancia ---

    def _distance_matrix_tfidf(self) -> np.ndarray:
        """d = 1 - coseno(TF-IDF), diagonal=0, simétrica."""
        n = len(self.doc_ids)
        D = np.zeros((n, n))
        
        # Precomputar vectores TF-IDF
        vectors = {}
        for i, doc_id in enumerate(self.doc_ids):
            tf = tf_vector(self.tokens_by_id[doc_id])
            vectors[doc_id] = tfidf_vector(tf, self.idf)
        
        # Vocabulario común
        vocab = sorted(set().union(*[set(v.keys()) for v in vectors.values()]))
        vec_matrix = np.array([[v.get(t, 0.0) for t in vocab] for v in vectors.values()])
        
        for i in range(n):
            for j in range(i + 1, n):
                cos = cosine_similarity(vec_matrix[i], vec_matrix[j])
                dist = 1.0 - cos
                D[i, j] = D[j, i] = dist
        
        return D

    def _distance_matrix_embeddings(self, embeddings: dict[str, list[float]]) -> np.ndarray:
        """d = 1 - coseno(embeddings), diagonal=0, simétrica."""
        n = len(self.doc_ids)
        D = np.zeros((n, n))
        
        for i, id_a in enumerate(self.doc_ids):
            vec_a = embeddings.get(id_a)
            if vec_a is None:
                continue
            for j, id_b in enumerate(self.doc_ids):
                if j <= i:
                    continue
                vec_b = embeddings.get(id_b)
                if vec_b is None:
                    continue
                cos = cosine_similarity(vec_a, vec_b)
                dist = 1.0 - cos
                D[i, j] = D[j, i] = dist
        
        return D

    def get_distance_matrix(self, source: DistanceSource) -> np.ndarray:
        """Obtiene matriz de distancia según fuente seleccionada."""
        if source == "tfidf":
            return self._distance_matrix_tfidf()
        elif source == "glove":
            return self._distance_matrix_embeddings(self.embeddings_glove)
        elif source == "fasttext":
            return self._distance_matrix_embeddings(self.embeddings_fasttext)
        else:
            raise ValueError(f"Fuente desconocida: {source}")

    # --- Clustering ---

    def run_all_linkages(self, D: np.ndarray) -> dict[str, LinkageMatrix]:
        """Ejecuta los 3 métodos de enlace sobre una matriz de distancia."""
        linkages = {}
        for method in ("single", "complete", "average"):
            linkages[method] = agglomerative_clustering(D, method)
        return linkages

    def evaluate_all(self, D: np.ndarray, linkages: dict[str, LinkageMatrix]) -> dict[str, ClusterResult]:
        """Evalúa todos los enlaces: cofenética + siluetas k=2..6."""
        return {method: evaluate_clustering(D, linkage) for method, linkage in linkages.items()}

    def select_best_linkage(self, evaluations: dict[str, ClusterResult]) -> str:
        """
        Selecciona el mejor enlace: mayor silueta promedio sobre k=2..6.
        Cophenética como dato complementario (no decide).
        """
        best_method = None
        best_avg_sil = -1.0
        
        for method, eval_result in evaluations.items():
            sil_values = list(eval_result.silhouettes.values())
            avg_sil = sum(sil_values) / len(sil_values) if sil_values else 0.0
            if avg_sil > best_avg_sil:
                best_avg_sil = avg_sil
                best_method = method
        
        return best_method or "average"

    # --- API Pública ---

    def run_clustering(
        self,
        source: DistanceSource,
        k: int = 3,
        output_dir: str | Path = "data/processed/plots",
    ) -> ClusteringOutput:
        """
        Ejecuta pipeline completo de clustering.
        
        Args:
            source: "tfidf", "glove" o "fasttext"
            k: Número de clusters para salida (2..6)
            output_dir: Directorio para guardar dendrogramas
        
        Returns:
            ClusteringOutput con todo lo necesario para la UI.
        """
        # 1. Matriz de distancia
        D = self.get_distance_matrix(source)
        
        # 2. Tres enlaces
        linkages = self.run_all_linkages(D)
        
        # 3. Evaluación
        evaluations = self.evaluate_all(D, linkages)
        
        # 4. Mejor enlace
        best_linkage = self.select_best_linkage(evaluations)
        
        # 5. Dendrogramas
        dendro_paths = plot_all_dendrograms(linkages, self.doc_labels, output_dir, prefix=f"dendro_{source}")
        
        # 6. Clusters para k elegido
        clusters_at_k = {}
        for method, linkage in linkages.items():
            clusters_at_k[method] = get_clusters_at_k(linkage, k)
        
        # 7. Preparar salida
        return ClusteringOutput(
            distance_source=source,
            distance_matrix=D.tolist(),
            linkages={m: l.to_list() for m, l in linkages.items()},
            dendrograms={m: str(p) for m, p in dendro_paths.items()},
            metrics={
                m: {
                    "cophenetic_corr": e.cophenetic_corr,
                    "silhouettes": e.silhouettes,
                    "best_k": e.best_k,
                }
                for m, e in evaluations.items()
            },
            best_linkage=best_linkage,
            clusters_at_k=clusters_at_k,
            doc_labels=self.doc_labels,
        )

    def list_sources(self) -> list[str]:
        """Fuentes de distancia disponibles."""
        available = ["tfidf"]
        if self.embeddings_glove:
            available.append("glove")
        if self.embeddings_fasttext:
            available.append("fasttext")
        return available


# Instancia global
_service_instance: ClusteringService | None = None


def get_clustering_service() -> ClusteringService:
    global _service_instance
    if _service_instance is None:
        _service_instance = ClusteringService()
    return _service_instance


def _demo():
    """Demo rápido."""
    svc = get_clustering_service()
    print("Fuentes disponibles:", svc.list_sources())
    print(f"Documentos: {len(svc.doc_ids)}")
    
    for source in svc.list_sources():
        print(f"\n=== Fuente: {source} ===")
        output = svc.run_clustering(source, k=3)
        print(f"  Mejor enlace: {output.best_linkage}")
        print(f"  Métricas:")
        for method, metrics in output.metrics.items():
            print(f"    {method}: cophenetic={metrics['cophenetic_corr']:.4f}, "
                  f"silhouettes={metrics['silhouettes']}, best_k={metrics['best_k']}")
        print(f"  Clusters k=3:")
        for method, clusters in output.clusters_at_k.items():
            print(f"    {method}: {clusters}")


if __name__ == "__main__":
    _demo()