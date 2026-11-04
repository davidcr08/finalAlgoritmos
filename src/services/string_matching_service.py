#!/usr/bin/env python3
"""
Servicio de búsqueda de patrones de texto (R3): orquesta KMP, Boyer-Moore y Rabin-Karp.

Proporciona:
- Búsqueda general en los 20 abstracts del corpus
- Demostración académica paso a paso con trazas detalladas
- Medición de rendimiento y análisis de complejidad
"""

import time
from dataclasses import dataclass, field
from typing import Any

from src.preprocessing.text_preprocessing import load_corpus
from src.string_matching.kmp import KMPResult, kmp_search, build_lps
from src.string_matching.boyer_moore import BMResult, boyer_moore_search, build_bad_char, build_good_suffix
from src.string_matching.rabin_karp import RKResult, rabin_karp_search, hash_of, power_mod


@dataclass
class SearchResult:
    """Resultado de búsqueda en un documento."""
    doc_id: str
    title: str
    positions_kmp: list[int]
    positions_bm: list[int]
    positions_rk: list[int]
    count: int


@dataclass
class SummaryResult:
    """Resumen comparativo de los tres algoritmos."""
    total_occurrences: int
    docs_with_matches: int
    kmp_comparisons: int
    bm_comparisons: int
    rk_comparisons: int
    rk_hash_comparisons: int
    rk_collisions: int
    kmp_preprocess_time: float
    kmp_search_time: float
    bm_preprocess_time: float
    bm_search_time: float
    rk_preprocess_time: float
    rk_search_time: float


@dataclass
class DemoResult:
    """Resultado completo de demostración académica."""
    doc_id: str
    title: str
    pattern: str
    abstract: str
    kmp_result: KMPResult
    bm_result: BMResult
    rk_result: RKResult
    kmp_preprocess_time: float
    kmp_search_time: float
    bm_preprocess_time: float
    bm_search_time: float
    rk_preprocess_time: float
    rk_search_time: float


class StringMatchingService:
    """Servicio que coordina los tres algoritmos de búsqueda de patrones."""

    def __init__(self):
        self.corpus = load_corpus()
        self.documents = [
            {
                "id": doc["id"],
                "title": doc["title"],
                "abstract": doc["abstract"],
            }
            for doc in self.corpus
        ]

    def list_documents(self) -> list[dict]:
        """Devuelve lista de documentos: id, title, abstract."""
        return self.documents

    def search_all(self, pattern: str) -> tuple[list[SearchResult], SummaryResult]:
        """
        Busca un patrón en los 20 abstracts con los tres algoritmos.

        Preprocesamiento: se ejecuta UNA vez por patrón (fuera del ciclo de documentos).
        Búsqueda: suma de los tiempos sobre los 20 documentos.

        Returns:
            (lista de resultados por documento, resumen comparativo)
        """
        if not pattern:
            return [], SummaryResult(
                total_occurrences=0, docs_with_matches=0,
                kmp_comparisons=0, bm_comparisons=0, rk_comparisons=0,
                rk_hash_comparisons=0, rk_collisions=0,
                kmp_preprocess_time=0.0, kmp_search_time=0.0,
                bm_preprocess_time=0.0, bm_search_time=0.0,
                rk_preprocess_time=0.0, rk_search_time=0.0
            )

        # --- PREPROCESAMIENTO (una vez por patrón) ---
        # KMP: tabla LPS
        t0 = time.perf_counter()
        lps = build_lps(pattern)
        kmp_preprocess_time = time.perf_counter() - t0

        # Boyer-Moore: tablas bad_char y good_suffix
        t0 = time.perf_counter()
        bad_char = build_bad_char(pattern)
        good_suffix = build_good_suffix(pattern)
        bm_preprocess_time = time.perf_counter() - t0

        # Rabin-Karp: pattern_hash y high (B^(m-1) mod Q)
        t0 = time.perf_counter()
        base = 256
        mod = 1_000_000_007
        pattern_hash = hash_of(pattern, base, mod)
        high = power_mod(base, len(pattern) - 1, mod)
        rk_preprocess_time = time.perf_counter() - t0

        # --- BÚSQUEDA SOBRE LOS 20 DOCUMENTOS ---
        results = []
        total_occurrences = 0
        docs_with_matches = 0
        kmp_comparisons = 0
        bm_comparisons = 0
        rk_comparisons = 0
        rk_hash_comparisons = 0
        rk_collisions = 0
        kmp_search_time = 0.0
        bm_search_time = 0.0
        rk_search_time = 0.0

        for doc in self.documents:
            text = doc["abstract"]

            # KMP con LPS precalculada
            t0 = time.perf_counter()
            kmp_res = kmp_search(text, pattern, trace=False, lps=lps)
            kmp_search_time += time.perf_counter() - t0
            kmp_comparisons += kmp_res.comparisons

            # Boyer-Moore con tablas precalculadas
            t0 = time.perf_counter()
            bm_res = boyer_moore_search(text, pattern, trace=False, bad_char=bad_char, good_suffix=good_suffix)
            bm_search_time += time.perf_counter() - t0
            bm_comparisons += bm_res.comparisons

            # Rabin-Karp con pattern_hash y high precalculados
            t0 = time.perf_counter()
            rk_res = rabin_karp_search(text, pattern, trace=False, pattern_hash=pattern_hash, high=high)
            rk_search_time += time.perf_counter() - t0
            rk_comparisons += rk_res.comparisons
            rk_hash_comparisons += rk_res.hash_comparisons
            rk_collisions += rk_res.collisions

            # Verificar consistencia de posiciones
            positions = kmp_res.positions
            if bm_res.positions != positions or rk_res.positions != positions:
                raise RuntimeError(
                    f"Discrepancia en {doc['id']}: "
                    f"KMP={positions}, BM={bm_res.positions}, RK={rk_res.positions}"
                )

            count = len(positions)
            if count > 0:
                total_occurrences += count
                docs_with_matches += 1

            results.append(SearchResult(
                doc_id=doc["id"],
                title=doc["title"],
                positions_kmp=kmp_res.positions,
                positions_bm=bm_res.positions,
                positions_rk=rk_res.positions,
                count=count,
            ))

        summary = SummaryResult(
            total_occurrences=total_occurrences,
            docs_with_matches=docs_with_matches,
            kmp_comparisons=kmp_comparisons,
            bm_comparisons=bm_comparisons,
            rk_comparisons=rk_comparisons,
            rk_hash_comparisons=rk_hash_comparisons,
            rk_collisions=rk_collisions,
            kmp_preprocess_time=kmp_preprocess_time,
            kmp_search_time=kmp_search_time,
            bm_preprocess_time=bm_preprocess_time,
            bm_search_time=bm_search_time,
            rk_preprocess_time=rk_preprocess_time,
            rk_search_time=rk_search_time,
        )

        return results, summary

    def demo(self, doc_id: str, pattern: str) -> DemoResult:
        """
        Demostración académica paso a paso con trazas detalladas.

        La medición de tiempos separa:
        - Preprocesamiento: una vez por patrón
        - Búsqueda: con trace=False, reutilizando estructuras
        - Generación de trazas: con trace=True, NO incluido en el tiempo de búsqueda

        Args:
            doc_id: Identificador del documento (A01..A20)
            pattern: Patrón a buscar

        Returns:
            DemoResult con resultados completos y trazas
        """
        doc = next((d for d in self.documents if d["id"] == doc_id), None)
        if not doc:
            raise ValueError(f"Documento {doc_id} no encontrado")

        text = doc["abstract"]

        # --- PREPROCESAMIENTO (una vez) ---
        t0 = time.perf_counter()
        lps = build_lps(pattern)
        kmp_preprocess_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        bad_char = build_bad_char(pattern)
        good_suffix = build_good_suffix(pattern)
        bm_preprocess_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        base = 256
        mod = 1_000_000_007
        pattern_hash = hash_of(pattern, base, mod)
        high = power_mod(base, len(pattern) - 1, mod)
        rk_preprocess_time = time.perf_counter() - t0

        # --- BÚSQUEDA SIN TRAZA (para medir tiempo real) ---
        t0 = time.perf_counter()
        _ = kmp_search(text, pattern, trace=False, lps=lps)
        kmp_search_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = boyer_moore_search(text, pattern, trace=False, bad_char=bad_char, good_suffix=good_suffix)
        bm_search_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = rabin_karp_search(text, pattern, trace=False, pattern_hash=pattern_hash, high=high)
        rk_search_time = time.perf_counter() - t0

        # --- GENERACIÓN DE TRAZAS (no incluida en tiempos de búsqueda) ---
        kmp_res = kmp_search(text, pattern, trace=True, lps=lps)
        bm_res = boyer_moore_search(text, pattern, trace=True, bad_char=bad_char, good_suffix=good_suffix)
        rk_res = rabin_karp_search(text, pattern, trace=True, pattern_hash=pattern_hash, high=high)

        return DemoResult(
            doc_id=doc["id"],
            title=doc["title"],
            pattern=pattern,
            abstract=text,
            kmp_result=kmp_res,
            bm_result=bm_res,
            rk_result=rk_res,
            kmp_preprocess_time=kmp_preprocess_time,
            kmp_search_time=kmp_search_time,
            bm_preprocess_time=bm_preprocess_time,
            bm_search_time=bm_search_time,
            rk_preprocess_time=rk_preprocess_time,
            rk_search_time=rk_search_time,
        )


# Instancia global para uso sencillo
_service_instance: StringMatchingService | None = None


def get_service() -> StringMatchingService:
    global _service_instance
    if _service_instance is None:
        _service_instance = StringMatchingService()
    return _service_instance


def _demo():
    """Demo rápido."""
    svc = get_service()
    docs = svc.list_documents()
    print("Documentos disponibles:")
    for d in docs[:3]:
        print(f"  {d['id']}: {d['title'][:60]}... ({len(d['abstract'])} chars)")

    # Búsqueda de prueba
    pattern = "algoritmo"
    print(f"\nBúsqueda de '{pattern}':")
    results, summary = svc.search_all(pattern)
    for r in results:
        if r.count > 0:
            print(f"  {r.doc_id}: {r.count} ocurrencias en {r.positions_kmp}")
    print(f"Total: {summary.total_occurrences} ocurrencias en {summary.docs_with_matches} docs")
    print(f"KMP comps: {summary.kmp_comparisons}, BM comps: {summary.bm_comparisons}, RK comps: {summary.rk_comparisons}")


if __name__ == "__main__":
    _demo()
