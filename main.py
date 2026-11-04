#!/usr/bin/env python3
"""
Punto de entrada general del proyecto finalAlgoritmos.

Ejecuta secuencialmente:
  R1 - Similitud textual (6 algoritmos)
  R2 - Clustering jerárquico (3 fuentes, 3 enlaces, evaluación)

Uso:
  python main.py
"""

import sys
import traceback


def run_r1() -> None:
    """Ejecuta el Requerimiento 1: Similitud textual."""
    print("=" * 60)
    print("INICIO REQUERIMIENTO 1 - Similitud Textual")
    print("=" * 60)

    from src.services.similarity_service import get_service

    svc = get_service()
    docs = svc.list_documents()

    print(f"Documentos cargados: {len(docs)} (A01..A{len(docs):02d})")
    print()

    # Caso de estudio con los 2 primeros documentos
    if len(docs) >= 2:
        a, b = docs[0]["id"], docs[1]["id"]
        print(f"Caso de estudio: {a} vs {b}")
        pair = svc.compare_pair(a, b, trace=True)
        for name, res in pair.results.items():
            print(f"  {name}: {res.similarity:.4f}")
        print()

    # Matriz 3x3 con los 3 primeros documentos
    if len(docs) >= 3:
        ids = [d["id"] for d in docs[:3]]
        print(f"Matriz 3x3: {ids}")
        mat = svc.compare_matrix(ids)
        for name, m in mat.matrices.items():
            print(f"  {name}:")
            for row in m:
                print(f"    {[f'{x:.3f}' for x in row]}")
        print()

    print("FIN REQUERIMIENTO 1")
    print("=" * 60)
    print()


def run_r2() -> None:
    """Ejecuta el Requerimiento 2: Clustering jerárquico."""
    print("=" * 60)
    print("INICIO REQUERIMIENTO 2 - Clustering Jerárquico")
    print("=" * 60)

    from src.services.clustering_service import get_clustering_service, DistanceSource

    svc = get_clustering_service()
    sources = svc.list_sources()

    print(f"Fuentes de distancia disponibles: {sources}")
    print(f"Documentos: {len(svc.doc_ids)}")
    print()

    for source in sources:
        print(f"--- Fuente: {source} ---")
        try:
            src: DistanceSource = source  # type: ignore[assignment]
            output = svc.run_clustering(src, k=3)

            print(f"  Mejor enlace (mayor silueta promedio k=2..6): {output.best_linkage}")
            print("  Métricas por enlace:")
            for method, metrics in output.metrics.items():
                sil_str = ", ".join(f"k={k}:{v:.4f}" for k, v in metrics["silhouettes"].items())
                print(f"    {method}: cophenetic={metrics['cophenetic_corr']:.4f}, "
                      f"siluetas=[{sil_str}], best_k={metrics['best_k']}")

            print("  Clusters k=3:")
            for method, clusters in output.clusters_at_k.items():
                cluster_sizes = [len(c) for c in clusters]
                print(f"    {method}: {clusters} (tamaños: {cluster_sizes})")

            print(f"  Dendrogramas generados: {list(output.dendrograms.keys())}")
            print()

        except FileNotFoundError as e:
            print(f"  [AVISO] {e}")
            print()
        except Exception as e:
            print(f"  [ERROR] Fallo en fuente '{source}': {e}")
            traceback.print_exc()
            print()

    print("FIN REQUERIMIENTO 2")
    print("=" * 60)


def main() -> int:
    """Función principal: ejecuta R1 y R2 secuencialmente."""
    print()
    print(">>> finalAlgoritmos - Ejecución completa R1 + R2")
    print()

    # R1: Similitud textual
    try:
        run_r1()
    except Exception as e:
        print(f"[ERROR] Requerimiento 1 falló: {e}")
        traceback.print_exc()
        print("=" * 60)
        return 1

    # R2: Clustering jerárquico
    try:
        run_r2()
    except Exception as e:
        print(f"[ERROR] Requerimiento 2 falló: {e}")
        traceback.print_exc()
        print("=" * 60)
        return 1

    print()
    print(">>> EJECUCIÓN COMPLETA EXITOSA")
    return 0


if __name__ == "__main__":
    sys.exit(main())