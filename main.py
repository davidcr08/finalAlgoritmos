#!/usr/bin/env python3
"""
Punto de entrada general del proyecto finalAlgoritmos.

Menú interactivo para ejecutar:
  R1 - Similitud textual (6 algoritmos)
  R2 - Clustering jerárquico (3 fuentes, 3 enlaces, evaluación)
  R3 - Búsqueda de patrones de texto (KMP, Boyer-Moore, Rabin-Karp)

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


def run_r3() -> None:
    """Ejecuta el Requerimiento 3: Búsqueda de patrones de texto."""
    from src.services.string_matching_service import get_service

    svc = get_service()

    while True:
        print("=" * 60)
        print("       R3 - BÚSQUEDA DE PATRONES DE TEXTO")
        print("=" * 60)
        print()
        print("1. Buscar patrón en los 20 abstracts")
        print("2. Demostración académica paso a paso")
        print("0. Volver al menú principal")
        print()
        choice = input("Seleccione una opción: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            run_r3_search(svc)
        elif choice == "2":
            run_r3_demo(svc)
        else:
            print("\n[ERROR] Opción inválida. Intente de nuevo.\n")


def run_r3_search(svc) -> None:
    """Opción 1: Búsqueda general en los 20 abstracts."""
    print()
    pattern = input("Ingrese el patrón a buscar: ").strip()
    if not pattern:
        print("[AVISO] Patrón vacío. No se realizó búsqueda.\n")
        return

    print(f"\nBuscando '{pattern}' en los 20 abstracts...\n")

    try:
        results, summary = svc.search_all(pattern)

        # Resultados por documento
        print("RESULTADOS POR DOCUMENTO")
        print("-" * 60)
        for r in results:
            print(f"Documento: {r.doc_id} - {r.title[:70]}")
            if r.count > 0:
                print(f"  Ocurrencias: {r.count}")
                print(f"  Posiciones (base 0): {r.positions_kmp}")
                # Verificar consistencia
                if r.positions_kmp != r.positions_bm or r.positions_kmp != r.positions_rk:
                    print(f"  [ADVERTENCIA] Discrepancia: KMP={r.positions_kmp}, BM={r.positions_bm}, RK={r.positions_rk}")
            else:
                print("  Sin coincidencias")
            print()

        # Resumen comparativo
        print("RESUMEN COMPARATIVO")
        print("-" * 60)
        print(f"Total de ocurrencias encontradas: {summary.total_occurrences}")
        print(f"Documentos con al menos una ocurrencia: {summary.docs_with_matches}")
        print()
        print("Métricas de rendimiento:")
        print(f"  KMP - Comparaciones: {summary.kmp_comparisons}, "
              f"Preprocesamiento: {summary.kmp_preprocess_time*1000:.3f} ms, "
              f"Búsqueda: {summary.kmp_search_time*1000:.3f} ms")
        print(f"  Boyer-Moore - Comparaciones: {summary.bm_comparisons}, "
              f"Preprocesamiento: {summary.bm_preprocess_time*1000:.3f} ms, "
              f"Búsqueda: {summary.bm_search_time*1000:.3f} ms")
        print(f"  Rabin-Karp - Comparaciones char: {summary.rk_comparisons}, "
              f"Hash comparisons: {summary.rk_hash_comparisons}, "
              f"Colisiones: {summary.rk_collisions}, "
              f"Preprocesamiento: {summary.rk_preprocess_time*1000:.3f} ms, "
              f"Búsqueda: {summary.rk_search_time*1000:.3f} ms")
        print()

        # Análisis de complejidad teórica
        print_teoretical_analysis(pattern, results)

    except Exception as e:
        print(f"[ERROR] Fallo en búsqueda: {e}")
        traceback.print_exc()
        print()


def run_r3_demo(svc) -> None:
    """Opción 2: Demostración académica paso a paso."""
    docs = svc.list_documents()

    print()
    print("DOCUMENTOS DISPONIBLES")
    print("-" * 60)
    for i, d in enumerate(docs, 1):
        print(f"  {i:2d}. {d['id']} - {d['title'][:70]}")
    print()

    try:
        idx = int(input("Seleccione documento (1-20): ").strip()) - 1
        if not (0 <= idx < len(docs)):
            print("[ERROR] Índice fuera de rango.\n")
            return
        doc = docs[idx]
    except ValueError:
        print("[ERROR] Entrada inválida.\n")
        return

    pattern = input("Ingrese el patrón a buscar: ").strip()
    if not pattern:
        print("[AVISO] Patrón vacío.\n")
        return

    print(f"\nEjecutando demostración académica para {doc['id']} con patrón '{pattern}'...\n")

    try:
        demo = svc.demo(doc["id"], pattern)
        print_demo_kmp(demo)
        print_demo_boyer_moore(demo)
        print_demo_rabin_karp(demo)
        print_complexity_analysis(demo)
    except Exception as e:
        print(f"[ERROR] Fallo en demostración: {e}")
        traceback.print_exc()
        print()


def print_demo_kmp(demo) -> None:
    """Demostración académica KMP."""
    print("=" * 70)
    print("A. DEMOSTRACIÓN KMP (Knuth-Morris-Pratt)")
    print("=" * 70)
    res = demo.kmp_result
    pattern = demo.pattern
    text = demo.abstract

    print(f"\n1. Patrón seleccionado: '{pattern}' (longitud m={len(pattern)})")
    print(f"   Texto: abstract de {demo.doc_id} (longitud n={len(text)})")

    print(f"\n2. Tabla LPS (Longest Proper Prefix which is also Suffix):")
    print(f"   pattern:  {' '.join(f'{c:>2}' for c in pattern)}")
    print(f"   índice:   {' '.join(f'{i:>2}' for i in range(len(pattern)))}")
    print(f"   LPS:      {' '.join(f'{v:>2}' for v in res.lps)}")

    print("\n3. Explicación de la tabla LPS:")
    for i, val in enumerate(res.lps):
        if val > 0:
            prefix = pattern[:val]
            print(f"   LPS[{i}] = {val}: prefijo '{prefix}' = sufijo de pattern[0..{i}]")

    print(f"\n4. Recorrido del texto (primeros 30 pasos de {len(res.steps)}):")
    for step in res.steps[:30]:
        i, j, event = step
        if event == "coincide":
            print(f"   i={i:4d}, j={j:2d}: COINCIDE  text[{i}]='{text[i]}' == pattern[{j}]='{pattern[j]}'")
        elif event == "falla":
            print(f"   i={i:4d}, j={j:2d}: FALLA     text[{i}]='{text[i]}' != pattern[{j}]='{pattern[j]}' -> j=LPS[{j-1}]={res.lps[j-1] if j>0 else 0}")
        elif event == "ocurrencia":
            print(f"   i={i:4d}, j={j:2d}: OCURRENCIA en posición {i-j}")
    if len(res.steps) > 30:
        print(f"   ... ({len(res.steps) - 30} pasos más)")

    print(f"\n5. Comparaciones totales: {res.comparisons}")
    print(f"6. Posiciones finales (base 0): {res.positions}")

    # Ventana de contexto para cada ocurrencia
    for pos in res.positions:
        start = max(0, pos - 20)
        end = min(len(text), pos + len(pattern) + 20)
        print(f"   Contexto en [{pos}]: ...{text[start:pos]}[{text[pos:pos+len(pattern)]}]{text[pos+len(pattern):end]}...")


def print_demo_boyer_moore(demo) -> None:
    """Demostración académica Boyer-Moore."""
    print("\n" + "=" * 70)
    print("B. DEMOSTRACIÓN BOYER-MOORE")
    print("=" * 70)
    res = demo.bm_result
    pattern = demo.pattern
    text = demo.abstract
    m = len(pattern)

    print(f"\n1. Patrón seleccionado: '{pattern}' (longitud m={m})")
    print(f"   Texto: abstract de {demo.doc_id} (longitud n={len(text)})")

    print("\n2. Tabla de Carácter Malo (bad character):")
    for ch in sorted(res.bad_char.keys()):
        print(f"   '{ch}': última posición en patrón = {res.bad_char[ch]}")

    print("\n3. Tabla de Sufijo Bueno (good suffix, versión fuerte):")
    print(f"   shift[j] = desplazamiento cuando falla en j-1 (sufijo pattern[j:] ya coincidió)")
    print(f"   shift[0] = desplazamiento tras ocurrencia completa")
    print(f"   Índices: {' '.join(f'{i:>3}' for i in range(m+1))}")
    print(f"   Shift:   {' '.join(f'{v:>3}' for v in res.good_suffix)}")

    print("\n4. Comparaciones de derecha a izquierda y desplazamientos:")
    for step in res.steps[:30]:
        s, j, event, jump = step
        if event == "ocurrencia":
            print(f"   s={s:4d}: OCURRENCIA en posición {s} -> salto good_suffix[0]={jump}")
        elif event == "falla":
            bc_char = text[s + j] if s + j < len(text) else '?'
            bc_shift = j - res.bad_char.get(bc_char, -1)
            gs_shift = res.good_suffix[j + 1]
            rule = "bad_char" if bc_shift >= gs_shift else "good_suffix"
            print(f"   s={s:4d}, j={j:2d}: FALLA text[{s+j}]='{bc_char}' vs pattern[{j}]='{pattern[j]}'")
            print(f"         bad_char shift={bc_shift}, good_suffix shift={gs_shift} -> usa {rule} -> salto={jump}")
    if len(res.steps) > 30:
        print(f"   ... ({len(res.steps) - 30} pasos más)")

    print(f"\n5. Comparaciones totales: {res.comparisons}")
    print(f"6. Posiciones finales (base 0): {res.positions}")

    for pos in res.positions:
        start = max(0, pos - 20)
        end = min(len(text), pos + len(pattern) + 20)
        print(f"   Contexto en [{pos}]: ...{text[start:pos]}[{text[pos:pos+len(pattern)]}]{text[pos+len(pattern):end]}...")


def print_demo_rabin_karp(demo) -> None:
    """Demostración académica Rabin-Karp."""
    print("\n" + "=" * 70)
    print("C. DEMOSTRACIÓN RABIN-KARP")
    print("=" * 70)
    res = demo.rk_result
    pattern = demo.pattern
    text = demo.abstract
    m = len(pattern)
    n = len(text)

    print(f"\n1. Patrón seleccionado: '{pattern}' (longitud m={m})")
    print(f"   Texto: abstract de {demo.doc_id} (longitud n={n})")

    print(f"\n2. Parámetros:")
    print(f"   Base (B): {res.base}")
    print(f"   Módulo (Q): {res.mod}")

    print(f"\n3. Hash del patrón: {res.pattern_hash}")
    print(f"   Cálculo: hash('{pattern}') = {res.pattern_hash} (mod {res.mod})")

    # Calcular hash de la primera ventana para mostrar
    if m <= n:
        first_window = text[:m]
        first_hash = res.steps[0][1] if res.steps else "N/A"
        print(f"\n4. Hash de la primera ventana text[0:{m}] = '{first_window}': {first_hash}")

    print(f"\n5. Rolling hash y búsqueda (primeras 20 ventanas de {res.hash_comparisons}):")
    for i, step in enumerate(res.steps[:20]):
        idx, w_hash, event = step
        if event == "ocurrencia":
            print(f"   Ventana {idx:4d}: hash={w_hash} == pattern_hash -> VERIFICACIÓN OK -> OCURRENCIA en {idx}")
        elif event == "colision":
            print(f"   Ventana {idx:4d}: hash={w_hash} == pattern_hash -> VERIFICACIÓN FALLA -> COLISIÓN")
        elif event == "hash distinto" and i < 5:  # solo mostrar primeros hash distintos
            print(f"   Ventana {idx:4d}: hash={w_hash} != {res.pattern_hash} -> hash distinto")

    if len(res.steps) > 20:
        print(f"   ... ({len(res.steps) - 20} pasos más)")

    print(f"\n6. Métricas:")
    print(f"   Comparaciones de hash (ventanas): {res.hash_comparisons}")
    print(f"   Comparaciones carácter a carácter (verificación): {res.comparisons}")
    print(f"   Colisiones (hash igual, texto distinto): {res.collisions}")
    print(f"7. Posiciones finales (base 0): {res.positions}")

    for pos in res.positions:
        start = max(0, pos - 20)
        end = min(len(text), pos + len(pattern) + 20)
        print(f"   Contexto en [{pos}]: ...{text[start:pos]}[{text[pos:pos+len(pattern)]}]{text[pos+len(pattern):end]}...")


def print_complexity_analysis(demo) -> None:
    """Análisis de complejidad teórica vs observado."""
    print("\n" + "=" * 70)
    print("ANÁLISIS DE COMPLEJIDAD TEÓRICA VS RENDIMIENTO OBSERVADO")
    print("=" * 70)

    n = len(demo.abstract)
    m = len(demo.pattern)

    print(f"\nParámetros del caso de estudio:")
    print(f"  n (longitud del texto) = {n}")
    print(f"  m (longitud del patrón) = {m}")
    print(f"  |Σ| (alfabeto observado) ≈ {len(set(demo.abstract))}")

    print("\n--- COMPLEJIDAD TEÓRICA ---")
    print("KMP:")
    print(f"  Preprocesamiento (tabla LPS): O(m) = O({m})")
    print(f"  Búsqueda: O(n) = O({n})")
    print(f"  Total: O(n + m)")

    print("\nBoyer-Moore (con bad character + good suffix fuerte):")
    print(f"  Preprocesamiento: O(m + |Σ|) = O({m} + {len(set(demo.abstract))})")
    print(f"  Búsqueda - Mejor caso: O(n/m) ≈ O({n}/{m}) = O({n//m if m else 'N/A'})")
    print(f"  Búsqueda - Promedio:O(n*m)")
    print(f"  Búsqueda - Peor caso (con good suffix fuerte): O(n*m)")
    print("  Nota: Sin optimización de Galil, el peor caso es O(n*m).")

    print("\nRabin-Karp:")
    print(f"  Preprocesamiento (hash del patrón + B^(m-1)): O(m) = O({m})")
    print(f"  Búsqueda - Promedio: O(n + m) = O({n} + {m})")
    print(f"  Búsqueda - Peor caso (muchas colisiones): O(n·m) = O({n}·{m})")

    print("\n--- RENDIMIENTO OBSERVADO (time.perf_counter) ---")
    print(f"KMP:")
    print(f"  Preprocesamiento: {demo.kmp_preprocess_time*1000:.4f} ms")
    print(f"  Búsqueda: {demo.kmp_search_time*1000:.4f} ms")
    print(f"  Comparaciones reales: {demo.kmp_result.comparisons}")

    print(f"\nBoyer-Moore:")
    print(f"  Preprocesamiento: {demo.bm_preprocess_time*1000:.4f} ms")
    print(f"  Búsqueda: {demo.bm_search_time*1000:.4f} ms")
    print(f"  Comparaciones reales: {demo.bm_result.comparisons}")

    print(f"\nRabin-Karp:")
    print(f"  Preprocesamiento: {demo.rk_preprocess_time*1000:.4f} ms")
    print(f"  Búsqueda: {demo.rk_search_time*1000:.4f} ms")
    print(f"  Comparaciones char: {demo.rk_result.comparisons}")
    print(f"  Comparaciones hash: {demo.rk_result.hash_comparisons}")
    print(f"  Colisiones: {demo.rk_result.collisions}")

    print("\n--- OBSERVACIONES ---")
    print("• Los tiempos son mediciones experimentales en una sola ejecución.")
    print("• No constituyen garantías de rendimiento general.")
    print("• KMP: comparaciones ≈ n (sin retroceso en texto)")
    print("• Boyer-Moore: comparaciones << n (saltos grandes en promedio)")
    print("• Rabin-Karp: hash comparisons ≈ n, char comparisons solo en verificaciones")


def print_teoretical_analysis(pattern: str, results: list) -> None:
    """Imprime análisis teórico breve para la búsqueda general."""
    from src.preprocessing.text_preprocessing import load_corpus
    corpus = load_corpus()
    total_text_len = sum(len(doc["abstract"]) for doc in corpus)
    m = len(pattern)

    print("ANÁLISIS DE COMPLEJIDAD TEÓRICA (resumen)")
    print("-" * 60)
    print(f"Corpus total: n ≈ {total_text_len} caracteres, patrón: m = {m}")
    print("KMP:           O(n + m) comparaciones, sin retroceso")
    print("Boyer-Moore:   O(n) promedio, O(n*m) peor caso")
    print("Rabin-Karp:    O(n + m) promedio, O(n·m) peor caso (colisiones)")
    print()
    print("NOTA: Estos son límites asintóticos. El rendimiento observado")
    print("      depende de la distribución del patrón en el texto.")
    print()


def main() -> int:
    """Función principal con menú interactivo."""
    print()
    print("=" * 50)
    print("           FINAL ALGORITMOS")
    print("=" * 50)

    while True:
        print()
        print("Seleccione el requerimiento que desea ejecutar:")
        print()
        print("1. R1 - Similitud textual")
        print("2. R2 - Clustering jerárquico")
        print("3. R3 - Búsqueda de patrones de texto")
        print("0. Salir")
        print()

        try:
            choice = input("Seleccione una opción: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSaliendo...")
            return 0

        if choice == "0":
            print("\nSaliendo...")
            return 0
        elif choice == "1":
            try:
                run_r1()
            except Exception as e:
                print(f"[ERROR] Requerimiento 1 falló: {e}")
                traceback.print_exc()
                print("=" * 60)
        elif choice == "2":
            try:
                run_r2()
            except Exception as e:
                print(f"[ERROR] Requerimiento 2 falló: {e}")
                traceback.print_exc()
                print("=" * 60)
        elif choice == "3":
            try:
                run_r3()
            except Exception as e:
                print(f"[ERROR] Requerimiento 3 falló: {e}")
                traceback.print_exc()
                print("=" * 60)
        else:
            print("\n[ERROR] Opción inválida. Intente de nuevo.")


if __name__ == "__main__":
    sys.exit(main())