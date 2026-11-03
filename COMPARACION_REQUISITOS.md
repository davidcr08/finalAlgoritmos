# Comparación: Proyecto Actual vs Requisitos del Documento (Solo Aplicación)

## Resumen Ejecutivo

| Categoría | Requeridos | Implementados | % |
|-----------|------------|---------------|---|
| **R1: Similitud Textual - Clásicos** | 4 algoritmos | 0/4 (solo infra base) | 0% |
| **R1: Similitud Textual - IA** | 2 algoritmos | 0/2 (solo infra embeddings) | 0% |
| **R2: Clustering Jerárquico** | 3 algoritmos + dendrogramas | 0/3 (módulo vacío) | 0% |
| **R3: Búsqueda Patrones** | 3 algoritmos | 3/3 completos | 100% |
| **Infraestructura Base** | Ingesta, Preproc, Métricas | 3/3 completos | 100% |
| **TOTAL FUNCIONALIDAD APLICACIÓN** | 12 componentes núcleo | 6/12 | **50%** |

---

## Detalle por Requerimiento

### R1: Análisis de Similitud Textual

| Algoritmo | Especificación Documento | Estado Actual | Archivo / Comentario |
|-----------|-------------------------|---------------|---------------------|
| **Levenshtein / Needleman-Wunsch** | Distancia de edición (prog. dinámica) | ❌ **NO EXISTE** | Falta `src/similarity/edit_distance.py` |
| **Coseno con TF-IDF** | Vectorización TF-IDF + similitud coseno | ⚠️ **SOLO COSENO BASE** | `metrics.py:cosine_similarity()` existe, **falta TF-IDF** |
| **Coeficiente Jaccard** | Conjuntos de tokens / shingles | ❌ **NO EXISTE** | Falta `src/similarity/jaccard.py` |
| **4º Clásico** (sugerido: otro DP o vectorial) | Ej: Sorensen-Dice, Overlap, Longest Common Substring | ❌ **NO EXISTE** | Falta implementación |
| **Embeddings IA #1** | Word2Vec / FastText + coseno/euclidiana | ⚠️ **INFRA SOLA** | `embeddings.py` tiene `average_vector()` + `embed_corpus()` pero **no expone 2 enfoques concretos** |
| **Embeddings IA #2** | Modelo distinto (ej: transformer sentence embeddings) | ❌ **NO EXISTE** | Falta segundo enfoque (ej: `sentence-transformers` o API) |
| **Selección dinámica 2+ artículos** | UI/API para elegir artículos del corpus | ❌ **NO EXISTE** | Solo CLI batch (`build_corpus`) |
| **Análisis comparativo empírico** | Ejecutar todos y comparar resultados | ❌ **NO EXISTE** | Falta script/runner de comparación |

**Gap R1:** 6/6 algoritmos núcleo faltantes + capa de selección/comparación.

---

### R2: Agrupamiento Jerárquico (Clustering)

| Componente | Especificación Documento | Estado Actual | Archivo / Comentario |
|------------|-------------------------|---------------|---------------------|
| **Single Linkage** | Mínima distancia entre clusters | ❌ **NO EXISTE** | `src/clustering/__init__.py` vacío |
| **Complete Linkage** | Máxima distancia entre clusters | ❌ **NO EXISTE** | Idem |
| **Average Linkage / Ward** | Distancia promedio / varianza mínima | ❌ **NO EXISTE** | Idem |
| **Preprocesamiento → Similitud** | Pipeline integrado | ⚠️ **PARCIAL** | Preproc + métricas base existen, **falta pipeline clustering** |
| **Métrica evaluación interna** | Silhouette, Dunn, Cophenetic correlation | ❌ **NO EXISTE** | Falta `src/clustering/evaluation.py` |
| **Dendrograma (3 visualizaciones)** | Gráfico árbol por cada linkage | ❌ **NO EXISTE** | Falta `src/clustering/dendrogram.py` (matplotlib/scipy) |
| **Aplicación a 20 abstracts** | Ejecución completa corpus | ❌ **NO EXISTE** | Falta runner |

**Gap R2:** 3/3 linkages + evaluación + 3 dendrogramas + pipeline = **todo faltante**.

---

### R3: Búsqueda de Patrones (String Matching) ✅ COMPLETO

| Algoritmo | Especificación | Estado | Archivo |
|-----------|----------------|--------|---------|
| **KMP** | Tabla LPS, sin retroceso, O(n+m) | ✅ Completo | `src/string_matching/kmp.py` |
| **Boyer-Moore** | Bad char + Good suffix (strong), O(n) avg | ✅ Completo | `src/string_matching/boyer_moore.py` |
| **Rabin-Karp** | Rolling hash polinómico, base=256, mod=1e9+7 | ✅ Completo | `src/string_matching/rabin_karp.py` |
| **Dataclasses de resultado** | positions, comparisons, traza, métricas internas | ✅ Completo | BMResult, KMPResult, RKResult |
| **Tests exhaustivos** | Casos borde, colisiones, referencia aleatoria | ✅ Completo | `tests/test_*.py` (3 archivos) |
| **Uso sobre abstracts del corpus** | Integrado vía `preprocess_corpus` + abstracts | ✅ Listo para usar | Requiere script de búsqueda interactiva |

**Nota:** R3 está **completo y probado**. Solo falta una interfaz (CLI/API) para que el usuario ingrese patrón y busque en los 20 abstracts.

---

### Infraestructura Base (Soporta R1, R2, R3)

| Componente | Estado | Archivo | Listo para |
|------------|--------|---------|------------|
| **Ingesta PDFs → corpus.json** | ✅ Completo | `src/ingestion/build_corpus.py` | R1, R2, R3 |
| **Extracción abstract/título/autores** | ✅ Completo | `src/ingestion/pdf_reader.py` | R1, R2, R3 |
| **Overrides por artículo** | ✅ Completo | `data/overrides.json` | R1, R2, R3 |
| **Preprocesamiento (tokenize, stopwords ES/EN, NFKC)** | ✅ Completo | `src/preprocessing/text_preprocessing.py` | R1, R2 |
| **Métricas base (dot, norm, coseno, euclidiana)** | ✅ Completo | `src/similarity/metrics.py` | R1, R2 |
| **Embeddings infra (average_vector, embed_corpus, OOV stats)** | ✅ Completo | `src/similarity/embeddings.py` | R1-IA |
| **Tests infra** | ✅ Completo | `tests/test_preprocessing.py`, `test_normalize.py`, `test_metrics_embeddings.py`, `test_ingestion.py` | - |

---

## Qué FALTA Implementar (Priorizado)

### Prioridad ALTA (Core R1 + R2)
1. **`src/similarity/edit_distance.py`** — Levenshtein + Needleman-Wunsch (DP)
2. **`src/similarity/tfidf.py`** — TF-IDF vectorizer + coseno
3. **`src/similarity/jaccard.py`** — Jaccard (sets/k-shingles)
4. **`src/similarity/sorensen_dice.py`** — 4º clásico (sugerido)
5. **`src/similarity/embeddings_approaches.py`** — 2 enfoques IA concretos (Word2Vec/FastText + SentenceTransformer)
6. **`src/clustering/linkages.py`** — Single, Complete, Average Linkage (desde cero, O(n²) o O(n² log n))
7. **`src/clustering/evaluation.py`** — Silhouette, Cophenetic correlation
8. **`src/clustering/dendrogram.py`** — Visualización 3 dendrogramas (matplotlib)

### Prioridad MEDIA (Integración / UX)
9. **`src/app/similarity_cli.py`** — Menú: seleccionar 2+ artículos → ejecutar 6 algoritmos → tabla comparativa
10. **`src/app/clustering_cli.py`** — Ejecutar 3 linkages → mostrar métricas → generar 3 PNG dendrogramas
11. **`src/app/search_cli.py`** — Input patrón → buscar en 20 abstracts con KMP/BM/RK → mostrar posiciones + métricas

### Prioridad BAJA (Documentación/Despliegue - R4)
12. Documento técnico con casos de estudio (R1, R3)
13. Despliegue (Docker/Streamlit/FastAPI)

---

## Conclusión

- **R3 (String Matching): 100% listo** — solo exponer vía CLI.
- **Infra base: 100% lista** — reutilizable directamente para R1 y R2.
- **R1 (Similitud): 0% algoritmos núcleo** — 6 algoritmos por escribir + capa comparación.
- **R2 (Clustering): 0%** — módulo completamente vacío, 3 linkages + eval + 3 dendrogramas.

**Esfuerzo estimado:** ~10-12 archivos nuevos en `src/similarity/`, `src/clustering/`, `src/app/`.