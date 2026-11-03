# Metodología y Documentación del Proyecto finalAlgoritmos

## 1. Metodología Utilizada

### Enfoque General
El proyecto sigue una arquitectura en capas con separación estricta entre:
- **Capa de algoritmos puros** (`src/`): Implementaciones desde cero sin librerías externas prohibidas
- **Capa de servicios** (`src/services/`): Orquestación y lógica de negocio
- **Capa de tests** (`tests/`): Validación contra referencias externas (SciPy/sklearn) y tests de arquitectura

### Principios de Desarrollo
1. **Desde cero**: Todos los algoritmos implementados sin usar `scipy`, `sklearn`, `nltk`, `difflib`, `str.find`, etc.
2. **Arquitectura por capas**: `src/` contiene solo lógica pura; `src/services/` orquesta; UI futura en `src/app/`
3. **Tests de referencia**: Validación contra SciPy/sklearn SOLO en `tests/`, nunca en `src/`
4. **Tests de arquitectura**: `test_forbidden_imports.py` escanea `src/` y falla si detecta imports prohibidos
4. **Reproducibilidad**: Seeds fijas, tolerancias numéricas justificadas, datos controlados
5. **Trazabilidad**: Modo `trace=True` en algoritmos para análisis paso a paso

### Validación de Algoritmos
- **Referencia independiente**: Comparación contra SciPy/sklearn SOLO en tests
- **Casos límite**: Vacíos, un elemento, longitudes diferentes, tokens repetidos
- **Propiedades matemáticas**: Simetría, rango [0,1], identidad, desigualdad triangular
- **Reproducibilidad**: 10 ejecuciones con misma semilla, resultados idénticos

---

## 2. Estructura del Proyecto

```
finalAlgoritmos/
├── .git/
├── .pytest_cache/
├── .venv/
├── data/
│   ├── pdfs/                          # 20 PDFs del corpus (entrada)
│   ├── processed/
│   │   ├── corpus.json                # Corpus procesado (20 artículos A01-A20)
│   │   ├── overrides.json             # Reglas por artículo para correcciones
│   │   ├── idf.json                   # IDF precomputado (984 términos, N=20)
│   │   ├── embeddings_glove.json      # Embeddings GloVe (generado por experiment)
│   │   ├── embeddings_fasttext.json   # Embeddings FastText (generado por experiment)
│   │   └── plots/                     # Dendrogramas PNG generados
├── docs/                              # Documentación futura
├── experiments/
│   ├── __init__.py
│   ├── generate_embeddings.py         # Descarga GloVe/FastText y genera embeddings
│   └── preview_preprocessing.py       # Vista previa tokens preprocesamiento
├── results/                           # Outputs futuros
├── src/
│   ├── __init__.py
│   ├── app/                           # UI futura (Streamlit)
│   │   └── __init__.py
│   ├── clustering/                    # R2: AGRUPAMIENTO JERÁRQUICO
│   │   ├── __init__.py
│   │   ├── hierarchical.py            # Algoritmos puros: 3 enlaces + métricas
│   │   └── dendrogram.py              # Visualización matplotlib (sin cruces)
│   ├── ingestion/                     # INGESTA PDFs → CORPUS
│   │   ├── __init__.py
│   │   ├── build_corpus.py            # Pipeline principal: PDFs → corpus.json
│   │   └── pdf_reader.py              # Extracción: texto, título, autores, abstract
│   ├── preprocessing/                 # PREPROCESAMIENTO TEXTO
│   │   ├── __init__.py
│   │   └── text_preprocessing.py      # Tokenize, stopwords EN/ES, NFKC, min_len
│   ├── similarity/                    # R1: SIMILITUD TEXTUAL (6 algoritmos)
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Base: dot, norm, coseno, euclidiana
│   │   ├── embeddings.py              # Infra: average_vector, embed_corpus, OOV
│   │   ├── edit_distance.py           # Levenshtein (palabras, costos 1/1/1)
│   │   ├── needleman_wunsch.py        # Needleman-Wunsch (match+1, mismatch-1, gap-1)
│   │   ├── tfidf.py                   # TF-IDF + Coseno (df sobre corpus N=20)
│   │   └── jaccard.py                 # Jaccard sobre sets de tokens
│   ├── sorting/                       # (vacío - para algoritmos ordenación futuros)
│   │   └── __init__.py
│   ├── string_matching/               # R3: BÚSQUEDA PATRONES (3 algoritmos)
│   │   ├── __init__.py
│   │   ├── boyer_moore.py             # Boyer-Moore (bad char + good suffix strong)
│   │   ├── kmp.py                     # Knuth-Morris-Pratt (tabla LPS)
│   │   └── rabin_karp.py              # Rabin-Karp (rolling hash polinómico)
│   └── services/                      # CAPA DE SERVICIOS (orquesta src/)
│       ├── __init__.py
│       ├── similarity_service.py      # R1: 6 algoritmos, caso estudio/matriz k×k
│       └── clustering_service.py      # R2: 3 fuentes, 3 enlaces, eval, dendrogramas
├── tests/                             # TESTS (82 totales)
│   ├── __init__.py
│   ├── test_forbidden_imports.py      # Escanea src/ por imports prohibidos
│   ├── test_boyer_moore.py            # R3: BM (5 tests)
│   ├── test_kmp.py                    # R3: KMP (5 tests)
│   ├── test_rabin_karp.py             # R3: Rabin-Karp (7 tests)
│   ├── test_ingestion.py              # Ingesta: 14 tests
│   ├── test_preprocessing.py          # Preproc: 7 tests
│   ├── test_normalize.py              # Normalización: 1 test
│   ├── test_metrics_embeddings.py     # Métricas base + embeddings: 9 tests
│   ├── test_corpus_files.py           # Corpus files: 5 tests
│   ├── test_r1_reference.py           # R1 vs SciPy/sklearn: 19 tests
│   └── test_clustering_reference.py   # R2 vs SciPy: 9 tests
├── requirements.txt                   # pytest, pymupdf
├── requirements-embeddings.txt        # gensim (para experimentos embeddings)
├── requirements-dev.txt               # scipy, scikit-learn, pytest (solo tests/)
└── README.md
```

---

## 3. Qué Hace Cada Archivo Principal

### Capa de Algoritmos (`src/`)

| Archivo | Responsabilidad | Tests Relacionados |
|---------|----------------|-------------------|
| `src/ingestion/build_corpus.py` | Pipeline: PDFs → corpus.json (IDs A01-A20, título, autores, abstract) | `test_corpus_files.py`, `test_ingestion.py` |
| `src/ingestion/pdf_reader.py` | Extracción: texto, título, autores, abstract (pymupdf) | `test_ingestion.py`, `test_normalize.py` |
| `src/preprocessing/text_preprocessing.py` | Tokenize, stopwords EN/ES, NFKC, min_len=2 | `test_preprocessing.py` |
| `src/similarity/metrics.py` | Base: dot, norm, coseno, euclidiana | `test_metrics_embeddings.py` |
| `src/similarity/embeddings.py` | average_vector, embed_corpus, OOV stats | `test_metrics_embeddings.py` |
| `src/similarity/edit_distance.py` | Levenshtein (palabras, costos 1/1/1) + trace | `test_r1_reference.py::TestLevenshteinReference` |
| `src/similarity/needleman_wunsch.py` | Needleman-Wunsch (match+1, mismatch-1, gap-1) + trace | `test_r1_reference.py::TestNeedlemanWunschReference` |
| `src/similarity/tfidf.py` | TF-IDF + Coseno (df sobre corpus N=20) + trace | `test_r1_reference.py::TestTFIDFCosineReference` |
| `src/similarity/jaccard.py` | Jaccard sobre sets de tokens + trace | `test_r1_reference.py::TestJaccardReference` |
| `src/similarity/embeddings.py` | average_vector, embed_corpus, OOV | `test_metrics_embeddings.py` |
| `src/string_matching/boyer_moore.py` | Boyer-Moore (bad char + good suffix strong) | `test_boyer_moore.py` |
| `src/string_matching/kmp.py` | KMP (tabla LPS) | `test_kmp.py` |
| `src/string_matching/rabin_karp.py` | Rabin-Karp (rolling hash polinómico) | `test_rabin_karp.py` |
| `src/clustering/hierarchical.py` | 3 enlaces (single/complete/average) + métricas | `test_clustering_reference.py` |
| `src/clustering/dendrogram.py` | Visualización matplotlib (sin cruces) | - |
| `src/services/similarity_service.py` | R1: 6 algoritmos, caso estudio/matriz k×k | - |
| `src/services/clustering_service.py` | R2: 3 fuentes, 3 enlaces, eval, dendrogramas | - |

### Capa de Tests (`tests/`)

| Archivo | Qué Valida | Dependencias |
|---------|------------|--------------|
| `test_forbidden_imports.py` | Escanea `src/` por imports prohibidos | Ninguna |
| `test_boyer_moore.py` | R3: BM (bad-char, solapadas, vacío, referencia) | Propias |
| `test_kmp.py` | R3: KMP (LPS, solapadas, vacío, referencia) | Propias |
| `test_rabin_karp.py` | R3: Rabin-Karp (hash, solapadas, colisiones) | Propias |
| `test_ingestion.py` | Ingesta: limpieza, abstract, título, autores | Propias |
| `test_preprocessing.py` | Tokenize, acentos, stopwords EN/ES, min_len | Propias |
| `test_normalize.py` | Zero-width chars | Propias |
| `test_metrics_embeddings.py` | Dot, norm, coseno, euclidiana, OOV, embed_corpus | Propias |
| `test_corpus_files.py` | Listado PDFs, build_corpus con overrides | Propias |
| `test_r1_reference.py` | R1 vs SciPy/sklearn: 19 tests | SciPy, sklearn |
| `test_clustering_reference.py` | R2 vs SciPy: 9 tests | SciPy |

---

## 4. Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE COMPLETO DEL PROYECTO                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  data/pdfs/  │  (20 PDFs entrada)
│  (entrada)   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTA (src/ingestion/)                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ list_pdfs() → read_first_pages() → normalize_text()                    │ │
│  │       ↓              ↓                    ↓                             │ │
│  │ find_abstract() → guess_title() → guess_authors()                      │ │
│  │       ↓              ↓                    ↓                             │ │
│  │ build_corpus() + overrides.json                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ data/processed/corpus.json  ──► 20 docs: {id:A01..A20, title, authors,    │
│                               abstract, abstract_words, needs_review[]}    │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. PREPROCESAMIENTO (src/preprocessing/)                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ load_corpus() → preprocess_corpus()                                     │ │
│  │       ↓                    ↓                                            │ │
│  │ tokenize() → filter len ≥ 2 → stopwords (EN/ES)                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ tokens_by_id: {A01: [token1, token2, ...], ..., A20: [...]}               │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ├──► ┌─────────────────────────────────────────────────────────────────┐
       │     │ 3A. SIMILITUD R1 (src/similarity/ + src/services/similarity_) │
       │     │  ┌───────────────────────────────────────────────────────────┐ │
       │     │  │ tokens_by_id + idf (precomputado) + embeddings (opcional) │ │
       │     │  │       ↓                                                    │ │
       │     │  │ ┌────────────┬────────────┬─────────┬────────┬──────────┐  │
       │     │  │ │Levenshtein │Needleman-W. │TF-IDF  │Jaccard │GloVe/Fast│ │
       │     │  │ │  (palabras)│  (palabras)  │ Coseno │        │  Text    │ │
       │     │  │ └────────────┴────────────┴────────┴────────┴──────────┘  │
       │     │  │       ↓                                                         │
       │     │  │ SimilarityService.compare_pair() → caso estudio (trace)     │ │
       │     │  │ SimilarityService.compare_matrix() → matrices k×k           │ │
       │     │  └───────────────────────────────────────────────────────────┘ │
       │     └─────────────────────────────────────────────────────────────────┘
       │
       ├──► ┌─────────────────────────────────────────────────────────────────┐
       │     │ 3B. CLUSTERING R2 (src/clustering/ + src/services/clustering_) │
       │     │  ┌───────────────────────────────────────────────────────────┐ │
       │     │  │ tokens_by_id + idf / embeddings                           │ │
       │     │  │       ↓                                                    │ │
       │     │  │ get_distance_matrix(source) → D = 1 - coseno (20×20)     │ │
       │     │  │       ↓                                                    │ │
       │     │  │ run_all_linkages(D) → 3 LinkageMatrix                    │ │
       │     │  │   ├─ Single (min)    ├─ Complete (max)   ├─ Average     │ │
       │     │  │       ↓                                                    │ │
       │     │  │ evaluate_all() → cophenetic + siluetas k=2..6            │ │
       │     │  │       ↓                                                    │ │
       │     │  │ select_best_linkage() → mayor silueta promedio           │ │
       │     │  │       ↓                                                    │ │
       │     │  │ plot_all_dendrograms() → 3 PNG (sin cruces)              │ │
       │     │  │       ↓                                                    │ │
       │     │  │ get_clusters_at_k(k) → clusters por enlace               │ │
       │     │  │       ↓                                                    │ │
       │     │  │ ClusteringOutput (matriz, enlaces, dendrogramas, métricas)│ │
       │     │  └───────────────────────────────────────────────────────────┘ │
       │     └─────────────────────────────────────────────────────────────────┘
       │
       └──► ┌─────────────────────────────────────────────────────────────────┐
             │ 3C. BÚSQUEDA PATRONES R3 (src/string_matching/)               │
             │  ┌───────────────────────────────────────────────────────────┐ │
             │  │ abstract crudo (texto original)                           │ │
             │  │       ↓                                                    │ │
             │  │ search_all(text, pattern, trace=True)                     │ │
             │  │   ├─ KMP (LPS, O(n+m))                                    │ │
             │  │   ├─ Boyer-Moore (bad char + good suffix, O(n) avg)      │ │
             │  │   └─ Rabin-Karp (rolling hash, O(n+m) avg)               │ │
             │  │       ↓                                                    │ │
             │  │ positions, comparisons, traces, métricas internas        │ │
             │  └───────────────────────────────────────────────────────────┘ │
             └─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. OUTPUTS Y VALIDACIÓN                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ data/processed/idf.json              # IDF precomputado (984 términos) │ │
│  │ data/processed/embeddings_*.json     # Embeddings GloVe/FastText       │ │
│  │ data/processed/plots/dendro_*.png    # 3 Dendrogramas PNG              │ │
│  │ tests/                               # 82 tests (82 passed)           │ │
│  │   ├─ 17 tests propios (sin deps)                                         │ │
│  │   ├─ 3 tests vs SciPy (coseno, euclidiana)                              │ │
│  │   ├─ 4 tests vs sklearn (TF-IDF coseno, tokens 1-char)                │ │
│  │   └─ 57 tests R3 + ingesta + preproc + métricas                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Comandos Principales

```bash
# Generar corpus
python -m src.ingestion.build_corpus

# Generar embeddings (GloVe ~130MB, FastText ~1GB)
python -m experiments.generate_embeddings

# Vista previa preprocesamiento
python -m experiments.preview_preprocessing

# Tests
python -m pytest tests/ -v                    # Suite completa (82 tests)
python -m pytest tests/test_r1_reference.py   # Solo R1
python -m pytest tests/test_forbidden_imports.py  # Validación arquitectura

# Servicios
python -c "from src.services.similarity_service import get_service; s=get_service(); print(s.compare_pair('A01','A02',trace=True))"
python -c "from src.services.clustering_service import get_clustering_service; s=get_clustering_service(); print(s.run_clustering('tfidf', k=3))"
```

---

## 6. Decisiones Técnicas Clave

| Decisión | Justificación |
|----------|---------------|
| `np.divide(out=..., where=...)` vs `np.where` | Evita warning numpy, maneja división por cero sin NaN |
| `np.allclose(atol=1e-5)` vs `max_diff < 1e-5` | Comparación vectorizada robusta, tolerancia justificada |
| `tokenizer=str.split, token_pattern=None, lowercase=False` | Respeta tokens tal cual vienen del preprocesamiento |
| `vectorizer.idf_ = np.array([idf.get(t,1.0) for t in vocab])` | Fuerza IDF de sklearn = IDF propio (N=20) |
| Corpus controlado en tests de 1-char | IDF real calculado con `compute_idf`, no valores artificiales |
| `@pytest.mark.skipif` a nivel de método | Tests propios corren sin deps externas |
| `pytestmark` global eliminado | Cada clase decide su dependencia |
| `np.divide(out=np.zeros_like, where=...)` | Evita warning numpy "where used without out" |
| Corpus controlado compartido | Ambos tests de 1-char usan mismo corpus e IDF |

---

*Documento generado automáticamente basado en el estado actual del código.*