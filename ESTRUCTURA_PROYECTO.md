# Estructura del Proyecto: finalAlgoritmos

## Árbol de Directorios

```
finalAlgoritmos/
├── .git/
├── .pytest_cache/
├── .venv/
├── data/
│   ├── pdfs/                    # PDFs del corpus (20 artículos esperados)
│   ├── processed/               # corpus.json generado
│   └── overrides.json           # Reglas por artículo para correcciones
├── docs/
├── experiments/
│   ├── __init__.py
│   ├── generate_embeddings.py   # Script: genera embeddings del corpus
│   └── preview_preprocessing.py # Script: vista previa del preprocesamiento
├── results/
├── src/
│   ├── __init__.py
│   ├── app/
│   │   └── __init__.py
│   ├── clustering/
│   │   └── __init__.py          # (vacío, pendiente)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── build_corpus.py      # Pipeline principal: PDFs -> corpus.json
│   │   └── pdf_reader.py        # Extracción: texto, título, autores, abstract
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── text_preprocessing.py # Tokenización, stopwords, normalización
│   ├── similarity/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # Promedio de vectores de palabras (GloVe/Word2Vec/FastText)
│   │   └── metrics.py           # Similitud coseno, distancia euclídea, producto punto
│   ├── sorting/
│   │   └── __init__.py          # (vacío, pendiente)
│   └── string_matching/
│       ├── __init__.py
│       ├── boyer_moore.py       # BM con regla del carácter malo + sufijo bueno fuerte
│       ├── kmp.py               # Knuth-Morris-Pratt con tabla LPS
│       └── rabin_karp.py        # Rabin-Karp con rolling hash polinómico
├── tests/
│   ├── __init__.py
│   ├── test_boyer_moore.py
│   ├── test_corpus_files.py
│   ├── test_ingestion.py
│   ├── test_kmp.py
│   ├── test_metrics_embeddings.py
│   ├── test_normalize.py
│   ├── test_preprocessing.py
│   └── test_rabin_karp.py
├── README.md
├── requirements.txt
└── requirements-embeddings.txt
```

---

## Clases Definidas

El proyecto usa **dataclasses** como contenedores de resultado para los algoritmos de búsqueda de strings. No hay clases de dominio complejas ni jerarquías de herencia.

### `src/string_matching/boyer_moore.py`

| Clase | Línea | Campos | Descripción |
|-------|-------|--------|-------------|
| **`BMResult`** | 16 | `positions: list[int]`<br>`comparisons: int`<br>`bad_char: dict[str, int]`<br>`good_suffix: list[int]`<br>`steps: list[tuple]` | Resultado de `boyer_moore_search()`. Incluye posiciones halladas, conteo de comparaciones, tablas de preprocesamiento y traza opcional de pasos. |

**Funciones principales:**
- `build_bad_char(pattern)` → `dict[str, int]`
- `build_good_suffix(pattern)` → `list[int]`
- `boyer_moore_search(text, pattern, trace=False)` → `BMResult`

---

### `src/string_matching/kmp.py`

| Clase | Línea | Campos | Descripción |
|-------|-------|--------|-------------|
| **`KMPResult`** | 10 | `positions: list[int]`<br>`comparisons: int`<br>`lps: list[int]`<br>`steps: list[tuple]` | Resultado de `kmp_search()`. Incluye posiciones, comparaciones, tabla LPS y traza opcional. |

**Funciones principales:**
- `build_lps(pattern)` → `list[int]`
- `kmp_search(text, pattern, trace=False)` → `KMPResult`

---

### `src/string_matching/rabin_karp.py`

| Clase | Línea | Campos | Descripción |
|-------|-------|--------|-------------|
| **`RKResult`** | 18 | `positions: list[int]`<br>`comparisons: int`<br>`hash_comparisons: int`<br>`collisions: int`<br>`pattern_hash: int`<br>`base: int`<br>`mod: int`<br>`steps: list[tuple]` | Resultado de `rabin_karp_search()`. Incluye métricas de hash, colisiones y traza opcional. |

**Funciones principales:**
- `power_mod(base, exp, mod)` → `int`
- `hash_of(s, base, mod)` → `int`
- `rabin_karp_search(text, pattern, base=256, mod=1_000_000_007, trace=False)` → `RKResult`

---

## Módulos Funcionales (sin clases)

### `src/ingestion/`

| Archivo | Funciones Exportadas |
|---------|---------------------|
| **`pdf_reader.py`** | `normalize_text()`, `clean_text()`, `read_first_pages()`, `find_abstract()`, `guess_title()`, `guess_authors()` |
| **`build_corpus.py`** | `list_pdfs()`, `build_corpus()`, `main()` |

### `src/preprocessing/text_preprocessing.py`

| Función | Descripción |
|---------|-------------|
| `tokenize(text)` | Minúsculas + tokens solo letras (regex Unicode) |
| `preprocess(text, remove_stopwords=True, languages=("en",), min_len=2)` | Pipeline completo |
| `load_corpus(path)` | Carga `corpus.json` |
| `preprocess_corpus(corpus, **kwargs)` | `dict[id] -> tokens` |

### `src/similarity/`

| Archivo | Funciones |
|---------|-----------|
| **`embeddings.py`** | `average_vector(tokens, lookup, dim)` → `(vector, used, missing)`<br>`embed_corpus(tokens_by_id, lookup, dim)` → `dict` con vectores y estadísticas OOV |
| **`metrics.py`** | `dot()`, `norm()`, `cosine_similarity()`, `euclidean_distance()` |

### `src/clustering/__init__.py`
> **Vacío** — módulo pendiente de implementación.

### `src/sorting/__init__.py`
> **Vacío** — módulo pendiente de implementación.

---

## Tests

Todos los tests son **funciones puras** (sin clases de test), organizados por módulo:

| Archivo | Cobertura |
|---------|-----------|
| `test_boyer_moore.py` | Tabla bad-char, ocurrencias solapadas, sin coincidencia, patrón vacío/largo, referencia aleatoria |
| `test_kmp.py` | LPS clásico, ocurrencias solapadas, sin coincidencia, patrón vacío/largo, referencia aleatoria |
| `test_rabin_karp.py` | Hash polinómico conocido, ocurrencias solapadas, sin coincidencia, patrón vacío/largo, colisiones con modulo pequeño, referencia aleatoria |
| `test_ingestion.py` | Limpieza de texto, extracción abstract (múltiples estilos), título, autores |
| `test_preprocessing.py` | Tokenización, acentos/ñ, stopwords EN/ES, longitud mínima, texto vacío |
| `test_normalize.py` | Caracteres de ancho cero |
| `test_metrics_embeddings.py` | Producto punto, norma, coseno, euclidiana, dimensiones distintas, relación coseno-euclidiana, promedio de vectores, OOV |
| `test_corpus_files.py` | Listado PDFs (orden natural, mayúsculas), build_corpus con overrides |

---

## Puntos de Entrada (CLI)

| Comando | Efecto |
|---------|--------|
| `python -m src.ingestion.build_corpus` | Procesa `data/pdfs/` → `data/processed/corpus.json` |
| `python -m experiments.generate_embeddings` | Genera embeddings del corpus (requiere modelo descargado) |
| `python -m experiments.preview_preprocessing` | Muestra tokens de abstracts |
| `pytest` | Ejecuta toda la suite de tests |

---

## Dependencias Principales

| Paquete | Uso |
|---------|-----|
| `pymupdf` (fitz) | Lectura de PDFs |
| `gensim` | Carga de vectores preentrenados (GloVe, Word2Vec, FastText) |
| `pytest` | Testing |

> **Nota:** `requirements-embeddings.txt` contiene dependencias adicionales para descargar modelos de embeddings (ej. `gensim-data`).