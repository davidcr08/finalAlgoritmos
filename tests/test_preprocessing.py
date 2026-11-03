from src.preprocessing.text_preprocessing import preprocess, tokenize


def test_tokenize_minusculas_y_descarta_numeros_y_signos():
    assert tokenize("GPT-4 is Great, in 2023!") == ["gpt", "is", "great", "in"]


def test_acentos_y_enie_se_conservan():
    assert tokenize("Generación de código en español") == ["generación", "de", "código", "en", "español"]


def test_stopwords_ingles_por_defecto():
    assert preprocess("The models are trained on data") == ["models", "trained", "data"]


def test_sin_quitar_stopwords():
    assert preprocess("The models", remove_stopwords=False) == ["the", "models"]


def test_stopwords_espanol_opcional():
    assert preprocess("Los modelos de lenguaje", languages=("es",)) == ["modelos", "lenguaje"]


def test_longitud_minima():
    assert preprocess("a b ai ml", remove_stopwords=False, min_len=2) == ["ai", "ml"]


def test_texto_vacio():
    assert preprocess("") == []