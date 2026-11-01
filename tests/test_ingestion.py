from src.ingestion.pdf_reader import clean_text, find_abstract, guess_authors, guess_title


def test_clean_text_une_palabras_cortadas_y_espacios():
    assert clean_text("gener-\native   models\nare  great") == "generative models are great"


def test_clean_text_reemplaza_ligaduras():
    assert clean_text("e\ufb03cient \ufb01ne-tuning") == "efficient fine-tuning"


def test_abstract_con_encabezado_y_keywords():
    text = "Mi titulo\nAutor Uno\nAbstract\nLarge models\nare useful.\nKeywords: ia, llm\n1 Introduction\nblabla"
    abstract, pos = find_abstract(text)
    assert abstract == "Large models are useful."
    assert pos > 0


def test_abstract_en_la_misma_linea_estilo_ieee():
    text = "Titulo\nAbstract\u2014Generative AI changes work.\nIndex Terms\u2014ai\nI. INTRODUCTION\nx"
    abstract, _ = find_abstract(text)
    assert abstract == "Generative AI changes work."


def test_abstract_mayusculas_y_corta_en_introduction():
    text = "T\nABSTRACT\nWe study agents.\nMore text here.\nINTRODUCTION\nBody"
    abstract, _ = find_abstract(text)
    assert abstract == "We study agents. More text here."


def test_sin_abstract():
    assert find_abstract("solo texto sin encabezado")[0] is None


def test_titulo_prefiere_metadatos_plausibles():
    assert guess_title("Linea 1\nLinea 2", "Un titulo de metadatos") == ("Un titulo de metadatos", "metadatos")
    assert guess_title("Primera linea del pdf\nOtra", "doc.pdf") == ("Primera linea del pdf", "primera linea")


def test_autores_filtra_afiliaciones_y_correos():
    front = "Mi titulo\nAna Perez, Luis Gomez\nUniversity of Somewhere\nana@x.org\n"
    assert guess_authors(front, "Mi titulo") == "Ana Perez, Luis Gomez"