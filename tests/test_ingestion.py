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


def test_abstract_con_letras_separadas_estilo_elsevier():
    text = "ARTICLE INFO\nKeywords:\nAI\nA B S T R A C T\nWe study things in depth.\nKeywords: x\n1. Introduction\nbody"
    abstract, _ = find_abstract(text)
    assert abstract == "We study things in depth."


def test_abstract_sin_marca_de_cierre_usa_end_phrase():
    text = "A B S T R A C T\nThis is the abstract.\nSince Alan Turing asked a question\nmore body"
    assert find_abstract(text)[0].startswith("This is the abstract.")   # sin regla, se pasa de largo
    abstract, _ = find_abstract(text, end_phrase="Since Alan Turing")
    assert abstract == "This is the abstract."


def test_abstract_con_keywords_antes_usa_start_trim():
    text = "A B S T R A C T\nLearning analytics\nEthical considerations\nThe real abstract text.\n1. Introduction\nbody"
    abstract, _ = find_abstract(text, start_trim="Ethical considerations")
    assert abstract == "The real abstract text."


def test_palabra_abstraction_no_se_confunde_con_encabezado():
    assert find_abstract("The abstraction layer is useful.\nIntroduction\nx")[0] is None


def test_sin_abstract():
    assert find_abstract("solo texto sin encabezado")[0] is None


def test_titulo_prefiere_metadatos_plausibles():
    assert guess_title("Linea 1\nLinea 2", "Un titulo de metadatos") == ("Un titulo de metadatos", "metadatos")
    assert guess_title("Primera linea del pdf\nOtra", "doc.pdf") == ("Primera linea del pdf", "primera linea")


def test_titulo_de_metadatos_interno_del_editor_se_descarta():
    assert guess_title("Titulo real del articulo\nx", "ET-04-2023-0117_proof 609..645")[1] == "primera linea"


def test_autores_filtra_afiliaciones_y_correos():
    front = "Mi titulo\nAna Perez, Luis Gomez\nUniversity of Somewhere\nana@x.org\n"
    assert guess_authors(front, "Mi titulo") == "Ana Perez, Luis Gomez"


def test_autores_descarta_licencias_etiquetas_y_titulo_repetido():
    front = (
        "Artificial intelligence in education\nOpen Access\nANA PEREZ, LUIS GOMEZ\n"
        "creativecommons.org/licenses/by/4.0/\nREVIEW ARTICLE\nCitation: Perez, A. Algo.\n"
    )
    assert guess_authors(front, "Artificial intelligence in education") == "ANA PEREZ, LUIS GOMEZ"