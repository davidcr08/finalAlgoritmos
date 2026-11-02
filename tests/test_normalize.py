from src.ingestion.pdf_reader import normalize_text


def test_quita_caracteres_de_ancho_cero():
    assert normalize_text("creat\u200biveco\u200bmmons") == "creativecommons"
    assert normalize_text("\ufeffhola") == "hola"