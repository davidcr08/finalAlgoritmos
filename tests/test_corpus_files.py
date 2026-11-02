import json

import pymupdf

from src.ingestion.build_corpus import build_corpus, list_pdfs


def _make_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def test_list_pdfs_ignora_otros_archivos_y_acepta_mayusculas(tmp_path):
    (tmp_path / "b.PDF").write_bytes(b"")
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "notas.txt").write_text("x")
    (tmp_path / ".gitkeep").write_text("")
    assert [p.name for p in list_pdfs(tmp_path)] == ["a.pdf", "b.PDF"]


def test_list_pdfs_carpeta_inexistente(tmp_path):
    assert list_pdfs(tmp_path / "no_existe") == []


def test_list_pdfs_orden_natural(tmp_path):
    for name in ("10. b.pdf", "2. a.pdf", "1. c.PDF", "20. d.pdf", "9. e.pdf"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_pdfs(tmp_path)] == [
        "1. c.PDF", "2. a.pdf", "9. e.pdf", "10. b.pdf", "20. d.pdf",
    ]


def test_build_corpus_ids_en_orden_natural_y_overrides_por_prefijo(tmp_path):
    _make_pdf(tmp_path / "10. Beta.pdf", "A B S T R A C T\nBeta abstract text here.\nKeywords: x\n1. Introduction\nbody")
    _make_pdf(tmp_path / "2. Alfa.pdf", "A B S T R A C T\nTema uno\nUltima keyword\nAlfa abstract real.\n1. Introduction\nbody")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"2.": {"abstract_start_trim": "Ultima keyword"}}), encoding="utf-8")

    corpus = build_corpus(tmp_path, overrides)

    assert [d["id"] for d in corpus] == ["A01", "A02"]
    assert corpus[0]["file"] == "2. Alfa.pdf"       # el 2 va antes que el 10
    assert corpus[0]["abstract"] == "Alfa abstract real."
    assert corpus[1]["abstract"] == "Beta abstract text here."


def test_override_por_nombre_de_archivo_reemplaza_el_abstract(tmp_path):
    _make_pdf(tmp_path / "1. X.pdf", "sin encabezado reconocible")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"1. X.pdf": {"abstract": "Texto puesto a mano."}}), encoding="utf-8")
    corpus = build_corpus(tmp_path, overrides)
    assert corpus[0]["abstract"] == "Texto puesto a mano."