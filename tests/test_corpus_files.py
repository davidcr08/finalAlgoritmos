from src.ingestion.build_corpus import list_pdfs


def test_list_pdfs_ignora_otros_archivos_y_acepta_mayusculas(tmp_path):
    (tmp_path / "b.PDF").write_bytes(b"")
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "notas.txt").write_text("x")
    (tmp_path / ".gitkeep").write_text("")
    assert [p.name for p in list_pdfs(tmp_path)] == ["a.pdf", "b.PDF"]


def test_list_pdfs_carpeta_inexistente(tmp_path):
    assert list_pdfs(tmp_path / "no_existe") == []