
import numpy as np
import pytest

from src.services.clustering_service import ClusteringService


def make_service(doc_ids, embeddings):
    """Construye el servicio sin cargar archivos reales del proyecto."""
    service = ClusteringService.__new__(ClusteringService)
    service.doc_ids = doc_ids
    service.embeddings_glove = embeddings
    service.embeddings_fasttext = embeddings
    return service


def test_embeddings_generan_matriz_simetrica():
    service = make_service(
        ["A01", "A02", "A03"],
        {
            "A01": [1.0, 0.0],
            "A02": [0.0, 1.0],
            "A03": [1.0, 1.0],
        },
    )

    D = service.get_distance_matrix("glove")

    assert D.shape == (3, 3)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    assert np.all(D >= 0.0)
    assert np.all(D <= 2.0)


def test_embedding_ausente_lanza_value_error():
    service = make_service(
        ["A01", "A02"],
        {"A01": [1.0, 0.0]},
    )

    with pytest.raises(ValueError, match="A02"):
        service.get_distance_matrix("glove")


def test_dimensiones_incompatibles_lanzan_value_error():
    service = make_service(
        ["A01", "A02"],
        {
            "A01": [1.0, 0.0],
            "A02": [1.0, 0.0, 1.0],
        },
    )

    with pytest.raises(ValueError):
        service.get_distance_matrix("glove")


def test_vector_nulo_no_se_confunde_con_embedding_ausente():
    service = make_service(
        ["A01", "A02"],
        {
            "A01": [0.0, 0.0],
            "A02": [1.0, 0.0],
        },
    )

    D = service.get_distance_matrix("glove")

    assert D[0, 1] == pytest.approx(1.0)
    assert D[1, 0] == pytest.approx(1.0)


def test_archivo_de_embeddings_ausente_informa_como_generarlo(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    service = make_service(["A01"], {})
    
    with pytest.raises(FileNotFoundError, match="generate_embeddings"):
        service.get_distance_matrix("fasttext")


def test_fuente_desconocida_lanza_value_error():
    service = make_service(["A01"], {"A01": [1.0, 0.0]})

    with pytest.raises(ValueError):
        service.get_distance_matrix("desconocida")
