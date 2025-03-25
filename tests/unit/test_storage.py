import pytest
from pathlib import Path
from src.core.storage.vector_db import VectorStore


@pytest.fixture
def test_vector_store(tmp_path):
    data_dir = tmp_path / "test_data"
    index_dir = tmp_path / "test_index"
    data_dir.mkdir()
    index_dir.mkdir()

    test_doc = Path("tests/test_data/sample_document.json")
    (data_dir / "sample_document.json").write_text(test_doc.read_text())

    return VectorStore(data_dir=str(data_dir), index_dir=str(index_dir))


def test_index_creation(test_vector_store):
    test_vector_store.create_index()
    assert test_vector_store.index is not None
    assert len(test_vector_store.documents) > 0


def test_search(test_vector_store):
    test_vector_store.create_index()
    results = test_vector_store.search("пожар", top_k=1, min_score=0.5)
    assert len(results) == 1
    assert "112" in results[0]["text"]
