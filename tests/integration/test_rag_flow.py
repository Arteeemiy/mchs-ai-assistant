import os
import pytest
from src.core.rag_system import RAGSystem


@pytest.mark.integration
def test_full_flow():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("API key not set")

    rag = RAGSystem(api_key)
    response = rag.process_query("Действия при возгорании в помещении")

    assert isinstance(response, str)
    assert len(response) > 50
