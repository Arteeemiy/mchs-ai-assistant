from src.core.embedding.embedder import OptimizedEmbedder
import numpy as np


def test_embedding_creation():
    embedder = OptimizedEmbedder()
    embeddings = embedder.embed(["test"])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 384)
