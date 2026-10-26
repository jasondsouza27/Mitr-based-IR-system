"""
Unit tests for Module 2: DenseVectorRetriever and FAISS search
"""

import pytest
import numpy as np
from threatmatch.dense import DenseVectorRetriever

def test_dense_retriever_initialization():
    retriever = DenseVectorRetriever()
    assert retriever.DIMENSION == 384
    assert retriever.embeddings is None

def test_dense_search_with_cache():
    # Test loading from precomputed cache
    cache_dir = "data/cache"
    import os
    import json
    if not os.path.exists(os.path.join(cache_dir, "embeddings.npy")):
        pytest.skip("Cache not generated yet.")

    with open("data/mitre_attack_corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    retriever = DenseVectorRetriever.load(cache_dir, corpus["techniques"])
    assert retriever.embeddings is not None
    assert retriever.embeddings.shape[1] == 384

    # Search for a semantic query
    results = retriever.search("credential dumping using mimikatz from memory", top_k=5)
    assert len(results) == 5
    # T1003.001 or credential access technique should be in top results
    assert results[0]["score"] > 0.3
    assert any("T1003" in r["id"] or "Credential" in " ".join(r["tactics"]) for r in results)
