"""
Integration tests for ThreatMatch Unified Retrieval Engine.
Validates:
  - BM25 retrieval
  - Dense FAISS retrieval
  - Hybrid RRF retrieval
  - Hybrid RRF + Rocchio PRF retrieval
  - Sub-second latency execution
  - Explainability and Tokenizer comparison
"""

import pytest
import os
from threatmatch.engine import ThreatMatchEngine

@pytest.fixture(scope="module")
def engine():
    corpus_path = "data/mitre_attack_corpus.json"
    cache_dir = "data/cache"
    if not os.path.exists(corpus_path) or not os.path.exists(os.path.join(cache_dir, "embeddings.npy")):
        pytest.skip("Corpus or cache not built yet.")
    return ThreatMatchEngine(corpus_path=corpus_path, cache_dir=cache_dir)

def test_engine_bm25_search(engine):
    resp = engine.search("powershell.exe -enc execution", mode="bm25", top_k=5)
    assert resp["mode"] == "bm25"
    assert len(resp["results"]) > 0
    assert resp["latency_ms"] < 1000.0  # Sub-second
    # T1059.001 should be top or near top
    assert any("T1059" in r["id"] for r in resp["results"][:3])

def test_engine_dense_search(engine):
    resp = engine.search("injecting malicious dynamic link library into svchost process", mode="dense", top_k=5)
    assert resp["mode"] == "dense"
    assert len(resp["results"]) == 5
    assert resp["latency_ms"] < 1000.0

def test_engine_hybrid_search(engine):
    resp = engine.search("Registry Run keys persistence HKLM CurrentVersion", mode="hybrid", top_k=5)
    assert resp["mode"] == "hybrid"
    assert len(resp["results"]) == 5
    assert "rrf_score" in resp["results"][0]
    assert resp["latency_ms"] < 1000.0

def test_engine_hybrid_prf_search(engine):
    resp = engine.search("dumping lsass process memory with procdump", mode="hybrid_prf", top_k=5)
    assert resp["mode"] == "hybrid_prf"
    assert resp["prf_info"] is not None
    assert len(resp["prf_info"]["expansion_terms"]) > 0
    assert len(resp["results"]) == 5
    assert resp["latency_ms"] < 1000.0

def test_engine_explain_bm25(engine):
    explanation = engine.explain_bm25("powershell.exe -enc", "T1059.001")
    assert explanation["doc_id"] == "T1059.001"
    assert "total_bm25_score" in explanation
    assert len(explanation["terms"]) >= 1

def test_engine_compare_tokenization(engine):
    res = engine.compare_tokenization("powershell.exe -enc T1059.001 CVE-2023-38606")
    assert res["technical_entities_count"] >= 3
    assert len(res["corrupted_by_standard"]) >= 1

def test_engine_search_with_dynamic_parameters(engine):
    resp = engine.search(
        "dumping lsass process memory with procdump",
        mode="hybrid_prf",
        top_k=5,
        bm25_k1=2.0,
        bm25_b=0.5,
        rrf_k=30,
        prf_alpha=1.5,
        prf_beta=1.0,
        prf_k_docs=2,
        prf_num_terms=4
    )
    assert resp["mode"] == "hybrid_prf"
    assert len(resp["results"]) == 5
    assert resp["prf_info"]["alpha"] == 1.5
    assert resp["prf_info"]["beta"] == 1.0
    assert resp["prf_info"]["k_feedback"] == 2
    assert len(resp["prf_info"]["expansion_terms"]) <= 4

def test_engine_compare_all_modes(engine):
    comparison = engine.compare_all_modes(
        "powershell.exe -enc aW1wb3J0 invoking script",
        top_k=5,
        bm25_k1=1.5,
        bm25_b=0.75,
        rrf_k=60
    )
    assert "bm25" in comparison
    assert "dense" in comparison
    assert "hybrid" in comparison
    assert "hybrid_prf" in comparison
    assert len(comparison["bm25"]["results"]) == 5
    assert len(comparison["dense"]["results"]) == 5
    assert len(comparison["hybrid"]["results"]) == 5
    assert len(comparison["hybrid_prf"]["results"]) == 5
    assert len(comparison["comparison_matrix"]) > 0
    assert "jaccard_bm25_dense" in comparison["metrics"]
    assert "total_pipeline_latency_ms" in comparison["metrics"]
    # Check that rank shift annotations exist
    assert "bm25_rank" in comparison["hybrid"]["results"][0]
    assert "hybrid_rank" in comparison["hybrid_prf"]["results"][0]


def test_engine_getattr_fallback_resolution(engine):
    """Verifies that an engine instance dynamically resolves methods via __getattr__ from threatmatch.engine."""
    # Test that valid engine methods can be resolved dynamically if looked up
    method = getattr(engine, "compare_all_modes")
    assert callable(method)

    # Test non-existent attribute cleanly raises standard AttributeError
    with pytest.raises(AttributeError, match="has no attribute 'non_existent_method'"):
        _ = engine.non_existent_method()


