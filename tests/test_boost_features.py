"""
Unit and Integration Tests for ThreatMatch Boost Features:
1. 4-Way Side-by-Side Comparison Pipeline
2. Dynamic IR Parameter Sandbox (k1, b, alpha, beta, k_docs, rrf_k)
3. 1-Click Examiner Demo Scenarios
"""

import pytest
import os
from threatmatch.engine import ThreatMatchEngine
from threatmatch.feedback import RocchioPRF
from threatmatch.fusion import ReciprocalRankFusion
from threatmatch.indexer import BM25Retriever

@pytest.fixture(scope="module")
def engine():
    corpus_path = "data/mitre_attack_corpus.json"
    cache_dir = "data/cache"
    if not os.path.exists(corpus_path) or not os.path.exists(os.path.join(cache_dir, "embeddings.npy")):
        pytest.skip("Corpus or cache not built yet.")
    return ThreatMatchEngine(corpus_path=corpus_path, cache_dir=cache_dir)

# -------------------------------------------------------------
# 1. Tests for 4-Way Side-by-Side Comparison
# -------------------------------------------------------------
def test_compare_all_modes_structure(engine):
    query = "powershell.exe -ExecutionPolicy Bypass -NoProfile -EncodedCommand aW1wb3J0"
    comp = engine.compare_all_modes(query, top_k=5)

    assert "bm25" in comp
    assert "dense" in comp
    assert "hybrid" in comp
    assert "hybrid_prf" in comp

    assert len(comp["bm25"]["results"]) == 5
    assert len(comp["dense"]["results"]) == 5
    assert len(comp["hybrid"]["results"]) == 5
    assert len(comp["hybrid_prf"]["results"]) == 5

    # Check rank shift annotations
    for r in comp["dense"]["results"]:
        assert "shift_vs_bm25" in r
    for r in comp["hybrid"]["results"]:
        assert "shift_vs_bm25" in r
        assert "shift_vs_dense" in r
    for r in comp["hybrid_prf"]["results"]:
        assert "shift_vs_hybrid" in r

    # Check cross-method comparison matrix
    assert len(comp["comparison_matrix"]) > 0
    for row in comp["comparison_matrix"]:
        assert "id" in row
        assert "name" in row
        assert "bm25_rank" in row
        assert "dense_rank" in row
        assert "hybrid_rank" in row
        assert "hybrid_prf_rank" in row
        assert "status" in row

    # Check consensus metrics
    m = comp["metrics"]
    assert 0.0 <= m["jaccard_bm25_dense"] <= 1.0
    assert 0.0 <= m["jaccard_hybrid_prf"] <= 1.0
    assert m["total_pipeline_latency_ms"] > 0.0

# -------------------------------------------------------------
# 2. Tests for Dynamic IR Parameter Sandbox
# -------------------------------------------------------------
def test_dynamic_bm25_parameters(engine):
    query = "powershell script execution"
    res1 = engine.search(query, mode="bm25", top_k=5, bm25_k1=0.5, bm25_b=0.2)
    res2 = engine.search(query, mode="bm25", top_k=5, bm25_k1=2.8, bm25_b=0.95)

    assert len(res1["results"]) > 0 and len(res2["results"]) > 0
    # Different k1 and b must yield different scores
    scores1 = [r["score"] for r in res1["results"]]
    scores2 = [r["score"] for r in res2["results"]]
    assert scores1 != scores2

def test_dynamic_rrf_parameters(engine):
    query = "lsass dump memory"
    res1 = engine.search(query, mode="hybrid", top_k=5, rrf_k=10)
    res2 = engine.search(query, mode="hybrid", top_k=5, rrf_k=90)

    assert len(res1["results"]) > 0 and len(res2["results"]) > 0
    # k=10 yields much higher RRF scores than k=90
    assert res1["results"][0]["rrf_score"] > res2["results"][0]["rrf_score"]

def test_dynamic_rocchio_prf_parameters(engine):
    query = "procdump lsass"
    res = engine.search(
        query,
        mode="hybrid_prf",
        top_k=5,
        prf_alpha=1.6,
        prf_beta=1.2,
        prf_k_docs=2,
        prf_num_terms=4
    )
    prf = res["prf_info"]
    assert prf is not None
    assert prf["alpha"] == 1.6
    assert prf["beta"] == 1.2
    assert prf["k_feedback"] == 2
    assert len(prf["expansion_terms"]) <= 4

# -------------------------------------------------------------
# 3. Tests for 1-Click Examiner Demo Scenarios
# -------------------------------------------------------------
@pytest.mark.parametrize("scenario_key,query,target_tech", [
    ("powershell_obfuscated", "powershell.exe -ExecutionPolicy Bypass -NoProfile -EncodedCommand aW1wb3J0 invoking hidden script", "T1059.001"),
    ("vocab_mismatch", "Stealthily injecting foreign executable code into running background process memory to evade antivirus", "T1055"),
    ("short_rocchio", "procdump lsass", "T1003.001"),
    ("cve_zeroday", "CVE-2023-38606 WebKit memory corruption vulnerability allowing arbitrary code execution in mobile browser", "T1203"),
    ("ransomware_multi", "vssadmin.exe delete shadows /all /quiet and bcdedit /set default bootstatuspolicy ignoreallfailures followed by file encryption", "T1490")
])
def test_examiner_demo_scenarios(engine, scenario_key, query, target_tech):
    resp = engine.search(query, mode="hybrid_prf", top_k=10)
    assert len(resp["results"]) > 0
    retrieved_ids = [r["id"] for r in resp["results"]]
    # Target technique or its parent/subtechnique should be present in retrieved candidates
    assert any(target_tech in rid for rid in retrieved_ids), f"Scenario '{scenario_key}' failed to retrieve {target_tech} in top 10: {retrieved_ids}"

# -------------------------------------------------------------
# 4. Edge Case & Robustness Tests
# -------------------------------------------------------------
def test_empty_and_whitespace_query_handling(engine):
    """Verifies that empty or whitespace-only queries return gracefully without crashing."""
    for empty_q in ["", "   ", "\t  \n"]:
        res_bm25 = engine.search(empty_q, mode="bm25")
        assert len(res_bm25["results"]) == 0

        res_dense = engine.search(empty_q, mode="dense")
        assert len(res_dense["results"]) == 0

        res_hybrid = engine.search(empty_q, mode="hybrid")
        assert len(res_hybrid["results"]) == 0

        res_prf = engine.search(empty_q, mode="hybrid_prf")
        assert len(res_prf["results"]) == 0

        comp = engine.compare_all_modes(empty_q, top_k=5)
        assert len(comp["bm25"]["results"]) == 0
        assert len(comp["dense"]["results"]) == 0
        assert len(comp["comparison_matrix"]) == 0
        assert comp["metrics"]["all_four_consensus_count"] == 0

def test_comparison_matrix_pyarrow_compatibility(engine):
    """Verifies that the cross-method matrix serializes cleanly to PyArrow Table without dtype errors."""
    import pandas as pd
    import pyarrow as pa

    query = "powershell script injection mimikatz"
    comp = engine.compare_all_modes(query, top_k=5)

    matrix = comp["comparison_matrix"]
    assert len(matrix) > 0

    # Ensure all rank entries are formatted uniform strings
    for row in matrix:
        assert isinstance(row["bm25_rank"], str)
        assert isinstance(row["dense_rank"], str)
        assert isinstance(row["hybrid_rank"], str)
        assert isinstance(row["hybrid_prf_rank"], str)
        assert row["status"] in {
            "Discovered by Rocchio PRF",
            "Boosted by Rocchio PRF",
            "Strong Dual-Modal Consensus",
            "Rescued by Dense Semantics",
            "Lexical Exact Match",
            "Hybrid Fusion Candidate"
        }

    # Verify PyArrow conversion does not throw ArrowInvalid
    df = pd.DataFrame(matrix)
    arrow_table = pa.Table.from_pandas(df)
    assert arrow_table.num_rows == len(matrix)

def test_extreme_parameter_sandbox_boundaries(engine):
    """Verifies that extreme slider parameter values operate stably without arithmetic exceptions."""
    query = "powershell command execution"

    # Minimal boundary values
    res_min = engine.search(
        query,
        mode="hybrid_prf",
        top_k=5,
        bm25_k1=0.1,
        bm25_b=0.0,
        rrf_k=1,
        prf_alpha=0.1,
        prf_beta=0.0,
        prf_k_docs=1,
        prf_num_terms=1
    )
    assert len(res_min["results"]) > 0

    # Maximal boundary values
    res_max = engine.search(
        query,
        mode="hybrid_prf",
        top_k=5,
        bm25_k1=3.0,
        bm25_b=1.0,
        rrf_k=100,
        prf_alpha=2.0,
        prf_beta=2.0,
        prf_k_docs=10,
        prf_num_terms=15
    )
    assert len(res_max["results"]) > 0

    # Compare all modes under custom parameters
    comp = engine.compare_all_modes(
        query,
        top_k=5,
        bm25_k1=0.2,
        bm25_b=0.1,
        rrf_k=5,
        prf_alpha=1.8,
        prf_beta=1.5
    )
    assert len(comp["hybrid_prf"]["results"]) == 5
