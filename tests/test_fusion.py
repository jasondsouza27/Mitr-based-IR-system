"""
Unit tests for Module 2: Reciprocal Rank Fusion (RRF)
"""

import pytest
from threatmatch.fusion import ReciprocalRankFusion

def test_rrf_scoring_formula():
    fuser = ReciprocalRankFusion(k=60)
    
    # Doc A is rank 1 in BM25, rank 2 in Dense
    # RRF(A) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    bm25_res = [
        {"id": "doc_A", "rank": 1, "score": 12.5, "name": "A"},
        {"id": "doc_B", "rank": 2, "score": 8.0, "name": "B"}
    ]
    dense_res = [
        {"id": "doc_B", "rank": 1, "score": 0.85, "name": "B"},
        {"id": "doc_A", "rank": 2, "score": 0.80, "name": "A"}
    ]

    fused = fuser.fuse(bm25_res, dense_res, top_k=5)
    assert len(fused) == 2
    # Both doc_A and doc_B have reciprocal sum 1/61 + 1/62, so scores should be identical
    assert pytest.approx(fused[0]["rrf_score"], rel=1e-4) == (1/61 + 1/62)
    assert pytest.approx(fused[1]["rrf_score"], rel=1e-4) == (1/61 + 1/62)

def test_rrf_disjoint_lists():
    fuser = ReciprocalRankFusion(k=60)
    bm25_res = [{"id": "doc_A", "rank": 1, "score": 10.0}]
    dense_res = [{"id": "doc_C", "rank": 1, "score": 0.9}]

    fused = fuser.fuse(bm25_res, dense_res, top_k=5)
    assert len(fused) == 2
    assert fused[0]["rrf_score"] == pytest.approx(1/61)
    assert fused[1]["rrf_score"] == pytest.approx(1/61)

def test_rrf_empty_lists():
    fuser = ReciprocalRankFusion(k=60)
    assert fuser.fuse([], [], top_k=5) == []

def test_rrf_dynamic_k():
    fuser = ReciprocalRankFusion(k=60)
    bm25_res = [{"id": "doc_A", "rank": 1, "score": 10.0}]
    dense_res = [{"id": "doc_B", "rank": 1, "score": 0.9}]

    # With dynamic k=10: score = 1 / (10 + 1) = 1/11
    fused = fuser.fuse(bm25_res, dense_res, top_k=5, k=10)
    assert len(fused) == 2
    assert fused[0]["rrf_score"] == pytest.approx(1 / 11)

