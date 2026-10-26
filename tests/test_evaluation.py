"""
Unit tests for Module 4: IR Evaluation Metrics (MAP, NDCG@10, MRR, P@k, Recall@k)
"""

import pytest
import math
from threatmatch.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    average_precision,
    ndcg_at_k
)

def test_precision_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    gt = {"doc1", "doc3"}

    assert precision_at_k(retrieved, gt, k=1) == 1.0
    assert precision_at_k(retrieved, gt, k=2) == 0.5
    assert precision_at_k(retrieved, gt, k=4) == 0.5
    assert precision_at_k(retrieved, gt, k=0) == 0.0

def test_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    gt = {"doc1", "doc3"}

    assert recall_at_k(retrieved, gt, k=1) == 0.5
    assert recall_at_k(retrieved, gt, k=3) == 1.0

def test_reciprocal_rank():
    # Relevant item at rank 1
    assert reciprocal_rank(["doc1", "doc2"], {"doc1"}) == 1.0
    # Relevant item at rank 2
    assert reciprocal_rank(["docX", "doc1"], {"doc1"}) == 0.5
    # Relevant item at rank 3
    assert reciprocal_rank(["docX", "docY", "doc1"], {"doc1"}) == pytest.approx(1.0 / 3.0)
    # Not found
    assert reciprocal_rank(["docX", "docY"], {"doc1"}) == 0.0

def test_average_precision():
    # If relevant docs are at rank 1 and 3 out of 2 total relevant docs:
    # P@1 = 1/1 = 1.0
    # P@3 = 2/3
    # AP = (1.0 + 2/3) / 2 = 1.6667 / 2 = 0.8333
    retrieved = ["doc1", "docX", "doc2"]
    gt = {"doc1", "doc2"}
    expected_ap = (1.0 + 2.0 / 3.0) / 2.0
    assert average_precision(retrieved, gt) == pytest.approx(expected_ap)

def test_ndcg_at_k_perfect_ranking():
    retrieved = ["doc1", "doc2", "docX"]
    gt = {"doc1", "doc2"}
    # Perfect order => NDCG must equal 1.0
    assert ndcg_at_k(retrieved, gt, k=10) == pytest.approx(1.0)

def test_ndcg_at_k_suboptimal_ranking():
    # Rel at rank 2 and 3 instead of 1 and 2
    retrieved = ["docX", "doc1", "doc2"]
    gt = {"doc1", "doc2"}
    score = ndcg_at_k(retrieved, gt, k=10)
    assert 0.0 < score < 1.0
