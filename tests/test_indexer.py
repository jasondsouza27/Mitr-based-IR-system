"""
Unit tests for Module 1: InvertedIndex and BM25Retriever
"""

import pytest
import os
import tempfile
from threatmatch.tokenizer import CyberRegexTokenizer
from threatmatch.indexer import InvertedIndex, BM25Retriever

@pytest.fixture
def sample_docs():
    return [
        {
            "id": "T1059.001",
            "name": "PowerShell",
            "search_text": "powershell.exe interactive command line -enc execution script environment"
        },
        {
            "id": "T1003.001",
            "name": "LSASS Memory",
            "search_text": "lsass.exe memory dump credentials procdump password plaintext"
        },
        {
            "id": "T1055.001",
            "name": "Dynamic-link Library Injection",
            "search_text": "process injection svchost.exe dynamic link library dll malicious"
        }
    ]

def test_inverted_index_construction(sample_docs):
    tokenizer = CyberRegexTokenizer()
    index = InvertedIndex(tokenizer=tokenizer)
    index.index_documents(sample_docs)

    assert index.total_docs == 3
    assert "powershell.exe" in index.postings
    assert "lsass.exe" in index.postings
    assert index.doc_freqs["powershell.exe"] == 1
    assert index.avg_doc_len > 0

def test_bm25_search_exact_match(sample_docs):
    tokenizer = CyberRegexTokenizer()
    index = InvertedIndex(tokenizer=tokenizer)
    index.index_documents(sample_docs)
    retriever = BM25Retriever(index)

    results = retriever.search("powershell.exe -enc", top_k=2)
    assert len(results) > 0
    assert results[0]["id"] == "T1059.001"
    assert results[0]["score"] > 0.0

def test_bm25_explain_breakdown(sample_docs):
    tokenizer = CyberRegexTokenizer()
    index = InvertedIndex(tokenizer=tokenizer)
    index.index_documents(sample_docs)
    retriever = BM25Retriever(index)

    explanation = retriever.explain("lsass.exe memory", "T1003.001")
    assert explanation["doc_id"] == "T1003.001"
    assert explanation["total_bm25_score"] > 0
    assert len(explanation["terms"]) >= 1

def test_bm25_empty_query(sample_docs):
    tokenizer = CyberRegexTokenizer()
    index = InvertedIndex(tokenizer=tokenizer)
    index.index_documents(sample_docs)
    retriever = BM25Retriever(index)

    results = retriever.search("", top_k=5)
    assert results == []

def test_index_save_and_load(sample_docs):
    tokenizer = CyberRegexTokenizer()
    index = InvertedIndex(tokenizer=tokenizer)
    index.index_documents(sample_docs)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "test_idx.pkl")
        index.save(save_path)

        loaded_index = InvertedIndex.load(save_path, tokenizer=tokenizer)
        assert loaded_index.total_docs == index.total_docs
        assert loaded_index.avg_doc_len == index.avg_doc_len
        assert "powershell.exe" in loaded_index.postings

def test_bm25_dynamic_parameters(sample_docs):
    tokenizer = CyberRegexTokenizer()
    index = InvertedIndex(tokenizer=tokenizer)
    index.index_documents(sample_docs)
    retriever = BM25Retriever(index, k1=1.5, b=0.75)

    res_default = retriever.search("powershell.exe", top_k=2)
    res_custom = retriever.search("powershell.exe", top_k=2, k1=2.5, b=0.2)
    assert len(res_default) > 0 and len(res_custom) > 0
    # Different k1/b yields different scores
    assert res_default[0]["score"] != res_custom[0]["score"]

def test_bm25_explain_dynamic_parameters(sample_docs):
    tokenizer = CyberRegexTokenizer()
    index = InvertedIndex(tokenizer=tokenizer)
    index.index_documents(sample_docs)
    retriever = BM25Retriever(index, k1=1.5, b=0.75)

    explanation = retriever.explain("lsass.exe", "T1003.001", k1=2.2, b=0.3)
    assert explanation["k1"] == 2.2
    assert explanation["b"] == 0.3

