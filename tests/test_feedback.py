"""
Unit tests for Module 3: Rocchio Pseudo-Relevance Feedback (PRF)
"""

import pytest
from threatmatch.tokenizer import CyberRegexTokenizer
from threatmatch.indexer import InvertedIndex
from threatmatch.feedback import RocchioPRF

@pytest.fixture
def sample_engine_index():
    docs = [
        {
            "id": "T1003.001",
            "name": "LSASS Memory",
            "search_text": "dumping lsass credentials mimikatz memory dump sekurlsa procdump password"
        },
        {
            "id": "T1059.001",
            "name": "PowerShell",
            "search_text": "powershell script execution download cradles encoded command"
        }
    ]
    tokenizer = CyberRegexTokenizer()
    index = InvertedIndex(tokenizer=tokenizer)
    index.index_documents(docs)
    return index

def test_rocchio_prf_query_expansion(sample_engine_index):
    prf = RocchioPRF(index=sample_engine_index, alpha=1.0, beta=0.75, k_feedback=1, num_expansion_terms=3)

    first_pass_results = [
        {"id": "T1003.001", "name": "LSASS Memory", "score": 5.0}
    ]

    original_query = "dumping lsass"
    result = prf.expand_query(original_query, first_pass_results)

    assert result.original_query == original_query
    # Expanded query should contain terms from T1003.001 (e.g. mimikatz, memory, dump, etc.)
    assert len(result.expansion_terms) > 0
    # Original query terms should NOT be duplicate-expanded as new terms
    extracted_terms = [t.term.lower() for t in result.expansion_terms]
    assert "dumping" not in extracted_terms
    assert "lsass" not in extracted_terms
    assert result.expanded_query != original_query
    assert any(term in result.expanded_query for term in ["mimikatz", "memory", "dump", "procdump"])

def test_rocchio_empty_first_pass(sample_engine_index):
    prf = RocchioPRF(index=sample_engine_index)
    result = prf.expand_query("unknown query", [])
    assert result.expanded_query == "unknown query"
    assert result.expansion_terms == []

def test_rocchio_dynamic_parameters(sample_engine_index):
    prf = RocchioPRF(index=sample_engine_index, alpha=1.0, beta=0.5, k_feedback=1, num_expansion_terms=1)
    first_pass_results = [
        {"id": "T1003.001", "name": "LSASS Memory", "score": 5.0}
    ]
    # Overriding alpha, beta, and num_terms
    result = prf.expand_query("lsass", first_pass_results, num_terms=4, alpha=1.8, beta=1.2)
    assert result.alpha == 1.8
    assert result.beta == 1.2
    assert len(result.expansion_terms) > 1

