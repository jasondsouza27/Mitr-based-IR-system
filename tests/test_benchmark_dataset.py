"""
Test that all 25 benchmark queries map to valid techniques in the MITRE corpus.
"""
import json
import os
import pytest

def test_evaluation_queries_ground_truth():
    corpus_path = "data/mitre_attack_corpus.json"
    queries_path = "data/evaluation_queries.json"

    assert os.path.exists(corpus_path), "Corpus file does not exist."
    assert os.path.exists(queries_path), "Evaluation queries file does not exist."

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    assert len(queries) == 25, f"Expected 25 benchmark queries, got {len(queries)}"

    valid_tech_ids = {t["id"] for t in corpus["techniques"]}

    missing_ids = []
    for q in queries:
        for gt in q["ground_truth"]:
            if gt not in valid_tech_ids:
                missing_ids.append((q["id"], gt))

    assert len(missing_ids) == 0, f"Some ground truth IDs are missing from corpus: {missing_ids}"

if __name__ == "__main__":
    test_evaluation_queries_ground_truth()
    print("All 25 queries have valid ground-truth IDs in the corpus!")
