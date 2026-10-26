"""
Benchmark runner script for ThreatMatch.
Executes evaluation over the 25 standardized incident scenarios
across all 4 strategies:
  1. Pure BM25
  2. Pure Dense Vector (FAISS)
  3. Hybrid Fusion (BM25 + FAISS via RRF)
  4. Hybrid Fusion + Rocchio PRF
Saves results to data/benchmark_results.json and prints formatted comparative table.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from threatmatch.engine import ThreatMatchEngine
from threatmatch.evaluation import BenchmarkEvaluator

def run_benchmark(
    corpus_path="data/mitre_attack_corpus.json",
    cache_dir="data/cache",
    queries_path="data/evaluation_queries.json",
    output_path="data/benchmark_results.json"
):
    print("=" * 70)
    print(" THREATMATCH: SYSTEM BENCHMARKING (CSDC 7023 / SEM-VII)")
    print("=" * 70)

    print("\nInitializing ThreatMatch Engine...")
    engine = ThreatMatchEngine(corpus_path=corpus_path, cache_dir=cache_dir)
    evaluator = BenchmarkEvaluator(engine)

    print(f"Loading {queries_path}...")
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Evaluating {len(queries)} realistic incident queries across 4 strategies...\n")

    results = evaluator.evaluate_all(queries, top_k=10)

    # Print Formatted Comparison Table
    print("\n" + "=" * 80)
    print(f"{'Strategy':<32} | {'MAP':<7} | {'NDCG@10':<7} | {'MRR':<7} | {'P@1':<7} | {'P@5':<7} | {'Latency':<8}")
    print("-" * 80)
    for row in results["summary_table"]:
        print(f"{row['Strategy']:<32} | {row['MAP']:<7.4f} | {row['NDCG@10']:<7.4f} | {row['MRR']:<7.4f} | {row['P@1']:<7.4f} | {row['P@5']:<7.4f} | {row['Latency (ms)']:<6.1f}ms")
    print("=" * 80 + "\n")

    # Save benchmark results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark results successfully saved to {output_path}!")
    return results

if __name__ == "__main__":
    run_benchmark()
