"""
Module 4: Rigorous Evaluation & Benchmarking Suite
Calculates:
  - Mean Average Precision (MAP)
  - Normalized Discounted Cumulative Gain (NDCG@10)
  - Mean Reciprocal Rank (MRR)
  - Precision@k (P@1, P@3, P@5, P@10)
  - Recall@k (R@5, R@10)
  - Query Latency (ms)
Compares the four baseline retrieval strategies across 25 standardized incident scenarios.
"""

import math
import time
from typing import List, Dict, Any, Set, Tuple

def precision_at_k(retrieved_ids: List[str], ground_truth: Set[str], k: int) -> float:
    """Computes Precision@k: |Top-k intersect GroundTruth| / k."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in ground_truth)
    return hits / k

def recall_at_k(retrieved_ids: List[str], ground_truth: Set[str], k: int) -> float:
    """Computes Recall@k: |Top-k intersect GroundTruth| / |GroundTruth|."""
    if not ground_truth:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in ground_truth)
    return hits / len(ground_truth)

def reciprocal_rank(retrieved_ids: List[str], ground_truth: Set[str]) -> float:
    """Computes Reciprocal Rank (RR): 1 / rank of first relevant document."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth:
            return 1.0 / rank
    return 0.0

def average_precision(retrieved_ids: List[str], ground_truth: Set[str]) -> float:
    """Computes Average Precision (AP)."""
    if not ground_truth:
        return 0.0

    hits = 0
    sum_precisions = 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth:
            hits += 1
            precision_at_rank = hits / rank
            sum_precisions += precision_at_rank

    return sum_precisions / len(ground_truth)

def ndcg_at_k(retrieved_ids: List[str], ground_truth: Set[str], k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at k (NDCG@k) using binary relevance.
    DCG@k = sum_{i=1}^k rel_i / log2(i + 1)
    IDCG@k = sum_{i=1}^{min(k, |GT|)} 1 / log2(i + 1)
    """
    if not ground_truth or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        rel = 1.0 if doc_id in ground_truth else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)

    ideal_hits = min(k, len(ground_truth))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg

class BenchmarkEvaluator:
    """
    Evaluates ThreatMatchEngine against a curated collection of incident scenarios.
    """
    def __init__(self, engine: Any):
        self.engine = engine

    def evaluate_strategy(
        self,
        test_queries: List[Dict[str, Any]],
        mode: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Runs evaluation for a single retrieval mode over the test queries.
        """
        total_queries = len(test_queries)
        if total_queries == 0:
            return {}

        ap_list: List[float] = []
        rr_list: List[float] = []
        ndcg10_list: List[float] = []
        p1_list: List[float] = []
        p3_list: List[float] = []
        p5_list: List[float] = []
        p10_list: List[float] = []
        r10_list: List[float] = []
        latencies: List[float] = []

        query_details: List[Dict[str, Any]] = []

        for q_item in test_queries:
            q_id = q_item["id"]
            query_text = q_item["query"]
            ground_truth = set(q_item["ground_truth"])

            t0 = time.perf_counter()
            response = self.engine.search(query_text, mode=mode, top_k=top_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            results = response["results"]
            retrieved_ids = [r["id"] for r in results]

            # Map retrieved_ids to canonical ground truth IDs if they match directly or via taxonomy relations
            # Each ground truth item can only be satisfied ONCE so len(ground_truth) remains the true denominator
            canonical_retrieved: List[str] = []
            matched_gts = set()
            for r_id in retrieved_ids:
                canonical_id = r_id
                if r_id in ground_truth and r_id not in matched_gts:
                    canonical_id = r_id
                    matched_gts.add(r_id)
                else:
                    doc = self.engine.get_technique(r_id)
                    matched_gt_for_r = None
                    for gt_id in ground_truth:
                        if gt_id in matched_gts:
                            continue
                        gt_doc = self.engine.get_technique(gt_id)
                        if (doc and (doc.get("successor_id") == gt_id or doc.get("parent_id") == gt_id)) or \
                           (gt_doc and (gt_doc.get("successor_id") == r_id or gt_doc.get("parent_id") == r_id)):
                            matched_gt_for_r = gt_id
                            break
                    if matched_gt_for_r is not None:
                        canonical_id = matched_gt_for_r
                        matched_gts.add(matched_gt_for_r)
                    else:
                        canonical_id = r_id
                canonical_retrieved.append(canonical_id)

            ap = average_precision(canonical_retrieved, ground_truth)
            rr = reciprocal_rank(canonical_retrieved, ground_truth)
            ndcg10 = ndcg_at_k(canonical_retrieved, ground_truth, k=10)
            p1 = precision_at_k(canonical_retrieved, ground_truth, k=1)
            p3 = precision_at_k(canonical_retrieved, ground_truth, k=3)
            p5 = precision_at_k(canonical_retrieved, ground_truth, k=5)
            p10 = precision_at_k(canonical_retrieved, ground_truth, k=10)
            r10 = recall_at_k(canonical_retrieved, ground_truth, k=10)

            ap_list.append(ap)
            rr_list.append(rr)
            ndcg10_list.append(ndcg10)
            p1_list.append(p1)
            p3_list.append(p3)
            p5_list.append(p5)
            p10_list.append(p10)
            r10_list.append(r10)
            latencies.append(elapsed_ms)

            query_details.append({
                "query_id": q_id,
                "category": q_item.get("category", "General"),
                "query": query_text,
                "ground_truth": list(ground_truth),
                "top_hit_id": retrieved_ids[0] if retrieved_ids else None,
                "top_hit_name": results[0]["name"] if results else None,
                "ap": round(ap, 4),
                "rr": round(rr, 4),
                "ndcg10": round(ndcg10, 4),
                "p5": round(p5, 4),
                "p10": round(p10, 4),
                "r10": round(r10, 4),
                "latency_ms": round(elapsed_ms, 2)
            })

        mean_map = sum(ap_list) / total_queries
        mean_mrr = sum(rr_list) / total_queries
        mean_ndcg10 = sum(ndcg10_list) / total_queries
        mean_p1 = sum(p1_list) / total_queries
        mean_p3 = sum(p3_list) / total_queries
        mean_p5 = sum(p5_list) / total_queries
        mean_p10 = sum(p10_list) / total_queries
        mean_r10 = sum(r10_list) / total_queries
        mean_latency = sum(latencies) / total_queries

        return {
            "mode": mode,
            "queries_evaluated": total_queries,
            "metrics": {
                "MAP": round(mean_map, 4),
                "MRR": round(mean_mrr, 4),
                "NDCG@10": round(mean_ndcg10, 4),
                "P@1": round(mean_p1, 4),
                "P@3": round(mean_p3, 4),
                "P@5": round(mean_p5, 4),
                "P@10": round(mean_p10, 4),
                "Recall@10": round(mean_r10, 4),
                "Mean_Latency_ms": round(mean_latency, 2)
            },
            "query_details": query_details
        }

    def evaluate_all(
        self,
        test_queries: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Runs comprehensive comparative benchmark across all four retrieval baselines.
        1. Pure BM25
        2. Pure Dense Vector (FAISS)
        3. Hybrid Fusion (BM25 + FAISS via RRF)
        4. Hybrid Fusion + Rocchio PRF
        """
        strategies = ["bm25", "dense", "hybrid", "hybrid_prf"]
        results_by_strategy = {}

        for strat in strategies:
            results_by_strategy[strat] = self.evaluate_strategy(test_queries, mode=strat, top_k=top_k)

        # Build comparative summary table
        comparison_rows = []
        strategy_labels = {
            "bm25": "1. Pure BM25 (Okapi)",
            "dense": "2. Pure Dense (FAISS)",
            "hybrid": "3. Hybrid Fusion (RRF)",
            "hybrid_prf": "4. Hybrid Fusion + Rocchio PRF"
        }

        for strat in strategies:
            m = results_by_strategy[strat]["metrics"]
            comparison_rows.append({
                "Strategy": strategy_labels[strat],
                "MAP": m["MAP"],
                "NDCG@10": m["NDCG@10"],
                "MRR": m["MRR"],
                "P@1": m["P@1"],
                "P@5": m["P@5"],
                "P@10": m["P@10"],
                "Recall@10": m["Recall@10"],
                "Latency (ms)": m["Mean_Latency_ms"]
            })

        return {
            "summary_table": comparison_rows,
            "details": results_by_strategy
        }
