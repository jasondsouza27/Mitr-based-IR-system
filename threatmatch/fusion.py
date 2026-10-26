"""
Module 2: Score Merging via Reciprocal Rank Fusion (RRF)
Implements:
  RRF(d) = sum_{m in {BM25, Dense}} 1 / (k + r_m(d))   (k = 60)
Combines sparse lexical rankings and dense semantic rankings.
"""

from typing import List, Dict, Any, Optional

class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) algorithm for fusing disparate retrieval results.
    Cormack, Clarke, and Büttcher (SIGIR 2009).
    """
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: int = 10,
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Merges BM25 and Dense search result lists using Reciprocal Rank Fusion.
        Allows overriding smoothing constant k dynamically.
        """
        doc_data: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}
        bm25_ranks: Dict[str, int] = {}
        bm25_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        dense_scores: Dict[str, float] = {}
        matched_terms_map: Dict[str, List[str]] = {}

        eff_k = self.k if k is None else k

        # 1. Process BM25 rankings
        for item in bm25_results:
            d_id = item["id"]
            rank = item["rank"]
            score = item["score"]
            bm25_ranks[d_id] = rank
            bm25_scores[d_id] = score
            matched_terms_map[d_id] = item.get("matched_terms", [])
            doc_data[d_id] = item
            rrf_scores[d_id] = rrf_scores.get(d_id, 0.0) + (1.0 / (eff_k + rank))

        # 2. Process Dense rankings
        for item in dense_results:
            d_id = item["id"]
            rank = item["rank"]
            score = item["score"]
            dense_ranks[d_id] = rank
            dense_scores[d_id] = score
            if d_id not in doc_data:
                doc_data[d_id] = item
            rrf_scores[d_id] = rrf_scores.get(d_id, 0.0) + (1.0 / (eff_k + rank))

        # 3. Sort by fused RRF score descending
        sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for final_rank, (doc_id, rrf_score) in enumerate(sorted_fused[:top_k], start=1):
            base_doc = doc_data[doc_id]
            results.append({
                "id": doc_id,
                "name": base_doc.get("name", ""),
                "tactics": base_doc.get("tactics", []),
                "platforms": base_doc.get("platforms", []),
                "description": base_doc.get("description", ""),
                "mitigations": base_doc.get("mitigations", []),
                "url": base_doc.get("url", ""),
                "score": float(rrf_score),
                "rrf_score": float(rrf_score),
                "bm25_rank": bm25_ranks.get(doc_id),
                "bm25_score": bm25_scores.get(doc_id),
                "dense_rank": dense_ranks.get(doc_id),
                "dense_score": dense_scores.get(doc_id),
                "matched_terms": matched_terms_map.get(doc_id, []),
                "rank": final_rank,
                "method": "Hybrid_RRF"
            })

        return results
