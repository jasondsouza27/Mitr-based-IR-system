"""
ThreatMatch Unified Retrieval Engine.
Coordinates:
  - CyberRegexTokenizer
  - BM25 Inverted Index
  - Dense Vector Index (FAISS + Sentence-Transformers)
  - Reciprocal Rank Fusion (RRF)
  - Rocchio Pseudo-Relevance Feedback (PRF)
Delivers sub-second incident triage and flexible search modes.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from threatmatch.tokenizer import CyberRegexTokenizer
from threatmatch.indexer import InvertedIndex, BM25Retriever
from threatmatch.dense import DenseVectorRetriever
from threatmatch.fusion import ReciprocalRankFusion
from threatmatch.feedback import RocchioPRF, PRFResult

class ThreatMatchEngine:
    """
    Unified retrieval engine for MITRE ATT&CK CTI.
    """
    def __init__(
        self,
        corpus_path: str = "data/mitre_attack_corpus.json",
        cache_dir: str = "data/cache",
        rrf_k: int = 60,
        prf_alpha: float = 1.0,
        prf_beta: float = 0.75,
        prf_k_docs: int = 3,
        prf_num_terms: int = 5
    ):
        self.corpus_path = corpus_path
        self.cache_dir = cache_dir

        # 1. Load Corpus
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found at {corpus_path}. Run build_corpus.py first.")

        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus_data = json.load(f)

        self.techniques = self.corpus_data["techniques"]
        self.tactics = self.corpus_data.get("tactics", {})
        self.tech_map = {t["id"]: t for t in self.techniques}

        # 2. Tokenizer
        self.tokenizer = CyberRegexTokenizer()

        # 3. Inverted Index & BM25
        bm25_cache = os.path.join(cache_dir, "inverted_index.pkl")
        if os.path.exists(bm25_cache):
            self.index = InvertedIndex.load(bm25_cache, tokenizer=self.tokenizer)
        else:
            self.index = InvertedIndex(tokenizer=self.tokenizer)
            self.index.index_documents(self.techniques, text_field="search_text")

        self.bm25 = BM25Retriever(self.index, k1=1.5, b=0.75)

        # 4. Dense Retriever & FAISS
        meta_cache = os.path.join(cache_dir, "meta.json")
        emb_cache = os.path.join(cache_dir, "embeddings.npy")
        if os.path.exists(meta_cache) and os.path.exists(emb_cache):
            self.dense = DenseVectorRetriever.load(cache_dir, self.techniques)
        else:
            self.dense = DenseVectorRetriever()
            self.dense.build_index(self.techniques, text_field="search_text")

        # Warm up the sentence transformer model so first query doesn't pay cold-start load time
        _ = self.dense.model.encode(["threatmatch warmup"], convert_to_numpy=True)

        # 5. Fusion & Feedback
        self.fuser = ReciprocalRankFusion(k=rrf_k)
        self.prf = RocchioPRF(
            index=self.index,
            alpha=prf_alpha,
            beta=prf_beta,
            k_feedback=prf_k_docs,
            num_expansion_terms=prf_num_terms
        )

    def _resolve_revoked_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Maps revoked/superseded MITRE ATT&CK techniques to their active successors
        so incident responders receive valid, actionable techniques.
        """
        resolved: List[Dict[str, Any]] = []
        seen_ids = set()

        for item in results:
            tech_id = item["id"]
            meta = self.tech_map.get(tech_id, {})
            is_revoked = meta.get("is_revoked", False)
            successor_id = meta.get("successor_id")

            if is_revoked and successor_id and successor_id in self.tech_map:
                active_meta = self.tech_map[successor_id]
                if successor_id in seen_ids:
                    continue
                seen_ids.add(successor_id)
                new_item = dict(item)
                new_item["id"] = successor_id
                new_item["name"] = active_meta.get("name", item.get("name", ""))
                new_item["tactics"] = active_meta.get("tactics", item.get("tactics", []))
                new_item["platforms"] = active_meta.get("platforms", item.get("platforms", []))
                new_item["description"] = active_meta.get("description", item.get("description", ""))
                new_item["mitigations"] = active_meta.get("mitigations", item.get("mitigations", []))
                new_item["url"] = active_meta.get("url", item.get("url", ""))
                new_item["superseded_from"] = tech_id
                resolved.append(new_item)
            else:
                if tech_id not in seen_ids:
                    seen_ids.add(tech_id)
                    resolved.append(item)

        for rank, r in enumerate(resolved, start=1):
            r["rank"] = rank

        return resolved

    def search(
        self,
        query: str,
        mode: str = "hybrid_prf",
        top_k: int = 10,
        pool_size: int = 50,
        bm25_k1: Optional[float] = None,
        bm25_b: Optional[float] = None,
        rrf_k: Optional[int] = None,
        prf_alpha: Optional[float] = None,
        prf_beta: Optional[float] = None,
        prf_k_docs: Optional[int] = None,
        prf_num_terms: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes search across the specified retrieval pipeline mode:
          - 'bm25': Pure Okapi BM25 Lexical Retrieval
          - 'dense': Pure Dense Vector Search (Sentence-Transformers + FAISS)
          - 'hybrid': Reciprocal Rank Fusion (BM25 + Dense)
          - 'hybrid_prf': Hybrid Fusion + Rocchio Pseudo-Relevance Feedback (2-pass)
        Allows dynamic overrides for BM25 (k1, b), RRF (k), and Rocchio PRF (alpha, beta, k_docs, num_terms).
        """
        t_start = time.perf_counter()
        mode_clean = mode.lower().strip()
        prf_result_data = None
        effective_query = query

        if not query or not query.strip():
            return {
                "query": query,
                "effective_query": query,
                "mode": mode,
                "top_k": top_k,
                "latency_ms": 0.0,
                "prf_info": None,
                "results": []
            }

        if mode_clean == "bm25":
            raw_results = self.bm25.search(query, top_k=pool_size, k1=bm25_k1, b=bm25_b)
            results = self._resolve_revoked_results(raw_results)[:top_k]

        elif mode_clean == "dense":
            raw_results = self.dense.search(query, top_k=pool_size)
            results = self._resolve_revoked_results(raw_results)[:top_k]

        elif mode_clean == "hybrid":
            bm25_pool = self.bm25.search(query, top_k=pool_size, k1=bm25_k1, b=bm25_b)
            dense_pool = self.dense.search(query, top_k=pool_size)
            fused = self.fuser.fuse(bm25_pool, dense_pool, top_k=pool_size, k=rrf_k)
            results = self._resolve_revoked_results(fused)[:top_k]

        elif mode_clean in ("hybrid_prf", "prf"):
            # Pass 1: Initial Hybrid Retrieval
            bm25_pass1 = self.bm25.search(query, top_k=pool_size, k1=bm25_k1, b=bm25_b)
            query_vec = self.dense.encode_query(query)
            dense_pass1 = self.dense.search_vector(query_vec, top_k=pool_size) if query_vec is not None else []
            initial_fusion = self.fuser.fuse(bm25_pass1, dense_pass1, top_k=pool_size, k=rrf_k)

            # Rocchio PRF on top-k candidates
            eff_alpha = self.prf.alpha if prf_alpha is None else prf_alpha
            eff_beta = self.prf.beta if prf_beta is None else prf_beta
            eff_k_docs = self.prf.k_feedback if prf_k_docs is None else prf_k_docs
            eff_num_terms = self.prf.num_expansion_terms if prf_num_terms is None else prf_num_terms

            prf_info: PRFResult = self.prf.expand_query(
                query,
                initial_fusion,
                num_terms=eff_num_terms,
                k_docs=eff_k_docs,
                alpha=eff_alpha,
                beta=eff_beta
            )
            effective_query = prf_info.expanded_query

            # Build term weights: original terms = eff_alpha, expansion terms = eff_beta * normalized weight (<= 0.35)
            q_orig_tokens = set(t.lower() for t in self.tokenizer.tokenize(query))
            term_weights: Dict[str, float] = {t: eff_alpha for t in q_orig_tokens}
            if prf_info.expansion_terms:
                max_w = max(t.weight for t in prf_info.expansion_terms) if prf_info.expansion_terms else 1.0
                for exp in prf_info.expansion_terms:
                    scaled_w = eff_beta * (exp.weight / max(max_w, 1e-6))
                    term_weights[exp.term.lower()] = max(0.1, min(scaled_w, 0.35))

            # Pass 2: Search with weighted query for BM25 and Rocchio dense vector for Dense
            bm25_pass2 = self.bm25.search(effective_query, top_k=pool_size, term_weights=term_weights, k1=bm25_k1, b=bm25_b)
            updated_dense_vec = self.prf.expand_dense_vector(
                query_vec,
                prf_info.feedback_doc_ids,
                self.dense,
                alpha=eff_alpha,
                beta=eff_beta
            )
            dense_pass2 = self.dense.search_vector(updated_dense_vec, top_k=pool_size) if updated_dense_vec is not None else []
            fused_pass2 = self.fuser.fuse(bm25_pass2, dense_pass2, top_k=pool_size, k=rrf_k)

            # Blend Pass 1 initial consensus (55%) with Pass 2 expanded refinement (45%)
            combined_scores: Dict[str, float] = {}
            doc_map: Dict[str, Dict[str, Any]] = {}
            for r in initial_fusion:
                doc_map[r["id"]] = r
                combined_scores[r["id"]] = combined_scores.get(r["id"], 0.0) + 0.55 * r["score"]
            for r in fused_pass2:
                if r["id"] not in doc_map:
                    doc_map[r["id"]] = r
                combined_scores[r["id"]] = combined_scores.get(r["id"], 0.0) + 0.45 * r["score"]

            sorted_final = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            fused_combined = []
            for rank, (d_id, sc) in enumerate(sorted_final[:pool_size], start=1):
                item = dict(doc_map[d_id])
                item["score"] = float(sc)
                item["rrf_score"] = float(sc)
                item["rank"] = rank
                item["method"] = "Hybrid_RRF_with_PRF"
                fused_combined.append(item)

            results = self._resolve_revoked_results(fused_combined)[:top_k]

            prf_result_data = {
                "original_query": prf_info.original_query,
                "expanded_query": prf_info.expanded_query,
                "feedback_doc_ids": prf_info.feedback_doc_ids,
                "feedback_doc_names": prf_info.feedback_doc_names,
                "expansion_terms": [
                    {
                        "term": t.term,
                        "weight": round(float(t.weight), 4),
                        "idf": round(float(t.idf), 4),
                        "freq": t.frequency_in_feedback
                    }
                    for t in prf_info.expansion_terms
                ],
                "alpha": prf_info.alpha,
                "beta": prf_info.beta,
                "k_feedback": prf_info.k_feedback
            }
        else:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Choose from 'bm25', 'dense', 'hybrid', 'hybrid_prf'.")

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "query": query,
            "effective_query": effective_query,
            "mode": mode,
            "top_k": top_k,
            "latency_ms": round(latency_ms, 2),
            "prf_info": prf_result_data,
            "results": results
        }

    def explain_bm25(
        self,
        query: str,
        doc_id: str,
        k1: Optional[float] = None,
        b: Optional[float] = None
    ) -> Dict[str, Any]:
        """Explains BM25 score component-by-component."""
        return self.bm25.explain(query, doc_id, k1=k1, b=b)

    def compare_all_modes(
        self,
        query: str,
        top_k: int = 5,
        bm25_k1: Optional[float] = None,
        bm25_b: Optional[float] = None,
        rrf_k: Optional[int] = None,
        prf_alpha: Optional[float] = None,
        prf_beta: Optional[float] = None,
        prf_k_docs: Optional[int] = None,
        prf_num_terms: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes all 4 retrieval baselines (BM25, Dense FAISS, Hybrid RRF, Hybrid RRF + PRF)
        on the exact same query, and computes cross-method rank shifts and consensus metrics.
        """
        bm25_res = self.search(
            query=query,
            mode="bm25",
            top_k=top_k,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b
        )
        dense_res = self.search(
            query=query,
            mode="dense",
            top_k=top_k
        )
        hybrid_res = self.search(
            query=query,
            mode="hybrid",
            top_k=top_k,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            rrf_k=rrf_k
        )
        hybrid_prf_res = self.search(
            query=query,
            mode="hybrid_prf",
            top_k=top_k,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            rrf_k=rrf_k,
            prf_alpha=prf_alpha,
            prf_beta=prf_beta,
            prf_k_docs=prf_k_docs,
            prf_num_terms=prf_num_terms
        )

        # Build rank lookups
        bm25_ranks = {r["id"]: r["rank"] for r in bm25_res["results"]}
        dense_ranks = {r["id"]: r["rank"] for r in dense_res["results"]}
        hybrid_ranks = {r["id"]: r["rank"] for r in hybrid_res["results"]}
        prf_ranks = {r["id"]: r["rank"] for r in hybrid_prf_res["results"]}

        # Annotate Dense results with rank comparison vs BM25
        for r in dense_res["results"]:
            d_id = r["id"]
            b_rank = bm25_ranks.get(d_id)
            r["bm25_rank"] = b_rank
            r["shift_vs_bm25"] = (b_rank - r["rank"]) if b_rank is not None else None

        # Annotate Hybrid results with rank comparison vs BM25 & Dense
        for r in hybrid_res["results"]:
            d_id = r["id"]
            b_rank = bm25_ranks.get(d_id)
            d_rank = dense_ranks.get(d_id)
            r["bm25_rank"] = b_rank
            r["dense_rank"] = d_rank
            r["shift_vs_bm25"] = (b_rank - r["rank"]) if b_rank is not None else None
            r["shift_vs_dense"] = (d_rank - r["rank"]) if d_rank is not None else None

        # Annotate Hybrid+PRF results with rank comparison vs Hybrid & BM25
        for r in hybrid_prf_res["results"]:
            d_id = r["id"]
            h_rank = hybrid_ranks.get(d_id)
            b_rank = bm25_ranks.get(d_id)
            d_rank = dense_ranks.get(d_id)
            r["hybrid_rank"] = h_rank
            r["bm25_rank"] = b_rank
            r["dense_rank"] = d_rank
            r["shift_vs_hybrid"] = (h_rank - r["rank"]) if h_rank is not None else None
            r["shift_vs_bm25"] = (b_rank - r["rank"]) if b_rank is not None else None

        # Aggregate unique techniques matrix
        all_ids = list(dict.fromkeys(
            [r["id"] for r in hybrid_prf_res["results"]] +
            [r["id"] for r in hybrid_res["results"]] +
            [r["id"] for r in dense_res["results"]] +
            [r["id"] for r in bm25_res["results"]]
        ))

        matrix = []
        for tech_id in all_ids:
            tech_obj = self.tech_map.get(tech_id, {})
            b_rk = bm25_ranks.get(tech_id)
            d_rk = dense_ranks.get(tech_id)
            h_rk = hybrid_ranks.get(tech_id)
            p_rk = prf_ranks.get(tech_id)

            if p_rk is not None and h_rk is None and b_rk is None and d_rk is None:
                status = "Discovered by Rocchio PRF"
            elif p_rk is not None and h_rk is not None and p_rk < h_rk:
                status = "Boosted by Rocchio PRF"
            elif b_rk is not None and d_rk is not None:
                status = "Strong Dual-Modal Consensus"
            elif d_rk is not None and b_rk is None:
                status = "Rescued by Dense Semantics"
            elif b_rk is not None and d_rk is None:
                status = "Lexical Exact Match"
            else:
                status = "Hybrid Fusion Candidate"

            matrix.append({
                "id": tech_id,
                "name": tech_obj.get("name", tech_id),
                "tactics": ", ".join(tech_obj.get("tactics", [])),
                "bm25_rank": f"#{b_rk}" if b_rk is not None else "—",
                "dense_rank": f"#{d_rk}" if d_rk is not None else "—",
                "hybrid_rank": f"#{h_rk}" if h_rk is not None else "—",
                "hybrid_prf_rank": f"#{p_rk}" if p_rk is not None else "—",
                "status": status
            })

        # Overlap and Jaccard metrics
        set_bm25 = set(bm25_ranks.keys())
        set_dense = set(dense_ranks.keys())
        set_hybrid = set(hybrid_ranks.keys())
        set_prf = set(prf_ranks.keys())

        jaccard_bm25_dense = len(set_bm25 & set_dense) / len(set_bm25 | set_dense) if (set_bm25 | set_dense) else 0.0
        jaccard_hybrid_prf = len(set_hybrid & set_prf) / len(set_hybrid | set_prf) if (set_hybrid | set_prf) else 0.0
        all_four_consensus = set_bm25 & set_dense & set_hybrid & set_prf

        return {
            "query": query,
            "top_k": top_k,
            "bm25": bm25_res,
            "dense": dense_res,
            "hybrid": hybrid_res,
            "hybrid_prf": hybrid_prf_res,
            "comparison_matrix": matrix,
            "metrics": {
                "jaccard_bm25_dense": round(jaccard_bm25_dense, 3),
                "jaccard_hybrid_prf": round(jaccard_hybrid_prf, 3),
                "all_four_consensus_count": len(all_four_consensus),
                "total_unique_candidates": len(all_ids),
                "total_pipeline_latency_ms": round(
                    bm25_res["latency_ms"] + dense_res["latency_ms"] +
                    hybrid_res["latency_ms"] + hybrid_prf_res["latency_ms"], 2
                )
            }
        }

    def compare_tokenization(self, text: str) -> Dict[str, Any]:
        """Compares cyber regex tokenization against standard NLP tokenization."""
        return self.tokenizer.compare_tokenizers(text)

    def get_technique(self, tech_id: str) -> Optional[Dict[str, Any]]:
        """Fetches complete technique metadata by ID."""
        return self.tech_map.get(tech_id)

    def __getattr__(self, name: str) -> Any:
        """Fallback dynamic resolution for engine methods across module reloads or stale instances."""
        import sys
        mod = sys.modules.get("threatmatch.engine")
        if mod is not None:
            cls = getattr(mod, "ThreatMatchEngine", None)
            if cls is not None and hasattr(cls, name):
                attr = getattr(cls, name)
                if callable(attr):
                    return attr.__get__(self, self.__class__)
                return attr
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


