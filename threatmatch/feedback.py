import re
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from threatmatch.tokenizer import CyberRegexTokenizer, STANDARD_STOPWORDS
from threatmatch.indexer import InvertedIndex

GENERIC_CTI_STOPWORDS = {
    "adversaries", "adversary", "may", "use", "using", "used", "technique", "techniques",
    "system", "systems", "windows", "linux", "macos", "citation", "also", "one", "two",
    "such", "within", "given", "common", "commonly", "including", "another", "often",
    "various", "certain", "different", "multiple", "potential", "allow", "allows",
    "provide", "provides", "either", "without", "need", "needs", "perform", "performs",
    "description", "detection", "mitigations", "platforms", "tactics", "see"
}

@dataclass
class ExpansionTerm:
    term: str
    weight: float
    idf: float
    frequency_in_feedback: int

@dataclass
class PRFResult:
    original_query: str
    expanded_query: str
    feedback_doc_ids: List[str]
    feedback_doc_names: List[str]
    expansion_terms: List[ExpansionTerm]
    alpha: float
    beta: float
    k_feedback: int

class RocchioPRF:
    """
    Rocchio Pseudo-Relevance Feedback engine.
    Performs query reformulation by discovering high-weight domain context terms.
    """
    def __init__(
        self,
        index: InvertedIndex,
        alpha: float = 1.0,
        beta: float = 0.5,
        k_feedback: int = 3,
        num_expansion_terms: int = 3
    ):
        self.index = index
        self.alpha = alpha
        self.beta = beta
        self.k_feedback = k_feedback
        self.num_expansion_terms = num_expansion_terms

    def expand_query(
        self,
        query: str,
        first_pass_results: List[Dict[str, Any]],
        num_terms: Optional[int] = None,
        k_docs: Optional[int] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None
    ) -> PRFResult:
        """
        Executes Rocchio PRF on first-pass retrieval results and generates the expanded query.
        Allows dynamic overrides for num_terms, k_docs, alpha, and beta.
        """
        num_terms = self.num_expansion_terms if num_terms is None else num_terms
        k_docs = self.k_feedback if k_docs is None else k_docs
        eff_alpha = self.alpha if alpha is None else alpha
        eff_beta = self.beta if beta is None else beta

        query_tokens = self.index.tokenizer.tokenize(query)
        q0_set = {t.lower() for t in query_tokens}

        # Select top k pseudo-relevant documents D_r
        relevant_candidates = first_pass_results[:k_docs]
        if not relevant_candidates:
            return PRFResult(
                original_query=query,
                expanded_query=query,
                feedback_doc_ids=[],
                feedback_doc_names=[],
                expansion_terms=[],
                alpha=eff_alpha,
                beta=eff_beta,
                k_feedback=k_docs
            )

        feedback_doc_ids = [doc["id"] for doc in relevant_candidates]
        feedback_doc_names = [doc.get("name", "") for doc in relevant_candidates]
        num_rel = len(relevant_candidates)

        # Accumulate feedback weights: sum_{d in D_r} w_d * d(t)
        term_feedback_weights: Dict[str, float] = {}
        term_feedback_counts: Dict[str, int] = {}

        for rank_idx, doc in enumerate(relevant_candidates, start=1):
            rank_weight = 1.0 / math.sqrt(rank_idx)
            doc_id = doc["id"]
            doc_obj = self.index.documents.get(doc_id, {})
            text = doc_obj.get("search_text", "")
            doc_tokens = self.index.tokenizer.tokenize(text)
            doc_len = max(len(doc_tokens), 1)

            # Compute TF in doc
            doc_tf: Dict[str, int] = {}
            for t in doc_tokens:
                doc_tf[t] = doc_tf.get(t, 0) + 1

            # Weight terms by TF-IDF (normalized by doc length and rank)
            for t, tf in doc_tf.items():
                t_lower = t.lower()
                # Skip stopwords, short strings, and generic terms
                if t_lower in STANDARD_STOPWORDS or t_lower in GENERIC_CTI_STOPWORDS or len(t) <= 2:
                    continue

                # CRITICAL: Do NOT expand with Technique/Tactic/Mitigation IDs
                # Adding technique IDs directly causes BM25 to drift to unrelated techniques
                if re.match(r'^(?:T\d{4}(?:\.\d{3})?|TA\d{4}|M\d{4})$', t, re.IGNORECASE):
                    continue

                idf_val = self.index.idf(t)
                if idf_val <= 0.1:
                    continue

                # Normalized TF * IDF scaled by rank decay
                norm_weight = rank_weight * (tf / doc_len) * idf_val
                term_feedback_weights[t] = term_feedback_weights.get(t, 0.0) + norm_weight
                term_feedback_counts[t] = term_feedback_counts.get(t, 0) + tf

        # Apply Rocchio beta factor: (beta / |D_r|) * sum
        expansion_candidates: List[ExpansionTerm] = []
        for term, accumulated_weight in term_feedback_weights.items():
            # Skip terms already present in initial query
            if term.lower() in q0_set:
                continue

            rocchio_weight = (eff_beta / num_rel) * accumulated_weight
            idf_val = self.index.idf(term)

            expansion_candidates.append(
                ExpansionTerm(
                    term=term,
                    weight=rocchio_weight,
                    idf=idf_val,
                    frequency_in_feedback=term_feedback_counts.get(term, 0)
                )
            )

        # Sort candidate expansion terms descending by Rocchio weight
        expansion_candidates.sort(key=lambda x: x.weight, reverse=True)
        top_expansion = expansion_candidates[:num_terms]

        # Reformulate query: original_query + added domain terms
        added_str = " ".join([item.term for item in top_expansion])
        expanded_query = f"{query} {added_str}".strip() if added_str else query

        return PRFResult(
            original_query=query,
            expanded_query=expanded_query,
            feedback_doc_ids=feedback_doc_ids,
            feedback_doc_names=feedback_doc_names,
            expansion_terms=top_expansion,
            alpha=eff_alpha,
            beta=eff_beta,
            k_feedback=num_rel
        )

    def expand_dense_vector(
        self,
        query_vec: np.ndarray,
        feedback_doc_ids: List[str],
        dense_retriever: Any,
        alpha: Optional[float] = None,
        beta: Optional[float] = None
    ) -> np.ndarray:
        """
        Applies Rocchio vector arithmetic in dense embedding space:
          q_new = normalize(alpha * q_0 + (beta / |D_r|) * sum_{d in D_r} w_d * d)
        """
        if query_vec is None or not feedback_doc_ids:
            return query_vec

        eff_alpha = self.alpha if alpha is None else alpha
        eff_beta = self.beta if beta is None else beta

        emb_list = []
        weights = []
        for i, doc_id in enumerate(feedback_doc_ids, start=1):
            emb = dense_retriever.get_document_embedding(doc_id)
            if emb is not None:
                emb_list.append(emb)
                weights.append(1.0 / math.sqrt(i))

        if not emb_list:
            return query_vec

        weights_arr = np.array(weights, dtype=np.float32).reshape(-1, 1)
        embs_arr = np.array(emb_list, dtype=np.float32)
        weighted_sum = np.sum(embs_arr * weights_arr, axis=0) / np.sum(weights)

        new_vec = eff_alpha * query_vec + eff_beta * weighted_sum
        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec = new_vec / norm
        return new_vec.astype("float32")
