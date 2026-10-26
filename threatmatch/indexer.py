"""
Module 1: Cyber-Specific Inverted Index & Okapi BM25 Lexical Retrieval Engine
Implements:
- Document & Term Frequency tracking (df_t, tf_{t,d})
- Inverted Index construction and persistence
- Exact Okapi BM25 scoring formula
- Term-level score attribution / explainability
"""

import math
import pickle
import os
from typing import List, Dict, Tuple, Any, Optional
from threatmatch.tokenizer import CyberRegexTokenizer

class InvertedIndex:
    """
    Inverted Index structure mapping terms to postings lists:
    postings[term] = [(doc_id, term_frequency), ...]
    """
    def __init__(self, tokenizer: Optional[CyberRegexTokenizer] = None):
        self.tokenizer = tokenizer or CyberRegexTokenizer()
        # term -> list of (doc_id, tf)
        self.postings: Dict[str, List[Tuple[str, int]]] = {}
        # term -> document frequency (number of docs containing term)
        self.doc_freqs: Dict[str, int] = {}
        # doc_id -> total token count in document
        self.doc_lengths: Dict[str, int] = {}
        # doc_id -> {term: tf} for O(1) lookups
        self.doc_term_freqs: Dict[str, Dict[str, int]] = {}
        # doc_id -> full document dict
        self.documents: Dict[str, Dict[str, Any]] = {}
        # Collection statistics
        self.total_docs: int = 0
        self.avg_doc_len: float = 0.0

    def index_documents(self, documents: List[Dict[str, Any]], text_field: str = "search_text") -> None:
        """
        Indexes a list of documents. Each document must have an 'id' and 'text_field'.
        """
        self.documents = {doc["id"]: doc for doc in documents}
        self.total_docs = len(documents)
        self.postings.clear()
        self.doc_freqs.clear()
        self.doc_lengths.clear()
        self.doc_term_freqs.clear()

        total_length = 0

        for doc in documents:
            doc_id = doc["id"]
            text = doc.get(text_field, "")
            tokens = self.tokenizer.tokenize(text)
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            total_length += doc_len

            # Count term frequencies in this document
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            self.doc_term_freqs[doc_id] = tf_map

            for term, tf in tf_map.items():
                if term not in self.postings:
                    self.postings[term] = []
                    self.doc_freqs[term] = 0
                self.postings[term].append((doc_id, tf))
                self.doc_freqs[term] += 1

        self.avg_doc_len = total_length / self.total_docs if self.total_docs > 0 else 0.0

    def idf(self, term: str) -> float:
        """
        Calculates Okapi BM25 Robertson-Spärck Jones IDF:
        IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
        Guaranteed to be non-negative.
        """
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))

    def get_tf(self, term: str, doc_id: str) -> int:
        """Returns the raw term frequency of a term in a document in O(1) time."""
        if self.doc_term_freqs and doc_id in self.doc_term_freqs:
            return self.doc_term_freqs[doc_id].get(term, 0)
        if term not in self.postings:
            return 0
        for d_id, tf in self.postings[term]:
            if d_id == doc_id:
                return tf
        return 0

    def save(self, filepath: str) -> None:
        """Serializes the index to disk using pickle."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "postings": self.postings,
            "doc_freqs": self.doc_freqs,
            "doc_lengths": self.doc_lengths,
            "doc_term_freqs": self.doc_term_freqs,
            "documents": self.documents,
            "total_docs": self.total_docs,
            "avg_doc_len": self.avg_doc_len
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str, tokenizer: Optional[CyberRegexTokenizer] = None) -> "InvertedIndex":
        """Loads a serialized index from disk."""
        index = cls(tokenizer=tokenizer)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        index.postings = data["postings"]
        index.doc_freqs = data["doc_freqs"]
        index.doc_lengths = data["doc_lengths"]
        index.doc_term_freqs = data.get("doc_term_freqs", {})
        index.documents = data["documents"]
        index.total_docs = data["total_docs"]
        index.avg_doc_len = data["avg_doc_len"]
        return index


class BM25Retriever:
    """
    Okapi BM25 Lexical Retrieval Engine.
    Formula:
      Score(D, Q) = sum_{t in Q} IDF(t) * [ f(t, D) * (k1 + 1) ] / [ f(t, D) + k1 * (1 - b + b * (|D| / avgdl)) ]
    """
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def search(
        self,
        query: str,
        top_k: int = 10,
        term_weights: Optional[Dict[str, float]] = None,
        k1: Optional[float] = None,
        b: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes BM25 ranking for a query, optionally weighting terms and overriding k1/b.
        Returns ranked list of documents with scores and metadata.
        """
        if not query or not query.strip():
            return []

        query_tokens = self.index.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # Accumulate BM25 scores across matching documents
        scores: Dict[str, float] = {}
        matched_terms: Dict[str, List[str]] = {}

        avgdl = self.index.avg_doc_len
        eff_k1 = self.k1 if k1 is None else k1
        eff_b = self.b if b is None else b

        for token in set(query_tokens):
            if token not in self.index.postings:
                continue

            term_idf = self.index.idf(token)
            if term_idf <= 0.0:
                continue

            q_weight = 1.0
            if term_weights:
                q_weight = term_weights.get(token.lower(), term_weights.get(token, 1.0))

            for doc_id, tf in self.index.postings[token]:
                doc_len = self.index.doc_lengths.get(doc_id, avgdl)
                # BM25 term weight
                tf_component = (tf * (eff_k1 + 1.0)) / (tf + eff_k1 * (1.0 - eff_b + eff_b * (doc_len / avgdl)))
                score_contribution = q_weight * term_idf * tf_component

                scores[doc_id] = scores.get(doc_id, 0.0) + score_contribution
                if doc_id not in matched_terms:
                    matched_terms[doc_id] = []
                matched_terms[doc_id].append(token)

        # Sort documents by BM25 score descending
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], start=1):
            doc = self.index.documents[doc_id]
            results.append({
                "id": doc["id"],
                "name": doc["name"],
                "tactics": doc.get("tactics", []),
                "platforms": doc.get("platforms", []),
                "description": doc.get("description", ""),
                "mitigations": doc.get("mitigations", []),
                "url": doc.get("url", ""),
                "score": float(score),
                "rank": rank,
                "matched_terms": matched_terms.get(doc_id, []),
                "method": "BM25"
            })

        return results

    def explain(
        self,
        query: str,
        doc_id: str,
        k1: Optional[float] = None,
        b: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Provides detailed mathematical breakdown of BM25 score for a specific document.
        """
        if doc_id not in self.index.documents:
            return {"error": f"Document {doc_id} not found."}

        query_tokens = self.index.tokenizer.tokenize(query)
        doc = self.index.documents[doc_id]
        doc_len = self.index.doc_lengths.get(doc_id, self.index.avg_doc_len)
        avgdl = self.index.avg_doc_len
        eff_k1 = self.k1 if k1 is None else k1
        eff_b = self.b if b is None else b

        term_breakdowns = []
        total_score = 0.0

        for token in set(query_tokens):
            tf = self.index.get_tf(token, doc_id)
            idf = self.index.idf(token)
            tf_comp = (tf * (eff_k1 + 1.0)) / (tf + eff_k1 * (1.0 - eff_b + eff_b * (doc_len / avgdl))) if tf > 0 else 0.0
            contrib = idf * tf_comp
            total_score += contrib

            term_breakdowns.append({
                "term": token,
                "tf": tf,
                "df": self.index.doc_freqs.get(token, 0),
                "idf": round(idf, 4),
                "tf_component": round(tf_comp, 4),
                "score_contribution": round(contrib, 4)
            })

        # Sort terms by score contribution descending
        term_breakdowns.sort(key=lambda x: x["score_contribution"], reverse=True)

        return {
            "doc_id": doc_id,
            "doc_name": doc.get("name", ""),
            "doc_length": doc_len,
            "avg_doc_length": round(avgdl, 2),
            "k1": eff_k1,
            "b": eff_b,
            "total_bm25_score": round(total_score, 4),
            "terms": term_breakdowns
        }

    def get_doc_tfidf_vector(self, doc_id: str) -> Dict[str, float]:
        """
        Returns a sparse TF-IDF dictionary for a document: term -> tf * idf.
        Used by Rocchio Pseudo-Relevance Feedback.
        """
        if self.index.doc_term_freqs and doc_id in self.index.doc_term_freqs:
            tf_map = self.index.doc_term_freqs[doc_id]
            return {term: tf * self.index.idf(term) for term, tf in tf_map.items()}

        vector: Dict[str, float] = {}
        for term, postings in self.index.postings.items():
            for d_id, tf in postings:
                if d_id == doc_id:
                    idf_val = self.index.idf(term)
                    vector[term] = tf * idf_val
                    break
        return vector
