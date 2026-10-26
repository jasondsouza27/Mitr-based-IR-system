"""
Module 2: Dense Vector Retrieval Engine
Uses Sentence-Transformers (all-MiniLM-L6-v2) to generate 384-dimensional dense vectors
and indexes them using FAISS with Inner Product (Cosine Similarity) on L2-normalized vectors.
Includes caching and fast serialization.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try importing faiss
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not available, falling back to NumPy cosine similarity.")

# Lazy import sentence_transformers only when embedding is generated

class DenseVectorRetriever:
    """
    Dense semantic retrieval using 384-d embeddings and FAISS inner-product index.
    """
    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSION = 384

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or self.MODEL_NAME
        self.device = device
        self._model: Optional[Any] = None
        self.doc_ids: List[str] = []
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[Any] = None

    @property
    def model(self) -> Any:
        """Lazy loader for SentenceTransformer to save memory and startup time."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def build_index(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "search_text",
        batch_size: int = 64,
        show_progress: bool = True
    ) -> None:
        """
        Generates dense embeddings for all documents and constructs the FAISS index.
        """
        self.documents = {doc["id"]: doc for doc in documents}
        self.doc_ids = [doc["id"] for doc in documents]
        texts = [doc.get(text_field, "") for doc in documents]

        # Generate dense embeddings
        raw_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        self.embeddings = raw_embeddings

        # Build FAISS IndexFlatIP (Cosine similarity for normalized vectors)
        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(self.DIMENSION)
            index.add(self.embeddings)
            self.faiss_index = index
        else:
            self.faiss_index = None

    def encode_query(self, query: str) -> Optional[np.ndarray]:
        """Encodes query into normalized 384-d vector."""
        if not query or not query.strip():
            return None
        vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")
        return vec[0]

    def get_document_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Returns precomputed embedding vector for doc_id."""
        if self.embeddings is None or not self.doc_ids:
            return None
        try:
            idx = self.doc_ids.index(doc_id)
            return self.embeddings[idx]
        except ValueError:
            return None

    def search_vector(self, query_vec: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Computes cosine similarity against query vector and returns top_k ranked documents.
        """
        if query_vec is None or not self.doc_ids or self.embeddings is None:
            return []

        if query_vec.ndim == 1:
            vec_2d = query_vec.reshape(1, -1).astype("float32")
        else:
            vec_2d = query_vec.astype("float32")

        norm = np.linalg.norm(vec_2d)
        if norm > 0:
            vec_2d = vec_2d / norm

        top_k = min(top_k, len(self.doc_ids))

        if FAISS_AVAILABLE and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(vec_2d, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            dot_products = np.dot(self.embeddings, vec_2d[0])
            indices = np.argsort(-dot_products)[:top_k]
            scores = dot_products[indices]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            doc_id = self.doc_ids[idx]
            doc = self.documents.get(doc_id, {})
            results.append({
                "id": doc_id,
                "name": doc.get("name", ""),
                "tactics": doc.get("tactics", []),
                "platforms": doc.get("platforms", []),
                "description": doc.get("description", ""),
                "mitigations": doc.get("mitigations", []),
                "url": doc.get("url", ""),
                "score": float(score),
                "rank": rank,
                "method": "Dense_FAISS"
            })

        return results

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Encodes query, computes cosine similarity, and returns top_k ranked documents.
        """
        if not query or not query.strip():
            return []
        vec = self.encode_query(query)
        if vec is None:
            return []
        return self.search_vector(vec, top_k=top_k)

    def save(self, cache_dir: str) -> None:
        """
        Persists precomputed embeddings, doc IDs, and FAISS index to disk.
        """
        os.makedirs(cache_dir, exist_ok=True)
        # 1. Save embeddings
        if self.embeddings is not None:
            np.save(os.path.join(cache_dir, "embeddings.npy"), self.embeddings)

        # 2. Save document metadata
        meta = {
            "doc_ids": self.doc_ids,
            "dimension": self.DIMENSION,
            "model_name": self.model_name
        }
        with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 3. Save FAISS index
        if FAISS_AVAILABLE and self.faiss_index is not None:
            faiss.write_index(self.faiss_index, os.path.join(cache_dir, "faiss.index"))

    @classmethod
    def load(
        cls,
        cache_dir: str,
        documents: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ) -> "DenseVectorRetriever":
        """
        Fast-loads precomputed embeddings and FAISS index from disk.
        """
        retriever = cls(model_name=model_name, device=device)
        retriever.documents = {doc["id"]: doc for doc in documents}

        meta_path = os.path.join(cache_dir, "meta.json")
        emb_path = os.path.join(cache_dir, "embeddings.npy")
        faiss_path = os.path.join(cache_dir, "faiss.index")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        retriever.doc_ids = meta["doc_ids"]

        if os.path.exists(emb_path):
            retriever.embeddings = np.load(emb_path)

        if FAISS_AVAILABLE and os.path.exists(faiss_path):
            retriever.faiss_index = faiss.read_index(faiss_path)
        elif retriever.embeddings is not None and FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(retriever.DIMENSION)
            index.add(retriever.embeddings)
            retriever.faiss_index = index

        return retriever
