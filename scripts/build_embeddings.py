"""
Embedding and Index Builder for ThreatMatch.
Precomputes dense 384-d vectors for all corpus documents using all-MiniLM-L6-v2
and builds the FAISS IndexFlatIP index. Also precomputes and saves the BM25 inverted index.
"""

import json
import os
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from threatmatch.tokenizer import CyberRegexTokenizer
from threatmatch.indexer import InvertedIndex
from threatmatch.dense import DenseVectorRetriever

def build_all_indices(
    corpus_path: str = "data/mitre_attack_corpus.json",
    cache_dir: str = "data/cache"
):
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    techniques = corpus["techniques"]
    print(f"Total techniques to index: {len(techniques)}")

    os.makedirs(cache_dir, exist_ok=True)

    # 1. Build & Save Inverted Index (BM25)
    print("\n--- Building BM25 Inverted Index ---")
    t0 = time.time()
    tokenizer = CyberRegexTokenizer()
    inv_index = InvertedIndex(tokenizer=tokenizer)
    inv_index.index_documents(techniques, text_field="search_text")
    bm25_cache = os.path.join(cache_dir, "inverted_index.pkl")
    inv_index.save(bm25_cache)
    print(f"BM25 Inverted Index created in {time.time() - t0:.2f}s:")
    print(f"  Vocabulary size: {len(inv_index.postings)}")
    print(f"  Average doc length: {inv_index.avg_doc_len:.2f} tokens")
    print(f"  Saved to: {bm25_cache}")

    # 2. Build & Save Dense Vector Index (FAISS)
    print("\n--- Building Dense Vector Index (Sentence-Transformers + FAISS) ---")
    t0 = time.time()
    dense_retriever = DenseVectorRetriever()
    print("Encoding corpus documents with all-MiniLM-L6-v2...")
    dense_retriever.build_index(techniques, text_field="search_text", batch_size=64, show_progress=True)
    dense_retriever.save(cache_dir)
    print(f"Dense embeddings & FAISS index generated in {time.time() - t0:.2f}s:")
    print(f"  Embeddings shape: {dense_retriever.embeddings.shape}")
    print(f"  Saved to: {cache_dir}")

    print("\n[SUCCESS] All indexes successfully built and cached for sub-second retrieval!")

if __name__ == "__main__":
    build_all_indices()
