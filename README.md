# ThreatMatch: Automated Cyber Threat Intelligence & MITRE ATT&CK Retrieval Engine

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Information%20Retrieval%20(CSDC%207023)-007acc?style=for-the-badge" alt="Domain">
  <img src="https://img.shields.io/badge/Framework-MITRE%20ATT%26CK%20v14.1-e11d48?style=for-the-badge" alt="MITRE">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/UI-Streamlit%20Dark%20Theme-ff4b4b?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Tests-62%20Passed-success?style=for-the-badge" alt="PyTest">
</p>

---

## 📌 Repository Information
- **GitHub Repository:** [https://github.com/jasondsouza27/Mitr-based-IR-system](https://github.com/jasondsouza27/Mitr-based-IR-system)
- **Target Environment:** Academic Capstone / Mini-Project (Mumbai University BE Computer Engineering, Sem-VII)
- **Course:** Information Retrieval (CSDC 7023)
- **Corpus:** MITRE ATT&CK Enterprise STIX 2.1 (858 Techniques & Sub-techniques, 268 Mitigations, 15 Tactics)

---

## 1. Abstract

In modern Security Operations Centers (SOCs), incident responders face an overwhelming volume of unstructured threat reports, endpoint telemetry, and intrusion alerts. Mapping these raw observations to standard taxonomies like the MITRE ATT&CK framework is critical for rapid threat containment, but manual triage is slow and error-prone. Traditional Information Retrieval (IR) pipelines fail because standard tokenizers strip punctuation essential to technical entities (such as `CVE-2023-38606`, `T1059.001`, or `-ExecutionPolicy`), and pure keyword search misses high-level semantic intent.

**ThreatMatch** is an enterprise-grade Information Retrieval system engineered to index and retrieve MITRE ATT&CK techniques, tactics, and mitigations from unstructured incident logs. The system features:
1. **Domain-Specific CyberRegex Tokenizer:** Preserves Indicators of Compromise (IoCs), command flags, CVEs, and Windows registry hives intact.
2. **Dual-Index Storage Architecture:** Combines an inverted index for sparse lexical matching alongside a dense vector index for semantic intent.
3. **Hybrid Ranking via Reciprocal Rank Fusion (RRF):** Merges Okapi BM25 and dense FAISS cosine scores using ordinal rank aggregation ($k=60$).
4. **Dual-Space Rocchio Pseudo-Relevance Feedback (PRF):** Automates query expansion by extracting top discriminating domain terms and shifting dense query embeddings.
5. **Interactive Streamlit Web Dashboard:** Features 1-Click Examiner Demo Scenarios, a 4-Way Side-by-Side Baseline Comparison View, and an Interactive IR Parameter Sandbox.
6. **Rigorous IR Evaluation Suite:** Benchmarked across 25 standardized incident scenarios using MAP, NDCG@10, MRR, Precision@k, and Recall@k.

---

## 2. System Architecture & Pipeline Workflow

```
               [ Unstructured Incident Log / SOC Alert ]
                                   │
                                   ▼
                      [ CyberRegex Tokenizer ]
      (Preserves Technique IDs, Registry Paths, Flags, CVEs, Ports)
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
        [ BM25 Lexical Index ]            [ Dense Vector Index ]
     (Inverted Index + Okapi BM25)     (all-MiniLM-L6-v2 + FAISS FlatIP)
                  │                                 │
                  └────────────────┬────────────────┘
                                   ▼
                    [ Reciprocal Rank Fusion (RRF) ]
                       RRF(d) = Σ 1 / (60 + r_m(d))
                                   │
                                   ▼
               [ Dual-Space Rocchio Relevance Feedback ]
           ├─► Sparse TF-IDF Term Expansion (with Rank Decay)
           └─► Dense Vector Space Centroid Shift
                                   │
                                   ▼
            [ Ranked MITRE ATT&CK Techniques & Mitigations ]
```

---

## 3. Mathematical Foundations & IR Modules

### Module 1: Cyber-Specific Inverted Index & Tokenization (`threatmatch/tokenizer.py`)
Standard NLP tokenizers (such as `\w+` or NLTK's `word_tokenize`) corrupt critical Indicators of Compromise (IoCs):
* `T1059.001` is split into `t1059` and `001`.
* `CVE-2023-38606` is split into `cve`, `2023`, `38606`.
* `-ExecutionPolicy Bypass` loses its command-flag hyphen `-`.
* `HKLM\Software\Microsoft\Windows\CurrentVersion\Run` loses backslashes and registry hive structure.

`CyberRegexTokenizer` applies pattern-matching rules with entity classification:
* **Technique IDs:** `\bT\d{4}(?:[\./]\d{3})?\b` (handles dot & slash notation, e.g. `T1059/001` $\rightarrow$ `T1059.001`)
* **Tactics & Mitigations:** `\bTA\d{4}\b`, `\bM\d{4}\b`
* **CVE Identifiers:** `\bCVE-\d{4}-\d{4,7}\b`
* **Registry Hives & Keys:** `\b(?:HKEY_[A-Z_]+|HKLM|HKCU|HKCR|HKU|HKCC)(?:\\[A-Za-z0-9_\-\. ]{1,64})+\b`
* **Process Artifacts:** `\b[A-Za-z0-9_\-]+\.(?:exe|dll|sys|ps1|bat|vbs|sh|py|bin|elf|scr|cpl|hta|jar)\b`
* **Command Flags:** `(?:^|[\s\(\[\{"\',;])(?:--?[A-Za-z0-9_\-]+|\/[A-Za-z0-9_\?]+)`
* **Network Artifacts & Ports:** `\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b`, cyber ports (`3389`, `445`, `8080`, `443`, `88`)

> **Stop-Word Policy:** Standard English stop-word filtering applies strictly to conversational words (`the`, `is`, `with`). Technical entities bypass stop-word filtering completely.

---

### Module 2: Hybrid Retrieval Engine (BM25 + FAISS + RRF)

#### 1. Okapi BM25 Lexical Scoring (`threatmatch/indexer.py`)
Calculates non-linear term frequency saturation and document length normalization using exact Robertson-Spärck Jones IDF:

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

$$\text{IDF}(t) = \ln\left(1 + \frac{N - df_t + 0.5}{df_t + 0.5}\right)$$

* **$k_1 = 1.5$:** Calibrates term frequency saturation.
* **$b = 0.75$:** Normalizes against verbose MITRE documentation lengths.

#### 2. Dense Semantic Vector Retrieval (`threatmatch/dense.py`)
* Generates 384-dimensional dense semantic vectors using `sentence-transformers/all-MiniLM-L6-v2`.
* Embeddings are $L_2$-normalized: $\vec{v} = \frac{\vec{u}}{\|\vec{u}\|_2}$.
* Indexed via `faiss.IndexFlatIP` (Inner Product on normalized vectors $\equiv$ exact Cosine Similarity):

$$\text{Score}_{\text{Dense}}(D, Q) = \langle \vec{q}, \vec{d} \rangle = \sum_{i=1}^{384} q_i \cdot d_i$$

#### 3. Reciprocal Rank Fusion (`threatmatch/fusion.py`)
Combines disparate score distributions ($[0, \infty)$ from BM25 and $[-1, 1]$ from Cosine Similarity) using Cormack's rank aggregation formula:

$$RRF(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + r_m(d)} \quad (\text{where } k = 60)$$

---

### Module 3: Dual-Space Rocchio Pseudo-Relevance Feedback (`threatmatch/feedback.py`)
Overcomes the **vocabulary mismatch problem** when analysts use short, operational descriptions (e.g. `procdump lsass`):
1. **Candidate Selection:** Assumes the top-$k$ documents (default $k=3$) from Pass 1 are relevant ($D_r$).
2. **Sparse TF-IDF Expansion:**
   $$\text{Weight}(t) = \frac{\beta}{|D_r|} \sum_{d \in D_r} \left( \frac{1}{\sqrt{\text{rank}(d)}} \cdot \frac{f(t, d)}{|d|} \cdot \text{IDF}(t) \right)$$
   *Appends discriminating context terms (`mimikatz`, `memory`, `dump`, `sekurlsa`) while filtering technique IDs to prevent query drift.*
3. **Dense Vector Shifting:**
   $$\vec{q}_{\text{dense, new}} = \text{normalize}\left(\alpha \vec{q}_0 + \frac{\beta}{|D_r|} \sum_{d \in D_r} \frac{1}{\sqrt{\text{rank}(d)}} \vec{e}_d\right)$$

---

## 4. Empirical Benchmark Results

Evaluated across a standardized test collection of **25 realistic SOC incident scenarios** mapped to ground-truth ATT&CK techniques:

| Strategy | MAP | NDCG@10 | MRR | Precision@1 | Precision@5 | Recall@10 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Pure BM25 (Okapi)** | 0.8233 | 0.8477 | 0.7967 | 0.7200 | 0.1920 | 0.9600 | **1.65 ms** |
| **2. Pure Dense (FAISS)** | 0.7680 | 0.8017 | 0.7280 | 0.6000 | 0.1920 | 0.9600 | 27.62 ms |
| **3. Hybrid Fusion (RRF)** | 0.8413 | 0.8654 | 0.8013 | 0.7200 | 0.1920 | **1.0000** | 25.60 ms |
| **4. Hybrid + Rocchio PRF** | **0.8467** | **0.8702** | **0.8067** | **0.7200** | **0.2000** | **1.0000** | **38.70 ms** |

### Benchmark Key Takeaways:
1. **Monotonic Improvement:** Performance increases steadily from single baselines $\rightarrow$ Hybrid Fusion $\rightarrow$ Hybrid + Rocchio PRF across all ranking metrics.
2. **100% Recall@10:** Hybrid Fusion achieves a perfect Recall of `1.0000`, meaning zero target techniques were missed across all 25 incident queries.
3. **Sub-40ms Response Time:** Total pipeline execution completes in **38.7 milliseconds**, well below the 1,000ms threshold required for real-time SOC incident containment.

---

## 5. Web Dashboard Features (`app.py`)

1. **🎯 1-Click Examiner Demo Scenarios:** Top header bar with 5 pre-loaded scenarios (`⚡ Obfuscated PowerShell`, `🧠 Vocabulary Mismatch`, `✨ Short Query Rocchio`, `🎯 Exact CVE`, `🛡️ Ransomware Multi-Stage`) with an educational briefing explaining the IR concepts demonstrated.
2. **🔀 4-Way Baseline Comparison Tab:** Simultaneous 4-column layout comparing Top-5 techniques across all strategies with dynamic rank-shift badges (`▲ +X`, `▼ X`, `★ Rescued`, `★ Discovered by PRF`), Jaccard similarity metrics, and a Cross-Method Consensus Matrix.
3. **🎛️ IR Parameter Sandbox Tab:** Real-time sliders for BM25 ($k_1, b$), RRF ($k$), and Rocchio PRF ($\alpha, \beta, k_{docs}$) with sensitivity analysis, side-by-side impact diffing, and a 1-click reset to academic defaults.
4. **🔍 Incident Triage & Search Tab:** Free-form incident search returning MITRE technique cards, tactic tags, platforms, official defense mitigations, and a BM25 term-contribution scoring attribution table.
5. **🧪 CyberRegex Tokenizer Inspector Tab:** Side-by-side comparison of standard NLP tokenizers versus `CyberRegexTokenizer` with entity badge classification.
6. **📊 Benchmarking & IR Metrics Tab:** Interactive Plotly radar charts, bar graphs, and individual query score cards.
7. **📚 MITRE ATT&CK Corpus Browser Tab:** Filter and explore 858 techniques, 15 tactics, and 268 mitigations.

---

## 6. Mumbai University Syllabus Coverage (CSDC 7023)

| Unit / Syllabus Pillar | ThreatMatch Implementation | Key Equation / Metric |
| :--- | :--- | :--- |
| **Unit I: Text Preprocessing & Inverted Index** | `CyberRegexTokenizer` regex scanner, stopword filtering bypass for technical tokens, postings lists `postings[t] = [(doc_id, tf)]`, document frequencies ($df_t$), average doc length ($avgdl$). | Postings List, $df_t, tf_{t,d}$ |
| **Unit II: Retrieval Models (Probabilistic & Vector Space)** | Okapi BM25 Robertson-Spärck Jones IDF model, 384-dimensional dense semantic vector space model, FAISS inner-product cosine similarity. | $BM25(D, Q)$, $\langle \vec{q}, \vec{d} \rangle$ |
| **Unit III: Relevance Feedback & Query Expansion** | Rocchio Pseudo-Relevance Feedback (PRF) vector arithmetic, top-$k$ feedback document selection, TF-IDF term discriminating extraction with rank decay. | $\vec{q}_{\text{new}} = \alpha \vec{q}_0 + \frac{\beta}{|D_r|}\sum \vec{d}$ |
| **Unit IV: System Evaluation** | Mean Average Precision (MAP), Normalized Discounted Cumulative Gain (NDCG@10), Mean Reciprocal Rank (MRR), Precision@k, Recall@k over 25 standardized incident scenarios. | MAP, NDCG@10, MRR, P@k, R@k |

---

## 7. Project Directory Structure

```
Mitr-based-IR-system/
├── data/
│   ├── enterprise_attack_raw.json   # Downloaded MITRE STIX 2.1 JSON (45.7 MB)
│   ├── mitre_attack_corpus.json     # 858 structured techniques & mitigations
│   ├── evaluation_queries.json      # 25 standardized incident queries with ground truth
│   ├── benchmark_results.json       # Precomputed evaluation metrics
│   └── cache/                       # Serialized BM25 index & FAISS dense embeddings
│       ├── embeddings.npy           # 384-d normalized document embeddings
│       ├── faiss.index              # FAISS FlatIP vector index
│       ├── inverted_index.pkl       # Serialized inverted index & postings lists
│       └── meta.json                # Document ID index mapping
│
├── threatmatch/
│   ├── __init__.py
│   ├── tokenizer.py                 # Module 1: CyberRegexTokenizer & entity extraction
│   ├── indexer.py                   # Module 1: InvertedIndex & Okapi BM25 retrieval
│   ├── dense.py                     # Module 2: DenseVectorRetriever (FAISS + all-MiniLM-L6-v2)
│   ├── fusion.py                    # Module 2: Reciprocal Rank Fusion (RRF)
│   ├── feedback.py                  # Module 3: Dual-space Rocchio PRF
│   ├── evaluation.py                # Module 4: BenchmarkEvaluator (MAP, NDCG, MRR)
│   └── engine.py                    # Unified ThreatMatch Retrieval Engine facade
│
├── tests/
│   ├── test_tokenizer.py            # Tokenizer regex & IoC preservation tests
│   ├── test_indexer.py              # Inverted index & BM25 scoring tests
│   ├── test_dense.py                # FAISS vector search tests
│   ├── test_fusion.py               # RRF rank math tests
│   ├── test_feedback.py             # Rocchio query expansion tests
│   ├── test_evaluation.py           # MAP, NDCG, MRR mathematical correctness tests
│   ├── test_benchmark_dataset.py    # Ground truth validation against corpus
│   ├── test_boost_features.py       # Parameter sandbox & comparison matrix tests
│   ├── test_ui_streamlit.py         # Headless Streamlit AppTest integration tests
│   └── test_engine.py               # End-to-end integration & sub-second latency tests
│
├── scripts/
│   ├── build_corpus.py              # Extracts structured corpus from STIX
│   ├── build_embeddings.py          # Precomputes FAISS & BM25 indexes to cache
│   └── run_benchmark.py             # Runs 25-query comparative benchmark
│
├── .streamlit/
│   └── config.toml                  # Streamlit server & file-watcher configuration
├── app.py                           # Streamlit SOC Analyst Web Dashboard
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Ignored cache & virtual environment files
└── README.md                        # Documentation & Syllabus Mapping
```

---

## 8. Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/jasondsouza27/Mitr-based-IR-system.git
cd Mitr-based-IR-system
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Automated Tests (62 Tests)
```bash
python -m pytest tests/ -v
```

### 4. Run System Benchmarks
```bash
python scripts/run_benchmark.py
```

### 5. Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 9. Dependencies (`requirements.txt`)
* `streamlit>=1.30.0`
* `rank-bm25>=0.2.2`
* `sentence-transformers>=2.2.2`
* `faiss-cpu>=1.7.4`
* `scikit-learn>=1.3.0`
* `numpy>=1.24.0`
* `pandas>=2.0.0`
* `plotly>=5.18.0`
* `pytest>=7.4.0`

---

## 10. License & Academic Attribution
Developed for the **Information Retrieval (CSDC 7023)** academic curriculum, Department of Computer Engineering, University of Mumbai.

- **Author:** Jason Dsouza ([@jasondsouza27](https://github.com/jasondsouza27))
- **MITRE ATT&CK®:** Copyright © 2026 The MITRE Corporation. Used under the MITRE ATT&CK Terms of Use.
