"""
Streamlit App UI Integration Tests using streamlit.testing.v1.AppTest.
Verifies interactive features:
1. 1-Click Examiner Demo Scenarios button click and state synchronization
2. Sandbox hyperparameter adjustment and reset to academic defaults
3. PyArrow serialization of Cross-Method Consensus matrix
"""

import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")

def test_streamlit_examiner_demo_scenario_sync():
    """Verifies that clicking an Examiner Demo Scenario button synchronizes query across UI textareas."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    # Initial state
    assert at.session_state["active_scenario_key"] == "powershell_obfuscated"

    # Click vocab_mismatch button in 4-Way Comparison tab
    btn = [b for b in at.button if b.key and "compare_sc_btn_vocab_mismatch" in b.key][0]
    btn.click().run()
    assert not at.exception

    # Verify state and textareas updated
    assert at.session_state["active_scenario_key"] == "vocab_mismatch"
    expected_q = "Stealthily injecting foreign executable"
    assert expected_q in at.session_state["global_query"]
    assert expected_q in at.text_area(key="compare_query_textarea").value
    assert expected_q in at.text_area(key="sandbox_query_input").value
    assert expected_q in at.text_area(key="triage_query_input").value

def test_streamlit_sandbox_reset_button():
    """Verifies that the Reset button in the Sandbox restores academic default hyperparameters."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    # Mutate parameters to non-defaults via slider widgets
    at.slider(key="sandbox_k1").set_value(2.8)
    at.slider(key="sandbox_b").set_value(0.2)
    at.slider(key="sandbox_rrf_k").set_value(15)
    at.slider(key="sandbox_alpha").set_value(1.8)
    at.slider(key="sandbox_beta").set_value(1.5)

    # Find reset button and click
    reset_btn = at.button(key="sandbox_reset_btn")
    reset_btn.click().run()
    assert not at.exception

    # Verify restored to academic defaults
    assert at.session_state["bm25_k1"] == 1.5
    assert at.session_state["bm25_b"] == 0.75
    assert at.session_state["rrf_k"] == 60
    assert at.session_state["prf_alpha"] == 1.0
    assert at.session_state["prf_beta"] == 0.75
    assert at.session_state["prf_k_docs"] == 3
    assert at.session_state["prf_num_terms"] == 5

def test_streamlit_compare_all_modes_execution():
    """Verifies that compare_all_modes executes cleanly in the Streamlit app without AttributeError."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    # Change the query and trigger rerun
    at.text_area(key="compare_query_textarea").input("vssadmin delete shadows /all /quiet").run()
    assert not at.exception
    assert "vssadmin" in at.session_state["global_query"]


def test_streamlit_all_examiner_demo_scenarios():
    """Verifies that all 5 Examiner Demo Scenarios execute compare_all_modes cleanly without exception."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    scenarios = [
        "powershell_obfuscated",
        "vocab_mismatch",
        "short_rocchio",
        "cve_zeroday",
        "ransomware_multi"
    ]
    for sc_key in scenarios:
        btns = [b for b in at.button if b.key and f"compare_sc_btn_{sc_key}" in b.key]
        assert len(btns) == 1
        btns[0].click().run()
        assert not at.exception
        assert at.session_state["active_scenario_key"] == sc_key


