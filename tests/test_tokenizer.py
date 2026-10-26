"""
Unit tests for Module 1: CyberRegexTokenizer
"""

import pytest
from threatmatch.tokenizer import CyberRegexTokenizer, Token

@pytest.fixture
def tokenizer():
    return CyberRegexTokenizer()

def test_preserves_mitre_technique_ids(tokenizer):
    text = "Adversary utilized T1059.001 and T1055 alongside T1003.001 for execution."
    tokens = tokenizer.tokenize(text)
    assert "T1059.001" in tokens
    assert "T1055" in tokens
    assert "T1003.001" in tokens

def test_preserves_cve_identifiers(tokenizer):
    text = "Vulnerability CVE-2023-38606 was exploited alongside CVE-2021-44228."
    tokens = tokenizer.tokenize(text)
    assert "CVE-2023-38606" in tokens
    assert "CVE-2021-44228" in tokens

def test_preserves_command_flags(tokenizer):
    text = "powershell.exe -enc -w hidden -nop /add /v /c"
    tokens = tokenizer.tokenize(text)
    assert "-enc" in tokens
    assert "-nop" in tokens
    assert "/add" in tokens
    assert "/v" in tokens

def test_preserves_registry_paths(tokenizer):
    text = "Key written to HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run and HKCU\\Software\\Classes."
    tokens = tokenizer.tokenize(text)
    assert any("CurrentVersion\\Run" in t for t in tokens)

def test_preserves_process_artifacts(tokenizer):
    text = "Observed processes: powershell.exe, lsass.exe, svchost.exe, malicious.dll, script.ps1"
    tokens = tokenizer.tokenize(text)
    assert "powershell.exe" in tokens
    assert "lsass.exe" in tokens
    assert "svchost.exe" in tokens
    assert "malicious.dll" in tokens
    assert "script.ps1" in tokens

def test_preserves_ip_and_ports(tokenizer):
    text = "Connection from 192.168.1.100:4444 to 10.0.0.5."
    tokens = tokenizer.tokenize(text)
    assert "192.168.1.100:4444" in tokens
    assert "10.0.0.5" in tokens

def test_technical_entities_bypass_stopwords(tokenizer):
    # 'it', 'the', 'and' should be removed, but technical entities are kept
    text = "it is the powershell.exe -enc and T1059.001"
    tokens = tokenizer.tokenize(text)
    assert "it" not in tokens
    assert "is" not in tokens
    assert "the" not in tokens
    assert "and" not in tokens
    assert "powershell.exe" in tokens
    assert "-enc" in tokens
    assert "T1059.001" in tokens

def test_compare_tokenizers_highlights_corruption(tokenizer):
    sample = "Attacker invoked powershell.exe -enc with T1059.001"
    comparison = tokenizer.compare_tokenizers(sample)
    assert comparison["technical_entities_count"] >= 3
    # Standard tokenizer destroys '-enc' and 'T1059.001'
    assert len(comparison["corrupted_by_standard"]) >= 1
    assert any("T1059.001" in x or "powershell.exe" in x or "-enc" in x for x in comparison["corrupted_by_standard"])

def test_empty_input(tokenizer):
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize(None) == []

def test_slash_technique_normalization(tokenizer):
    tokens = tokenizer.tokenize("Adversary abused T1059/001 and T1055/001.")
    assert "T1059.001" in tokens
    assert "T1059" in tokens
    assert "T1055.001" in tokens

def test_flags_with_colons_and_equals(tokenizer):
    tokens = tokenizer.tokenize("schtasks /create /tn:SecurityUpdater --policy=bypass /sc:onlogon")
    assert "/create" in tokens
    assert "/tn" in tokens
    assert "securityupdater" in tokens
    assert "--policy" in tokens
    assert "bypass" in tokens

def test_registry_with_spaces(tokenizer):
    tokens = tokenizer.tokenize("Key at HKLM\\Software\\Policies\\Microsoft\\Windows Defender\\Real-Time Protection")
    assert any("Windows Defender" in t for t in tokens)

def test_standalone_cyber_ports(tokenizer):
    tokens = tokenizer.tokenize("RDP on port 3389 and SMB on 445")
    assert "3389" in tokens
    assert "445" in tokens

def test_process_base_subtokens(tokenizer):
    tokens = tokenizer.tokenize("powershell.exe procdump.exe")
    assert "powershell.exe" in tokens
    assert "powershell" in tokens
    assert "procdump.exe" in tokens
    assert "procdump" in tokens
