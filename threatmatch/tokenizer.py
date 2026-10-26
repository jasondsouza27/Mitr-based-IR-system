"""
Module 1: Cyber-Specific Inverted Index & Tokenization
CyberRegexTokenizer: Preserves MITRE IDs, CVEs, Registry Paths, Process Artifacts, Command Flags.
Bypasses conventional stop-word removal for technical entities.
"""

import re
from dataclasses import dataclass
from typing import List, Set, Dict, Any

# Standard English stopwords (filtered only from non-technical tokens)
STANDARD_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

@dataclass
class Token:
    value: str
    token_type: str
    is_technical: bool
    start_pos: int
    end_pos: int

class CyberRegexTokenizer:
    """
    Domain-specific regex tokenizer tailored for cybersecurity threat intelligence.
    Preserves:
      - MITRE Technique IDs (e.g. T1059, T1059.001)
      - MITRE Tactic IDs (e.g. TA0002)
      - MITRE Mitigation IDs (e.g. M1047)
      - CVE identifiers (e.g. CVE-2023-38606)
      - Windows Registry hives and keys (e.g. HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run)
      - Windows and Unix File paths (e.g. C:\\Windows\\System32\\cmd.exe, /etc/shadow)
      - Process artifacts and script extensions (.exe, .dll, .sys, .ps1, .bat, etc.)
      - Command-line flags and parameters (e.g. -enc, -w hidden, -nop, /add, /c, /v, --bypass)
      - IPv4 addresses and port designations (e.g. 192.168.1.50:4444)
    """

    MASTER_REGEX = re.compile(
        r'(?P<TECHNIQUE>\bT\d{4}(?:[\./]\d{3})?\b)|'
        r'(?P<TACTIC>\bTA\d{4}\b)|'
        r'(?P<MITIGATION>\bM\d{4}\b)|'
        r'(?P<CVE>\bCVE-\d{4}-\d{4,7}\b)|'
        r'(?P<REGISTRY>\b(?:HKEY_[A-Z_]+|HKLM|HKCU|HKCR|HKU|HKCC)(?:\\[A-Za-z0-9_\-\. ]{1,64})+\b)|'
        r'(?P<REGISTRY_HIVE>\b(?:HKEY_[A-Z_]+|HKLM|HKCU|HKCR|HKU|HKCC)\b)|'
        r'(?P<WIN_PATH>\b[A-Za-z]:\\(?:[A-Za-z0-9_\-\. ]+\\)*[A-Za-z0-9_\-\. ]+\b)|'
        r'(?P<UNIX_PATH>/(?:etc|bin|usr|var|tmp|home|proc|dev|opt|root|sys|Library|private)/[A-Za-z0-9_\-\./]+)|'
        r'(?P<PROCESS>\b[A-Za-z0-9_\-]+\.(?:exe|dll|sys|ps1|bat|vbs|sh|py|bin|elf|scr|cpl|hta|jar|vbe|wsf|msi)\b)|'
        r'(?P<IP_PORT>\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b)|'
        r'(?P<PORT>\b(?:3389|445|443|8080|8443|88|53|22|21|23|25|135|139|389|636|1433|1521|3306|5432|5985|5986|4444)\b)|'
        r'(?P<FLAG>(?:^|[\s\(\[\{"\',;])(?:--?[A-Za-z0-9_\-]+|\/[A-Za-z0-9_\?]+)(?=[=:\s]|$|[;,"\'\)]))|'
        r'(?P<WORD>\b[A-Za-z][A-Za-z0-9_\-]*\b)',
        re.IGNORECASE
    )

    def __init__(self, filter_stopwords: bool = True, lowercase_standard: bool = True, extract_path_subtokens: bool = True):
        self.filter_stopwords = filter_stopwords
        self.lowercase_standard = lowercase_standard
        self.extract_path_subtokens = extract_path_subtokens

    def tokenize_detailed(self, text: str) -> List[Token]:
        """
        Tokenizes input text and returns structured Token objects with type information.
        """
        if not text:
            return []

        tokens: List[Token] = []
        for match in self.MASTER_REGEX.finditer(text):
            kind = match.lastgroup
            raw_val = match.group(kind).strip()
            start, end = match.span(kind)

            if kind == "TECHNIQUE":
                # Normalize IDs to uppercase with dot notation (T1059/001 -> T1059.001)
                val = raw_val.upper().replace("/", ".")
                tokens.append(Token(value=val, token_type="technique", is_technical=True, start_pos=start, end_pos=end))
                # If it's a sub-technique, also index the parent technique ID
                if self.extract_path_subtokens and "." in val:
                    parent_tech = val.split(".")[0]
                    tokens.append(Token(value=parent_tech, token_type="technique_parent", is_technical=True, start_pos=start, end_pos=end))

            elif kind in ("TACTIC", "MITIGATION", "CVE"):
                val = raw_val.upper()
                tokens.append(Token(value=val, token_type=kind.lower(), is_technical=True, start_pos=start, end_pos=end))

            elif kind in ("REGISTRY", "REGISTRY_HIVE"):
                tokens.append(Token(value=raw_val, token_type="registry", is_technical=True, start_pos=start, end_pos=end))
                if self.extract_path_subtokens and "\\" in raw_val:
                    subparts = [p.strip() for p in raw_val.split("\\") if p.strip()]
                    if len(subparts) > 1:
                        # Add terminal key name (e.g. Run)
                        leaf = subparts[-1].lower()
                        if leaf and leaf not in STANDARD_STOPWORDS and len(leaf) > 1:
                            tokens.append(Token(value=leaf, token_type="registry_subkey", is_technical=True, start_pos=start, end_pos=end))

            elif kind in ("WIN_PATH", "UNIX_PATH"):
                tokens.append(Token(value=raw_val, token_type="file_path", is_technical=True, start_pos=start, end_pos=end))
                # Optionally add filename artifact if path ends with process / executable / script
                if self.extract_path_subtokens:
                    basename = raw_val.replace("\\", "/").split("/")[-1]
                    if "." in basename:
                        tokens.append(Token(value=basename.lower(), token_type="process_artifact", is_technical=True, start_pos=start, end_pos=end))

            elif kind == "PROCESS":
                # Processes normalized to lowercase for consistent indexing
                p_lower = raw_val.lower()
                tokens.append(Token(value=p_lower, token_type="process_artifact", is_technical=True, start_pos=start, end_pos=end))
                # Also emit base process name (e.g. powershell from powershell.exe, lsass from lsass.exe)
                if self.extract_path_subtokens and "." in p_lower:
                    base_name = p_lower.split(".")[0]
                    if len(base_name) > 1 and base_name not in STANDARD_STOPWORDS:
                        tokens.append(Token(value=base_name, token_type="process_name", is_technical=True, start_pos=start, end_pos=end))

            elif kind == "IP_PORT":
                tokens.append(Token(value=raw_val, token_type="network_artifact", is_technical=True, start_pos=start, end_pos=end))

            elif kind == "PORT":
                tokens.append(Token(value=raw_val, token_type="network_port", is_technical=True, start_pos=start, end_pos=end))

            elif kind == "FLAG":
                clean_f = raw_val.lstrip(' \t\n\r([{"\',;').lower()
                tokens.append(Token(value=clean_f, token_type="command_flag", is_technical=True, start_pos=start, end_pos=end))
                # Also emit clean flag word without prefix dashes or slashes
                if self.extract_path_subtokens:
                    clean_flag = clean_f.lstrip("-/").strip()
                    if len(clean_flag) > 2 and clean_flag not in STANDARD_STOPWORDS:
                        tokens.append(Token(value=clean_flag, token_type="flag_name", is_technical=True, start_pos=start, end_pos=end))

            elif kind == "WORD":
                val = raw_val.lower() if self.lowercase_standard else raw_val
                # Stopword removal applies ONLY to standard non-technical words
                if self.filter_stopwords and val in STANDARD_STOPWORDS:
                    continue
                # Skip trivial single characters
                if len(val) <= 1:
                    continue
                tokens.append(Token(value=val, token_type="standard_term", is_technical=False, start_pos=start, end_pos=end))

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes input text and returns a list of clean string tokens for index or query.
        """
        detailed = self.tokenize_detailed(text)
        return [t.value for t in detailed]

    @staticmethod
    def standard_nlp_tokenize(text: str) -> List[str]:
        """
        Simulates standard conventional NLP tokenization (e.g. whitespace + punctuation stripping / \\w+).
        Useful for side-by-side comparison in UI and benchmark demonstration.
        """
        if not text:
            return []
        raw_tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in raw_tokens if t not in STANDARD_STOPWORDS and len(t) > 1]

    def compare_tokenizers(self, text: str) -> Dict[str, Any]:
        """
        Returns comparison between CyberRegexTokenizer and Standard Tokenizer,
        highlighting preserved vs corrupted technical entities.
        """
        cyber_tokens = self.tokenize_detailed(text)
        std_tokens = self.standard_nlp_tokenize(text)

        technical_tokens = [t.value for t in cyber_tokens if t.is_technical]
        corrupted_entities = []

        # Check if technical entities survived intact in standard tokenization
        for t in technical_tokens:
            if t.lower() not in std_tokens:
                corrupted_entities.append(t)

        return {
            "input_text": text,
            "cyber_tokens": [t.value for t in cyber_tokens],
            "cyber_detailed": [
                {"value": t.value, "type": t.token_type, "technical": t.is_technical}
                for t in cyber_tokens
            ],
            "standard_tokens": std_tokens,
            "technical_entities_count": len(technical_tokens),
            "corrupted_by_standard": list(dict.fromkeys(corrupted_entities)),
            "preservation_rate": (
                (len(technical_tokens) - len(corrupted_entities)) / len(technical_tokens)
                if technical_tokens else 1.0
            )
        }
