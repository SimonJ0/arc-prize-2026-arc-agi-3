"""
Feature Extraction Engine tailored for Chatbot Arena Human Preference Predictions.
Computes verbosity, structural formatting, markdown signals, lexical overlap,
multi-turn dialogue dynamics, and tie interaction signals with built-in symmetric duality support.
"""

import json
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

LIST_PATTERN = re.compile(r"^\s*([*\-+]|\d+\.)\s+", re.MULTILINE)
HEADER_PATTERN = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
TABLE_PATTERN = re.compile(r"\|(?:\s*:?---+:?\s*\|)+")
LATEX_PATTERN = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^\$]+?\$|\\[a-zA-Z]+)")
DISCLAIMER_PATTERN = re.compile(
    r"\b(as an ai|as a large language model|i cannot|i am unable to|i apologize|sorry)\b",
    re.IGNORECASE,
)


def safe_parse_dialogue(val: Any) -> List[str]:
    """
    Safely decodes JSON dialogue arrays, handling malformed strings and escapes.
    Returns a list of conversation turns.
    """
    if val is None:
        return [""]
    if isinstance(val, list):
        return [str(turn) if turn is not None else "" for turn in val]
    if not isinstance(val, str) or not val.strip():
        return [""]

    val_clean = val.strip()
    if val_clean.startswith("[") and val_clean.endswith("]"):
        try:
            parsed = json.loads(val_clean)
            if isinstance(parsed, list):
                return [str(turn) if turn is not None else "" for turn in parsed]
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            pass

    # Fallback: unescape newlines if present in raw string
    return [val_clean.replace("\\n", "\n").replace('\\"', '"')]


def compute_turn_metrics(turns: List[str]) -> Dict[str, Any]:
    """Extracts structural and length statistics across dialogue turns."""
    n_turns = max(1, len(turns))
    full_text = "\n\n".join(turns)
    last_turn = turns[-1] if turns else ""

    char_lens = [len(t) for t in turns]
    word_lens = [len(t.split()) for t in turns]

    return {
        "n_turns": float(n_turns),
        "total_chars": float(sum(char_lens)),
        "total_words": float(sum(word_lens)),
        "last_turn_chars": float(len(last_turn)),
        "last_turn_words": float(len(last_turn.split())),
        "turn_char_growth": float(char_lens[-1] - char_lens[0]) if n_turns > 1 else 0.0,
        "full_text": full_text,
        "last_turn_text": last_turn,
    }


class FeatureExtractor:
    """
    Extracts tabular features from (prompt, response_a, response_b) triples.
    All difference and ratio features have explicit sign symmetry for TTA.
    """

    def __init__(self):
        self.feature_names: List[str] = []

    def extract_row(self, prompt: Any, resp_a: Any, resp_b: Any) -> Dict[str, float]:
        """Extracts features for a single sample."""
        feats = {}

        # 0. Multi-Turn Dialogue Decoding
        p_turns = safe_parse_dialogue(prompt)
        a_turns = safe_parse_dialogue(resp_a)
        b_turns = safe_parse_dialogue(resp_b)

        p_stats = compute_turn_metrics(p_turns)
        a_stats = compute_turn_metrics(a_turns)
        b_stats = compute_turn_metrics(b_turns)

        text_a, text_b = a_stats["full_text"], b_stats["full_text"]
        text_prompt = p_stats["full_text"]

        # 1. Length & Verbosity features
        char_a = a_stats["total_chars"]
        char_b = b_stats["total_chars"]
        word_a = a_stats["total_words"]
        word_b = b_stats["total_words"]
        word_prompt = p_stats["total_words"]

        words_a_list = text_a.split()
        words_b_list = text_b.split()
        words_prompt_list = text_prompt.split()

        feats["char_len_a"] = float(char_a)
        feats["char_len_b"] = float(char_b)
        feats["word_len_a"] = float(word_a)
        feats["word_len_b"] = float(word_b)
        feats["prompt_word_len"] = float(word_prompt)

        # Differences and Ratios (Sign flips under A <-> B swap)
        feats["char_diff"] = float(char_a - char_b)
        feats["word_diff"] = float(word_a - word_b)
        feats["char_ratio"] = float((char_a - char_b) / (char_a + char_b + 1.0))
        feats["word_ratio"] = float((word_a - word_b) / (word_a + word_b + 1.0))
        feats["log_char_diff"] = float(np.log1p(char_a) - np.log1p(char_b))
        feats["log_word_diff"] = float(np.log1p(word_a) - np.log1p(word_b))

        # Multi-Turn & Final-Turn Dynamics
        feats["num_turns"] = float(p_stats["n_turns"])
        last_char_a = a_stats["last_turn_chars"]
        last_char_b = b_stats["last_turn_chars"]
        last_word_a = a_stats["last_turn_words"]
        last_word_b = b_stats["last_turn_words"]

        feats["last_char_len_a"] = float(last_char_a)
        feats["last_char_len_b"] = float(last_char_b)
        feats["last_word_len_a"] = float(last_word_a)
        feats["last_word_len_b"] = float(last_word_b)
        feats["last_char_diff"] = float(last_char_a - last_char_b)
        feats["last_word_diff"] = float(last_word_a - last_word_b)
        feats["last_char_ratio"] = float((last_char_a - last_char_b) / (last_char_a + last_char_b + 1.0))

        feats["turn_growth_a"] = float(a_stats["turn_char_growth"])
        feats["turn_growth_b"] = float(b_stats["turn_char_growth"])
        feats["turn_growth_diff"] = float(a_stats["turn_char_growth"] - b_stats["turn_char_growth"])

        # 2. Structural & Markdown Formatting
        code_a = text_a.count("```")
        code_b = text_b.count("```")
        feats["code_blocks_a"] = float(code_a)
        feats["code_blocks_b"] = float(code_b)
        feats["code_blocks_diff"] = float(code_a - code_b)

        lists_a = len(LIST_PATTERN.findall(text_a))
        lists_b = len(LIST_PATTERN.findall(text_b))
        feats["lists_a"] = float(lists_a)
        feats["lists_b"] = float(lists_b)
        feats["lists_diff"] = float(lists_a - lists_b)

        bold_a = text_a.count("**")
        bold_b = text_b.count("**")
        feats["bold_a"] = float(bold_a)
        feats["bold_b"] = float(bold_b)
        feats["bold_diff"] = float(bold_a - bold_b)

        head_a = len(HEADER_PATTERN.findall(text_a))
        head_b = len(HEADER_PATTERN.findall(text_b))
        feats["head_a"] = float(head_a)
        feats["head_b"] = float(head_b)
        feats["head_diff"] = float(head_a - head_b)

        nl_a = text_a.count("\n")
        nl_b = text_b.count("\n")
        feats["newlines_a"] = float(nl_a)
        feats["newlines_b"] = float(nl_b)
        feats["newlines_diff"] = float(nl_a - nl_b)

        # Tables, LaTeX & Disclaimers
        table_a = len(TABLE_PATTERN.findall(text_a))
        table_b = len(TABLE_PATTERN.findall(text_b))
        feats["tables_a"] = float(table_a)
        feats["tables_b"] = float(table_b)
        feats["tables_diff"] = float(table_a - table_b)

        latex_a = len(LATEX_PATTERN.findall(text_a))
        latex_b = len(LATEX_PATTERN.findall(text_b))
        feats["latex_a"] = float(latex_a)
        feats["latex_b"] = float(latex_b)
        feats["latex_diff"] = float(latex_a - latex_b)

        disclaim_a = len(DISCLAIMER_PATTERN.findall(text_a))
        disclaim_b = len(DISCLAIMER_PATTERN.findall(text_b))
        feats["disclaimer_a"] = float(disclaim_a)
        feats["disclaimer_b"] = float(disclaim_b)
        feats["disclaimer_diff"] = float(disclaim_a - disclaim_b)

        # 3. Lexical Overlap with Prompt
        set_prompt = set(w.lower() for w in words_prompt_list if len(w) > 2)
        set_a = set(w.lower() for w in words_a_list if len(w) > 2)
        set_b = set(w.lower() for w in words_b_list if len(w) > 2)

        if len(set_prompt) > 0:
            overlap_a = len(set_prompt.intersection(set_a))
            overlap_b = len(set_prompt.intersection(set_b))
            union_a = len(set_prompt.union(set_a))
            union_b = len(set_prompt.union(set_b))
            feats["prompt_jaccard_a"] = float(overlap_a / union_a) if union_a > 0 else 0.0
            feats["prompt_jaccard_b"] = float(overlap_b / union_b) if union_b > 0 else 0.0
            feats["prompt_overlap_diff"] = float(overlap_a - overlap_b)
            feats["prompt_jaccard_diff"] = feats["prompt_jaccard_a"] - feats["prompt_jaccard_b"]
        else:
            feats["prompt_jaccard_a"] = 0.0
            feats["prompt_jaccard_b"] = 0.0
            feats["prompt_overlap_diff"] = 0.0
            feats["prompt_jaccard_diff"] = 0.0

        # 4. Tie-Detector Interaction Features (Symmetric)
        if len(set_a.union(set_b)) > 0:
            feats["resp_ab_jaccard"] = float(len(set_a.intersection(set_b)) / len(set_a.union(set_b)))
        else:
            feats["resp_ab_jaccard"] = 1.0

        feats["is_identical"] = float(text_a.strip() == text_b.strip())
        feats["abs_norm_diff"] = float(abs(char_a - char_b) / (char_a + char_b + 1.0))

        # Character 3-gram Jaccard
        if len(text_a) >= 3 and len(text_b) >= 3:
            ng_a = set(text_a[i : i + 3] for i in range(len(text_a) - 2))
            ng_b = set(text_b[i : i + 3] for i in range(len(text_b) - 2))
            u_ng = len(ng_a.union(ng_b))
            feats["jaccard_3g"] = float(len(ng_a.intersection(ng_b)) / u_ng) if u_ng > 0 else 1.0
        else:
            feats["jaccard_3g"] = 1.0 if text_a.strip() == text_b.strip() else 0.0

        # 5. Linguistic complexity
        feats["avg_word_len_a"] = float(char_a / word_a) if word_a > 0 else 0.0
        feats["avg_word_len_b"] = float(char_b / word_b) if word_b > 0 else 0.0
        feats["avg_word_len_diff"] = feats["avg_word_len_a"] - feats["avg_word_len_b"]

        feats["questions_diff"] = float(text_a.count("?") - text_b.count("?"))
        feats["exclamations_diff"] = float(text_a.count("!") - text_b.count("!"))

        return feats

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts feature DataFrame for all rows in df."""
        records = []
        for prompt, resp_a, resp_b in zip(df["prompt"], df["response_a"], df["response_b"]):
            records.append(self.extract_row(prompt, resp_a, resp_b))

        feature_df = pd.DataFrame(records)
        self.feature_names = list(feature_df.columns)
        return feature_df

    def extract_swapped_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts features with Response A and Response B positions inverted.
        Used for Position Symmetry evaluation and Test-Time Augmentation (TTA).
        """
        records = []
        for prompt, resp_a, resp_b in zip(df["prompt"], df["response_a"], df["response_b"]):
            records.append(self.extract_row(prompt, resp_b, resp_a))

        return pd.DataFrame(records)

