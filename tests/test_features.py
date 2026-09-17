"""
Unit tests for feature extraction, multi-turn dialogue decoding, and symmetry duality.
"""

import json

import numpy as np
import pandas as pd

from src.features.extractor import FeatureExtractor, safe_parse_dialogue


def test_safe_parse_dialogue():
    # 1. Valid JSON array
    valid_json = '["Turn 1 text", "Turn 2 text"]'
    parsed = safe_parse_dialogue(valid_json)
    assert parsed == ["Turn 1 text", "Turn 2 text"]

    # 2. Plain text string (single turn)
    plain = "Just a single prompt without JSON."
    parsed_plain = safe_parse_dialogue(plain)
    assert parsed_plain == ["Just a single prompt without JSON."]

    # 3. Empty or None
    assert safe_parse_dialogue("") == [""]
    assert safe_parse_dialogue(None) == [""]

    # 4. Malformed JSON with brackets
    malformed = '["Turn 1 text", "Incomplete turn'
    parsed_malformed = safe_parse_dialogue(malformed)
    assert len(parsed_malformed) == 1


def test_feature_extractor_symmetry():
    extractor = FeatureExtractor()
    df = pd.DataFrame(
        [
            {
                "prompt": json.dumps(["What is Python?", "How do you define a function?"]),
                "response_a": json.dumps(
                    [
                        "Python is a programming language. ```python\nprint(1)\n```",
                        "Here is how: | function | syntax |\n|---|---|\n| def | def foo(): pass |",
                    ]
                ),
                "response_b": json.dumps(
                    [
                        "Short answer turn 1.",
                        "Short answer turn 2 with apology: I apologize for being brief.",
                    ]
                ),
            }
        ]
    )

    feats_norm = extractor.extract_features(df)
    feats_swap = extractor.extract_swapped_features(df)

    # Difference features should strictly negate
    assert np.isclose(feats_norm.loc[0, "char_diff"], -feats_swap.loc[0, "char_diff"])
    assert np.isclose(feats_norm.loc[0, "word_diff"], -feats_swap.loc[0, "word_diff"])
    assert np.isclose(feats_norm.loc[0, "char_ratio"], -feats_swap.loc[0, "char_ratio"])
    assert np.isclose(feats_norm.loc[0, "code_blocks_diff"], -feats_swap.loc[0, "code_blocks_diff"])
    assert np.isclose(feats_norm.loc[0, "last_char_diff"], -feats_swap.loc[0, "last_char_diff"])
    assert np.isclose(feats_norm.loc[0, "last_word_diff"], -feats_swap.loc[0, "last_word_diff"])
    assert np.isclose(feats_norm.loc[0, "last_char_ratio"], -feats_swap.loc[0, "last_char_ratio"])
    assert np.isclose(feats_norm.loc[0, "turn_growth_diff"], -feats_swap.loc[0, "turn_growth_diff"])
    assert np.isclose(feats_norm.loc[0, "tables_diff"], -feats_swap.loc[0, "tables_diff"])
    assert np.isclose(feats_norm.loc[0, "disclaimer_diff"], -feats_swap.loc[0, "disclaimer_diff"])

    # Symmetric properties should remain identical
    assert feats_norm.loc[0, "num_turns"] == 2.0
    assert feats_norm.loc[0, "num_turns"] == feats_swap.loc[0, "num_turns"]
    assert feats_norm.loc[0, "prompt_word_len"] == feats_swap.loc[0, "prompt_word_len"]
    assert np.isclose(feats_norm.loc[0, "resp_ab_jaccard"], feats_swap.loc[0, "resp_ab_jaccard"])
    assert np.isclose(feats_norm.loc[0, "jaccard_3g"], feats_swap.loc[0, "jaccard_3g"])
    assert np.isclose(feats_norm.loc[0, "abs_norm_diff"], feats_swap.loc[0, "abs_norm_diff"])
