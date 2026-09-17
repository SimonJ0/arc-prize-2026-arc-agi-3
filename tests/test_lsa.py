"""
Unit tests for Dense Latent Semantic Analysis (LSA) feature vectorizer.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.lsa_vectorizer import DenseLSAVectorizer


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "prompt": [
                "What is Python?",
                "How do I sort a list in C++?",
                "Explain quantum computing in simple terms.",
            ],
            "response_a": [
                "Python is an interpreted, high-level programming language.",
                "Use std::sort from the <algorithm> header with iterators.",
                "Quantum computers use qubits that can exist in superpositions.",
            ],
            "response_b": [
                "Python is a snake found in tropical regions.",
                "You can write a bubble sort loop in C++.",
                "It is very fast classical computers that run on light.",
            ],
        }
    )


def test_lsa_symmetry(sample_data):
    vec = DenseLSAVectorizer(n_components=4, max_features=100)
    all_texts = pd.concat(
        [sample_data["prompt"], sample_data["response_a"], sample_data["response_b"]]
    ).tolist()
    vec.fit(all_texts)

    feats_normal = vec.extract_lsa_features(sample_data, swap=False)
    feats_swapped = vec.extract_lsa_features(sample_data, swap=True)

    # Differential features should be exactly negated under swap
    np.testing.assert_allclose(
        feats_normal["lsa_cos_diff"].values, -feats_swapped["lsa_cos_diff"].values, atol=1e-6
    )
    np.testing.assert_allclose(
        feats_normal["lsa_l2_diff"].values, -feats_swapped["lsa_l2_diff"].values, atol=1e-6
    )

    for i in range(4):
        col = f"lsa_dim_diff_{i}"
        np.testing.assert_allclose(feats_normal[col].values, -feats_swapped[col].values, atol=1e-6)

    # Pairwise similarity between responses should be symmetric (identical)
    np.testing.assert_allclose(
        feats_normal["lsa_cos_a_b"].values, feats_swapped["lsa_cos_a_b"].values, atol=1e-6
    )


def test_lsa_handles_empty_and_nan():
    vec = DenseLSAVectorizer(n_components=4, max_features=100)
    vec.fit(["hello world", "test sample", "programming"])

    df_edge = pd.DataFrame(
        {
            "prompt": ["", None, "   "],
            "response_a": [None, "", "a"],
            "response_b": ["", "b", None],
        }
    )

    feats = vec.extract_lsa_features(df_edge)
    assert not feats.isna().any().any()
    assert not np.isinf(feats.values).any()
    assert feats.shape == (3, 9)  # 5 scalar features + 4 dim diff features
