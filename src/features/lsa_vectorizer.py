"""
Dense Latent Semantic Analysis (LSA) Feature Vectorizer.
Extracts low-dimensional dense semantic representations of prompts and responses
using n-gram TF-IDF and TruncatedSVD with strict position symmetry guarantees.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.extractor import safe_parse_dialogue


def _clean_dialogue_texts(texts: list[Any]) -> list[str]:
    """Decodes multi-turn dialogue lists into unified clean text strings."""
    return ["\n\n".join(safe_parse_dialogue(t)) for t in texts]


class DenseLSAVectorizer:
    """
    Computes dense semantic embeddings using TruncatedSVD over paired TF-IDF features.
    Extracts semantic alignment, cosine similarities, and coordinate differentials.
    """

    def __init__(
        self,
        n_components: int = 16,
        max_features: int = 5000,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.max_features = max_features
        self.random_state = random_state

        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.is_fitted = False

    def fit(self, texts: list[Any]) -> "DenseLSAVectorizer":
        """Fits vocabulary and latent semantic components strictly on training text."""
        clean_texts = _clean_dialogue_texts(texts)
        tfidf_mat = self.tfidf.fit_transform(clean_texts)
        self.svd.fit(tfidf_mat)
        self.is_fitted = True
        return self

    def transform(self, texts: list[Any]) -> np.ndarray:
        """Projects texts into the latent semantic space."""
        if not self.is_fitted:
            raise RuntimeError("DenseLSAVectorizer must be fitted before calling transform().")
        clean_texts = _clean_dialogue_texts(texts)
        tfidf_mat = self.tfidf.transform(clean_texts)
        return self.svd.transform(tfidf_mat)

    def extract_lsa_features(self, df: pd.DataFrame, swap: bool = False) -> pd.DataFrame:
        """
        Computes dense LSA features for pairs of responses and prompts.
        Guaranteed to be symmetric under position swap.
        """
        col_a = "response_b" if swap else "response_a"
        col_b = "response_a" if swap else "response_b"

        prompts = df["prompt"].tolist()
        resps_a = df[col_a].tolist()
        resps_b = df[col_b].tolist()

        lsa_p = self.transform(prompts)
        lsa_a = self.transform(resps_a)
        lsa_b = self.transform(resps_b)

        # Normalize latent vectors for cosine similarity computation
        def _normalize_rows(mat: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms

        norm_p = _normalize_rows(lsa_p)
        norm_a = _normalize_rows(lsa_a)
        norm_b = _normalize_rows(lsa_b)

        cos_p_a = np.sum(norm_p * norm_a, axis=1)
        cos_p_b = np.sum(norm_p * norm_b, axis=1)
        cos_a_b = np.sum(norm_a * norm_b, axis=1)

        l2_diff = np.linalg.norm(lsa_p - lsa_a, axis=1) - np.linalg.norm(lsa_p - lsa_b, axis=1)
        cos_diff = cos_p_a - cos_p_b

        feat_dict = {
            "lsa_cos_p_a": cos_p_a,
            "lsa_cos_p_b": cos_p_b,
            "lsa_cos_diff": cos_diff,
            "lsa_l2_diff": l2_diff,
            "lsa_cos_a_b": cos_a_b,
        }

        # Add coordinate-wise differential features: lsa_a[i] - lsa_b[i]
        coord_diff = lsa_a - lsa_b
        if coord_diff.shape[1] < self.n_components:
            pad_width = self.n_components - coord_diff.shape[1]
            coord_diff = np.pad(coord_diff, ((0, 0), (0, pad_width)), mode="constant")

        for i in range(self.n_components):
            feat_dict[f"lsa_dim_diff_{i}"] = coord_diff[:, i]

        return pd.DataFrame(feat_dict, index=df.index)
