"""
TF-IDF Differential Vectorizer and Cross-Feature Engine.
Computes sparse differential text representations and cosine overlaps.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances


class TextVectorizer:
    """
    Computes TF-IDF differential features (TFIDF(response_a) - TFIDF(response_b))
    and prompt-response semantic overlaps.
    """

    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer: Optional[TfidfVectorizer] = None

    def fit(self, texts: List[str]) -> "TextVectorizer":
        """Fits shared vocabulary on corpus of prompts and responses."""
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            stop_words="english",
        )
        self.vectorizer.fit(texts)
        return self

    def transform_differential(
        self,
        df: pd.DataFrame,
        swap: bool = False,
    ) -> csr_matrix:
        """
        Computes sparse feature matrix:
        - TFIDF(A) - TFIDF(B)
        - prompt-response cosine similarities
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted before transforming.")

        if swap:
            col_a, col_b = "response_b", "response_a"
        else:
            col_a, col_b = "response_a", "response_b"

        tfidf_a = self.vectorizer.transform(df[col_a])
        tfidf_b = self.vectorizer.transform(df[col_b])
        diff_matrix = tfidf_a - tfidf_b

        # Compute cosine similarities
        tfidf_prompt = self.vectorizer.transform(df["prompt"])
        
        # 1 - cosine distance = cosine similarity (guard against NaN for zero-norm vectors)
        sim_prompt_a = np.nan_to_num(1.0 - paired_cosine_distances(tfidf_prompt, tfidf_a), nan=0.0).reshape(-1, 1)
        sim_prompt_b = np.nan_to_num(1.0 - paired_cosine_distances(tfidf_prompt, tfidf_b), nan=0.0).reshape(-1, 1)
        sim_ab = np.nan_to_num(1.0 - paired_cosine_distances(tfidf_a, tfidf_b), nan=0.0).reshape(-1, 1)
        sim_diff = (sim_prompt_a - sim_prompt_b).reshape(-1, 1)

        extra_features = csr_matrix(np.hstack([sim_prompt_a, sim_prompt_b, sim_ab, sim_diff]))
        return hstack([diff_matrix, extra_features]).tocsr()
