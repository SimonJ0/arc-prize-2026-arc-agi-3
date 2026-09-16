"""
Cross-validation splitters with anti-leakage guarantees.
"""

import hashlib
from typing import Generator, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold


class DatasetSplitter:
    """
    Generates reproducible, leak-free cross-validation folds.
    Supports standard StratifiedKFold or StratifiedGroupKFold on prompt hashes.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
        group_by_prompt: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns list of (train_indices, val_indices).
        
        Args:
            df: DataFrame containing 'target' and 'prompt'.
            group_by_prompt: If True, prevents identical prompts from appearing in both train and val.
        """
        y = df["target"].values

        if group_by_prompt and "prompt" in df.columns:
            # Hash prompt to create groups deterministically across all Python sessions
            groups = df["prompt"].apply(
                lambda p: int(hashlib.md5(str(p).encode("utf-8")).hexdigest(), 16) % (10**9)
            ).values
            sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = list(sgkf.split(df, y, groups))
        else:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = list(skf.split(df, y))

        return splits

    def add_fold_column(self, df: pd.DataFrame, group_by_prompt: bool = True) -> pd.DataFrame:
        """Adds a 'fold' column (0 to n_splits-1) to the dataframe."""
        df = df.copy()
        df["fold"] = -1
        splits = self.split(df, group_by_prompt=group_by_prompt)
        for fold_idx, (_, val_idx) in enumerate(splits):
            df.loc[val_idx, "fold"] = fold_idx
        return df
