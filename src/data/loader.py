"""
Data ingestion and preprocessing for LLM Classification Finetuning.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


CLASSES = ["winner_model_a", "winner_model_b", "winner_tie"]


def get_target_class_indices(df: pd.DataFrame) -> np.ndarray:
    """Converts 3 one-hot target columns to class indices (0: a, 1: b, 2: tie)."""
    return np.argmax(df[CLASSES].values, axis=1)


class DataLoader:
    """Handles dataset loading, text sanitation, and subsetting for fast iteration."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load_train(self, sample_size: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
        """Loads train.csv with null replacement and optional subsampling."""
        path = self.data_dir / "train.csv"
        if not path.exists():
            raise FileNotFoundError(f"Training data not found at {path}")

        df = pd.read_csv(path)
        
        # Clean string nulls
        for col in ["prompt", "response_a", "response_b"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        # Add integer label column for convenient stratification
        df["target"] = get_target_class_indices(df)

        if sample_size and sample_size < len(df):
            from sklearn.model_selection import train_test_split
            df, _ = train_test_split(
                df,
                train_size=sample_size,
                stratify=df["target"],
                random_state=random_state,
            )
            df = df.reset_index(drop=True)

        return df

    def load_test(self) -> pd.DataFrame:
        """Loads test.csv."""
        path = self.data_dir / "test.csv"
        if not path.exists():
            raise FileNotFoundError(f"Test data not found at {path}")

        df = pd.read_csv(path)
        for col in ["prompt", "response_a", "response_b"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        return df

    def load_sample_submission(self) -> pd.DataFrame:
        """Loads sample_submission.csv."""
        path = self.data_dir / "sample_submission.csv"
        if not path.exists():
            raise FileNotFoundError(f"Sample submission not found at {path}")
        return pd.read_csv(path)
