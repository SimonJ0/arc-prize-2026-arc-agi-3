"""
Unit tests for submission generation and kernel validation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.submit.generator import SubmissionGenerator


def test_generate_submission_csv(tmp_path):
    gen = SubmissionGenerator(output_dir=str(tmp_path))
    test_df = pd.DataFrame({"id": [1, 2, 3], "prompt": ["a", "b", "c"]})
    raw_preds = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.33, 0.33, 0.34]])

    sub_file = gen.generate_submission_csv(test_df, raw_preds, filename="test_sub.csv")
    assert sub_file.exists()

    df_out = pd.read_csv(sub_file)
    assert list(df_out.columns) == ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert len(df_out) == 3
    assert np.allclose(df_out[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1), 1.0)


def test_generate_standalone_kernel(tmp_path):
    gen = SubmissionGenerator(output_dir=str(tmp_path))
    kernel_file = gen.generate_standalone_kaggle_kernel(output_filename="test_kernel.py")
    assert kernel_file.exists()
    content = kernel_file.read_text(encoding="utf-8")
    assert "extract_features" in content
    assert "submission.csv" in content


def test_generate_dynamic_kernel_with_weights(tmp_path):
    gen = SubmissionGenerator(output_dir=str(tmp_path))
    weights = {"baseline_length_prior": 0.25, "structural_features_lightgbm": 0.75}
    kernel_file = gen.generate_standalone_kaggle_kernel(
        lgb_model_str="dummy_tree_str",
        ensemble_weights=weights,
        temperature=1.05,
        output_filename="dynamic_kernel.py",
    )
    assert kernel_file.exists()
    content = kernel_file.read_text(encoding="utf-8")
    assert "MODEL_BLOB_LZMA" in content
    import base64, lzma, pickle, re
    match = re.search(r"MODEL_BLOB_LZMA = ['\"]([^'\"]+)['\"]", content)
    assert match is not None
    pkg = pickle.loads(lzma.decompress(base64.b64decode(match.group(1))))
    assert pkg.get("booster") == "dummy_tree_str"
    assert "0.25" in content
    assert "1.05" in content
