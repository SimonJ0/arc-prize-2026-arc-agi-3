"""
Unit tests for Transformer Cross-Encoder pipeline specification and script generation.
"""

import numpy as np

from src.models.transformer_head import (
    TransformerModelConfig,
    TransformerPipelineSpec,
    compute_symmetric_probabilities,
    format_cross_encoder_text,
    format_swapped_cross_encoder_text,
)


def test_text_formatting():
    p = "What is gravity?"
    a = "A fundamental interaction that causes mutual attraction."
    b = "Magic force."

    text_norm = format_cross_encoder_text(p, a, b)
    assert "[Response A]:\nA fundamental interaction" in text_norm
    assert "[Response B]:\nMagic force." in text_norm

    text_swap = format_swapped_cross_encoder_text(p, a, b)
    assert "[Response A]:\nMagic force." in text_swap
    assert "[Response B]:\nA fundamental interaction" in text_swap


def test_symmetric_probabilities():
    p_norm = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    p_swap = np.array([[0.25, 0.65, 0.1], [0.75, 0.15, 0.1]])

    p_sym = compute_symmetric_probabilities(p_norm, p_swap)

    assert p_sym.shape == (2, 3)
    np.testing.assert_allclose(p_sym.sum(axis=1), np.ones(2), atol=1e-6)

    # First row:
    # winner_a = 0.5 * (0.60 + 0.65) = 0.625
    # winner_b = 0.5 * (0.30 + 0.25) = 0.275
    # tie = 0.5 * (0.10 + 0.10) = 0.100
    expected_row_0 = np.array([0.625, 0.275, 0.100])
    np.testing.assert_allclose(p_sym[0], expected_row_0, atol=1e-6)


def test_generate_kaggle_finetune_script(tmp_path):
    spec = TransformerPipelineSpec(
        TransformerModelConfig(pretrained_model_name="microsoft/deberta-v3-small")
    )
    out_script = tmp_path / "finetune_deberta.py"
    generated_path = spec.generate_kaggle_finetune_script(str(out_script))

    assert generated_path.exists()
    content = generated_path.read_text(encoding="utf-8")
    assert "microsoft/deberta-v3-small" in content
    assert "AutoModelForSequenceClassification" in content
    # Verify Python syntax validity
    compile(content, str(generated_path), "exec")
