"""
Transformer Cross-Encoder Architecture Blueprint and Specification.
Implements pairwise cross-encoding with symmetric pair augmentation
for fine-tuning models like DeBERTa-v3 or Gemma/Llama reward models.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TransformerModelConfig:
    pretrained_model_name: str = "microsoft/deberta-v3-small"
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    symmetric_loss_weight: float = 0.5
    fp16: bool = True
    gradient_accumulation_steps: int = 2


def format_cross_encoder_text(
    prompt: str, resp_a: str, resp_b: str, max_chars_per_resp: int = 1500
) -> str:
    """
    Formats the pair into a structured cross-encoder prompt sequence.
    Truncates overly long responses cleanly to stay within token budgets.
    """
    clean_p = str(prompt).strip()[:1000]
    clean_a = str(resp_a).strip()[:max_chars_per_resp]
    clean_b = str(resp_b).strip()[:max_chars_per_resp]
    return f"Prompt:\n{clean_p}\n\n[Response A]:\n{clean_a}\n\n[Response B]:\n{clean_b}"


def format_swapped_cross_encoder_text(
    prompt: str, resp_a: str, resp_b: str, max_chars_per_resp: int = 1500
) -> str:
    """Formats the pair with positions inverted for symmetric cross-entropy loss."""
    return format_cross_encoder_text(prompt, resp_b, resp_a, max_chars_per_resp=max_chars_per_resp)


def compute_symmetric_probabilities(p_norm: np.ndarray, p_swap: np.ndarray) -> np.ndarray:
    """
    Combines normal and swapped probability predictions with exact position symmetry.
    p[:, 0] = winner_model_a
    p[:, 1] = winner_model_b
    p[:, 2] = winner_tie
    """
    p_sym = np.zeros_like(p_norm)
    p_sym[:, 0] = 0.5 * (p_norm[:, 0] + p_swap[:, 1])
    p_sym[:, 1] = 0.5 * (p_norm[:, 1] + p_swap[:, 0])
    p_sym[:, 2] = 0.5 * (p_norm[:, 2] + p_swap[:, 2])

    # Normalize rows
    row_sums = p_sym.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return p_sym / row_sums


class TransformerPipelineSpec:
    """
    Specification for DeBERTa-v3 / Transformer fine-tuning in Kaggle environments.
    Handles tokenization format, pair symmetry training loss, script generation, and inference.
    """

    def __init__(self, config: TransformerModelConfig | None = None):
        self.config = config or TransformerModelConfig()

    def generate_kaggle_finetune_script(
        self, output_path: str = "submissions/kaggle_deberta_finetuning.py"
    ) -> Path:
        """
        Generates a standalone PyTorch / HuggingFace script tailored to Kaggle's T4/P100 GPUs,
        utilizing mixed precision (fp16), evaluation on multi-class log loss, and symmetric pair loss.
        """
        script = f'''"""
Kaggle Fine-tuning Script for DeBERTa-v3 Cross-Encoder.
"""

import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
import pandas as pd
import numpy as np

MODEL_NAME = "{self.config.pretrained_model_name}"
MAX_LEN = {self.config.max_length}
LR = {self.config.learning_rate}
BATCH_SIZE = {self.config.batch_size}
EPOCHS = {self.config.epochs}
FP16 = {self.config.fp16}
GRAD_ACC = {self.config.gradient_accumulation_steps}
OUTPUT_DIR = "/kaggle/working/deberta_best_model"

def format_text(p, a, b):
    p_clean = str(p)[:1000]
    a_clean = str(a)[:1500]
    b_clean = str(b)[:1500]
    return f"Prompt:\\n{{p_clean}}\\n\\n[Response A]:\\n{{a_clean}}\\n\\n[Response B]:\\n{{b_clean}}"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
    probs /= probs.sum(axis=1, keepdims=True)

    n = len(labels)
    y_one_hot = np.zeros((n, 3))
    y_one_hot[np.arange(n), labels] = 1.0
    loss = -np.mean(np.sum(y_one_hot * np.log(probs), axis=1))
    return {{"log_loss": float(loss)}}

if __name__ == "__main__":
    print(f"Loading base model: {{MODEL_NAME}}...")
    train_csv = None
    for p in ["/kaggle/input/llm-classification-finetuning/train.csv", "data/train.csv"]:
        if os.path.exists(p):
            train_csv = p
            break
    if not train_csv:
        raise FileNotFoundError("Could not locate train.csv")

    df = pd.read_csv(train_csv)
    print(f"Loaded training data: {{len(df)}} rows.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. Normal order
    texts_norm = [format_text(p, a, b) for p, a, b in zip(df['prompt'], df['response_a'], df['response_b'])]
    targets_norm = np.argmax(df[['winner_model_a', 'winner_model_b', 'winner_tie']].values, axis=1)

    # 2. Symmetric swapped order: label 0 <-> 1, 2 stays 2
    swap_map = {{0: 1, 1: 0, 2: 2}}
    texts_swap = [format_text(p, b, a) for p, a, b in zip(df['prompt'], df['response_a'], df['response_b'])]
    targets_swap = np.array([swap_map[t] for t in targets_norm])

    # Combine for full position invariance
    all_texts = texts_norm + texts_swap
    all_targets = np.concatenate([targets_norm, targets_swap])

    ds = Dataset.from_dict({{"text": all_texts, "label": all_targets}})
    ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LEN), batched=True)
    split_ds = ds.train_test_split(test_size=0.1, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="log_loss",
        greater_is_better=False,
        fp16=FP16 and torch.cuda.is_available(),
        gradient_accumulation_steps=GRAD_ACC,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_ds["train"],
        eval_dataset=split_ds["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting cross-encoder fine-tuning...")
    trainer.train()
    print(f"Fine-tuning complete. Saving best model to {{OUTPUT_DIR}}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
'''
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(script, encoding="utf-8")
        return out_file
