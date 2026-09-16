"""
Submission File and Kaggle Inference Kernel Generator.
Produces verified submission.csv and standalone Kaggle kernels for code competition execution.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.core.metrics import CLASSES, normalize_probabilities
from src.core.gates import ValidationGatekeeper


class SubmissionGenerator:
    """Produces verified competition submission files and standalone Kaggle inference kernels."""

    def __init__(self, output_dir: str = "submissions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gatekeeper = ValidationGatekeeper()

    def generate_submission_csv(
        self,
        test_df: pd.DataFrame,
        preds: np.ndarray,
        filename: str = "submission.csv",
    ) -> Path:
        """
        Creates, normalizes, and gate-checks submission.csv.
        """
        preds_norm = normalize_probabilities(preds)

        sub_df = pd.DataFrame({
            "id": test_df["id"].values,
            "winner_model_a": preds_norm[:, 0],
            "winner_model_b": preds_norm[:, 1],
            "winner_tie": preds_norm[:, 2],
        })

        # Strict validation
        gate_res = self.gatekeeper.verify_submission_format(sub_df, test_df)
        if not gate_res.passed:
            raise ValueError(f"Submission verification failed: {gate_res.message}")

        out_path = self.output_dir / filename
        sub_df.to_csv(out_path, index=False)
        return out_path

    def generate_standalone_kaggle_kernel(
        self,
        lgb_model_str: Optional[str] = None,
        ensemble_weights: Optional[dict] = None,
        temperature: float = 1.0,
        lsa_data: Optional[dict] = None,
        output_filename: str = "kaggle_submission_kernel.py",
    ) -> Path:
        """
        Generates a standalone, zero-dependency Kaggle inference kernel
        capable of discovering test data, extracting features, running model inference
        with TTA symmetry, and outputting submission.csv within Kaggle's notebook environment.
        Compressed with LZMA to guarantee kernel source size is strictly < 1MB for Kaggle API.
        """
        package_dict = {}
        if lgb_model_str is not None:
            package_dict["booster"] = lgb_model_str
        if lsa_data is not None:
            raw_vocab = lsa_data["vocab"]
            if isinstance(raw_vocab, (list, tuple)):
                vocab_str = "\n".join(raw_vocab)
            else:
                vocab_str = str(raw_vocab)
            package_dict["lsa"] = {
                "vocab": vocab_str,
                "idf": lsa_data["idf"].astype(np.float16),
                "comp": lsa_data["comp"].astype(np.float16),
            }

        if package_dict:
            import base64
            import lzma
            import pickle
            model_blob_b64 = base64.b64encode(
                lzma.compress(pickle.dumps(package_dict), preset=9 | lzma.PRESET_EXTREME)
            ).decode("ascii")
            model_blob_repr = repr(model_blob_b64)
        else:
            model_blob_repr = "None"

        if ensemble_weights is not None:
            if hasattr(ensemble_weights, "tolist"):
                weights_repr = repr(ensemble_weights.tolist())
            else:
                weights_repr = repr(ensemble_weights)
        else:
            weights_repr = "{}"

        kernel_code = rf'''"""
Kaggle Submission Pipeline for LLM Classification Finetuning.
"""

import json
import os
import re
import numpy as np
import pandas as pd
from scipy.special import expit

MODEL_BLOB_LZMA = {model_blob_repr}
ENSEMBLE_WEIGHTS = {weights_repr}
TEMPERATURE = {temperature}

LIST_PATTERN = re.compile(r"^\s*([*\-+]|\d+\.)\s+", re.MULTILINE)
HEADER_PATTERN = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
TABLE_PATTERN = re.compile(r"\|(?:\s*:?---+:?\s*\|)+")
LATEX_PATTERN = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^\$]+?\$|\\[a-zA-Z]+)")
DISCLAIMER_PATTERN = re.compile(
    r"\b(as an ai|as a large language model|i cannot|i am unable to|i apologize|sorry)\b",
    re.IGNORECASE,
)

def safe_parse_dialogue(val):
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
        except Exception:
            pass
    return [val_clean.replace("\\n", "\n").replace('\\"', '"')]

def compute_turn_metrics(turns):
    n_turns = max(1, len(turns))
    full_text = "\n\n".join(turns)
    last_turn = turns[-1] if turns else ""
    char_lens = [len(t) for t in turns]
    word_lens = [len(t.split()) for t in turns]
    return {{
        "n_turns": float(n_turns),
        "total_chars": float(sum(char_lens)),
        "total_words": float(sum(word_lens)),
        "last_turn_chars": float(len(last_turn)),
        "last_turn_words": float(len(last_turn.split())),
        "turn_char_growth": float(char_lens[-1] - char_lens[0]) if n_turns > 1 else 0.0,
        "full_text": full_text,
        "last_turn_text": last_turn,
    }}

def extract_features(df, swap=False):
    col_a = "response_b" if swap else "response_a"
    col_b = "response_a" if swap else "response_b"

    records = []
    for prompt, resp_a, resp_b in zip(df["prompt"].fillna(""), df[col_a].fillna(""), df[col_b].fillna("")):
        p_turns = safe_parse_dialogue(prompt)
        a_turns = safe_parse_dialogue(resp_a)
        b_turns = safe_parse_dialogue(resp_b)

        p_stats = compute_turn_metrics(p_turns)
        a_stats = compute_turn_metrics(a_turns)
        b_stats = compute_turn_metrics(b_turns)

        text_a, text_b = a_stats["full_text"], b_stats["full_text"]
        text_prompt = p_stats["full_text"]

        char_a = a_stats["total_chars"]
        char_b = b_stats["total_chars"]
        word_a = a_stats["total_words"]
        word_b = b_stats["total_words"]
        word_prompt = p_stats["total_words"]

        words_a_list = text_a.split()
        words_b_list = text_b.split()
        words_prompt_list = text_prompt.split()

        char_diff = char_a - char_b
        word_diff = word_a - word_b
        norm_denom = char_a + char_b + 1.0

        last_char_a = a_stats["last_turn_chars"]
        last_char_b = b_stats["last_turn_chars"]
        last_word_a = a_stats["last_turn_words"]
        last_word_b = b_stats["last_turn_words"]
        last_char_diff = last_char_a - last_char_b
        last_word_diff = last_word_a - last_word_b
        last_norm_denom = last_char_a + last_char_b + 1.0

        code_a = text_a.count("```")
        code_b = text_b.count("```")
        lists_a = len(LIST_PATTERN.findall(text_a))
        lists_b = len(LIST_PATTERN.findall(text_b))
        bold_a = text_a.count("**")
        bold_b = text_b.count("**")
        head_a = len(HEADER_PATTERN.findall(text_a))
        head_b = len(HEADER_PATTERN.findall(text_b))
        nl_a = text_a.count("\n")
        nl_b = text_b.count("\n")

        tables_a = len(TABLE_PATTERN.findall(text_a))
        tables_b = len(TABLE_PATTERN.findall(text_b))
        latex_a = len(LATEX_PATTERN.findall(text_a))
        latex_b = len(LATEX_PATTERN.findall(text_b))
        disclaim_a = len(DISCLAIMER_PATTERN.findall(text_a))
        disclaim_b = len(DISCLAIMER_PATTERN.findall(text_b))

        set_p = set(w.lower() for w in words_prompt_list if len(w) > 2)
        set_a = set(w.lower() for w in words_a_list if len(w) > 2)
        set_b = set(w.lower() for w in words_b_list if len(w) > 2)

        j_a = len(set_p.intersection(set_a)) / len(set_p.union(set_a)) if len(set_p.union(set_a)) > 0 else 0.0
        j_b = len(set_p.intersection(set_b)) / len(set_p.union(set_b)) if len(set_p.union(set_b)) > 0 else 0.0
        j_ab = len(set_a.intersection(set_b)) / len(set_a.union(set_b)) if len(set_a.union(set_b)) > 0 else 1.0

        if len(text_a) >= 3 and len(text_b) >= 3:
            ng_a = set(text_a[i : i + 3] for i in range(len(text_a) - 2))
            ng_b = set(text_b[i : i + 3] for i in range(len(text_b) - 2))
            u_ng = len(ng_a.union(ng_b))
            jaccard_3g = float(len(ng_a.intersection(ng_b)) / u_ng) if u_ng > 0 else 1.0
        else:
            jaccard_3g = 1.0 if text_a.strip() == text_b.strip() else 0.0

        records.append({{
            "char_len_a": float(char_a),
            "char_len_b": float(char_b),
            "word_len_a": float(word_a),
            "word_len_b": float(word_b),
            "prompt_word_len": float(word_prompt),
            "char_diff": float(char_diff),
            "word_diff": float(word_diff),
            "char_ratio": float(char_diff / norm_denom),
            "word_ratio": float(word_diff / (word_a + word_b + 1.0)),
            "log_char_diff": float(np.log1p(char_a) - np.log1p(char_b)),
            "log_word_diff": float(np.log1p(word_a) - np.log1p(word_b)),
            "num_turns": float(p_stats["n_turns"]),
            "last_char_len_a": float(last_char_a),
            "last_char_len_b": float(last_char_b),
            "last_word_len_a": float(last_word_a),
            "last_word_len_b": float(last_word_b),
            "last_char_diff": float(last_char_diff),
            "last_word_diff": float(last_word_diff),
            "last_char_ratio": float(last_char_diff / last_norm_denom),
            "turn_growth_a": float(a_stats["turn_char_growth"]),
            "turn_growth_b": float(b_stats["turn_char_growth"]),
            "turn_growth_diff": float(a_stats["turn_char_growth"] - b_stats["turn_char_growth"]),
            "code_blocks_a": float(code_a),
            "code_blocks_b": float(code_b),
            "code_blocks_diff": float(code_a - code_b),
            "lists_a": float(lists_a),
            "lists_b": float(lists_b),
            "lists_diff": float(lists_a - lists_b),
            "bold_a": float(bold_a),
            "bold_b": float(bold_b),
            "bold_diff": float(bold_a - bold_b),
            "head_a": float(head_a),
            "head_b": float(head_b),
            "head_diff": float(head_a - head_b),
            "newlines_a": float(nl_a),
            "newlines_b": float(nl_b),
            "newlines_diff": float(nl_a - nl_b),
            "tables_a": float(tables_a),
            "tables_b": float(tables_b),
            "tables_diff": float(tables_a - tables_b),
            "latex_a": float(latex_a),
            "latex_b": float(latex_b),
            "latex_diff": float(latex_a - latex_b),
            "disclaimer_a": float(disclaim_a),
            "disclaimer_b": float(disclaim_b),
            "disclaimer_diff": float(disclaim_a - disclaim_b),
            "prompt_jaccard_a": float(j_a),
            "prompt_jaccard_b": float(j_b),
            "prompt_overlap_diff": float(len(set_p.intersection(set_a)) - len(set_p.intersection(set_b))),
            "prompt_jaccard_diff": float(j_a - j_b),
            "resp_ab_jaccard": float(j_ab),
            "is_identical": float(text_a.strip() == text_b.strip()),
            "abs_norm_diff": float(abs(char_diff) / norm_denom),
            "jaccard_3g": float(jaccard_3g),
            "avg_word_len_a": float(char_a / word_a) if word_a > 0 else 0.0,
            "avg_word_len_b": float(char_b / word_b) if word_b > 0 else 0.0,
            "avg_word_len_diff": float((char_a / word_a) - (char_b / word_b)) if word_a > 0 and word_b > 0 else 0.0,
            "questions_diff": float(text_a.count("?") - text_b.count("?")),
            "exclamations_diff": float(text_a.count("!") - text_b.count("!")),
        }})
    return pd.DataFrame(records)

def extract_lsa_features(df, lsa_dict, n_components=None, swap=False):
    col_a = "response_b" if swap else "response_a"
    col_b = "response_a" if swap else "response_b"

    prompts = ["\n\n".join(safe_parse_dialogue(x)) for x in df["prompt"].fillna("")]
    resps_a = ["\n\n".join(safe_parse_dialogue(x)) for x in df[col_a].fillna("")]
    resps_b = ["\n\n".join(safe_parse_dialogue(x)) for x in df[col_b].fillna("")]

    raw_v = lsa_dict["vocab"]
    if isinstance(raw_v, str):
        vocab = {{w: i for i, w in enumerate(raw_v.split("\n"))}}
    else:
        vocab = {{w: i for i, w in enumerate(raw_v)}}
    idf = lsa_dict["idf"].astype(np.float64)
    components = lsa_dict["comp"].astype(np.float64)
    if n_components is None:
        n_components = components.shape[0]

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    cv = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2), stop_words="english")

    def _transform(texts):
        X = cv.transform(texts).astype(np.float64)
        if X.nnz > 0:
            np.log(X.data, out=X.data)
            X.data += 1.0
            X.data *= idf[X.indices]
            X = normalize(X, norm="l2")
            return X.dot(components.T)
        return np.zeros((len(texts), n_components), dtype=np.float64)

    lsa_p = _transform(prompts)
    lsa_a = _transform(resps_a)
    lsa_b = _transform(resps_b)

    def _norm(mat):
        n = np.linalg.norm(mat, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return mat / n

    norm_p = _norm(lsa_p)
    norm_a = _norm(lsa_a)
    norm_b = _norm(lsa_b)

    cos_p_a = np.sum(norm_p * norm_a, axis=1)
    cos_p_b = np.sum(norm_p * norm_b, axis=1)
    cos_a_b = np.sum(norm_a * norm_b, axis=1)
    l2_diff = np.linalg.norm(lsa_p - lsa_a, axis=1) - np.linalg.norm(lsa_p - lsa_b, axis=1)
    cos_diff = cos_p_a - cos_p_b

    feat_dict = {{
        "lsa_cos_p_a": cos_p_a,
        "lsa_cos_p_b": cos_p_b,
        "lsa_cos_diff": cos_diff,
        "lsa_l2_diff": l2_diff,
        "lsa_cos_a_b": cos_a_b,
    }}
    coord_diff = lsa_a - lsa_b
    if coord_diff.shape[1] < n_components:
        coord_diff = np.pad(coord_diff, ((0, 0), (0, n_components - coord_diff.shape[1])), mode="constant")
    for i in range(n_components):
        feat_dict[f"lsa_dim_diff_{{i}}"] = coord_diff[:, i]
    return pd.DataFrame(feat_dict, index=df.index)

def predict_length_prior(df):
    feats_norm = extract_features(df, swap=False)
    ratios = feats_norm["char_ratio"].values
    sig_norm = expit(1.8 * ratios)
    sig_swap = expit(1.8 * (-ratios))
    p_tie = 0.31
    rem = 1.0 - p_tie
    p_a_norm, p_b_norm = rem * sig_norm, rem * (1.0 - sig_norm)
    p_a_swap, p_b_swap = rem * sig_swap, rem * (1.0 - sig_swap)
    p_a = 0.5 * (p_a_norm + p_b_swap)
    p_b = 0.5 * (p_b_norm + p_a_swap)
    p_tie_arr = np.full_like(p_a, p_tie)
    probs = np.column_stack([p_a, p_b, p_tie_arr])
    return probs / probs.sum(axis=1, keepdims=True)

def main():
    test_path = None
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "test.csv" in files:
                test_path = os.path.join(root, "test.csv")
                print(f"Discovered test.csv at: {{test_path}}")
                break

    if not test_path or not os.path.exists(test_path):
        for candidate in ["data/raw/test.csv", "test.csv"]:
            if os.path.exists(candidate):
                test_path = candidate
                break

    if not test_path:
        raise FileNotFoundError("Could not find test.csv in /kaggle/input or local directories.")

    print(f"Loading test data from: {{test_path}}")
    test_df = pd.read_csv(test_path)

    p_length = predict_length_prior(test_df)
    final_probs = p_length

    booster = None
    lsa_dict = None
    if MODEL_BLOB_LZMA is not None:
        try:
            import base64
            import lzma
            import pickle
            pkg = pickle.loads(lzma.decompress(base64.b64decode(MODEL_BLOB_LZMA)))
            if "booster" in pkg:
                import lightgbm as lgb
                booster = lgb.Booster(model_str=pkg["booster"])
            if "lsa" in pkg:
                lsa_dict = pkg["lsa"]
        except Exception as e:
            print(f"Error unpacking model blob: {{e}}")

    if booster is not None:
        try:
            X_norm = extract_features(test_df, swap=False)
            X_swap = extract_features(test_df, swap=True)

            if lsa_dict is not None:
                X_lsa_norm = extract_lsa_features(test_df, lsa_dict=lsa_dict, swap=False)
                X_lsa_swap = extract_lsa_features(test_df, lsa_dict=lsa_dict, swap=True)
                X_norm = pd.concat([X_norm.reset_index(drop=True), X_lsa_norm.reset_index(drop=True)], axis=1)
                X_swap = pd.concat([X_swap.reset_index(drop=True), X_lsa_swap.reset_index(drop=True)], axis=1)

            p_norm = booster.predict(X_norm.values)
            p_swap = booster.predict(X_swap.values)

            p_sym_lgb = np.zeros_like(p_norm)
            p_sym_lgb[:, 0] = 0.5 * (p_norm[:, 0] + p_swap[:, 1])
            p_sym_lgb[:, 1] = 0.5 * (p_norm[:, 1] + p_swap[:, 0])
            p_sym_lgb[:, 2] = 0.5 * (p_norm[:, 2] + p_swap[:, 2])
            p_sym_lgb = p_sym_lgb / p_sym_lgb.sum(axis=1, keepdims=True)

            # Check if ensemble weights exist
            if isinstance(ENSEMBLE_WEIGHTS, dict) and len(ENSEMBLE_WEIGHTS) > 0:
                w_prior = ENSEMBLE_WEIGHTS.get("baseline_length_prior", 0.0)
                w_boost = sum(v for k, v in ENSEMBLE_WEIGHTS.items() if k != "baseline_length_prior")
                total_w = w_prior + w_boost
                if total_w > 0:
                    final_probs = (w_prior / total_w) * p_length + (w_boost / total_w) * p_sym_lgb
                else:
                    final_probs = p_sym_lgb
            elif isinstance(ENSEMBLE_WEIGHTS, (list, tuple)) and len(ENSEMBLE_WEIGHTS) >= 2:
                w_prior = ENSEMBLE_WEIGHTS[0]
                w_boost = sum(ENSEMBLE_WEIGHTS[1:])
                total_w = w_prior + w_boost
                if total_w > 0:
                    final_probs = (w_prior / total_w) * p_length + (w_boost / total_w) * p_sym_lgb
                else:
                    final_probs = p_sym_lgb
            else:
                final_probs = p_sym_lgb
            print("Successfully executed LightGBM booster inference.")
        except Exception as e:
            print(f"LightGBM inference fallback to length prior: {{e}}")
            final_probs = p_length

    # Apply temperature calibration if requested
    if abs(TEMPERATURE - 1.0) > 1e-4:
        logits = np.log(np.clip(final_probs, 1e-15, 1.0 - 1e-15)) / TEMPERATURE
        shift = logits - np.max(logits, axis=1, keepdims=True)
        exp_shift = np.exp(shift)
        final_probs = exp_shift / exp_shift.sum(axis=1, keepdims=True)

    # Normalize final probabilities
    row_sums = final_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    final_probs = final_probs / row_sums

    sub = pd.DataFrame({{
        "id": test_df["id"],
        "winner_model_a": final_probs[:, 0],
        "winner_model_b": final_probs[:, 1],
        "winner_tie": final_probs[:, 2],
    }})

    sub.to_csv("submission.csv", index=False)
    print("Generated submission.csv successfully!")
    print(sub.head())

if __name__ == "__main__":
    main()
'''
        out_path = self.output_dir / output_filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(kernel_code)
        return out_path
