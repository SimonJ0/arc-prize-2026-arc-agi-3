"""
The Permanent Self-Driving Research Loop Controller.
Orchestrates autonomous hypotheses execution, 5-fold cross-validation,
gate checks, diagnostic HTML generation, ensembling, and the Iron Rule submission gate.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.core.adaptive import AdaptiveHypothesisGenerator
from src.core.calibration import TemperatureCalibrator
from src.core.gates import GateResult, ValidationGatekeeper
from src.core.metrics import (
    CLASSES,
    compute_log_loss,
    evaluate_predictions,
)
from src.data.loader import DataLoader
from src.data.splitter import DatasetSplitter
from src.evaluation.diagnostics import DiagnosticEngine
from src.evaluation.html_reporter import HtmlReportGenerator
from src.features.extractor import FeatureExtractor
from src.features.lsa_vectorizer import DenseLSAVectorizer
from src.features.text_vectorizer import TextVectorizer
from src.models.ensemble import EnsembleBlender
from src.models.gbdt_classifier import LightGBMPredictor
from src.models.length_prior import LengthPriorPredictor
from src.models.tfidf_linear import TfidfLogisticPredictor
from src.submit.generator import SubmissionGenerator
from src.submit.submission_gate import SubmissionAuthorizationGate


class SelfDrivingResearchLoop:
    """
    Permanent self-improving research loop modeled after Tom's BirdCLEF-2026 solution.
    """

    def __init__(
        self,
        config_path: str = "configs/default_config.yaml",
        hypotheses_path: str = "configs/hypotheses.yaml",
    ):
        self.config_path = Path(config_path)
        self.hypotheses_path = Path(hypotheses_path)
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        with open(hypotheses_path, encoding="utf-8") as f:
            self.hypotheses_data = yaml.safe_load(f)

        self.data_loader = DataLoader(data_dir=self.config["paths"]["raw_data_dir"])
        self.splitter = DatasetSplitter(
            n_splits=self.config["cross_validation"]["n_splits"],
            random_state=self.config["cross_validation"]["random_state"],
        )
        self.gatekeeper = ValidationGatekeeper(
            min_loss_improvement=self.config["gating"]["min_loss_improvement"],
            max_symmetry_divergence=self.config["gating"]["max_symmetry_divergence"],
            max_ensemble_correlation=self.config["gating"]["max_ensemble_correlation"],
        )
        self.diagnostics_engine = DiagnosticEngine()
        self.html_reporter = HtmlReportGenerator(output_dir=self.config["paths"]["reports_dir"])
        self.submission_generator = SubmissionGenerator(
            output_dir=self.config["paths"]["submissions_dir"]
        )
        self.auth_gate = SubmissionAuthorizationGate(
            reports_dir=self.config["paths"]["reports_dir"]
        )

        self.experiments_dir = Path(self.config["paths"]["experiments_dir"])
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.experiments_dir / "registry.json"
        self.state_path = self.experiments_dir / "ensemble_state.npz"

        self.registry = self._load_registry()
        self.best_log_loss: float | None = self._get_current_best_loss()
        self.ensemble_oofs: list[np.ndarray] = []
        self.ensemble_names: list[str] = []
        self.trained_models: dict[str, list[Any]] = {}
        self.temperature_calibrators: dict[str, TemperatureCalibrator] = {}
        self._last_blender: EnsembleBlender | None = None

        self._load_ensemble_state()

    def _save_ensemble_state(self):
        """Persists ensemble pool OOF arrays and model names to disk."""
        if not self.ensemble_oofs:
            return
        np.savez_compressed(
            self.state_path,
            oofs=np.array(self.ensemble_oofs),
            names=np.array(self.ensemble_names),
        )

    def _load_ensemble_state(self):
        """Restores ensemble pool from disk if available."""
        if self.state_path.exists():
            try:
                data = np.load(self.state_path, allow_pickle=True)
                self.ensemble_oofs = [oof for oof in data["oofs"]]
                self.ensemble_names = [str(n) for n in data["names"]]
                print(
                    f"[STATE] Restored {len(self.ensemble_oofs)} models in ensemble pool from {self.state_path.name}"
                )
            except Exception as e:
                print(f"[STATE] Could not load ensemble state: {e}")

    def _load_registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            with open(self.registry_path, encoding="utf-8") as f:
                return json.load(f)
        return {"experiments": [], "current_best": None}

    def _save_registry(self):
        def _json_convert(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, default=_json_convert)

    def _get_current_best_loss(self) -> float | None:
        best = None
        for exp in self.registry.get("experiments", []):
            if exp.get("status") == "PASSED":
                loss = exp.get("metrics", {}).get("log_loss")
                if loss is not None:
                    if best is None or loss < best:
                        best = loss
        return best

    def update_leaderboard_md(self):
        """Generates clean Markdown leaderboard of all iterations."""
        reports_dir = Path(self.config["paths"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for exp in self.registry.get("experiments", []):
            m = exp.get("metrics", {})
            rows.append(
                {
                    "ID": exp.get("id"),
                    "Model": exp.get("model_name"),
                    "Status": exp.get("status"),
                    "Log Loss": f"{m.get('log_loss', 9.99):.5f}",
                    "Brier": f"{m.get('brier_score', 9.99):.5f}",
                    "ECE": f"{m.get('ece', 9.99):.4f}",
                    "Sym Div": f"{exp.get('symmetry_divergence', 0.0):.4f}",
                    "Timestamp": exp.get("timestamp", ""),
                }
            )

        df_board = pd.DataFrame(rows)
        if not df_board.empty:
            df_board = df_board.sort_values(by="Log Loss", ascending=True)
            table_md = df_board.to_markdown(index=False)
        else:
            table_md = "No experiments logged yet."

        content = f"""# Autonomous Research Leaderboard

Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Competition: `{self.config["competition"]["name"]}`
Optimization Metric: `Multi-Class Log Loss (Lower is better)`

{table_md}
"""
        with open(reports_dir / "leaderboard.md", "w", encoding="utf-8") as f:
            f.write(content)

    def run_experiment(
        self, hypothesis_spec: dict[str, Any], df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Executes a single experiment cycle end-to-end:
        1. Feature preparation
        2. 5-Fold Cross-Validation
        3. TTA & Symmetry evaluation
        4. Validation Gates Check
        5. Diagnostics & HTML Dashboard
        6. Ensembling & Authorization Request
        """
        exp_id = hypothesis_spec["id"]
        model_name = hypothesis_spec["name"]
        model_type = hypothesis_spec["model_type"]
        hyp_text = hypothesis_spec["hypothesis"]
        params = hypothesis_spec.get("parameters", {})

        print(f"\n[{'=' * 30}]")
        print(f"Executing {exp_id}: {model_name} ({model_type})")
        print(f"Hypothesis: {hyp_text}")
        print(f"[{'=' * 30}]\n")

        # 1. Feature preparation & Splits
        splits = self.splitter.split(
            df_train, group_by_prompt=self.config["cross_validation"]["group_by_prompt"]
        )
        y_true = df_train[CLASSES].values
        y_indices = df_train["target"].values
        n_samples = len(df_train)

        oof_preds = np.zeros((n_samples, 3))
        oof_swapped = np.zeros((n_samples, 3))
        fold_losses: list[float] = []
        feat_importances: dict[str, float] = {}
        fold_models: list[Any] = []

        if model_type == "length_prior":
            extractor = FeatureExtractor()
            X_feats = extractor.extract_features(df_train)
            X_swap = extractor.extract_swapped_features(df_train)

            for _fold_idx, (train_idx, val_idx) in enumerate(splits):
                model: Any = LengthPriorPredictor(beta=params.get("beta", 1.6))
                model.fit(X_feats.iloc[train_idx], y_indices[train_idx])

                p_sym, p_norm, p_sw = model.predict_proba_with_tta(
                    X_feats.iloc[val_idx],
                    X_swap.iloc[val_idx],
                )
                oof_preds[val_idx] = p_sym
                oof_swapped[val_idx] = p_sw

                fold_loss = compute_log_loss(y_true[val_idx], p_sym)
                fold_losses.append(fold_loss)
                fold_models.append(model)

        elif model_type == "lightgbm":
            extractor = FeatureExtractor()
            X_feats = extractor.extract_features(df_train)
            X_swap = extractor.extract_swapped_features(df_train)

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                X_train_f = X_feats.iloc[train_idx]
                y_train_f = y_indices[train_idx]

                # Symmetric data augmentation if configured
                if params.get("symmetric_training", True):
                    X_train_swap = X_swap.iloc[train_idx]
                    swap_map = {0: 1, 1: 0, 2: 2}
                    y_train_swap = np.array([swap_map[y] for y in y_train_f])
                    X_train_f = pd.concat([X_train_f, X_train_swap], ignore_index=True)
                    y_train_f = np.concatenate([y_train_f, y_train_swap])

                model = LightGBMPredictor(name=f"lgb_fold_{fold_idx}", params=params)
                model.fit(X_train_f, y_train_f)

                p_sym, p_norm, p_sw = model.predict_proba_with_tta(
                    X_feats.iloc[val_idx],
                    X_swap.iloc[val_idx],
                )
                oof_preds[val_idx] = p_sym
                oof_swapped[val_idx] = p_sw

                fold_loss = compute_log_loss(y_true[val_idx], p_sym)
                fold_losses.append(fold_loss)
                fold_models.append(model)

                # Aggregate importances
                for feat, imp in model.feature_importances.items():
                    feat_importances[feat] = feat_importances.get(feat, 0.0) + imp

            # Average importances
            for feat in feat_importances:
                feat_importances[feat] = feat_importances[feat] / len(splits)

        elif model_type == "tfidf_logistic":
            for _fold_idx, (train_idx, val_idx) in enumerate(splits):
                vectorizer = TextVectorizer(max_features=params.get("max_features", 5000))
                train_fold_df = df_train.iloc[train_idx]
                val_fold_df = df_train.iloc[val_idx]

                # Fit ONLY on training fold text to eliminate leakage
                train_texts = pd.concat(
                    [
                        train_fold_df["prompt"],
                        train_fold_df["response_a"],
                        train_fold_df["response_b"],
                    ]
                ).tolist()
                vectorizer.fit(train_texts)

                X_diff_train = vectorizer.transform_differential(train_fold_df, swap=False)
                X_diff_val = vectorizer.transform_differential(val_fold_df, swap=False)
                X_diff_val_swap = vectorizer.transform_differential(val_fold_df, swap=True)

                model = TfidfLogisticPredictor(C=params.get("C", 1.0))
                model.fit(X_diff_train, y_indices[train_idx])

                p_sym, p_norm, p_sw = model.predict_proba_with_tta(
                    X_diff_val,
                    X_diff_val_swap,
                )
                oof_preds[val_idx] = p_sym
                oof_swapped[val_idx] = p_sw

                fold_loss = compute_log_loss(y_true[val_idx], p_sym)
                fold_losses.append(fold_loss)
                fold_models.append((model, vectorizer))

        elif model_type == "lsa_lightgbm":
            extractor = FeatureExtractor()
            X_feats = extractor.extract_features(df_train)
            X_swap = extractor.extract_swapped_features(df_train)

            n_comps = params.get("n_components", 16)
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                train_fold_df = df_train.iloc[train_idx]
                val_fold_df = df_train.iloc[val_idx]

                lsa_vec = DenseLSAVectorizer(
                    n_components=n_comps, max_features=params.get("max_features", 5000)
                )
                train_texts = pd.concat(
                    [
                        train_fold_df["prompt"],
                        train_fold_df["response_a"],
                        train_fold_df["response_b"],
                    ]
                ).tolist()
                lsa_vec.fit(train_texts)

                # Training fold LSA features
                X_lsa_train = lsa_vec.extract_lsa_features(train_fold_df, swap=False)
                X_train_f = pd.concat(
                    [
                        X_feats.iloc[train_idx].reset_index(drop=True),
                        X_lsa_train.reset_index(drop=True),
                    ],
                    axis=1,
                )
                y_train_f = y_indices[train_idx]

                if params.get("symmetric_training", True):
                    X_lsa_train_swap = lsa_vec.extract_lsa_features(train_fold_df, swap=True)
                    X_train_swap = pd.concat(
                        [
                            X_swap.iloc[train_idx].reset_index(drop=True),
                            X_lsa_train_swap.reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    swap_map = {0: 1, 1: 0, 2: 2}
                    y_train_swap = np.array([swap_map[y] for y in y_train_f])
                    X_train_f = pd.concat([X_train_f, X_train_swap], ignore_index=True)
                    y_train_f = np.concatenate([y_train_f, y_train_swap])

                model = LightGBMPredictor(name=f"lsa_lgb_fold_{fold_idx}", params=params)
                model.fit(X_train_f, y_train_f)

                # Validation fold LSA features
                X_lsa_val = lsa_vec.extract_lsa_features(val_fold_df, swap=False)
                X_lsa_val_swap = lsa_vec.extract_lsa_features(val_fold_df, swap=True)
                X_val_norm = pd.concat(
                    [
                        X_feats.iloc[val_idx].reset_index(drop=True),
                        X_lsa_val.reset_index(drop=True),
                    ],
                    axis=1,
                )
                X_val_swap = pd.concat(
                    [
                        X_swap.iloc[val_idx].reset_index(drop=True),
                        X_lsa_val_swap.reset_index(drop=True),
                    ],
                    axis=1,
                )

                p_sym, p_norm, p_sw = model.predict_proba_with_tta(X_val_norm, X_val_swap)
                oof_preds[val_idx] = p_sym
                oof_swapped[val_idx] = p_sw

                fold_loss = compute_log_loss(y_true[val_idx], p_sym)
                fold_losses.append(fold_loss)
                fold_models.append((model, lsa_vec))

                for feat, imp in model.feature_importances.items():
                    feat_importances[feat] = feat_importances.get(feat, 0.0) + imp

            for feat in feat_importances:
                feat_importances[feat] = feat_importances[feat] / len(splits)

        elif model_type == "ensemble":
            if len(self.ensemble_oofs) < 2:
                raise ValueError("Ensemble requires at least 2 existing models in the pool.")

            blender = EnsembleBlender(names=self.ensemble_names)
            n_blend = params.get("n_blend_folds", 5)
            # Use nested cross-validation blending to prevent in-sample overfitting
            blender.fit_with_nested_cv(self.ensemble_oofs, y_true, n_blend_folds=n_blend)
            self._last_blender = blender
            oof_preds = blender.blend(self.ensemble_oofs)
            # Under response swap, symmetric ensemble predictions swap win_a and win_b probabilities
            oof_swapped = np.column_stack([oof_preds[:, 1], oof_preds[:, 0], oof_preds[:, 2]])
            fold_losses = [blender.optimal_log_loss or 0.0] * len(splits)
            print(f"Optimal ensemble weights (nested CV): {blender.get_weight_summary()}")

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Retain trained fold models for fold-bagged test prediction
        self.trained_models[exp_id] = fold_models

        # Tier 2: Post-Hoc Probability Calibration via Temperature Scaling
        calibrator = TemperatureCalibrator()
        calibrator.fit(y_true, oof_preds)
        calibrated_oofs = calibrator.calibrate(oof_preds)
        raw_loss = compute_log_loss(y_true, oof_preds)
        cal_loss = compute_log_loss(y_true, calibrated_oofs)

        if cal_loss < raw_loss:
            print(
                f"  [CALIBRATION] Temperature scaling (T={calibrator.temperature:.3f}) improved log loss: {raw_loss:.5f} -> {cal_loss:.5f}"
            )
            oof_preds = calibrated_oofs
            oof_swapped = calibrator.calibrate(oof_swapped)
        self.temperature_calibrators[exp_id] = calibrator

        # Compute full metrics suite
        metrics = evaluate_predictions(y_true, oof_preds, preds_swapped=oof_swapped)
        overall_loss = metrics["log_loss"]
        sym_div = metrics.get("symmetry_divergence", 0.0)

        # 2. Gatekeeper Checks
        # Validate leakage across all K folds
        leak_check = GateResult(
            passed=True,
            gate_name="Leakage Gate",
            message=f"All {len(splits)} folds verified leak-free.",
        )
        for _fold_idx, (t_idx, v_idx) in enumerate(splits):
            fold_leak = self.gatekeeper.check_leakage(t_idx, v_idx, oof_preds[v_idx])
            if not fold_leak.passed:
                leak_check = fold_leak
                break

        cv_check = self.gatekeeper.check_cv_improvement(
            candidate_log_loss=overall_loss,
            best_log_loss=self.best_log_loss,
            allow_slight_regression_for_diversity=(model_type != "ensemble"),
        )
        sym_check = self.gatekeeper.check_symmetry(oof_preds, oof_swapped)
        div_check = self.gatekeeper.check_diversity(oof_preds, self.ensemble_oofs)
        anomaly_check = self.gatekeeper.check_anomalies(oof_preds, fold_losses)

        # Enforce diversity requirement:
        # If a candidate model had slight regression, it was conditionally allowed by CV gate ONLY for diversity.
        # If it fails diversity check, it must be rejected.
        if model_type != "ensemble" and cv_check.passed:
            cv_delta = cv_check.details.get("delta", 0.0)
            if cv_delta > -self.config["gating"]["min_loss_improvement"]:
                if not div_check.passed:
                    cv_check = GateResult(
                        passed=False,
                        gate_name="Diversity Gate",
                        message=f"REJECTED: Model regressed (+{cv_delta:.5f}) and is too correlated with existing models ({div_check.message}).",
                        details={"cv_delta": cv_delta, "div_details": div_check.details},
                    )

        all_gates_passed = (
            leak_check.passed and cv_check.passed and sym_check.passed and anomaly_check.passed
        )

        print(f"Results for {exp_id}:")
        print(f"  OOF Log Loss: {overall_loss:.5f} (Fold Mean: {np.mean(fold_losses):.5f})")
        print(f"  Brier Score:  {metrics['brier_score']:.5f} | ECE: {metrics['ece']:.4f}")
        print(f"  Symmetry Div: {sym_div:.5f}")
        print(
            f"  Gate Checks: Leakage={leak_check.passed}, CV={cv_check.passed}, Symmetry={sym_check.passed}, Diversity={div_check.passed}, Anomaly={anomaly_check.passed}"
        )

        delta = (overall_loss - self.best_log_loss) if self.best_log_loss is not None else None

        # 3. Deep Diagnostics & HTML Report
        diag_data = self.diagnostics_engine.run_full_diagnostics(
            df=df_train,
            y_true=y_true,
            y_pred=oof_preds,
            fold_scores=fold_losses,
            feature_importances=feat_importances if feat_importances else None,
        )
        report_path = self.html_reporter.generate(
            exp_id=exp_id,
            model_name=model_name,
            hypothesis=hyp_text,
            diagnostics=diag_data,
            delta=delta,
            symmetry_div=sym_div,
            gate_passed=all_gates_passed,
        )
        print(f"  HTML Report: {report_path}")

        # 4. Handle Promotion & Submission Gate
        auth_req_path = None

        if all_gates_passed:
            status = "PASSED"
            # Add to ensemble pool if non-ensemble model
            if model_type != "ensemble":
                if div_check.passed or len(self.ensemble_oofs) == 0:
                    self.ensemble_oofs.append(oof_preds.copy())
                    self.ensemble_names.append(model_name)
                    self._save_ensemble_state()

            # Tier 3: Per-experiment model artifact serialization & checkpointing
            models_dir = self.experiments_dir / "models" / exp_id
            models_dir.mkdir(parents=True, exist_ok=True)
            if exp_id in self.trained_models:
                for f_idx, mod in enumerate(self.trained_models[exp_id]):
                    target_m = mod[0] if isinstance(mod, tuple) else mod
                    target_lsa = mod[1] if isinstance(mod, tuple) else None
                    if hasattr(target_m, "booster_model_string") and target_m.booster_model_string:
                        (models_dir / f"booster_fold_{f_idx}.txt").write_text(
                            target_m.booster_model_string, encoding="utf-8"
                        )
                    if target_lsa is not None and hasattr(target_lsa, "tfidf"):
                        import pickle

                        with open(models_dir / f"lsa_data_fold_{f_idx}.pkl", "wb") as f:
                            pickle.dump(
                                {
                                    "vocab": list(target_lsa.tfidf.get_feature_names_out()),
                                    "idf": target_lsa.tfidf.idf_,
                                    "comp": target_lsa.svd.components_,
                                },
                                f,
                            )

            if self.best_log_loss is None or overall_loss < self.best_log_loss:
                previous_best = self.best_log_loss
                self.best_log_loss = overall_loss
                delta_str = (
                    f"{-(overall_loss - previous_best):.5f} improvement"
                    if previous_best
                    else "initial baseline"
                )
                print(f"  >>> NEW BEST MODEL PROMOTED: {overall_loss:.5f} ({delta_str}) <<<")

                # Generate fold-bagged test predictions
                test_preds = self._predict_test_bagged(
                    exp_id, model_type, params, df_train, df_test
                )
                sub_file = self.submission_generator.generate_submission_csv(
                    test_df=df_test,
                    preds=test_preds,
                    filename=f"submission_{exp_id}.csv",
                )
                # Keep canonical submission.csv updated
                self.submission_generator.generate_submission_csv(
                    test_df=df_test,
                    preds=test_preds,
                    filename="submission.csv",
                )

                # Dynamic kernel generation with trained LightGBM booster and ensemble weights
                lgb_model_str = None
                ensemble_weights = None
                lsa_data = None
                if (
                    model_type in ["lightgbm", "lsa_lightgbm"]
                    and exp_id in self.trained_models
                    and len(self.trained_models[exp_id]) > 0
                ):
                    m = self.trained_models[exp_id][0]
                    target_lsa = None
                    if isinstance(m, tuple):
                        target_m = m[0]
                        target_lsa = m[1]
                    else:
                        target_m = m
                    if hasattr(target_m, "booster_model_string") and target_m.booster_model_string:
                        lgb_model_str = target_m.booster_model_string
                    if (
                        target_lsa is not None
                        and hasattr(target_lsa, "tfidf")
                        and hasattr(target_lsa, "svd")
                    ):
                        lsa_data = {
                            "vocab": list(target_lsa.tfidf.get_feature_names_out()),
                            "idf": target_lsa.tfidf.idf_,
                            "comp": target_lsa.svd.components_,
                        }
                elif model_type == "ensemble":
                    if self._last_blender is not None:
                        ensemble_weights = self._last_blender.get_weight_summary()
                    for _mid, fmods in self.trained_models.items():
                        for m in fmods:
                            target_m = m[0] if isinstance(m, tuple) else m
                            target_lsa = m[1] if isinstance(m, tuple) else None
                            if (
                                lgb_model_str is None
                                and hasattr(target_m, "booster_model_string")
                                and target_m.booster_model_string
                            ):
                                lgb_model_str = target_m.booster_model_string
                            if (
                                lsa_data is None
                                and target_lsa is not None
                                and hasattr(target_lsa, "tfidf")
                            ):
                                lsa_data = {
                                    "vocab": list(target_lsa.tfidf.get_feature_names_out()),
                                    "idf": target_lsa.tfidf.idf_,
                                    "comp": target_lsa.svd.components_,
                                }
                    # Fallback to disk checkpoints if member models were trained in a previous run
                    if lgb_model_str is None:
                        models_base = self.experiments_dir / "models"
                        for candidate_id in ["H005", "H002"]:
                            b_file = models_base / candidate_id / "booster_fold_0.txt"
                            if b_file.exists():
                                lgb_model_str = b_file.read_text(encoding="utf-8")
                                l_file = models_base / candidate_id / "lsa_data_fold_0.pkl"
                                if l_file.exists():
                                    import pickle

                                    with open(l_file, "rb") as f:
                                        lsa_data = pickle.load(f)
                                break

                temp_val = calibrator.temperature if calibrator is not None else 1.0
                self.submission_generator.generate_standalone_kaggle_kernel(
                    lgb_model_str=lgb_model_str,
                    ensemble_weights=ensemble_weights,
                    temperature=temp_val,
                    lsa_data=lsa_data,
                )

                # Iron Rule: Generate formal authorization request with previous best
                auth_req_path = self.auth_gate.request_authorization(
                    exp_id=exp_id,
                    model_name=model_name,
                    oof_log_loss=overall_loss,
                    best_log_loss=previous_best,
                    submission_csv_path=sub_file,
                    preview_df=pd.read_csv(sub_file),
                )
                print(
                    f"  [IRON RULE] Submission Authorization Request generated at: {auth_req_path}"
                )
        else:
            status = "REJECTED"
            rej_reason = (
                cv_check.message
                if not cv_check.passed
                else (anomaly_check.message if not anomaly_check.passed else sym_check.message)
            )
            print(f"  [GATE REJECTED] Model rejected by gates: {rej_reason}")

        # 5. Record Experiment in Registry
        exp_record = {
            "id": exp_id,
            "model_name": model_name,
            "model_type": model_type,
            "hypothesis": hyp_text,
            "status": status,
            "metrics": metrics,
            "symmetry_divergence": sym_div,
            "gates": {
                "leakage": bool(leak_check.passed),
                "cv": bool(cv_check.passed),
                "symmetry": bool(sym_check.passed),
                "diversity": bool(div_check.passed),
                "anomaly": bool(anomaly_check.passed),
            },
            "html_report": str(report_path),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.registry["experiments"].append(exp_record)
        self.registry["current_best"] = self.best_log_loss
        self._save_registry()
        self.update_leaderboard_md()

        return exp_record

    def _predict_test(
        self, model_type: str, params: dict[str, Any], df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> np.ndarray:
        """Generates test predictions with full TTA for submission."""
        extractor = FeatureExtractor()
        X_train = extractor.extract_features(df_train)
        X_test = extractor.extract_features(df_test)
        X_test_swap = extractor.extract_swapped_features(df_test)

        if model_type == "length_prior":
            model: Any = LengthPriorPredictor(beta=params.get("beta", 1.6))
            model.fit(X_train, df_train["target"].values)
            p_sym, _, _ = model.predict_proba_with_tta(X_test, X_test_swap)
            return p_sym

        elif model_type == "lightgbm":
            X_train_fit = X_train
            y_train_fit = df_train["target"].values
            # Synchronize with CV: use symmetric training augmentation if enabled
            if params.get("symmetric_training", True):
                X_train_swap = extractor.extract_swapped_features(df_train)
                swap_map = {0: 1, 1: 0, 2: 2}
                y_train_swap = np.array([swap_map[y] for y in y_train_fit])
                X_train_fit = pd.concat([X_train, X_train_swap], ignore_index=True)
                y_train_fit = np.concatenate([y_train_fit, y_train_swap])

            model = LightGBMPredictor(params=params)
            model.fit(X_train_fit, y_train_fit)
            p_sym, _, _ = model.predict_proba_with_tta(X_test, X_test_swap)
            return p_sym

        elif model_type == "tfidf_logistic":
            vectorizer = TextVectorizer(max_features=params.get("max_features", 5000))
            # Fit vocabulary only on training data
            train_texts = pd.concat(
                [df_train["prompt"], df_train["response_a"], df_train["response_b"]]
            ).tolist()
            vectorizer.fit(train_texts)
            X_diff_train = vectorizer.transform_differential(df_train, swap=False)
            X_diff_test = vectorizer.transform_differential(df_test, swap=False)
            X_diff_test_swap = vectorizer.transform_differential(df_test, swap=True)

            model = TfidfLogisticPredictor(C=params.get("C", 1.0))
            model.fit(X_diff_train, df_train["target"].values)
            p_sym, _, _ = model.predict_proba_with_tta(X_diff_test, X_diff_test_swap)
            return p_sym

        elif model_type == "lsa_lightgbm":
            extractor = FeatureExtractor()
            X_train = extractor.extract_features(df_train)
            X_test = extractor.extract_features(df_test)
            X_test_swap = extractor.extract_swapped_features(df_test)

            lsa_vec = DenseLSAVectorizer(
                n_components=params.get("n_components", 16),
                max_features=params.get("max_features", 5000),
            )
            train_texts = pd.concat(
                [df_train["prompt"], df_train["response_a"], df_train["response_b"]]
            ).tolist()
            lsa_vec.fit(train_texts)

            X_lsa_train = lsa_vec.extract_lsa_features(df_train, swap=False)
            X_train_fit = pd.concat(
                [X_train.reset_index(drop=True), X_lsa_train.reset_index(drop=True)], axis=1
            )
            y_train_fit = df_train["target"].values

            if params.get("symmetric_training", True):
                X_swap_train = extractor.extract_swapped_features(df_train)
                X_lsa_train_swap = lsa_vec.extract_lsa_features(df_train, swap=True)
                X_train_swap = pd.concat(
                    [X_swap_train.reset_index(drop=True), X_lsa_train_swap.reset_index(drop=True)],
                    axis=1,
                )
                swap_map = {0: 1, 1: 0, 2: 2}
                y_train_swap = np.array([swap_map[y] for y in y_train_fit])
                X_train_fit = pd.concat([X_train_fit, X_train_swap], ignore_index=True)
                y_train_fit = np.concatenate([y_train_fit, y_train_swap])

            X_lsa_test = lsa_vec.extract_lsa_features(df_test, swap=False)
            X_lsa_test_swap = lsa_vec.extract_lsa_features(df_test, swap=True)
            X_test_norm = pd.concat(
                [X_test.reset_index(drop=True), X_lsa_test.reset_index(drop=True)], axis=1
            )
            X_test_sw = pd.concat(
                [X_test_swap.reset_index(drop=True), X_lsa_test_swap.reset_index(drop=True)], axis=1
            )

            model = LightGBMPredictor(params=params)
            model.fit(X_train_fit, y_train_fit)
            p_sym, _, _ = model.predict_proba_with_tta(X_test_norm, X_test_sw)
            return p_sym

        elif model_type == "ensemble":
            # Dynamically look up parameters by name or type instead of fragile hardcoded indices
            hyp_by_name = {h["name"]: h for h in self.hypotheses_data.get("hypotheses", [])}
            hyp_by_type = {h["model_type"]: h for h in self.hypotheses_data.get("hypotheses", [])}

            pred_list = []
            for name in self.ensemble_names:
                matched_hyp = hyp_by_name.get(name)
                if matched_hyp is not None:
                    p = self._predict_test(
                        matched_hyp["model_type"],
                        matched_hyp.get("parameters", {}),
                        df_train,
                        df_test,
                    )
                elif "length" in name.lower():
                    p = self._predict_test(
                        "length_prior",
                        hyp_by_type.get("length_prior", {}).get("parameters", {}),
                        df_train,
                        df_test,
                    )
                elif "lightgbm" in name.lower() or "structural" in name.lower():
                    p = self._predict_test(
                        "lightgbm",
                        hyp_by_type.get("lightgbm", {}).get("parameters", {}),
                        df_train,
                        df_test,
                    )
                elif "tfidf" in name.lower():
                    p = self._predict_test(
                        "tfidf_logistic",
                        hyp_by_type.get("tfidf_logistic", {}).get("parameters", {}),
                        df_train,
                        df_test,
                    )
                else:
                    p = np.full((len(df_test), 3), 1.0 / 3.0)
                pred_list.append(p)

            blender = EnsembleBlender(names=self.ensemble_names)
            blender.fit_with_nested_cv(self.ensemble_oofs, df_train[CLASSES].values)
            return blender.blend(pred_list)

        else:
            return np.full((len(df_test), 3), 1.0 / 3.0)

    def _predict_test_bagged(
        self,
        exp_id: str,
        model_type: str,
        params: dict[str, Any],
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> np.ndarray:
        """Generates bagged test predictions across all K CV folds with TTA and calibration."""
        if model_type == "ensemble":
            print("  [BAGGING] Generating blended predictions from bagged member models.")
            hyp_by_name = {h["name"]: h for h in self.hypotheses_data.get("hypotheses", [])}
            hyp_by_type = {h["model_type"]: h for h in self.hypotheses_data.get("hypotheses", [])}
            pred_list = []
            for name in self.ensemble_names:
                matched_hyp = hyp_by_name.get(name)
                if matched_hyp is not None:
                    p = self._predict_test_bagged(
                        matched_hyp["id"],
                        matched_hyp["model_type"],
                        matched_hyp.get("parameters", {}),
                        df_train,
                        df_test,
                    )
                elif "length" in name.lower():
                    h = hyp_by_type.get("length_prior", {})
                    p = self._predict_test_bagged(
                        h.get("id", "H001"),
                        "length_prior",
                        h.get("parameters", {}),
                        df_train,
                        df_test,
                    )
                elif "lightgbm" in name.lower() or "structural" in name.lower():
                    h = hyp_by_type.get("lightgbm", {})
                    p = self._predict_test_bagged(
                        h.get("id", "H002"), "lightgbm", h.get("parameters", {}), df_train, df_test
                    )
                elif "tfidf" in name.lower():
                    h = hyp_by_type.get("tfidf_logistic", {})
                    p = self._predict_test_bagged(
                        h.get("id", "H003"),
                        "tfidf_logistic",
                        h.get("parameters", {}),
                        df_train,
                        df_test,
                    )
                else:
                    p = np.full((len(df_test), 3), 1.0 / 3.0)
                pred_list.append(p)

            blender = self._last_blender
            if blender is None or blender.weights is None or len(blender.weights) != len(pred_list):
                blender = EnsembleBlender(names=self.ensemble_names)
                blender.fit_with_nested_cv(self.ensemble_oofs, df_train[CLASSES].values)
            preds = blender.blend(pred_list)

        else:
            fold_models = self.trained_models.get(exp_id, [])
            if not fold_models:
                print(
                    f"  [BAGGING] No fold models cached for {exp_id}, falling back to full-fit prediction."
                )
                preds = self._predict_test(model_type, params, df_train, df_test)
            else:
                print(
                    f"  [BAGGING] Generating test predictions bagged across {len(fold_models)} CV fold models."
                )
                if model_type == "length_prior":
                    extractor = FeatureExtractor()
                    X_test = extractor.extract_features(df_test)
                    X_test_swap = extractor.extract_swapped_features(df_test)
                    fold_preds = [
                        m.predict_proba_with_tta(X_test, X_test_swap)[0] for m in fold_models
                    ]
                    preds = np.mean(fold_preds, axis=0)

                elif model_type == "lightgbm":
                    extractor = FeatureExtractor()
                    X_test = extractor.extract_features(df_test)
                    X_test_swap = extractor.extract_swapped_features(df_test)
                    fold_preds = [
                        m.predict_proba_with_tta(X_test, X_test_swap)[0] for m in fold_models
                    ]
                    preds = np.mean(fold_preds, axis=0)

                elif model_type == "lsa_lightgbm":
                    extractor = FeatureExtractor()
                    X_feats_test = extractor.extract_features(df_test)
                    X_feats_test_swap = extractor.extract_swapped_features(df_test)

                    fold_preds = []
                    for mod, lsa_vec in fold_models:
                        X_lsa_test = lsa_vec.extract_lsa_features(df_test, swap=False)
                        X_lsa_test_swap = lsa_vec.extract_lsa_features(df_test, swap=True)
                        X_test_norm = pd.concat(
                            [
                                X_feats_test.reset_index(drop=True),
                                X_lsa_test.reset_index(drop=True),
                            ],
                            axis=1,
                        )
                        X_test_swap = pd.concat(
                            [
                                X_feats_test_swap.reset_index(drop=True),
                                X_lsa_test_swap.reset_index(drop=True),
                            ],
                            axis=1,
                        )
                        p_sym, _, _ = mod.predict_proba_with_tta(X_test_norm, X_test_swap)
                        fold_preds.append(p_sym)
                    preds = np.mean(fold_preds, axis=0)

                elif model_type == "tfidf_logistic":
                    fold_preds = []
                    for m, vec in fold_models:
                        X_diff_test = vec.transform_differential(df_test, swap=False)
                        X_diff_test_swap = vec.transform_differential(df_test, swap=True)
                        p_sym, _, _ = m.predict_proba_with_tta(X_diff_test, X_diff_test_swap)
                        fold_preds.append(p_sym)
                    preds = np.mean(fold_preds, axis=0)

                else:
                    preds = self._predict_test(model_type, params, df_train, df_test)

        # Apply post-hoc temperature calibration if available
        if exp_id in self.temperature_calibrators:
            preds = self.temperature_calibrators[exp_id].calibrate(preds)

        return preds

    def run_all_hypotheses(self, sample_size: int | None = 10000, force_rerun: bool = False):
        """Runs through all hypotheses in the queue with crash resilience and resumability."""
        if force_rerun:
            self.best_log_loss = None
            self.ensemble_oofs = []
            self.ensemble_names = []
            if self.state_path.exists():
                try:
                    self.state_path.unlink()
                except Exception:
                    pass

        passed_ids = {
            exp.get("id")
            for exp in self.registry.get("experiments", [])
            if exp.get("status") == "PASSED"
        }

        print(f"Loading data (sample_size={sample_size})...")
        df_train = self.data_loader.load_train(sample_size=sample_size)
        df_test = self.data_loader.load_test()
        print(f"Data loaded. Training rows: {len(df_train)}, Test rows: {len(df_test)}")

        for hyp in self.hypotheses_data.get("hypotheses", []):
            hyp_id = hyp.get("id")
            if not force_rerun and hyp_id in passed_ids:
                print(
                    f"\n[SKIP] Hypothesis {hyp_id} ({hyp.get('name')}) already PASSED in registry. Set force_rerun=True to re-execute."
                )
                continue

            try:
                self.run_experiment(hyp, df_train, df_test)
            except Exception as e:
                print(f"\n[ERROR] Experiment {hyp.get('id')} crashed: {e}")
                self.registry["experiments"].append(
                    {
                        "id": hyp.get("id"),
                        "model_name": hyp.get("name"),
                        "model_type": hyp.get("model_type"),
                        "hypothesis": hyp.get("hypothesis"),
                        "status": "ERROR",
                        "error": str(e),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                self._save_registry()
                self.update_leaderboard_md()

        # Tier 3: Adaptive Hypothesis Generation
        adaptive_gen = AdaptiveHypothesisGenerator(
            registry_path=str(self.registry_path),
            hypotheses_path=str(self.hypotheses_path),
        )
        proposed = adaptive_gen.propose_next_hypothesis()
        registered = adaptive_gen.register_proposed_hypothesis(proposed)
        if registered:
            print("\n[ADAPTIVE RESEARCH LOOP] Proposing Next Experiment:")
            print(f"  ID: {proposed['id']} ({proposed['name']})")
            print(f"  Hypothesis: {proposed['hypothesis']}")
            print(f"  Rationale: {proposed.get('rationale')}")

        print("\nResearch cycle complete! All diagnostic dashboards and leaderboard updated.")
