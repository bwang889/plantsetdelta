from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
)

from pycaret.classification import (
    setup,
    compare_models,
    save_model,
    add_metric,
    pull,
    predict_model,
    get_config,
)

from .visualization import save_model_metric_plots
from .feature_selection import CorrelationTopKSelector

__all__ = ["train_best_model"]


_MODEL_ABBREV = {
    "CatBoost Classifier": "catboost",
    "Extreme Gradient Boosting": "xgboost",
    "Light Gradient Boosting Machine": "lightgbm",
    "Random Forest Classifier": "rf",
    "Extra Trees Classifier": "et",
    "Gradient Boosting Classifier": "gbc",
    "Quadratic Discriminant Analysis": "qda",
    "Logistic Regression": "lr",
    "K Neighbors Classifier": "knn",
    "Ada Boost Classifier": "ada",
    "Naive Bayes": "nb",
    "Ridge Classifier": "ridge",
    "Linear Discriminant Analysis": "lda",
    "SVM - Linear Kernel": "svm",
    "Decision Tree Classifier": "dt",
    "Dummy Classifier": "dummy",
}


def _score_column(pred_df: pd.DataFrame, class_label) -> Optional[str]:
    wanted = str(class_label)
    normalized = wanted[:-2] if wanted.endswith(".0") else wanted
    for prefix in ("prediction_score_", "Score_"):
        for col in pred_df.columns:
            if str(col).startswith(prefix):
                suffix = str(col)[len(prefix):]
                suffix_norm = suffix[:-2] if suffix.endswith(".0") else suffix
                if suffix == wanted or suffix_norm == normalized:
                    return col
    return None


def _get_probability_matrix(pred_df: pd.DataFrame, classes: list) -> Optional[np.ndarray]:
    columns = [_score_column(pred_df, label) for label in classes]
    if any(col is None for col in columns):
        return None
    try:
        scores = pred_df[columns].astype(float).to_numpy()
    except (TypeError, ValueError):
        return None
    if scores.ndim != 2 or not np.isfinite(scores).all():
        return None
    return scores


def _compute_holdout_metrics(pred: pd.DataFrame, target_col: str):
    y_true = pred[target_col].to_numpy()
    label_col = "prediction_label" if "prediction_label" in pred.columns else "Label"
    if label_col not in pred.columns:
        raise RuntimeError("predict_model output is missing prediction_label/Label.")
    y_pred = pred[label_col].to_numpy()

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    try:
        classes = np.unique(y_true).tolist()
    except TypeError:
        classes = sorted(pd.unique(y_true).tolist(), key=lambda value: str(value))
    auc = float("nan")
    if len(classes) >= 2:
        probabilities = _get_probability_matrix(pred, classes)
        if probabilities is not None:
            try:
                if len(classes) == 2:
                    auc = float(roc_auc_score(y_true, probabilities[:, 1]))
                else:
                    auc = float(roc_auc_score(
                        y_true, probabilities, labels=classes,
                        multi_class="ovr", average="macro"
                    ))
            except ValueError:
                auc = float("nan")

    return acc, auc, rec, prec, f1, kappa, mcc, bacc


def _prepare_run_output_dirs(models_dir: Path, results_cv_dir: Path, results_test_dir: Path):
    # Keep previous outputs intact. training_manifest.json explicitly lists only
    # models/training partitions produced by the current run, so stale files are
    # never included in the current feature-importance summary.
    for folder in (models_dir, results_cv_dir, results_test_dir):
        folder.mkdir(parents=True, exist_ok=True)


def train_best_model(
    feature_csv: Path,
    target_col: str,
    out_dir: Path,
    n_runs: int = 10,
    include_models: Optional[Union[List[str], str]] = None,
    n_jobs: int = 4,
    group_column: Optional[str] = None,
) -> Path:
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "best_models"
    models_dir.mkdir(exist_ok=True)
    results_cv_dir = out_dir / "model_results_cv"
    results_test_dir = out_dir / "model_results_test"
    _prepare_run_output_dirs(models_dir, results_cv_dir, results_test_dir)

    df = pd.read_csv(feature_csv)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the input file.")
    if df.empty:
        raise ValueError("Training data is empty.")
    if df[target_col].isna().any():
        raise ValueError("Target column contains missing labels.")
    if df.columns.duplicated().any():
        raise ValueError("Training data contains duplicate column names.")
    if df.drop(columns=[target_col]).shape[1] == 0:
        raise ValueError("No feature columns found.")
    if df[target_col].nunique() < 2:
        raise ValueError("At least two target classes are required.")
    if df[target_col].value_counts().min() < 2:
        raise ValueError("Each target class must contain at least two samples.")
    if group_column is not None:
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found in training data.")
        if df[group_column].isna().any():
            raise ValueError(f"Group column '{group_column}' contains missing values.")
        if df[group_column].nunique() < 3:
            raise ValueError("Group-aware holdout and CV require at least three distinct groups.")

    run_id = uuid.uuid4().hex[:10]
    model_scores = []
    all_cv_results = []
    all_test_results = []
    manifest_models = []

    for seed in range(n_runs):
        print(f"Training round {seed + 1}/{n_runs} (seed={seed})...")
        setup_kwargs = {
            "target": target_col,
            "session_id": seed,
            "html": False,
            "verbose": False,
            "n_jobs": n_jobs,
            "custom_pipeline": [("label_correlation_selector", CorrelationTopKSelector())],
        }
        if group_column is None:
            cv_folds = min(10, int(df[target_col].value_counts().min()))
            setup_kwargs.update({
                "data": df,
                "fold": cv_folds,
                "fold_shuffle": True,
            })
        else:
            # Keep related genes in one partition for both the outer holdout
            # and inner CV. Try multiple deterministic splits to retain all classes.
            splitter = GroupShuffleSplit(n_splits=100, test_size=0.3, random_state=seed)
            split_found = None
            for train_idx, test_idx in splitter.split(df, df[target_col], groups=df[group_column]):
                train_part = df.iloc[train_idx]
                test_part = df.iloc[test_idx]
                if (train_part[target_col].nunique() == df[target_col].nunique()
                    and test_part[target_col].nunique() == df[target_col].nunique()
                    and train_part[group_column].nunique() >= 2):
                    split_found = (train_idx, test_idx)
                    break
            if split_found is None:
                raise ValueError(
                    f"Could not create a group-disjoint 70/30 holdout containing all classes "
                    f"for group column '{group_column}'."
                )
            train_idx, test_idx = split_found
            train_part = df.iloc[train_idx].copy()
            test_part = df.iloc[test_idx].copy()
            cv_folds = min(5, int(train_part[group_column].nunique()))
            setup_kwargs.update({
                "data": train_part,
                "test_data": test_part,
                "ignore_features": [group_column],
                "fold_strategy": "groupkfold",
                "fold_groups": group_column,
                "fold": cv_folds,
            })
        setup(**setup_kwargs)

        # The selector is part of PyCaret's preprocessing pipeline. It is cloned
        # and fitted within each CV training fold, never on that fold's validation data.
        add_metric(
            id="balanced_accuracy",
            name="Balanced Accuracy",
            score_func=balanced_accuracy_score,
            greater_is_better=True,
        )

        include_arg = None if not include_models or include_models == "all" else include_models
        models_list = compare_models(
            sort="balanced_accuracy",
            include=include_arg,
            n_select=999,
        )
        if not isinstance(models_list, list):
            models_list = [models_list]
        if not models_list:
            raise RuntimeError("PyCaret did not return any trainable models.")

        cv_results = pull()
        if not isinstance(cv_results, pd.DataFrame) or cv_results.empty:
            raise RuntimeError("PyCaret pull() returned empty CV results.")
        if "Balanced Accuracy" not in cv_results.columns:
            raise RuntimeError("CV results do not contain Balanced Accuracy.")

        cv_results = cv_results.copy()
        cv_results["batch"] = seed
        cv_results.to_csv(results_cv_dir / f"model_comparison_cv_batch_{seed}.csv", index=False)
        all_cv_results.append(cv_results)

        best = models_list[0]
        best_score = float(cv_results["Balanced Accuracy"].iloc[0])
        if not np.isfinite(best_score):
            raise RuntimeError(f"Invalid CV Balanced Accuracy for seed {seed}.")
        model_scores.append((best, best_score))

        model_name = f"best_model_{run_id}_{seed + 1}"
        save_model(best, str(models_dir / model_name))
        model_file = models_dir / f"{model_name}.pkl"

        # Save only the training partition for later feature-importance estimation.
        # PyCaret's "train" slot is the original, untransformed training
        # partition; X_train is transformed and must not be used here.
        training_data = get_config("train")
        if not isinstance(training_data, pd.DataFrame) or target_col not in training_data.columns:
            raise RuntimeError(
                "PyCaret did not expose the original training partition; cannot "
                "save a leakage-free feature-importance dataset."
            )
        training_data = training_data.copy()
        training_path = results_cv_dir / f"training_data_{run_id}_seed_{seed}.csv"
        training_data.to_csv(training_path, index=False)
        manifest_models.append({
            "model": str(Path("best_models") / model_file.name),
            "training_data": str(Path("model_results_cv") / training_path.name),
            "seed": seed,
            "cv_balanced_accuracy": best_score,
        })

        test_rows = []
        model_names = cv_results["Model"].tolist() if "Model" in cv_results.columns else []
        for index, model in enumerate(models_list):
            name = model_names[index] if index < len(model_names) else f"model_{index + 1}"
            try:
                try:
                    prediction = predict_model(model, raw_score=True)
                except Exception:
                    prediction = predict_model(model)
                metrics = _compute_holdout_metrics(prediction, target_col)
            except Exception as exc:
                print(f"[plantsetdelta] Holdout metrics unavailable for {name}, seed {seed}: {exc}")
                metrics = (float("nan"),) * 8
            acc, auc, rec, prec, f1, kappa, mcc, bacc = metrics
            test_rows.append({
                "Model": name, "Accuracy": acc, "AUC": auc, "Recall": rec,
                "Prec.": prec, "F1": f1, "Kappa": kappa, "MCC": mcc,
                "Balanced Accuracy": bacc, "batch": seed,
            })

        test_df = pd.DataFrame(test_rows)
        test_df.to_csv(results_test_dir / f"model_comparison_test_batch_{seed}.csv", index=False)
        all_test_results.append(test_df)

    all_cv_df = pd.concat(all_cv_results, ignore_index=True)
    if "Model" in all_cv_df.columns:
        all_cv_df["Model"] = all_cv_df["Model"].map(lambda value: _MODEL_ABBREV.get(value, value))
    all_cv_df.to_csv(out_dir / "all_model_comparison_results_cv.csv", index=False)

    all_test_df = pd.concat(all_test_results, ignore_index=True)
    if "Model" in all_test_df.columns:
        all_test_df["Model"] = all_test_df["Model"].map(lambda value: _MODEL_ABBREV.get(value, value))
    all_test_df.to_csv(out_dir / "all_model_comparison_results_test.csv", index=False)

    for frame, metric, filename, title in (
        (all_cv_df, "Balanced Accuracy", "balanced_acc_cv_plot.pdf", "Model Comparison (CV): Balanced Accuracy"),
        (all_cv_df, "AUC", "auc_cv_plot.pdf", "Model Comparison (CV): AUC"),
        (all_test_df, "Balanced Accuracy", "balanced_acc_test_plot.pdf", "Model Comparison (Test): Balanced Accuracy"),
        (all_test_df, "AUC", "auc_test_plot.pdf", "Model Comparison (Test): AUC"),
    ):
        if metric in frame.columns and frame[metric].notna().any():
            save_model_metric_plots(
                all_results_df=frame, out_dir=out_dir, metric_col=metric,
                out_pdf=filename, title=title,
            )

    best_final, best_score = max(model_scores, key=lambda item: item[1])
    print(f"Selecting final best model by Balanced Accuracy (CV): {best_score:.4f}")
    model_path = out_dir / "best_model"
    save_model(best_final, str(model_path))

    manifest = {
        "run_id": run_id,
        "n_runs": n_runs,
        "target": target_col,
        "group_column": group_column,
        "feature_selection": "binary: absolute Pearson correlation; multiclass: ANOVA F-score; top max(1, floor(n_training_samples * 0.2)); fitted within each CV fold",
        "models": manifest_models,
    }
    (out_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return model_path.with_suffix(".pkl")
