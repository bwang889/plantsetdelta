from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd

from pycaret.classification import (
    setup,
    compare_models,
    save_model,
    add_metric,
    pull,
    predict_model,
)
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

from .visualization import save_model_metric_plots

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


def _safe_binary_auc(y_true, y_score) -> float:
    try:
        uniq = pd.unique(pd.Series(y_true))
        if len(uniq) != 2:
            return 0.0
        classes = sorted(list(uniq))
        y_bin = pd.Series(y_true).map({classes[0]: 0, classes[1]: 1}).values
        return float(roc_auc_score(y_bin, y_score))
    except Exception:
        return 0.0


def _get_prob_y1(pred_df: pd.DataFrame) -> np.ndarray:
    """
    Return P(y=1) for binary AUC from PyCaret predict_model(..., raw_score=True).

    Priority:
      1) prediction_score_1 (your current environment)
      2) Score_1 (other variants)
      3) prediction_score / Score fallback with inference when necessary
    """
    if "prediction_score_1" in pred_df.columns:
        return pred_df["prediction_score_1"].astype(float).values
    if "prediction_score_1.0" in pred_df.columns:
        return pred_df["prediction_score_1.0"].astype(float).values

    if "Score_1" in pred_df.columns:
        return pred_df["Score_1"].astype(float).values
    if "Score_1.0" in pred_df.columns:
        return pred_df["Score_1.0"].astype(float).values

    score_col = None
    for c in ("prediction_score", "Score"):
        if c in pred_df.columns:
            score_col = c
            break

    if score_col is None:
        raise RuntimeError(
            "No usable probability columns found. Need one of "
            "prediction_score_1 / Score_1 / prediction_score / Score. "
            f"Available tail cols: {list(pred_df.columns)[-30:]}"
        )

    s = pred_df[score_col].astype(float)

    if float(s.min()) < 0.5:
        return s.values

    if "prediction_label" in pred_df.columns:
        yhat = pred_df["prediction_label"].astype(int).values
    elif "Label" in pred_df.columns:
        yhat = pred_df["Label"].astype(int).values
    else:
        raise RuntimeError(
            f"Found '{score_col}' but no prediction_label/Label to infer P(y=1). "
            f"Available tail cols: {list(pred_df.columns)[-30:]}"
        )

    return np.where(yhat == 1, s.values, 1.0 - s.values)


def _compute_holdout_metrics(
    pred: pd.DataFrame,
    target_col: str,
) -> Tuple[float, float, float, float, float, float, float, float]:
    y_true = pred[target_col].values
    if "prediction_label" in pred.columns:
        y_pred = pred["prediction_label"].values
    elif "Label" in pred.columns:
        y_pred = pred["Label"].values
    else:
        raise RuntimeError("predict_model output missing prediction_label/Label column.")

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred, average="macro"))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    auc = 0.0
    if len(pd.unique(y_true)) == 2:
        try:
            y_score = _get_prob_y1(pred)
            auc = _safe_binary_auc(y_true, y_score)
        except Exception:
            auc = 0.0

    return acc, auc, rec, prec, f1, kappa, mcc, bacc


def train_best_model(
    feature_csv: Path,
    target_col: str,
    out_dir: Path,
    n_runs: int = 10,
    include_models: Optional[Union[List[str], str]] = None,
    n_jobs: int = 4,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = out_dir / "best_models"
    models_dir.mkdir(exist_ok=True)

    results_cv_dir = out_dir / "model_results_cv"
    results_cv_dir.mkdir(exist_ok=True)

    results_test_dir = out_dir / "model_results_test"
    results_test_dir.mkdir(exist_ok=True)

    df = pd.read_csv(feature_csv)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the input file.")

    model_scores = []
    all_cv_results = []
    all_test_results = []

    for seed in range(n_runs):
        print(f"Training round {seed + 1}/{n_runs} (seed={seed})...")

        setup(
            data=df,
            target=target_col,
            session_id=seed,
            html=False,
            verbose=False,
            n_jobs=n_jobs,
        )

        add_metric(
            id="balanced_accuracy",
            name="Balanced Accuracy",
            score_func=balanced_accuracy_score,
            greater_is_better=True,
        )

        include_arg = None
        if include_models and include_models != "all":
            include_arg = include_models

        models_list = compare_models(
            sort="balanced_accuracy",
            include=include_arg,
            n_select=999,
        )
        if not isinstance(models_list, list):
            models_list = [models_list]

        cv_results = pull()
        if not isinstance(cv_results, pd.DataFrame) or cv_results.empty:
            raise RuntimeError("PyCaret pull() returned empty results.")

        cv_results = cv_results.copy()
        cv_results["batch"] = seed
        cv_path = results_cv_dir / f"model_comparison_cv_batch_{seed}.csv"
        cv_results.to_csv(cv_path, index=False)
        all_cv_results.append(cv_results)

        best = models_list[0]
        best_score = 0.0
        if "Balanced Accuracy" in cv_results.columns:
            best_score = float(cv_results["Balanced Accuracy"].iloc[0])
        model_scores.append((best, best_score))

        save_model(best, str(models_dir / f"best_model_{seed+1}"))

        model_names = cv_results["Model"].tolist() if "Model" in cv_results.columns else [f"model_{i}" for i in range(len(models_list))]
        n = min(len(models_list), len(model_names))
        models_list = models_list[:n]
        model_names = model_names[:n]

        test_rows = []
        for m, mname in zip(models_list, model_names):
            pred = predict_model(m, raw_score=True)

            acc, auc, rec, prec, f1, kappa, mcc, bacc = _compute_holdout_metrics(
                pred=pred,
                target_col=target_col,
            )

            test_rows.append(
                {
                    "Model": mname,
                    "Accuracy": acc,
                    "AUC": auc,
                    "Recall": rec,
                    "Prec.": prec,
                    "F1": f1,
                    "Kappa": kappa,
                    "MCC": mcc,
                    "Balanced Accuracy": bacc,
                    "batch": seed,
                }
            )

        test_df = pd.DataFrame(test_rows)
        test_path = results_test_dir / f"model_comparison_test_batch_{seed}.csv"
        test_df.to_csv(test_path, index=False)
        all_test_results.append(test_df)

    all_cv_df = pd.concat(all_cv_results, ignore_index=True) if all_cv_results else pd.DataFrame()
    if not all_cv_df.empty:
        if "Model" in all_cv_df.columns:
            all_cv_df["Model"] = all_cv_df["Model"].map(lambda x: _MODEL_ABBREV.get(x, x))
        (out_dir / "all_model_comparison_results_cv.csv").write_text("", encoding="utf-8")
        all_cv_df.to_csv(out_dir / "all_model_comparison_results_cv.csv", index=False)

    all_test_df = pd.concat(all_test_results, ignore_index=True) if all_test_results else pd.DataFrame()
    if not all_test_df.empty:
        if "Model" in all_test_df.columns:
            all_test_df["Model"] = all_test_df["Model"].map(lambda x: _MODEL_ABBREV.get(x, x))
        all_test_df.to_csv(out_dir / "all_model_comparison_results_test.csv", index=False)

    if not all_cv_df.empty:
        if "Balanced Accuracy" in all_cv_df.columns:
            save_model_metric_plots(
                all_results_df=all_cv_df,
                out_dir=out_dir,
                metric_col="Balanced Accuracy",
                out_pdf="balanced_acc_cv_plot.pdf",
                title="Model Comparison (CV): Balanced Accuracy",
            )
        if "AUC" in all_cv_df.columns:
            save_model_metric_plots(
                all_results_df=all_cv_df,
                out_dir=out_dir,
                metric_col="AUC",
                out_pdf="auc_cv_plot.pdf",
                title="Model Comparison (CV): AUC",
            )

    if not all_test_df.empty:
        if "Balanced Accuracy" in all_test_df.columns:
            save_model_metric_plots(
                all_results_df=all_test_df,
                out_dir=out_dir,
                metric_col="Balanced Accuracy",
                out_pdf="balanced_acc_test_plot.pdf",
                title="Model Comparison (Test): Balanced Accuracy",
            )
        if "AUC" in all_test_df.columns:
            save_model_metric_plots(
                all_results_df=all_test_df,
                out_dir=out_dir,
                metric_col="AUC",
                out_pdf="auc_test_plot.pdf",
                title="Model Comparison (Test): AUC",
            )

    best_final, best_score = max(model_scores, key=lambda x: x[1])
    print(f"Selecting final best model by Balanced Accuracy (CV): {best_score:.4f}")

    model_path = out_dir / "best_model"
    save_model(best_final, str(model_path))
    return model_path.with_suffix(".pkl")