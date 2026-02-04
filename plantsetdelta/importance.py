from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

from .visualization import save_top_features_plot

__all__ = ["export_top_features"]


def export_top_features(
    model_path: Path,
    feature_csv: Path,
    target_col: str,
    out_dir: Path,
    top_n: int = 10,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feature_csv)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in: {feature_csv}")

    X = df.drop(columns=[target_col, "gene_id"], errors="ignore")
    y = df[target_col]
    feature_names = list(X.columns)

    model_dir = Path(model_path).parent / "best_models"
    if not model_dir.exists():
        raise FileNotFoundError(f"{model_dir} does not exist. Unable to read per-seed models.")

    model_files = sorted(model_dir.glob("best_model_*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No best_model_*.pkl found in: {model_dir}")

    feature_stats: Dict[str, Dict[str, float]] = {}

    for model_file in model_files:
        model_pipeline = joblib.load(model_file)

        feature_imp_dict: Dict[str, float]

        if hasattr(model_pipeline, "steps") and len(model_pipeline.steps) > 0:
            estimator = model_pipeline.steps[-1][1]
        else:
            estimator = model_pipeline

        if hasattr(estimator, "feature_importances_"):
            importance = np.asarray(estimator.feature_importances_, dtype=float)
            if importance.shape[0] != len(feature_names):
                perm = permutation_importance(
                    model_pipeline,
                    X,
                    y,
                    n_repeats=3,
                    random_state=1,
                    n_jobs=1,
                )
                feature_imp_dict = dict(zip(feature_names, perm.importances_mean))
            else:
                feature_imp_dict = dict(zip(feature_names, importance))
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=float)
            if coef.ndim == 2:
                coef = coef[0]
            if coef.shape[0] != len(feature_names):
                perm = permutation_importance(
                    model_pipeline,
                    X,
                    y,
                    n_repeats=3,
                    random_state=1,
                    n_jobs=1,
                )
                feature_imp_dict = dict(zip(feature_names, perm.importances_mean))
            else:
                feature_imp_dict = dict(zip(feature_names, np.abs(coef)))
        else:
            perm = permutation_importance(
                model_pipeline,
                X,
                y,
                n_repeats=3,
                random_state=1,
                n_jobs=1,
            )
            feature_imp_dict = dict(zip(feature_names, perm.importances_mean))

        values = np.array(list(feature_imp_dict.values()), dtype=float)
        denom = np.ptp(values)
        if denom > 0:
            values_norm = (values - values.min()) / denom
        else:
            values_norm = np.zeros_like(values)

        norm_feature_imp_dict = dict(zip(feature_imp_dict.keys(), values_norm))

        top_feats = sorted(norm_feature_imp_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for feat, imp in top_feats:
            if feat not in feature_stats:
                feature_stats[feat] = {"count": 0.0, "total_imp": 0.0}
            feature_stats[feat]["count"] += 1.0
            feature_stats[feat]["total_imp"] += float(imp)

    summary_df = pd.DataFrame(
        [
            {
                "Feature": feat,
                "Count": int(stat["count"]),
                "Avg_Importance": stat["total_imp"] / stat["count"] if stat["count"] > 0 else 0.0,
            }
            for feat, stat in feature_stats.items()
        ]
    )

    summary_df = summary_df.sort_values(
        by=["Count", "Avg_Importance"],
        ascending=[False, False],
    ).reset_index(drop=True)

    top_df = (
        summary_df.head(top_n)[["Feature", "Count", "Avg_Importance"]]
        .sort_values(by="Avg_Importance", ascending=False)
        .reset_index(drop=True)
    )

    top_csv = out_dir / f"top{top_n}_features.csv"
    top_pdf = out_dir / f"top{top_n}_lollipop.pdf"

    top_df.to_csv(top_csv, index=False)
    save_top_features_plot(top_df, top_pdf)

    return top_csv, top_pdf