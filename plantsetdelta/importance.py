from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

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
    if top_n < 1:
        raise ValueError("top_n must be >= 1.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feature_csv)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in: {feature_csv}")
    model_path = Path(model_path)
    manifest_path = model_path.parent / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    group_column = manifest.get("group_column")
    drop_columns = [target_col, "gene_id"]
    if group_column:
        drop_columns.append(group_column)
    X_all = df.drop(columns=drop_columns, errors="ignore")
    y_all = df[target_col]
    if X_all.empty or X_all.shape[1] == 0:
        raise ValueError("No feature columns found.")
    if y_all.isna().any():
        raise ValueError("Target column contains missing labels.")

    model_dir = model_path.parent / "best_models"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_entries = manifest.get("models", [])
        if not model_entries:
            raise FileNotFoundError("Training manifest contains no current-run models.")
        runs = [
            (
                model_path.parent / entry["model"],
                model_path.parent / entry["training_data"],
            )
            for entry in model_entries
        ]
    else:
        # Backward compatibility for outputs produced by older PlantSetDelta versions.
        if not model_dir.exists():
            raise FileNotFoundError(f"{model_dir} does not exist.")
        runs = [(path, None) for path in sorted(model_dir.glob("best_model_*.pkl"))]

    if not runs:
        raise FileNotFoundError("No per-seed models found.")

    feature_stats: Dict[str, Dict[str, float]] = {}
    for model_file, training_file in runs:
        if not model_file.exists():
            raise FileNotFoundError(f"Model listed in training manifest is missing: {model_file}")
        model_pipeline = joblib.load(model_file)

        # Use only the corresponding training partition for importance estimation;
        # the held-out validation/test rows are not used to rank features.
        if training_file is not None:
            if not training_file.exists():
                raise FileNotFoundError(f"Training partition listed in manifest is missing: {training_file}")
            train_df = pd.read_csv(training_file)
            if target_col not in train_df.columns:
                raise ValueError(f"{training_file} does not contain target column {target_col}.")
            X = train_df.drop(columns=drop_columns, errors="ignore")
            y = train_df[target_col]
        else:
            X, y = X_all, y_all

        missing = sorted(set(X_all.columns) - set(X.columns))
        extra = sorted(set(X.columns) - set(X_all.columns))
        if missing or extra:
            raise ValueError(
                f"Feature columns do not match for {model_file}; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        X = X.loc[:, X_all.columns]

        # Permutation importance provides a common input-feature scale across
        # tree, linear and non-linear estimators, including fitted selectors.
        perm = permutation_importance(
            model_pipeline,
            X,
            y,
            n_repeats=5,
            random_state=1,
            scoring="balanced_accuracy",
            n_jobs=1,
        )
        values = np.asarray(perm.importances_mean, dtype=float)
        if len(values) != len(X.columns):
            raise RuntimeError(f"Unexpected permutation-importance length for {model_file}.")
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to within-model percentile ranks so model-specific importance
        # scales are not mixed directly.
        ranks = pd.Series(values, index=X.columns).rank(method="average", pct=True)
        top_features = ranks.sort_values(ascending=False).head(top_n)
        for feature, score in top_features.items():
            stat = feature_stats.setdefault(feature, {"count": 0.0, "total_imp": 0.0})
            stat["count"] += 1.0
            stat["total_imp"] += float(score)

    summary_df = pd.DataFrame([
        {
            "Feature": feature,
            "Count": int(stat["count"]),
            "Avg_Importance": stat["total_imp"] / stat["count"],
        }
        for feature, stat in feature_stats.items()
    ])
    if summary_df.empty:
        raise RuntimeError("No feature importance values were generated.")
    summary_df = summary_df.sort_values(
        by=["Count", "Avg_Importance"],
        ascending=[False, False],
    ).reset_index(drop=True)

    top_df = summary_df.head(top_n)[["Feature", "Count", "Avg_Importance"]]
    top_df = top_df.sort_values(
        by=["Count", "Avg_Importance"], ascending=[False, False]
    ).reset_index(drop=True)

    top_csv = out_dir / f"top{top_n}_features.csv"
    top_pdf = out_dir / f"top{top_n}_lollipop.pdf"
    top_df.to_csv(top_csv, index=False)
    save_top_features_plot(top_df, top_pdf)
    return top_csv, top_pdf
