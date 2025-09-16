from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from .visualization import save_top_features_plot

__all__ = ["export_top_features"]

def export_top_features(model_path: Path, feature_csv: Path, target_col: str, out_dir: Path, top_n: int = 10):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(feature_csv)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns

    model_dir = model_path.parent / "best_models"
    if not model_dir.exists():
        raise FileNotFoundError(f"{model_dir} does not exist. Unable to read multiple models.")

    model_files = sorted(model_dir.glob("best_model_*.pkl"))
    feature_stats = {}

    for model_file in model_files:
        model_pipeline = joblib.load(model_file)
        best = model_pipeline.steps[-1][1]

        if hasattr(best, 'feature_importances_'):
            importance = best.feature_importances_
            feature_imp_dict = dict(zip(feature_names, importance))
        elif hasattr(best, 'coef_'):
            coefficients = best.coef_[0]
            feature_imp_dict = dict(zip(feature_names, np.abs(coefficients)))
        elif isinstance(best, GaussianNB):
            top_corr = X.corrwith(y).abs().nlargest(100).index
            X_reduced = X[top_corr]
            gnb = GaussianNB()
            gnb.fit(X_reduced, y)
            perm_result = permutation_importance(gnb, X_reduced, y, n_repeats=3, random_state=1)
            feature_imp_dict = dict(zip(top_corr, perm_result.importances_mean))
        else:
            perm_result = permutation_importance(best, X, y, n_repeats=3, random_state=1)
            feature_imp_dict = dict(zip(feature_names, perm_result.importances_mean))

        values = np.array(list(feature_imp_dict.values()))
        values_norm = (values - values.min()) / (values.ptp() if np.ptp(values) > 0 else 1)
        norm_feature_imp_dict = dict(zip(feature_imp_dict.keys(), values_norm))

        top_feats = sorted(norm_feature_imp_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for feat, imp in top_feats:
            if feat not in feature_stats:
                feature_stats[feat] = {'count': 0, 'total_imp': 0.0}
            feature_stats[feat]['count'] += 1
            feature_stats[feat]['total_imp'] += imp

    summary_df = pd.DataFrame([
        {
            'Feature': feat,
            'Count': stat['count'],
            'Avg_Importance': stat['total_imp'] / stat['count']
        }
        for feat, stat in feature_stats.items()
    ])
    summary_df = summary_df.sort_values(
        by=['Count', 'Avg_Importance'],
        ascending=[False, False]
    ).reset_index(drop=True)

    top_df = summary_df.head(top_n)[['Feature', 'Count', 'Avg_Importance']].sort_values(by='Avg_Importance', ascending=False)

    top_csv = out_dir / f"top{top_n}_features.csv"
    top_pdf = out_dir / f"top{top_n}_lollipop.pdf"
    top_df.to_csv(top_csv, index=False)
    save_top_features_plot(top_df, top_pdf)

    return top_csv, top_pdf