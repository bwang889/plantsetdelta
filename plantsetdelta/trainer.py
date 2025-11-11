from pathlib import Path
from typing import Optional
import pandas as pd
from pycaret.classification import setup, compare_models, save_model, add_metric, pull
from sklearn.metrics import balanced_accuracy_score

__all__ = ["train_best_model"]

def train_best_model(
    feature_csv: Path,
    target_col: str,
    out_dir: Path,
    seeds: int = 10,
    include_models: Optional[list[str]] = None
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "best_models"
    models_dir.mkdir(exist_ok=True)

    df = pd.read_csv(feature_csv)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the input file.")

    model_scores = []  # Store (model, score) tuples from each training run

    for seed in range(seeds):
        print(f"Training round {seed + 1}/{seeds}...")

        setup(
            df,
            target=target_col,
            session_id=seed,
            html=False,
            verbose=False
        )

        # Add custom evaluation metric
        add_metric(
            id='balanced_accuracy',
            name='Balanced Accuracy',
            score_func=balanced_accuracy_score,
            greater_is_better=True
        )

        best = compare_models(
            sort="balanced_accuracy",
            include=include_models if include_models != "all" else None
        )

        # Retrieve the balanced accuracy score from the leaderboard
        leaderboard = pull()
        model_name = type(best).__name__
        score_row = leaderboard[leaderboard['Model'] == model_name]
        if score_row.empty:
            score = 0
        else:
            score = score_row["Balanced Accuracy"].values[0]

        model_scores.append((best, score))
        save_model(best, str(models_dir / f"best_model_{seed+1}"))

    # Select the best model based on the highest balanced accuracy score
    best_final, _ = max(model_scores, key=lambda x: x[1])
    model_path = out_dir / "best_model"
    save_model(best_final, str(model_path))

    return model_path.with_suffix(".pkl")