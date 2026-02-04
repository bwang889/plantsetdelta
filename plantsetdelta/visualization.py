from __future__ import annotations

from pathlib import Path
import random
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "save_top_features_plot",
    "apply_model_plot_style",
    "save_model_metric_plots",
]


def save_top_features_plot(summary_df: pd.DataFrame, out_file: Path):
    top_n = summary_df.head(10)

    plt.figure(figsize=(4, 3))
    plt.hlines(
        y=top_n["Feature"],
        xmin=0,
        xmax=top_n["Avg_Importance"],
        color="grey",
        linewidth=1.5,
    )
    plt.plot(
        top_n["Avg_Importance"],
        top_n["Feature"],
        "o",
        markersize=12,
        color="#5a576c",
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.gca().set_facecolor("white")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.spines["left"].set_position(("axes", -0.05))
    ax.spines["left"].set_linewidth(1)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["bottom"].set_color("black")

    plt.tick_params(axis="both", colors="black", labelsize=8, length=5, width=1)
    plt.grid(False)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, format="pdf", bbox_inches="tight", transparent=True, dpi=400)
    plt.close()


def apply_model_plot_style():
    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["figure.titlesize"] = 10
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["axes.linewidth"] = 1.0


def _map_model_names(df: pd.DataFrame) -> pd.DataFrame:
    model_abbreviations = {
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
    out = df.copy()
    if "Model" in out.columns:
        out["Model"] = out["Model"].map(lambda x: model_abbreviations.get(x, x))
    return out


def _box_strip_plot(
    df: pd.DataFrame,
    metric: str,
    out_file: Path,
    title: str,
    ylabel: str,
    filter_bal_acc_auc: bool = False,
):
    apply_model_plot_style()

    d = df.copy()
    if metric not in d.columns:
        raise ValueError(f"Metric '{metric}' not found in columns: {list(d.columns)}")

    if filter_bal_acc_auc and ("Balanced Accuracy" in d.columns) and ("AUC" in d.columns):
        d = d[(d["Balanced Accuracy"] >= 0.5) & (d["AUC"] >= 0.6)]

    d = _map_model_names(d)

    order = (
        d.groupby("Model")[metric]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    data = [d.loc[d["Model"] == m, metric].dropna().values for m in order]

    fig, ax = plt.subplots(figsize=(4, 3.5))

    bp = ax.boxplot(
        data,
        labels=order,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        whis=1.5,
    )
    for box in bp["boxes"]:
        box.set(facecolor="white", edgecolor="black", linewidth=1)
    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=1)
    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1)
    for median in bp["medians"]:
        median.set(color="black", linewidth=1)

    rng = random.Random(1)
    for i, m in enumerate(order, start=1):
        vals = d.loc[d["Model"] == m, metric].dropna().values
        if len(vals) == 0:
            continue
        xs = [i + rng.uniform(-0.15, 0.15) for _ in range(len(vals))]
        ax.scatter(xs, vals, color="#5a576c", alpha=0.5, s=16, linewidths=0)

    ax.set_ylabel(ylabel, color="black")
    ax.set_xlabel("Model", color="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.spines["left"].set_position(("axes", -0.05))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", color="black")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.set_title(title, color="black")

    plt.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight", transparent=True, dpi=1200)
    plt.close()


def save_model_metric_plots(
    all_results_df: pd.DataFrame,
    out_dir: Path,
    metric_col: str,
    out_pdf: str,
    title: str,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if metric_col == "Balanced Accuracy":
        ylabel = "Balanced Accuracy"
    elif metric_col == "Accuracy":
        ylabel = "Accuracy"
    else:
        ylabel = metric_col

    _box_strip_plot(
        df=all_results_df,
        metric=metric_col,
        out_file=out_dir / out_pdf,
        title=title,
        ylabel=ylabel,
        filter_bal_acc_auc=False,
    )