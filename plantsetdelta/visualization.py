from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

__all__ = ["save_top_features_plot"]


def save_top_features_plot(summary_df: pd.DataFrame, out_file: Path):
    """
    Save a lollipop plot of top N features by average importance.

    Args:
        summary_df (pd.DataFrame): A DataFrame containing 'Feature' and 'Avg_Importance' columns.
        out_file (Path): Output path for the PDF file.
    """

    # Plot top 10 features (assumes the input DataFrame is already sorted)
    top_n = summary_df.head(10)

    plt.figure(figsize=(4, 3))
    plt.hlines(y=top_n['Feature'], xmin=0, xmax=top_n['Avg_Importance'], color='grey', linewidth=1.5)
    plt.plot(top_n['Avg_Importance'], top_n['Feature'], 'o', markersize=12, color='#5a576c')

    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()
    plt.gca().set_facecolor('white')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('axes', -0.05))
    ax.spines['left'].set_position(('axes', -0.05))
    ax.spines['left'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')

    plt.tick_params(axis='both', colors='black', labelsize=8, length=5, width=1)
    plt.grid(False)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, format='pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()