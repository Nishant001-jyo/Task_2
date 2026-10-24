"""Shared helpers: paths and plotting."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def save_bar_chart(df, metric, title, filename, higher_is_better=True):
    """Bar chart comparing one metric across models."""
    ordered = df.sort_values(metric, ascending=not higher_is_better)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(ordered.index, ordered[metric], color="#4C78A8")
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.invert_yaxis()
    for i, v in enumerate(ordered[metric]):
        ax.text(v, i, f" {v:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
