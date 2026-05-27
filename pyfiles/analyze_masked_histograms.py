"""Plot histograms and summarize skewness/kurtosis for masked datasets."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = {
    "mixed_brown_masked": Path("data/mixed_data_1st/mixed_brown_masked.csv"),
    "mixed_sabr_masked": Path("data/mixed_data_1st/mixed_sabr_masked.csv"),
}
OUTPUT_ROOT = Path("plots")
STATS_OUTPUT = Path("data/mixed_data_1st/masked_histogram_stats.csv")
MASK_IDS = range(1, 6)
SUFFIXES = ("sp500", "DGS10")
HISTOGRAM_BINS = 50


def mask_columns(suffix: str) -> list[str]:
    """Return mask columns for the given suffix in mask order."""
    return [f"mask{mask_id}_{suffix}" for mask_id in MASK_IDS]


def describe_columns(
    df: pd.DataFrame,
    dataset_name: str,
    suffix: str,
) -> list[dict[str, float | int | str]]:
    """Calculate skewness and kurtosis for the requested columns."""
    stats = []
    for column in mask_columns(suffix):
        series = df[column].dropna()
        stats.append(
            {
                "dataset": dataset_name,
                "series": suffix,
                "mask": column.split("_", maxsplit=1)[0],
                "column": column,
                "count": int(series.count()),
                "mean": series.mean(),
                "std": series.std(),
                "skewness": series.skew(),
                "kurtosis": series.kurt(),
            }
        )
    return stats


def plot_histograms(
    df: pd.DataFrame,
    dataset_name: str,
    suffix: str,
    output_path: Path,
) -> None:
    """Plot mask1 to mask5 histograms for one suffix."""
    columns = mask_columns(suffix)
    fig, axes = plt.subplots(
        nrows=len(columns),
        ncols=1,
        figsize=(12, 12),
        sharex=True,
    )
    fig.suptitle(f"{dataset_name} {suffix}: histograms")

    for ax, column in zip(axes, columns):
        series = df[column].dropna()
        ax.hist(series, bins=HISTOGRAM_BINS, alpha=0.8, edgecolor="black")
        ax.axvline(series.mean(), color="tab:red", linewidth=1.2, label="mean")
        ax.set_title(column)
        ax.set_ylabel("frequency")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run histogram plotting and skewness/kurtosis summary."""
    all_stats = []

    for dataset_name, data_path in DATASETS.items():
        df = pd.read_csv(data_path)
        output_dir = OUTPUT_ROOT / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for suffix in SUFFIXES:
            all_stats.extend(describe_columns(df, dataset_name, suffix))
            plot_histograms(
                df=df,
                dataset_name=dataset_name,
                suffix=suffix,
                output_path=output_dir / f"{suffix}_histograms_combined.png",
            )

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(STATS_OUTPUT, index=False)

    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(stats_df.to_string(index=False))
    print(f"\nSaved statistics: {STATS_OUTPUT}")


if __name__ == "__main__":
    main()
