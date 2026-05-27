"""Plot moving averages and rolling volatility for masked Brownian data."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("mixed_data_1st/mixed_brown_masked.csv")
OUTPUT_DIR = Path("plots/mixed_brown_masked")
MASK_IDS = range(1, 6)
SUFFIXES = ("sp500", "DGS10")
MOVING_AVERAGE_WINDOW = 50
VOLATILITY_WINDOW = 100


def load_data(path: Path) -> pd.DataFrame:
    """Load the masked Brownian dataset."""
    return pd.read_csv(path)


def rolling_volatility(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling volatility as a rolling standard deviation."""
    return series.rolling(window=window).std()


def plot_single_series(
    series: pd.Series,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot one time series and save it as a PNG file."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_combined_series(
    series_by_name: dict[str, pd.Series],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot five masked series in one stacked figure and save it."""
    fig, axes = plt.subplots(
        nrows=len(series_by_name),
        ncols=1,
        figsize=(12, 12),
        sharex=True,
    )
    fig.suptitle(title)

    for ax, (name, series) in zip(axes, series_by_name.items()):
        ax.plot(series.index, series, linewidth=1.0)
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_for_suffix(
    df: pd.DataFrame,
    suffix: str,
    metric_name: str,
    window: int,
    output_dir: Path,
) -> None:
    """Calculate and plot a rolling metric for all masks of one suffix."""
    series_by_name = {}

    for mask_id in MASK_IDS:
        column = f"mask{mask_id}_{suffix}"
        if metric_name == "moving_average":
            metric = df[column].rolling(window=window).mean()
            ylabel = f"moving average (window={window})"
        elif metric_name == "volatility":
            metric = rolling_volatility(df[column], window=window)
            ylabel = f"volatility (window={window})"
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        series_by_name[column] = metric
        plot_single_series(
            metric,
            title=f"{column}: {metric_name}",
            ylabel=ylabel,
            output_path=output_dir / f"{column}_{metric_name}_w{window}.png",
        )

    plot_combined_series(
        series_by_name,
        title=f"{suffix}: {metric_name} (window={window})",
        ylabel=ylabel,
        output_path=output_dir / f"{suffix}_{metric_name}_w{window}_combined.png",
    )


def main() -> None:
    """Run the masked Brownian data analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)

    for suffix in SUFFIXES:
        plot_metric_for_suffix(
            df,
            suffix=suffix,
            metric_name="moving_average",
            window=MOVING_AVERAGE_WINDOW,
            output_dir=OUTPUT_DIR,
        )
        plot_metric_for_suffix(
            df,
            suffix=suffix,
            metric_name="volatility",
            window=VOLATILITY_WINDOW,
            output_dir=OUTPUT_DIR,
        )


if __name__ == "__main__":
    main()
