"""Plot rolling volatility for the training SP500 and DGS10 data."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/train_sp500_us10y.csv")
OUTPUT_DIR = Path("plots/train_sp500_us10y")
DATE_COLUMN = "Unnamed: 0"
VALUE_COLUMNS = ("sp500", "DGS10")
VOLATILITY_WINDOW = 100


def load_data(path: Path) -> pd.DataFrame:
    """Load the training dataset with a datetime index."""
    df = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    return df.set_index(DATE_COLUMN)


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
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_combined_series(
    series_by_name: dict[str, pd.Series],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot all volatility series in one stacked figure and save it."""
    fig, axes = plt.subplots(
        nrows=len(series_by_name),
        ncols=1,
        figsize=(12, 7),
        sharex=True,
    )
    fig.suptitle(title)

    for ax, (name, series) in zip(axes, series_by_name.items()):
        ax.plot(series.index, series, linewidth=1.0)
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("date")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run volatility plotting for the training dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)
    ylabel = f"volatility (window={VOLATILITY_WINDOW})"
    volatility_by_name = {}

    for column in VALUE_COLUMNS:
        volatility = rolling_volatility(df[column], VOLATILITY_WINDOW)
        volatility_by_name[column] = volatility
        plot_single_series(
            volatility,
            title=f"{column}: volatility (window={VOLATILITY_WINDOW})",
            ylabel=ylabel,
            output_path=OUTPUT_DIR / f"{column}_volatility_w{VOLATILITY_WINDOW}.png",
        )

    plot_combined_series(
        volatility_by_name,
        title=f"train_sp500_us10y: volatility (window={VOLATILITY_WINDOW})",
        ylabel=ylabel,
        output_path=OUTPUT_DIR / f"volatility_w{VOLATILITY_WINDOW}_combined.png",
    )


if __name__ == "__main__":
    main()
