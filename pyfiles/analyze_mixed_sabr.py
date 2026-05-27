"""Plot rolling volatility for the masked SABR dataset."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/mixed_data_1st/mixed_sabr_masked.csv")
OUTPUT_DIR = Path("plots/mixed_sabr_masked")
MASK_IDS = range(1, 6)
SUFFIXES = ("sp500", "DGS10")
VOLATILITY_WINDOW = 100


def load_data(path: Path) -> pd.DataFrame:
    """Load the masked SABR dataset."""
    return pd.read_csv(path)


def mask_columns(suffix: str) -> list[str]:
    """Return mask columns for the given suffix in mask order."""
    return [f"mask{mask_id}_{suffix}" for mask_id in MASK_IDS]


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
    """Plot volatility series in one stacked figure and save it."""
    fig_height = max(4, 2.4 * len(series_by_name))
    fig, axes = plt.subplots(
        nrows=len(series_by_name),
        ncols=1,
        figsize=(12, fig_height),
        sharex=True,
    )
    fig.suptitle(title)

    if len(series_by_name) == 1:
        axes = [axes]

    for ax, (name, series) in zip(axes, series_by_name.items()):
        ax.plot(series.index, series, linewidth=1.0)
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run volatility plotting for the masked SABR dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)
    ylabel = f"volatility (window={VOLATILITY_WINDOW})"
    all_volatility_by_name = {}

    for suffix in SUFFIXES:
        suffix_volatility_by_name = {}

        for column in mask_columns(suffix):
            volatility = rolling_volatility(df[column], VOLATILITY_WINDOW)
            suffix_volatility_by_name[column] = volatility
            all_volatility_by_name[column] = volatility

            plot_single_series(
                volatility,
                title=f"{column}: volatility (window={VOLATILITY_WINDOW})",
                ylabel=ylabel,
                output_path=OUTPUT_DIR / f"{column}_volatility_w{VOLATILITY_WINDOW}.png",
            )

        plot_combined_series(
            suffix_volatility_by_name,
            title=f"mixed_sabr_masked {suffix}: volatility (window={VOLATILITY_WINDOW})",
            ylabel=ylabel,
            output_path=OUTPUT_DIR / f"{suffix}_volatility_w{VOLATILITY_WINDOW}_combined.png",
        )

    plot_combined_series(
        all_volatility_by_name,
        title=f"mixed_sabr_masked: volatility (window={VOLATILITY_WINDOW})",
        ylabel=ylabel,
        output_path=OUTPUT_DIR / f"volatility_w{VOLATILITY_WINDOW}_combined.png",
    )


if __name__ == "__main__":
    main()
