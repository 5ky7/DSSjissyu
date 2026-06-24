"""Plot rolling correlations between sp500 and DGS10 rolling volatilities."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/mixed_data_1st/mixed_sabr_masked.csv")
OUTPUT_DIR = Path("plots/mixed_sabr_masked/volatility_correlation")
MASK_IDS = range(1, 6)
WINDOWS = (5, 15, 20)


def mask_column(mask_id: int, suffix: str) -> str:
    """Return the column name for a mask and suffix."""
    return f"mask{mask_id}_{suffix}"


def volatility_correlation(
    sp500_returns: pd.Series,
    dgs10_returns: pd.Series,
    window: int,
) -> pd.Series:
    """Return rolling correlation between rolling volatilities."""
    sp500_volatility = sp500_returns.rolling(window=window).std()
    dgs10_volatility = dgs10_returns.rolling(window=window).std()
    return sp500_volatility.rolling(window=window).corr(dgs10_volatility)


def plot_window(
    df: pd.DataFrame,
    window: int,
    output_path: Path,
) -> None:
    """Plot all mask volatility correlations for one window in one PNG."""
    fig, axes = plt.subplots(
        nrows=len(MASK_IDS),
        ncols=1,
        figsize=(13, 3.0 * len(MASK_IDS)),
        sharex=True,
        sharey=True,
    )

    for ax, mask_id in zip(axes, MASK_IDS):
        sp500_col = mask_column(mask_id, "sp500")
        dgs10_col = mask_column(mask_id, "DGS10")
        vol_corr = volatility_correlation(
            sp500_returns=df[sp500_col],
            dgs10_returns=df[dgs10_col],
            window=window,
        )

        ax.plot(vol_corr.index, vol_corr, color="black", linewidth=1.0)
        ax.set_title(f"mask{mask_id}")
        ax.set_ylabel("vol corr")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("index")
    fig.suptitle(
        "sp500-DGS10 rolling volatility correlation by mask "
        f"(vol window={window}, corr window={window})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run volatility-correlation plotting for all configured windows."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    for window in WINDOWS:
        output_path = OUTPUT_DIR / f"all_masks_sp500_DGS10_vol_corr_w{window}.png"
        plot_window(df=df, window=window, output_path=output_path)
        print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
