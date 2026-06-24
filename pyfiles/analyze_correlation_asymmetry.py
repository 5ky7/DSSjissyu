"""Plot absolute rolling correlations and downside-return event markers."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


DATA_PATH = Path("data/mixed_data_1st/mixed_sabr_masked.csv")
OUTPUT_DIR = Path("plots/mixed_sabr_masked/abs_correlation_asymmetry_w14")
STATS_OUTPUT = Path(
    "data/mixed_data_1st/mixed_sabr_abs_correlation_asymmetry_w14_stats.csv"
)
MASK_IDS = range(1, 6)
ROLLING_WINDOW = 14


def mask_column(mask_id: int, suffix: str) -> str:
    """Return the column name for a mask and suffix."""
    return f"mask{mask_id}_{suffix}"


def event_masks(sp500_returns: pd.Series) -> tuple[float, float, pd.Series, pd.Series]:
    """Return downside thresholds and event masks for lower 1% and 5% returns."""
    q01 = sp500_returns.quantile(0.01)
    q05 = sp500_returns.quantile(0.05)
    bottom_1 = sp500_returns <= q01
    bottom_5 = sp500_returns <= q05
    return q01, q05, bottom_1, bottom_5


def shifted_event_positions(event_mask: pd.Series, shift: int) -> pd.Index:
    """Return event positions shifted by row position, dropping out-of-range events."""
    event_positions = event_mask.to_numpy().nonzero()[0] + shift
    return pd.Index(event_positions[event_positions < len(event_mask.index)])


def shifted_event_indices(event_mask: pd.Series, shift: int) -> pd.Index:
    """Return event indices shifted by row position, dropping out-of-range events."""
    return event_mask.index.take(shifted_event_positions(event_mask, shift))


def summarize_mask(
    mask_id: int,
    sp500_returns: pd.Series,
    rolling_corr: pd.Series,
) -> dict[str, float | int | str]:
    """Summarize whether absolute correlation rises during downside-return events."""
    q01, q05, bottom_1, bottom_5 = event_masks(sp500_returns)
    bottom_1_positions = shifted_event_positions(bottom_1, ROLLING_WINDOW)
    bottom_5_positions = shifted_event_positions(bottom_5, ROLLING_WINDOW)
    shifted_bottom_5 = pd.Series(False, index=sp500_returns.index)
    shifted_bottom_5.iloc[bottom_5_positions] = True

    valid_corr = rolling_corr.notna()
    baseline = (~shifted_bottom_5) & valid_corr
    bottom_1_corr = rolling_corr.iloc[bottom_1_positions].dropna()
    bottom_5_corr = rolling_corr.iloc[bottom_5_positions].dropna()
    baseline_corr = rolling_corr[baseline]

    baseline_mean = baseline_corr.mean()
    bottom_1_mean = bottom_1_corr.mean()
    bottom_5_mean = bottom_5_corr.mean()

    return {
        "mask": f"mask{mask_id}",
        "sp500_q01": q01,
        "sp500_q05": q05,
        "bottom_1_count": int(bottom_1.sum()),
        "bottom_5_count": int(bottom_5.sum()),
        "shifted_bottom_1_count": len(bottom_1_positions),
        "shifted_bottom_5_count": len(bottom_5_positions),
        "rolling_corr_mean_non_shifted_bottom_5": baseline_mean,
        "rolling_corr_mean_shifted_bottom_1": bottom_1_mean,
        "rolling_corr_mean_shifted_bottom_5": bottom_5_mean,
        "uplift_shifted_bottom_1_vs_non_shifted_bottom_5": bottom_1_mean
        - baseline_mean,
        "uplift_shifted_bottom_5_vs_non_shifted_bottom_5": bottom_5_mean
        - baseline_mean,
    }


def plot_mask(
    mask_id: int,
    sp500_returns: pd.Series,
    rolling_corr: pd.Series,
    output_path: Path,
) -> None:
    """Plot absolute rolling correlation with vertical lines for downside events."""
    q01, q05, bottom_1, bottom_5 = event_masks(sp500_returns)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    corr_line = ax.plot(
        rolling_corr.index,
        rolling_corr,
        color="black",
        linewidth=1.1,
        label=f"abs rolling corr (window={ROLLING_WINDOW})",
    )[0]

    for event_index in shifted_event_indices(bottom_5, ROLLING_WINDOW):
        ax.axvline(event_index, color="tab:blue", alpha=0.22, linewidth=0.8)
    for event_index in shifted_event_indices(bottom_1, ROLLING_WINDOW):
        ax.axvline(event_index, color="tab:orange", alpha=0.38, linewidth=0.9)

    ax.set_title(
        f"mask{mask_id}: sp500-DGS10 absolute rolling correlation and downside events "
        f"(events shifted +{ROLLING_WINDOW})"
    )
    ax.set_xlabel("index")
    ax.set_ylabel("absolute correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=[
            corr_line,
            Line2D([0], [0], color="tab:orange", linewidth=1.2),
            Line2D([0], [0], color="tab:blue", linewidth=1.2),
        ],
        labels=[
            f"abs rolling corr (window={ROLLING_WINDOW})",
            f"sp500 bottom 1% shifted +{ROLLING_WINDOW} (<= {q01:.4g})",
            f"sp500 bottom 5% shifted +{ROLLING_WINDOW} (<= {q05:.4g})",
        ],
        loc="best",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_combined_masks(
    plot_data: list[tuple[int, pd.Series, pd.Series]],
    output_path: Path,
) -> None:
    """Plot all mask absolute rolling correlations with downside-event lines in one PNG."""
    fig, axes = plt.subplots(
        nrows=len(plot_data),
        ncols=1,
        figsize=(13, 3.0 * len(plot_data)),
        sharex=True,
        sharey=True,
    )

    for ax, (mask_id, sp500_returns, rolling_corr) in zip(axes, plot_data):
        q01, q05, bottom_1, bottom_5 = event_masks(sp500_returns)
        ax.plot(
            rolling_corr.index,
            rolling_corr,
            color="black",
            linewidth=1.0,
            label=f"abs rolling corr (window={ROLLING_WINDOW})",
        )

        for event_index in shifted_event_indices(bottom_5, ROLLING_WINDOW):
            ax.axvline(event_index, color="tab:blue", alpha=0.20, linewidth=0.8)
        for event_index in shifted_event_indices(bottom_1, ROLLING_WINDOW):
            ax.axvline(event_index, color="tab:orange", alpha=0.36, linewidth=0.9)

        ax.set_title(f"mask{mask_id} (q01 <= {q01:.4g}, q05 <= {q05:.4g})")
        ax.set_ylabel("abs corr")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("index")
    fig.suptitle(
        f"sp500-DGS10 absolute rolling correlation by mask "
        f"(window={ROLLING_WINDOW}, events shifted +{ROLLING_WINDOW})"
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=1.0),
            Line2D([0], [0], color="tab:orange", linewidth=1.2),
            Line2D([0], [0], color="tab:blue", linewidth=1.2),
        ],
        labels=[
            f"abs rolling corr (window={ROLLING_WINDOW})",
            f"sp500 bottom 1% shifted +{ROLLING_WINDOW}",
            f"sp500 bottom 5% shifted +{ROLLING_WINDOW}",
        ],
        loc="lower center",
        ncols=3,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run absolute rolling-correlation plotting and event summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    stats = []
    plot_data = []

    for mask_id in MASK_IDS:
        sp500_col = mask_column(mask_id, "sp500")
        dgs10_col = mask_column(mask_id, "DGS10")
        sp500_returns = df[sp500_col]
        dgs10_returns = df[dgs10_col]
        rolling_corr = sp500_returns.rolling(ROLLING_WINDOW).corr(dgs10_returns).abs()
        plot_data.append((mask_id, sp500_returns, rolling_corr))

        plot_mask(
            mask_id=mask_id,
            sp500_returns=sp500_returns,
            rolling_corr=rolling_corr,
            output_path=OUTPUT_DIR
            / f"mask{mask_id}_sp500_DGS10_abs_corr_w{ROLLING_WINDOW}.png",
        )
        stats.append(summarize_mask(mask_id, sp500_returns, rolling_corr))

    plot_combined_masks(
        plot_data=plot_data,
        output_path=OUTPUT_DIR
        / f"all_masks_sp500_DGS10_abs_corr_w{ROLLING_WINDOW}.png",
    )

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(STATS_OUTPUT, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(stats_df.to_string(index=False))
    print(f"\nSaved plots: {OUTPUT_DIR}")
    print(f"Saved statistics: {STATS_OUTPUT}")


if __name__ == "__main__":
    main()
