"""Create stylized-fact diagnostics for the real data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.diagnostics.stylized_facts import DiagnosticConfig, stylized_fact_tables


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/daily_features.csv")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--figures-dir", default="reports/figures")
    return parser.parse_args()


def main() -> None:
    """Run diagnostics."""
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, parse_dates=["date"])
    columns = ["sp500_return", "dgs10_diff", "bond_return_proxy"]
    pairs = [
        ("sp500_return", "dgs10_diff"),
        ("sp500_return", "bond_return_proxy"),
    ]
    tables = stylized_fact_tables(df, columns, pairs, DiagnosticConfig())
    for name, table in tables.items():
        table.to_csv(tables_dir / f"real_data_{name}.csv", index=False)
    tables["marginal"].to_csv(tables_dir / "real_data_stylized_facts.csv", index=False)
    _plot_rolling_corr(df, figures_dir)
    print(f"Wrote diagnostics to {tables_dir} and {figures_dir}")


def _plot_rolling_corr(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot rolling correlations for Version A and B."""
    date = df["date"]
    sp500 = df["sp500_return"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for column, label in (
        ("dgs10_diff", "SP500 vs DGS10 diff"),
        ("bond_return_proxy", "SP500 vs bond return proxy"),
    ):
        corr = sp500.rolling(252).corr(df[column])
        ax.plot(date, corr, linewidth=1.0, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Real data rolling correlation (window=252)")
    ax.set_xlabel("date")
    ax.set_ylabel("correlation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "real_data_rolling_corr.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
