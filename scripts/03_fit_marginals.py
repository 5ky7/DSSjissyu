"""Fit marginal GARCH-family models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.marginal_garch import fit_marginals


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/daily_features.csv")
    parser.add_argument("--comparison-output", default="reports/tables/marginal_model_comparison.csv")
    parser.add_argument("--residual-output", default="data/processed/standardized_residuals.csv")
    parser.add_argument("--version", choices=["A", "B"], default="B")
    return parser.parse_args()


def main() -> None:
    """Fit marginal models."""
    args = parse_args()
    df = pd.read_csv(args.input)
    columns = (
        ["sp500_return", "dgs10_diff"]
        if args.version == "A"
        else ["sp500_return", "bond_return_proxy"]
    )
    comparison, residuals = fit_marginals(df, columns)
    comparison_output = Path(args.comparison_output)
    residual_output = Path(args.residual_output)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    residual_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_output, index=False)
    residuals.to_csv(residual_output, index=False)
    print(f"Wrote {comparison_output} and {residual_output}")


if __name__ == "__main__":
    main()
