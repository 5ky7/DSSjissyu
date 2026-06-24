"""Fit and simulate copula models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.copula import (
    fit_dynamic_gaussian_copula,
    fit_gaussian_copula,
    fit_t_copula,
    simulate_copula,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/daily_features.csv")
    parser.add_argument("--params-output", default="reports/tables/copula_params.csv")
    parser.add_argument("--dynamic-corr-output", default="data/processed/copula_dynamic_correlations.csv")
    parser.add_argument("--generated-output", default="data/generated/copula/generated_paths.csv")
    parser.add_argument("--version", choices=["A", "B"], default="B")
    parser.add_argument("--path-length", type=int, default=1260)
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument("--rolling-window", type=int, default=252)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def main() -> None:
    """Fit copulas and generate paths."""
    args = parse_args()
    df = pd.read_csv(args.input)
    columns = (
        ["sp500_return", "dgs10_diff"]
        if args.version == "A"
        else ["sp500_return", "bond_return_proxy"]
    )
    gaussian = fit_gaussian_copula(df, columns)
    t_copula = fit_t_copula(df, columns)
    dynamic, dynamic_corr = fit_dynamic_gaussian_copula(
        df,
        columns,
        rolling_window=args.rolling_window,
    )
    generated = pd.concat(
        [
            simulate_copula(df, columns, gaussian, args.path_length, args.n_paths, args.seed),
            simulate_copula(df, columns, t_copula, args.path_length, args.n_paths, args.seed + 1),
            simulate_copula(
                df,
                columns,
                dynamic,
                args.path_length,
                args.n_paths,
                args.seed + 2,
                dynamic_correlations=dynamic_corr,
            ),
        ],
        ignore_index=True,
    )

    params = pd.DataFrame([gaussian.to_dict(), t_copula.to_dict(), dynamic.to_dict()])
    _write_frame(params, args.params_output)
    _write_frame(pd.DataFrame({"conditional_corr": dynamic_corr}), args.dynamic_corr_output)
    _write_frame(generated, args.generated_output)
    print(f"Wrote {args.params_output}, {args.dynamic_corr_output}, and {args.generated_output}")


def _write_frame(df: pd.DataFrame, output: str) -> None:
    """Write a DataFrame to CSV."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
