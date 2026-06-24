"""Fit and simulate an ADCC model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.adcc import fit_adcc, simulate_adcc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/daily_features.csv")
    parser.add_argument("--residuals", default="data/processed/standardized_residuals.csv")
    parser.add_argument("--params-output", default="reports/tables/adcc_params.csv")
    parser.add_argument("--corr-output", default="data/processed/adcc_correlations.csv")
    parser.add_argument("--generated-output", default="data/generated/adcc/generated_paths.csv")
    parser.add_argument("--path-length", type=int, default=1260)
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main() -> None:
    """Fit ADCC and generate paths."""
    args = parse_args()
    residuals = pd.read_csv(args.residuals)
    columns = residuals[["series", "selected_model"]].drop_duplicates()["series"].tolist()
    pivot = _pivot_residuals(residuals, columns)
    fit, corr = fit_adcc(pivot)

    features = pd.read_csv(args.features)
    means = features[columns].mean().to_numpy()
    volatilities = _pivot_volatilities(residuals, columns).to_numpy()
    generated = simulate_adcc(
        fit,
        means,
        volatilities,
        args.path_length,
        args.n_paths,
        args.seed,
    )
    generated = generated.rename(columns={"x0": columns[0], "x1": columns[1]})

    _write_frame(pd.DataFrame([fit.to_dict()]), args.params_output)
    _write_frame(corr, args.corr_output)
    _write_frame(generated, args.generated_output)
    print(f"Wrote {args.params_output}, {args.corr_output}, and {args.generated_output}")


def _pivot_residuals(residuals: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pivot residual table from long to two-column wide format."""
    residuals = residuals.copy()
    residuals["row"] = residuals.groupby("series").cumcount()
    return residuals.pivot(index="row", columns="series", values="standardized_residual")[columns].dropna()


def _pivot_volatilities(residuals: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pivot volatility table from long to wide format."""
    residuals = residuals.copy()
    residuals["row"] = residuals.groupby("series").cumcount()
    wide = residuals.pivot(index="row", columns="series", values="conditional_volatility")[columns].dropna()
    return wide.replace([np.inf, -np.inf], np.nan).dropna()


def _write_frame(df: pd.DataFrame, output: str) -> None:
    """Write a DataFrame to CSV."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
