"""Fit and simulate an empirical regime-switching model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.regime import fit_empirical_regime_model, simulate_empirical_regime


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/daily_features.csv")
    parser.add_argument("--params-output", default="reports/tables/regime_params.csv")
    parser.add_argument("--transition-output", default="reports/tables/regime_transition_matrix.csv")
    parser.add_argument("--generated-output", default="data/generated/regime/generated_paths.csv")
    parser.add_argument("--version", choices=["A", "B"], default="B")
    parser.add_argument("--path-length", type=int, default=1260)
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument("--rolling-window", type=int, default=252)
    parser.add_argument("--seed", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    """Fit empirical regimes and generate paths."""
    args = parse_args()
    df = pd.read_csv(args.input)
    columns = (
        ["sp500_return", "dgs10_diff"]
        if args.version == "A"
        else ["sp500_return", "bond_return_proxy"]
    )
    fit, transitions, regime_data = fit_empirical_regime_model(
        df,
        columns,
        rolling_window=args.rolling_window,
    )
    generated = simulate_empirical_regime(
        regime_data,
        transitions,
        columns,
        args.path_length,
        args.n_paths,
        args.seed,
    )
    _write_frame(pd.DataFrame([fit.to_dict()]), args.params_output)
    _write_frame(transitions.reset_index(names="from_regime"), args.transition_output)
    _write_frame(generated, args.generated_output)
    print(f"Wrote {args.params_output}, {args.transition_output}, and {args.generated_output}")


def _write_frame(df: pd.DataFrame, output: str) -> None:
    """Write a DataFrame to CSV."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
